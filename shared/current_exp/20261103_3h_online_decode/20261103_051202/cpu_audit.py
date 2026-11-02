#!/usr/bin/env python3
"""Round-2 CPU audit: physical-slot remainder conservation and quota checks."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/chengyetai/workspace/current_exp/20261102_load_balance_e0')
OUT=Path(__file__).resolve().parent
EP=16; PER_RANK=17


def mapping(mode: str):
    layers=[]
    for layer in range(78):
        slots=[]
        for r in range(EP):
            base=r*16
            slots.extend(range(base,base+16))
            extra=base
            if mode=='layer49_top2' and layer==49:
                if r==0: extra=43
                elif r==1: extra=40
            slots.append(extra)
        layers.append(slots)
    return np.asarray(layers,dtype=np.int64)


def old_mapping():
    return np.asarray(list(range(256))+[43,40]+[r*16 for r in range(2,16)],dtype=np.int64)


def allocate_by_physical_index(pmap, counts):
    """Distribute every expert count over physical occurrences, remainder by physical order."""
    vectors=[[] for _ in range(EP)]
    by_expert=np.zeros(256,dtype=np.int64)
    for physical_index, logical in enumerate(pmap.tolist()):
        rank=physical_index//PER_RANK
        vector_index=physical_index%PER_RANK
        occurrences=np.flatnonzero(pmap==logical)
        order=int(np.flatnonzero(occurrences==physical_index)[0])
        share=int(counts[logical])//len(occurrences) + (int(counts[logical])%len(occurrences) if order==0 else 0)
        vectors[rank].append(share)
        by_expert[logical]+=share
    assert all(len(v)==PER_RANK for v in vectors)
    assert np.array_equal(by_expert,np.asarray([counts[e] for e in range(256)],dtype=np.int64))
    assert sum(map(sum,vectors))==sum(counts.values())
    return vectors, by_expert.tolist()


def build_plan_vectors(pmap, counts, q43, q40):
    """Build candidate residency: all non-target extra slots disabled.

    e43/e40 are split between rank2 primary and rank0/rank1 extra slots.
    Every other expert stays on its primary slot only, so an empty adapter
    has zero tasks in all extra slots while retaining the same buffers.
    """
    logicals=pmap.tolist()
    owner43=next(i for i,e in enumerate(logicals) if e==43 and i%PER_RANK<16)
    owner40=next(i for i,e in enumerate(logicals) if e==40 and i%PER_RANK<16)
    r43,i43=divmod(owner43,PER_RANK); r40,i40=divmod(owner40,PER_RANK)
    vectors=[[0]*PER_RANK for _ in range(EP)]
    for rank in range(EP):
        for idx in range(16):
            e=logicals[rank*PER_RANK+idx]
            if e not in (43,40):
                vectors[rank][idx]=int(counts[e])
    vectors[r43][i43]=int(counts[43])-q43
    vectors[r40][i40]=int(counts[40])-q40
    vectors[0][16]=q43
    vectors[1][16]=q40
    by_expert=[0]*256
    for rank in range(EP):
        for idx,e in enumerate(logicals[rank*PER_RANK:(rank+1)*PER_RANK]):
            by_expert[e]+=vectors[rank][idx]
    expected=[int(counts[e]) for e in range(256)]
    assert by_expert==expected, 'expert task conservation failed'
    assert sum(map(sum,vectors))==sum(counts.values()), 'total task conservation failed'
    return vectors, by_expert


def exhaustive_row_quota(base_vectors, n43, n40):
    loads=np.asarray([sum(v) for v in base_vectors],dtype=np.int64)
    b0,b1,b2=int(loads[0]),int(loads[1]),int(loads[2])
    floor=int(max(loads[3:].tolist()))
    best=None
    x40=np.arange(n40+1,dtype=np.int64)
    for x43 in range(n43+1):
        obj=np.maximum.reduce([
            np.full_like(x40,floor),
            np.full_like(x40,b0+x43),
            b1+x40,
            b2-x43-x40,
        ])
        idx=int(np.argmin(obj)); value=int(obj[idx])
        candidate=(value,x43+int(x40[idx]),x43,int(x40[idx]))
        if best is None or candidate<best:
            best=candidate
    return {'max_rows':best[0],'q43':best[2],'q40':best[3],
            'affected_loads_after':[b0+best[2],b1+best[3],b2-best[2]-best[3]],
            'unaffected_floor':floor,'loads_before':loads.tolist()}


def main():
    counts_rows=list(csv.DictReader(open(ROOT/'results/s0_layer49_expert_loads.csv')))
    splits={
      'calibration':{int(r['expert']):int(r['calibration_count']) for r in counts_rows},
      'holdout':{int(r['expert']):int(r['holdout_count']) for r in counts_rows},
    }
    result={'scope':'CPU-only minimal corrections; no NPU rerun','splits':{}}
    for split,counts in splits.items():
        corr=mapping('layer49_top2')[49]
        old=old_mapping()
        corr_vec,corr_by=allocate_by_physical_index(corr,counts)
        old_vec,old_by=allocate_by_physical_index(old,counts)
        # Empty adapter: candidate residency, every 17th slot exactly zero.
        empty,_=build_plan_vectors(corr,counts,0,0)
        # Independent exhaustive integer enumeration for corrected row minimax.
        base,_=build_plan_vectors(corr,counts,0,0)
        quota=exhaustive_row_quota(base,counts[43],counts[40])
        # Candidate plan and conservation.
        candidate,cand_by=build_plan_vectors(corr,counts,quota['q43'],quota['q40'])
        result['splits'][split]={
          'corrected_rank_loads':[sum(v) for v in corr_vec],
          'old_rank_loads':[sum(v) for v in old_vec],
          'corrected_total':int(sum(map(sum,corr_vec))),
          'old_total':int(sum(map(sum,old_vec))),
          'expected_total':int(sum(counts.values())),
          'corrected_by_expert_conserved':corr_by==[counts[e] for e in range(256)],
          'old_by_expert_conserved':old_by==[counts[e] for e in range(256)],
          'empty_adapter_extra_slots_zero':all(v[16]==0 for v in empty),
          'empty_adapter_owner_primary_only':empty[2][next(i for i,e in enumerate(corr.tolist()) if e==43 and i%PER_RANK<16)%PER_RANK]==counts[43],
          'row_quota_exhaustive':quota,
          'candidate_conserved':cand_by==[counts[e] for e in range(256)],
          'candidate_extra_slots':[v[16] for v in candidate],
          'candidate_rank_loads':[sum(v) for v in candidate],
        }
    # Decode per-step audit from the specified real dump.
    import torch
    dump=ROOT/'logs/20261102_081322/eplb_windows/000000_1793579172.815737.pt'
    x=torch.load(dump,map_location='cpu',weights_only=False)
    d=x['logical_count_decode'][:,49,:].to(torch.int64)
    totals=d.sum(1).tolist(); nz=(d>0).sum(1).tolist(); maxima=d.max(1).values.tolist()
    result['decode_trace_audit']={
      'path':str(dump),'shape':list(d.shape),'steps':int(d.shape[0]),
      'per_step_task_totals':totals,'per_step_nonzero_experts':nz,'per_step_max_expert':maxima,
      'unique_task_totals':sorted(set(totals)),'unique_nonzero_expert_counts':sorted(set(nz)),
      'all_totals_128':all(v==128 for v in totals),
      'all_counts_share_common_factor_16':all(all(v%16==0 for v in row.tolist()) for row in d),
      'tp16_repetition_signature': 'rejected' if not all(all(v%16==0 for v in row.tolist()) for row in d) else 'possible',
      'inferred_effective_batch_if_top8': 128/8,
      'contract':'Each layer49 decode step sums to 128 tasks and has 33-50 nonzero experts; this is consistent with 16 tokens x top8, not one token duplicated by TP16.',
    }
    (OUT/'CPU_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
