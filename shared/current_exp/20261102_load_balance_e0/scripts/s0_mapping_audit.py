#!/usr/bin/env python3
"""S0: offline mapping, slot-rank, count and conservation audit."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch

EP=16; BASE=256; EXTRA=16; PHYS=BASE+EXTRA; PER_RANK=PHYS//EP

def interleaved_mapping(mode: str):
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
            elif mode=='multi3_top2':
                table={49:{0:43,1:40},36:{0:178,1:82},54:{0:205,1:126}}
                extra=table.get(layer,{}).get(r,base)
            slots.append(extra)
        layers.append(slots)
    return np.asarray(layers,dtype=np.int64)

def rank_of(slot):
    return int(slot)//PER_RANK

def count_rank_load(counts, pmap):
    # counts: [78,256], pmap: [78,272], equal split across duplicate slots
    logical=counts[49].astype(np.int64)
    rank_load=np.zeros(EP,dtype=np.int64)
    for r in range(EP):
        for slot in pmap[49, r*PER_RANK:(r+1)*PER_RANK]:
            e=int(slot); n=sum(1 for x in pmap[49] if int(x)==e)
            share=logical[e]//n + (1 if logical[e]%n and (slot == np.where(pmap[49]==e)[0][0]) else 0)
            # deterministic remainder: put remainder on first physical occurrence
            first=np.where(pmap[49]==e)[0][0]
            share=logical[e]//n + (logical[e]%n if slot==first else 0)
            rank_load[r]+=share
    return rank_load

def validator_traps(counts, expected_tasks):
    # counts: [78,256], MoE layers only have 8N tokens and top-8 entries.
    total=int(counts[3:].sum())
    traps={}
    traps['baseline']={'total':total,'ok':total==expected_tasks}
    dup=counts.copy(); dup[49,43]+=1
    traps['duplicate_task']={'total':int(dup[3:].sum()),'ok':int(dup[3:].sum())==expected_tasks}
    drop=counts.copy(); drop[49,43]-=1
    traps['missing_task']={'total':int(drop[3:].sum()),'ok':int(drop[3:].sum())==expected_tasks}
    changed=counts.copy(); changed[49,43]-=1; changed[49,40]+=1
    traps['wrong_expert']={'total':int(changed[3:].sum()),'ok':int(changed[3:].sum())==expected_tasks and np.array_equal(changed,counts)}
    return traps

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--route-dir',required=True); ap.add_argument('--old-name',default='e1_layer49_top2.json'); a=ap.parse_args()
    root=Path('/home/chengyetai/workspace/current_exp/20261102_load_balance_e0')
    old=json.load(open(root/'mappings'/a.old_name))['physical_to_logical_map']
    old=np.asarray(old,dtype=np.int64)
    corrected=interleaved_mapping('layer49_top2')
    (root/'mappings_s0_corrected').mkdir(exist_ok=True)
    for mode in ('adapter','layer49_top2','multi3_top2'):
        json.dump({'physical_to_logical_map':interleaved_mapping(mode).tolist()},open(root/'mappings_s0_corrected'/f's0_{mode}.json','w'),separators=(',',':'))
    dumps=sorted(Path(a.route_dir).glob('00000*.pt'))
    traces={}
    for d in dumps:
        x=torch.load(d,map_location='cpu',weights_only=False)
        traces[d.name]={'prefill':x['logical_count_prefill'].sum(0).numpy().astype(np.int64),'decode':x['logical_count_decode'].sum(0).numpy().astype(np.int64)}
    trace_name='000000_1793579172.815737.pt'
    counts=traces[trace_name]['prefill']
    result={'ep_size':EP,'slots_per_rank':PER_RANK,'old_mapping':{},'corrected_mapping':{},'traces':{},'traps':validator_traps(counts,4096*75*8)}
    for label,m in [('old',old),('corrected',corrected)]:
        place={str(e):[int(i) for i in np.where(m[49]==e)[0]] for e in (43,40)}
        result[f'{label}_mapping']={'expert43_slots':place['43'],'expert40_slots':place['40'],'expert43_ranks':[rank_of(x) for x in place['43']],'expert40_ranks':[rank_of(x) for x in place['40']],'rank_slot_counts':[PER_RANK]*EP,'layer49_rank_load_cal':count_rank_load(counts,m).tolist()}
        result[f'{label}_mapping']['layer49_max_over_mean_cal']=float(max(result[f'{label}_mapping']['layer49_rank_load_cal'])/np.mean(result[f'{label}_mapping']['layer49_rank_load_cal']))
        hold=traces['000001_1793579178.2768657.pt']['prefill']
        result[f'{label}_mapping']['layer49_rank_load_hold']=count_rank_load(hold,m).tolist()
        result[f'{label}_mapping']['layer49_max_over_mean_hold']=float(max(result[f'{label}_mapping']['layer49_rank_load_hold'])/np.mean(result[f'{label}_mapping']['layer49_rank_load_hold']))
    for name,t in traces.items():
        result['traces'][name]={'prefill_total':int(t['prefill'][3:].sum()),'decode_total':int(t['decode'][3:].sum()),'prefill_conserved':int(t['prefill'][3:].sum())==4096*75*8}
    # full 256 expert load table for layer49 calibration/holdout
    with open(root/'results'/'s0_layer49_expert_loads.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['expert','calibration_count','holdout_count'])
        hold=traces['000001_1793579178.2768657.pt']['prefill']
        for e in range(256): w.writerow([e,int(counts[49,e]),int(hold[49,e])])
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'out':a.out,'old':result['old_mapping'],'corrected':result['corrected_mapping'],'traces':result['traces'],'traps':result['traps']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
