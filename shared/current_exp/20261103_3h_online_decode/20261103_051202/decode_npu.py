#!/usr/bin/env python3
"""Round-2 decode six-cell local-chain proxy with fixed artificial routes."""
from __future__ import annotations
import csv, hashlib, json, math, platform, statistics, sys, time
from collections import Counter
from pathlib import Path
import torch, torch_npu
sys.path.insert(0,'/old/scripts')
from s2_local_cost import ExpertBank, load_experts, make_case, sync  # noqa: E402

OUT=Path('/exp'); WARM=20; BLOCKS=5; TARGET_MS=250.0
SEED=20261103
EPS=(16,32,64)
EXPERT_IDS=[43,40,192,67,82,172,112,79,36,41,106,69,132,63,191,109]


def event_ms(fn,calls):
    sync(); s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True)
    host=time.perf_counter(); s.record()
    for _ in range(calls): fn()
    e.record(); e.synchronize(); total=s.elapsed_time(e)
    return total,total/calls,(time.perf_counter()-host)*1000/calls


def calibrate(fn):
    for _ in range(WARM): fn()
    sync(); _,per,_=event_ms(fn,1)
    calls=max(20,math.ceil(TARGET_MS/max(per,1e-6)))
    # Iterate on a real block, with headroom for run-to-run frequency changes.
    for _ in range(4):
        observed,_,_=event_ms(fn,calls)
        if observed>=TARGET_MS: break
        calls=max(calls,math.ceil(calls*TARGET_MS/max(observed,1e-6)*1.2))
    return calls


def make_routes():
    g=torch.Generator().manual_seed(SEED)
    tokens=[]
    for _ in range(80):
        ids=torch.randperm(256,generator=g)[:8].sort().values.tolist()
        assert len(set(ids))==8
        tokens.append(ids)
    return tokens


def rank_vectors(tokens,ep):
    per=256//ep; vectors=[]
    for r in range(ep):
        c=[0]*per
        for token in tokens:
            for expert in token:
                if r*per<=expert<(r+1)*per: c[expert-r*per]+=1
        vectors.append(tuple(c))
    return vectors


def step_sets():
    out=[]
    for name,indices in [('dev',range(0,4)),('holdout',range(4,8))]:
        for i in indices:
            out.append({'batch':1,'split':name,'sample':i,'token_indices':[int(i)],'identity':'fixed_seed_artificial_batch1','recombined':False})
    for name,groups in [('dev',[range(8+8*i,16+8*i) for i in range(4)]),('holdout',[range(40+8*i,48+8*i) for i in range(4)])]:
        for i,tokens in enumerate(groups):
            out.append({'batch':8,'split':name,'sample':i,'token_indices':list(tokens),'identity':'fixed_seed_artificial_batch8_offline_recombined','recombined':True})
    return out


def worktype(v): return tuple(sorted(x for x in v if x))
def active(v): return sum(x>0 for x in v)
def r_all(vectors):
    total=sum(map(sum,vectors))
    return max(map(sum,vectors))/(total/len(vectors)) if total else 0.0

def r_active(vectors):
    loads=[sum(v) for v in vectors if sum(v)>0]
    return max(loads)/(sum(loads)/len(loads)) if loads else 0.0


def main():
    torch.npu.set_device(0)
    if torch.npu.device_count()!=1: raise RuntimeError(torch.npu.device_count())
    sync(); free0,total_hbm=torch.npu.mem_get_info(); torch.npu.reset_peak_memory_stats()
    raw=load_experts(Path('/models/glm-q'),EXPERT_IDS)
    banks={16:ExpertBank('ep16',EXPERT_IDS,raw),32:ExpertBank('ep32',EXPERT_IDS[:8],raw),64:ExpertBank('ep64',EXPERT_IDS[:4],raw)}
    routes=make_routes(); sets=step_sets()

    plans={'identity':'fixed seed 20261103; top8 unique logical experts per token; no replicas; contiguous logical expert to rank mapping',
           'eps':list(EPS),'route_tokens':routes,'steps':{},'real_trace_contract':{
             'status':'not usable for batch1/8 token reconstruction',
             'reason':'Real layer49 decode dump has 128 tasks per step and 33-44 nonzero experts; consistent with batch16 x top8, not TP16 duplication. Only expert histograms are stored, so token top8 membership and router weights cannot be reconstructed.',
             'per_step_total':128}}
    # Construct all steps and unique work types.
    work_cases={}
    for ep in EPS:
        bank=banks[ep]
        for s in sets:
            key=(ep,s['batch'],s['sample'],s['split'])
            vectors=rank_vectors([routes[i] for i in s['token_indices']],ep)
            plans['steps'][str(key)]={'ep':ep,'batch':s['batch'],'split':s['split'],'sample':s['sample'],
              'token_indices':s['token_indices'],'identity':s['identity'],'offline_recombined':s['recombined'],
              'rank_vectors':[list(v) for v in vectors],'active_ranks':sum(active(v)>0 for v in vectors),
              'R_all_ranks':r_all(vectors),'R_active_ranks':r_active(vectors),'total_tasks':sum(map(sum,vectors)),
              'work_types':[list(worktype(v)) for v in vectors]}
            for idx,v in enumerate(vectors):
                wt=worktype(v)
                if wt: work_cases.setdefault((ep,wt),{'rank_index':idx,'counts':list(v)})

    # Calibrate and measure each (physical slot count, active-count multiset) once.
    functions={}; labels=[]
    for key,meta in work_cases.items():
        ep,wt=key; bank=banks[ep]; counts=list(meta['counts'])
        case=make_case(bank,counts,f'ep{ep}_wt{wt}','decode_worktype')
        functions[key]=lambda b=bank,c=case:b.chain(c); labels.append(key)
    calls={k:calibrate(fn) for k,fn in functions.items()}
    records={k:[] for k in functions}; csv_rows=[]
    for block in range(BLOCKS):
        order=list(labels)
        if block%2: order=list(reversed(order))
        for key in order:
            fn=functions[key]; n=calls[key]
            total,per,host=event_ms(fn,n)
            rec={'ep':key[0],'work_type':json.dumps(list(key[1])),'block':block,'calls':n,'total_ms':total,'ms_per_call':per,'host_ms_per_call':host}
            records[key].append(rec); csv_rows.append(rec)
    timing={}
    for key,recs in records.items():
        vals=[r['ms_per_call'] for r in recs]
        timing[key]={'median_ms':statistics.median(vals),'min_ms':min(vals),'max_ms':max(vals),
                     'calls_per_block':calls[key],'block_total_ms':[r['total_ms'] for r in recs],
                     'all_blocks_ge_100ms':all(r['total_ms']>=100 for r in recs)}

    # Validate that expert-slot permutation within one physical bank does not change work-type time.
    permutation_validation=[]
    for ep in EPS:
        keys=[k for k in timing if k[0]==ep and len(k[1])>=2]
        if not keys: continue
        key=max(keys,key=lambda k:sum(k[1]))
        counts=list(work_cases[key]['counts']); non=[x for x in counts if x]
        per=256//ep
        permuted=[0]*per
        for i,x in enumerate(reversed(non)): permuted[i]=x
        c1=make_case(banks[ep],counts,f'perm_a_ep{ep}','permutation'); c2=make_case(banks[ep],permuted,f'perm_b_ep{ep}','permutation')
        f1=lambda c=c1,b=banks[ep]:b.chain(c); f2=lambda c=c2,b=banks[ep]:b.chain(c)
        n1,n2=calibrate(f1),calibrate(f2); vals1=[]; vals2=[]
        for block in range(BLOCKS):
            order=[(1,f1,n1,vals1),(2,f2,n2,vals2)]
            if block%2: order=list(reversed(order))
            for _,fn,n,vals in order:
                _,per,_=event_ms(fn,n); vals.append(per)
        m1,m2=statistics.median(vals1),statistics.median(vals2)
        permutation_validation.append({'ep':ep,'work_type':list(key[1]),'layout_a_ms':m1,'layout_b_ms':m2,
          'relative_difference':abs(m1-m2)/max(m1,m2),'pass_at_5pct':abs(m1-m2)/max(m1,m2)<=0.05})

    # Reconstruct every step from measured work types.
    grid=[]
    for ep in EPS:
        for batch in (1,8):
            totals=[]
            for s in sets:
                if s['batch']!=batch: continue
                key=(ep,batch,s['sample'],s['split'])
                rec=plans['steps'][str(key)]
                fs=[timing[(ep,tuple(wt))]['median_ms'] if wt else 0.0 for wt in rec['work_types']]
                rec['local_chain_ms_by_rank']=fs
                rec['max_local_chain_ms']=max(fs)
                rec['sum_local_chain_ms']=sum(fs)
                totals.append(rec)
            grid.append({'ep':ep,'batch':batch,'sample_count':len(totals),
              'median_compute_proxy_ms':statistics.median([x['max_local_chain_ms'] for x in totals]),
              'min_compute_proxy_ms':min(x['max_local_chain_ms'] for x in totals),
              'max_compute_proxy_ms':max(x['max_local_chain_ms'] for x in totals),
              'active_ranks':[x['active_ranks'] for x in totals],
              'R_all_ranks':[x['R_all_ranks'] for x in totals],
              'R_active_ranks':[x['R_active_ranks'] for x in totals],
              'dev_median_compute_proxy_ms':statistics.median([x['max_local_chain_ms'] for x in totals if x['split']=='dev']),
              'holdout_median_compute_proxy_ms':statistics.median([x['max_local_chain_ms'] for x in totals if x['split']=='holdout'])})
    # Pair same logical sample indices across EP; per-step deltas then aggregate.
    bykey={(g['ep'],g['batch']):g for g in grid}
    for g in grid:
        base=bykey[(16,g['batch'])]
        steps=[]
        for s in sets:
            if s['batch']!=g['batch']: continue
            k=(g['ep'],g['batch'],s['sample'],s['split']); k16=(16,g['batch'],s['sample'],s['split'])
            d=plans['steps'][str(k16)]['max_local_chain_ms']-plans['steps'][str(k)]['max_local_chain_ms']
            steps.append({'split':s['split'],'sample':s['sample'],'ep16_ms':plans['steps'][str(k16)]['max_local_chain_ms'],'candidate_ms':plans['steps'][str(k)]['max_local_chain_ms'],'delta_ms':d})
        deltas=[x['delta_ms'] for x in steps]
        g['paired_deltas_vs_ep16']=steps
        g['median_delta_ms']=statistics.median(deltas)
        g['min_delta_ms']=min(deltas); g['max_delta_ms']=max(deltas)
        g['communication_budget_ms']=max(0.0,statistics.median(deltas))
        g['relative_compute_saving_vs_ep16']=statistics.median(deltas)/max(bykey[(16,g['batch'])]['median_compute_proxy_ms'],1e-12)
    sync(); free1,_=torch.npu.mem_get_info()
    result={'scope':'fixed-seed artificial batch1/8 routes; representative resident expert weights; eager local chain; no network',
      'status':'COMPLETE' if all(v['all_blocks_ge_100ms'] for v in timing.values()) and all(v['pass_at_5pct'] for v in permutation_validation) else 'PARTIAL',
      'identity':'model_estimate_from_single_npu_measured_work_types',
      'plans':plans,'grid':grid,'work_type_timing':{f'{k[0]}:{list(k[1])}':v for k,v in timing.items()},
      'permutation_validation':permutation_validation,
      'environment':{'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device_count':torch.npu.device_count(),'device_name':torch.npu.get_device_name(0)},
      'memory':{'free_before':int(free0),'free_after':int(free1),'total':int(total_hbm),'peak_allocated':int(torch.npu.max_memory_allocated()),'peak_reserved':int(torch.npu.max_memory_reserved())},
      'unmeasured':['real token-level decode top-k','router weights','actual dispatch source ranks','HCCL payload/time','production graph path','EP64 TPOT']}
    (OUT/'DECODE_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'TIMINGS_DECODE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    print(json.dumps({'status':result['status'],'grid':grid,'worktypes':len(timing),'permutation':permutation_validation,'memory':result['memory']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
