#!/usr/bin/env python3
"""Trace-driven 75-layer MoE online-copy simulation.

Measured local chain curve comes from the previous single-NPU S2 run.  Network
and dispatch parameters are explicitly swept, not claimed as measurements.
"""
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path
import numpy as np
import torch

TRACE=Path('/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/logs/20261102_081322/eplb_windows/000000_1793579172.815737.pt')
S2=Path('/home/chengyetai/workspace/Share/shared/current_exp/20261102_load_balance_e0/evidence/s0_s4_p2/S2_LOCAL_COST.json')
OUT=Path(__file__).resolve().parent
EXPERT_BYTES=37_769_216
ACTIVE_OVERHEAD_MS=0.020
LAYER_IDS=list(range(3,78))

s2=json.loads(S2.read_text())
curve=[(int(c['rows']),float(c['timing']['device_ms_median'])) for c in s2['shape_curve'] if 'timing' in c]
xs=np.array([x for x,_ in curve],dtype=float); ys=np.array([y for _,y in curve],dtype=float)
last_slope=(ys[-1]-ys[-2])/(xs[-1]-xs[-2])

def single_time(rows):
    if rows<=0:return 0.0
    if rows<=xs[-1]: return float(np.interp(rows,xs,ys))
    return float(ys[-1]+(rows-xs[-1])*last_slope)

def rank_time(counts):
    total=sum(counts); active=sum(x>0 for x in counts)
    return single_time(total)+ACTIVE_OVERHEAD_MS*max(0,active-1)

def rank_counts(counts,ep):
    per=256//ep
    return [tuple(int(x) for x in counts[r*per:(r+1)*per]) for r in range(ep)]

def allocate_expert(vectors, owner, target, expert, count):
    # vectors include an extra destination slot at the end for each rank.
    moved=count
    # Objective is max of owner/target loads. Search the small convex region.
    owner_load=sum(vectors[owner]); target_load=sum(vectors[target])
    base=owner_load-target_load
    centre=int(round((count+base)/2))
    candidates=set()
    for x in range(max(0,centre-3),min(count,centre+3)+1): candidates.add(x)
    candidates.update((0,count))
    best=None
    for x in candidates:
        ol=owner_load-x; tl=target_load+x
        # Owner expert group remains active unless all tasks leave.
        ov=list(vectors[owner]); tv=list(vectors[target])
        ov[expert-owner*(len(ov)-1)]=count-x
        tv[-1]=x
        score=max(rank_time(ov),rank_time(tv))
        if best is None or score<best[0]: best=(score,x,ov,tv)
    vectors[owner]=best[2]; vectors[target]=best[3]
    return count-best[1]

def online_layer(counts,max_copies):
    vectors=[list(v)+[0] for v in rank_counts(counts,16)]
    temp_used=set(); copies=[]; moved=0
    # Candidate experts are current hot experts. Spend copy budget only on
    # experts whose owner rank is above the current mean rank load.
    candidates=sorted(range(256),key=lambda e:int(counts[e]),reverse=True)
    for expert in candidates:
        if len(copies)>=max_copies: break
        c=int(counts[expert])
        if c<=0: break
        owner=expert//16
        mean=sum(sum(v) for v in vectors)/len(vectors)
        if sum(vectors[owner])<=mean: continue
        targets=[r for r in range(16) if r!=owner and r not in temp_used]
        if not targets: break
        target=min(targets,key=lambda r:sum(vectors[r]))
        moved_here=allocate_expert(vectors,owner,target,expert,c)
        if moved_here<=0: continue
        temp_used.add(target); copies.append({'expert':expert,'source':owner,'target':target,'moved':moved_here})
        moved+=moved_here
    return vectors,copies,moved

def summarize_ep(hist,steps,ep):
    rows=[]
    for step in range(steps):
        maxes=[];Rs=[];active=[]
        for layer in LAYER_IDS:
            vec=rank_counts(hist[step,layer],ep)
            times=[rank_time(v) for v in vec]
            maxes.append(max(times))
            loads=[sum(v) for v in vec]; Rs.append(max(loads)/(sum(loads)/ep))
            active.append(sum(sum(v)>0 for v in vec))
        rows.append({'step':step,'moe_critical_proxy_ms':sum(maxes),'R':Rs,'active_ranks':active})
    return {'ep':ep,'steps':len(rows),'median_moe_critical_proxy_ms':statistics.median(r['moe_critical_proxy_ms'] for r in rows),
            'min_moe_critical_proxy_ms':min(r['moe_critical_proxy_ms'] for r in rows),
            'max_moe_critical_proxy_ms':max(r['moe_critical_proxy_ms'] for r in rows),
            'median_R':statistics.median(v for r in rows for v in r['R']),
            'max_R':max(v for r in rows for v in r['R']),
            'per_step':rows}

def simulate_online(hist,steps,max_copies):
    per_step=[]
    for step in range(steps):
        base=[]; cand=[]; copyn=0; moved=0
        for layer in LAYER_IDS:
            b=rank_counts(hist[step,layer],16)
            bt=[rank_time(v) for v in b]
            av,cp,mv=online_layer(hist[step,layer],max_copies)
            at=[rank_time(v) for v in av]
            base.append(max(bt));cand.append(max(at));copyn+=len(cp);moved+=mv
        per_step.append({'step':step,'baseline_ms':sum(base),'online_compute_ms':sum(cand),
                         'compute_saving_ms':sum(base)-sum(cand),'copy_count':copyn,
                         'copy_bytes':copyn*EXPERT_BYTES,'moved_tasks':moved})
    return per_step

def sweep(per_step):
    rows=[]
    for dispatch_us in (0,0.1,0.5,1.0,2.0):
        ratios=[]; nets=[]
        for x in per_step:
            if x['copy_count']<=0: continue
            net=x['compute_saving_ms']-x['moved_tasks']*dispatch_us/1000.0
            nets.append(net); ratios.append(net/x['copy_count']*1000.0)
        per_copy_us=statistics.median(ratios) if ratios else None
        rows.append({'dispatch_us_per_moved_task':dispatch_us,
                     'median_net_ms_per_step':statistics.median(nets) if nets else None,
                     'break_even_remote_copy_us_per_expert':per_copy_us,
                     'positive_step_fraction':sum(v>0 for v in nets)/len(nets) if nets else 0.0})
    return {'median_compute_saving_ms_per_step':statistics.median(x['compute_saving_ms'] for x in per_step),
            'median_copy_count_per_step':statistics.median(x['copy_count'] for x in per_step),
            'median_moved_tasks_per_step':statistics.median(x['moved_tasks'] for x in per_step),'sweep':rows}

def main():
    x=torch.load(TRACE,map_location='cpu',weights_only=False)
    result={'method':'trace-driven full 75-layer current-count temp-copy simulation',
      'trace':str(TRACE),'layers':LAYER_IDS,'expert_state_bytes':EXPERT_BYTES,
      'cost_model':{'source':'NPU-measured S2 single-expert curve','single_expert_curve':curve,
                    'active_overhead_ms_per_extra_expert':ACTIVE_OVERHEAD_MS,
                    'formula':'single_time(total_rows)+0.020ms*(active_experts-1)',
                    'limitations':['eager local chain','no HCCL','no production graph','representative expert weights']},
      'baseline':{},'online_ep16':{}}
    for phase,steps in [('prefill',4),('decode',64)]:
        hist=x[f'logical_count_{phase}'].to(torch.int64)
        result['baseline'][phase]={str(ep):summarize_ep(hist,steps,ep) for ep in (16,32,64)}
        result['online_ep16'][phase]={}
        for k in (1,2,4,8,16):
            ps=simulate_online(hist,steps,k)
            result['online_ep16'][phase][str(k)]={'per_step':ps,'sweep':sweep(ps)}
    (OUT/'SIMULATION_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for phase in ('prefill','decode'):
        for ep,v in result['baseline'][phase].items():
            rows.append({'kind':'baseline','phase':phase,'ep':ep,'copies_per_layer':0,'median_moe_ms':v['median_moe_critical_proxy_ms'],'median_saving_ms':0,'median_copy_count':0,'median_copy_bytes':0,'median_moved_tasks':0,'break_even_copy_us':''})
        base=result['baseline'][phase]['16']['median_moe_critical_proxy_ms']
        for k,v in result['online_ep16'][phase].items():
            sw=v['sweep']; rows.append({'kind':'online_ep16','phase':phase,'ep':16,'copies_per_layer':k,
              'median_moe_ms':statistics.median(x['online_compute_ms'] for x in v['per_step']),
              'median_saving_ms':sw['median_compute_saving_ms_per_step'],'median_copy_count':sw['median_copy_count_per_step'],
              'median_copy_bytes':sw['median_copy_count_per_step']*EXPERT_BYTES,'median_moved_tasks':sw['median_moved_tasks_per_step'],
              'break_even_copy_us':sw['sweep'][0]['break_even_remote_copy_us_per_expert']})
    with (OUT/'SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps({phase:{ep:{'moe_ms':v['median_moe_critical_proxy_ms'],'R':v['median_R']} for ep,v in result['baseline'][phase].items()} for phase in ('prefill','decode')},indent=2))
    for phase in ('prefill','decode'):
        print('\n',phase)
        for k,v in result['online_ep16'][phase].items(): print('K',k,v['sweep'])

if __name__=='__main__':main()
