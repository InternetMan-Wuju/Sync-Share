#!/usr/bin/env python3
"""Round-2 online mechanism: current-count planner, integer quota, real state copy, local chains."""
from __future__ import annotations
import csv, hashlib, json, math, platform, statistics, sys, time
from pathlib import Path
import torch, torch_npu

sys.path.insert(0,'/old/scripts')
from s2_local_cost import ExpertBank, load_experts, make_case, sync  # noqa: E402
from s1_npu_numerics import Expert as S1Expert  # noqa: E402
from sglang.srt.hardware_backend.npu.utils import npu_format_cast  # noqa: E402

OUT=Path('/exp')
SEED=20261103
WARM=20
BLOCKS=5
TARGET_MS=120.0


def sha_tensor(t):
    sync(); x=t.detach().cpu().contiguous()
    if x.dtype==torch.bfloat16: x=x.view(torch.int16)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def event_ms(fn, calls):
    sync(); s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True)
    host=time.perf_counter(); s.record()
    for _ in range(calls): fn()
    e.record(); e.synchronize(); total=s.elapsed_time(e)
    return total,total/calls,(time.perf_counter()-host)*1000/calls


def calibrate(fn):
    for _ in range(WARM): fn()
    sync()
    total,per,_=event_ms(fn,1)
    calls=max(20,math.ceil(TARGET_MS/max(per,1e-6)))
    # Verify the chosen block length, and enlarge once if the first estimate was short.
    observed,_,_=event_ms(fn,calls)
    if observed<TARGET_MS:
        calls=max(calls,math.ceil(calls*TARGET_MS/max(observed,1e-6)))
    return calls


def measure_interleaved(functions, labels, csv_rows, group):
    calls={k:calibrate(fn) for k,fn in functions.items()}
    records={k:[] for k in functions}
    for block in range(BLOCKS):
        order=list(labels)
        if block%2: order=list(reversed(order))
        direction='forward' if block%2==0 else 'reverse'
        for order_index,key in enumerate(order):
            fn=functions[key]; n=calls[key]
            total,per,host=event_ms(fn,n)
            rec={'group':group,'case':key,'block':block,'calls':n,'total_ms':total,'ms_per_call':per,'host_ms_per_call':host,'order':order_index,'direction':direction}
            records[key].append(rec); csv_rows.append(rec)
    summary={}
    for key,recs in records.items():
        vals=[r['ms_per_call'] for r in recs]
        summary[key]={'median_ms':statistics.median(vals),'min_ms':min(vals),'max_ms':max(vals),
                      'calls_per_block':calls[key],'block_total_ms':[r['total_ms'] for r in recs],
                      'all_blocks_ge_100ms':all(r['total_ms']>=100 for r in recs),
                      'all_blocks_ge_20_calls':all(r['calls']>=20 for r in recs)}
    return summary


def planner(counts):
    # Pure CPU reference: host already has counts.  One hot expert owner + one receiver.
    a,b=int(counts[0]),int(counts[1]); total=a+b; target=total//2
    # Move only from the hotter owner; no reverse or multi-step case in this round.
    move=max(0,min(a,a-target))
    return {'input_counts':[a,b],'target':target,'move':move,'new_counts':[a-move,b+move],
            'copy_required':move>0,'quota_receiver':move,'quota_owner':a-move}


def cpu_planner_bench(counts):
    for _ in range(100): planner(counts)
    samples=[]
    for _ in range(15):
        start=time.perf_counter_ns()
        for _ in range(20000): planner(counts)
        samples.append((time.perf_counter_ns()-start)/20000/1000.0)  # microseconds
    return {'samples_us':samples,'median_us':statistics.median(samples),'min_us':min(samples),'max_us':max(samples)}


def dequant_reference(bank, dense):
    qx=dense['quantized_x'].cpu().float(); xs=dense['x_scale'].cpu().float().flatten().unsqueeze(1)
    x=qx*xs
    w={p:bank.source[f'{p}.weight'].float()*bank.source[f'{p}.weight_scale'].float().flatten().unsqueeze(1)
       for p in ('gate_proj','up_proj','down_proj')}
    gate=torch.mm(x,w['gate_proj'].T); up=torch.mm(x,w['up_proj'].T)
    act=torch.nn.functional.silu(gate)*up
    scale=torch.amax(act.abs(),dim=1,keepdim=True)/127.0
    scale=torch.where(scale==0,torch.ones_like(scale),scale)
    qa=torch.clamp(torch.round(act/scale),-128,127)
    return torch.mm(qa*scale,w['down_proj'].T)


def compare(c,r):
    sync(); diff=c.float()-r.float()
    return {'bitwise_equal':bool(torch.equal(c,r)),'max_abs':float(diff.abs().max()),
            'nrmse':float(torch.linalg.vector_norm(diff)/max(torch.linalg.vector_norm(r.float()),1e-12)),
            'nonfinite':int((~torch.isfinite(c)).sum())}


def correctness_bridge(s1_banks, timed_banks):
    result={'experts':{},'split':{}}
    for eid,s1 in s1_banks.items():
        x=torch.randn((17,6144),generator=torch.Generator().manual_seed(SEED+eid),dtype=torch.float32).to(torch.bfloat16).npu()
        dense=s1.dense_chain(x)
        ref=dequant_reference(s1,dense)
        grouped=s1.grouped_chain(dense['quantized_x'],dense['x_scale'],[17,0])
        case=make_case(timed_banks[eid],[17],f'bridge_e{eid}','bridge')
        case['x']=x
        timed=timed_banks[eid].chain(case)
        result['experts'][str(eid)]={
          'offset_abs_max':float(s1.raw['gate_proj.weight_offset'].abs().max()),
          'weight_shape':list(s1.source['gate_proj.weight'].shape),
          'weight_dtype':str(s1.source['gate_proj.weight'].dtype),
          'scale_source_dtype':str(s1.source['gate_proj.weight_scale'].dtype),
          'scale_kernel_dtype':str(s1.weight_scale['gate_proj'].dtype),
          'timed_weight_layout':list(timed_banks[eid].w13.shape),
          'dense_vs_reference':compare(dense['output'].cpu(),ref),
          's1_grouped_vs_dense':compare(grouped['output'],dense['output']),
          'timed_fused_vs_dense':compare(timed,dense['output']),
        }
    timed43=timed_banks[43]
    ranked_test=make_case(timed43,[17],'split_e43','split')
    base=timed43.chain(ranked_test); x=ranked_test['x']
    left=make_case(timed43,[9],'split_left','split'); left['x']=x[:9]
    right=make_case(timed43,[8],'split_right','split'); right['x']=x[9:]
    split=torch.cat([timed43.chain(left),timed43.chain(right)],0)
    result['split']['fixed_quant_9_8_vs_full_17']=compare(split,base)
    empty=torch.empty((0,6144),dtype=torch.bfloat16,device='npu:0')
    try:
        timed43.chain({'x':empty,'permutation':torch.empty(0,dtype=torch.int64,device='npu:0'),'inverse':torch.empty(0,dtype=torch.int64,device='npu:0'),'group_list':torch.zeros(1,dtype=torch.int64,device='npu:0')})
        result['split']['empty_branch']='supported'
    except Exception as exc: result['split']['empty_branch']=f'{type(exc).__name__}: {exc}'
    one=make_case(timed43,[1],'split_one','split')
    result['split']['one_row_finite']=bool(torch.isfinite(timed43.chain(one)).all())
    result['all_offsets_zero']=all(float(bank.raw['gate_proj.weight_offset'].abs().max())==0 for bank in s1_banks.values())
    result['pass']=result['all_offsets_zero'] and result['split']['empty_branch']=='supported' and result['split']['fixed_quant_9_8_vs_full_17']['nonfinite']==0 and all(
        v['dense_vs_reference']['nrmse']<=0.02 and v['dense_vs_reference']['nonfinite']==0 and v['timed_fused_vs_dense']['nonfinite']==0
        for v in result['experts'].values())
    return result


def main():
    torch.npu.set_device(0)
    if torch.npu.device_count()!=1: raise RuntimeError(torch.npu.device_count())
    sync(); free0,total=torch.npu.mem_get_info(); torch.npu.reset_peak_memory_stats()
    model=Path('/models/glm-q'); ids=[43,40]; raw=load_experts(model,ids)
    bank43=ExpertBank('e43',[43],{43:raw[43]})
    bank40=ExpertBank('e40',[40],{40:raw[40]})
    # The candidate rank1 state holds both experts and is also used by the empty adapter.
    rank1_bank=ExpertBank('rank1_e40_e43',[40,43],{40:raw[40],43:raw[43]})
    banks={43:bank43,40:bank40}
    s1_banks={eid:S1Expert(eid,raw[eid]) for eid in ids}
    correctness=correctness_bridge(s1_banks,banks)

    # Copy the real e43 formatted state into rank1's independent group-1 slots.
    source_groups={'w13':bank43.w13[0],'down':bank43.down[0],
                   'w13_scale':bank43.w13_scale[0],'down_scale':bank43.down_scale[0]}
    copy_count=0
    def copy_state():
        nonlocal copy_count
        rank1_bank.w13[1].copy_(source_groups['w13'])
        rank1_bank.down[1].copy_(source_groups['down'])
        rank1_bank.w13_scale[1].copy_(source_groups['w13_scale'])
        rank1_bank.down_scale[1].copy_(source_groups['down_scale'])
        copy_count+=1
    copy_calls=calibrate(copy_state)
    copy_blocks=[]
    for block in range(BLOCKS):
        total_ms,per,host=event_ms(copy_state,copy_calls)
        copy_blocks.append({'block':block,'calls':copy_calls,'total_ms':total_ms,'ms_per_call':per,'host_ms_per_call':host})
    copy_bytes=int(source_groups['w13'].numel()*source_groups['w13'].element_size()+source_groups['down'].numel()*source_groups['down'].element_size()+source_groups['w13_scale'].numel()*source_groups['w13_scale'].element_size()+source_groups['down_scale'].numel()*source_groups['down_scale'].element_size())
    copy_check={
      'bytes':copy_bytes,'source_vs_destination':{
        name:compare(rank1_bank_dest,src) for (name,rank1_bank_dest),src in [
          (('w13',rank1_bank.w13[1]),source_groups['w13']),(('down',rank1_bank.down[1]),source_groups['down']),
          (('w13_scale',rank1_bank.w13_scale[1]),source_groups['w13_scale']),(('down_scale',rank1_bank.down_scale[1]),source_groups['down_scale'])]},
      'version':copy_count,
    }

    # Format prep timing from raw transposed source tensors.
    w13_raw=torch.cat([raw[43]['gate_proj.weight'].npu(),raw[43]['up_proj.weight'].npu()],0).transpose(0,1).contiguous()
    down_raw=raw[43]['down_proj.weight'].npu().transpose(0,1).contiguous()
    def format_prep():
        _=npu_format_cast(w13_raw); _=npu_format_cast(down_raw)
    format_calls=calibrate(format_prep); format_blocks=[]
    for block in range(BLOCKS):
        total_ms,per,host=event_ms(format_prep,format_calls)
        format_blocks.append({'block':block,'calls':format_calls,'total_ms':total_ms,'ms_per_call':per,'host_ms_per_call':host})

    csv_rows=[]
    cases={}
    # A empty adapter: rank1 extra slot has zero tasks, same two-expert resident state.
    cases['hot']={'A':{'r0_counts':[960],'r1_counts':[64,0]},'B':{'r0_counts':[512],'r1_counts':[64,448]}}
    cases['uniform']={'A':{'r0_counts':[512],'r1_counts':[512,0]},'B':{'r0_counts':[512],'r1_counts':[512,0]}}
    functions={}; labels=[]
    for case_name in ('hot','uniform'):
        for layout in ('A_empty','B_plan'):
            counts=cases[case_name][layout[0]]
            c0=make_case(bank43,counts['r0_counts'],f'{case_name}_{layout}_r0','online')
            c1=make_case(rank1_bank,counts['r1_counts'],f'{case_name}_{layout}_r1','online')
            key0=f'{case_name}:{layout}:r0'; key1=f'{case_name}:{layout}:r1'
            functions[key0]=lambda c=c0: bank43.chain(c)
            functions[key1]=lambda c=c1: rank1_bank.chain(c)
            labels.extend([key0,key1])
        # Planner/case ledger
        pc=planner([960,64] if case_name=='hot' else [512,512])
        cases[case_name]['planner']=pc
        cases[case_name]['planner_bench']=cpu_planner_bench([960,64] if case_name=='hot' else [512,512])
        cases[case_name]['copy_bytes']=copy_bytes if pc['copy_required'] else 0
        cases[case_name]['copy_required']=pc['copy_required']
    timings=measure_interleaved(functions,labels,csv_rows,'online')
    for case_name in ('hot','uniform'):
        a0=timings[f'{case_name}:A_empty:r0']['median_ms']; a1=timings[f'{case_name}:A_empty:r1']['median_ms']
        b0=timings[f'{case_name}:B_plan:r0']['median_ms']; b1=timings[f'{case_name}:B_plan:r1']['median_ms']
        cases[case_name]['A_local_ms']={'r0':a0,'r1':a1,'max_proxy':max(a0,a1),'sum_sequential':a0+a1}
        cases[case_name]['B_local_ms']={'r0':b0,'r1':b1,'max_proxy':max(b0,b1),'sum_sequential':b0+b1}
        d=max(a0,a1)-max(b0,b1)
        cases[case_name]['delta_compute_ms']=d
        cases[case_name]['delta_compute_frac']=d/max(a0,a1)
        cases[case_name]['planner_over_delta_frac']=cases[case_name]['planner_bench']['median_us']/1000.0/max(d,1e-12)
    sync(); free1,_=torch.npu.mem_get_info()
    result={'scope':'CPU planner + eager local full expert chains on one NPU; no cross-rank execution/network',
      'status':'COMPLETE' if correctness['pass'] else 'FAIL_CORRECTNESS','correctness':correctness,
      'planner':{k:{'plan':v['planner'],'bench':v['planner_bench']} for k,v in cases.items()},
      'cases':cases,'timings':timings,
      'copy':{'bytes':copy_bytes,'timing_blocks':copy_blocks,'device_ms_median':statistics.median([b['ms_per_call'] for b in copy_blocks]),
              'host_ms_median':statistics.median([b['host_ms_per_call'] for b in copy_blocks]),'check':copy_check},
      'format_prep':{'timing_blocks':format_blocks,'device_ms_median':statistics.median([b['ms_per_call'] for b in format_blocks]),
                     'host_ms_median':statistics.median([b['host_ms_per_call'] for b in format_blocks])},
      'environment':{'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device_count':torch.npu.device_count(),'device_name':torch.npu.get_device_name(0)},
      'memory':{'free_before':int(free0),'free_after':int(free1),'total':int(total),'peak_allocated':int(torch.npu.max_memory_allocated()),'peak_reserved':int(torch.npu.max_memory_reserved())},
      'unmeasured':['global count exchange','remote HCCL copy','dispatch/combine delta','parallel contention','production graph integration']}
    (OUT/'ONLINE_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'TIMINGS_ONLINE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
