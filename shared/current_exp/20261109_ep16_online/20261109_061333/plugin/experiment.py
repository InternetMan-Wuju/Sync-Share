"""Isolated EP16 test harness. Active modes are forced calibration, NOT service policy.

Real input routing, concurrent ranks, native dispatch/compute/combine. No trained
cost model, profitability gate, graph compatibility or deployment claim.
"""
import json, time, math
import torch
import torch.distributed as dist
from planner import plan, source_quotas

def now(): return time.perf_counter_ns()
def elapsed(t): return (now()-t)/1e6

def local_counts(ids):
    logical=torch.where(ids>=0,(ids//17)*16+ids%17,ids)
    # Invalid padding is never assigned to expert0.
    valid=(ids>=0)&(ids<272)&(ids%17<16)
    bins=torch.where(valid,logical,256).reshape(-1).long()
    count=torch.bincount(bins,minlength=257)[:256]
    return logical,valid,count

def collect(rt, hidden, top, cfg, verify=False):
    t=now(); ids=top.topk_ids
    logical,valid,count=local_counts(ids)
    dist.all_gather_into_tensor(rt.count_recv.reshape(-1),count,group=rt.group)
    by_source=rt.count_recv.cpu().tolist(); timing={'count_exchange_d2h_ms':elapsed(t)}
    totals=[sum(row[e] for row in by_source) for e in range(256)]
    if verify:
        local={'rows':int(hidden.shape[0]),'k':int(ids.shape[-1]),'invalid':int((~valid).sum().item()),'temporary':int(((ids>=0)&(ids%17==16)).sum().item())}
        all_rows=[None]*16;dist.all_gather_object(all_rows,local,group=rt.cpu_group)
        assert all(x['k']==8 and x['invalid']==0 and x['temporary']==0 for x in all_rows),all_rows
        assert sum(x['rows'] for x in all_rows)==cfg.get('expected_global_rows',8192),all_rows
        assert sum(totals)==sum(x['rows']*x['k'] for x in all_rows)==cfg.get('expected_global_rows',8192)*8
    t=now(); p=plan(totals);timing['planner_ms']=elapsed(t)
    if cfg.get('force_quota') is not None:
        assert verify and cfg.get('test_only') and p is not None
        q=int(cfg['force_quota']);assert 0<q<totals[p['expert']]
        p=dict(p);p['q']=q;p['new_loads']=p['old_loads'].copy()
        p['new_loads'][p['src']]-=q;p['new_loads'][p['dst']]+=q
        p['saving_tasks']=max(p['old_loads'])-max(p['new_loads'])
        p['test_forced_quota']=True
    return logical,by_source,totals,p,timing

def remap(rt,logical,top,p,by_source,verify=False):
    quotas=source_quotas(by_source,p['expert'],p['q'])
    ids=top.topk_ids;flat=logical.reshape(-1);mask=flat==p['expert'];ordinal=mask.to(torch.int32).cumsum(0)
    move=mask&(ordinal<=quotas[rt.rank]);newids=torch.where(move,p['dst']*17+16,ids.reshape(-1)).reshape_as(ids)
    new=top._replace(topk_ids=newids)
    if verify:
        local=int(move.sum().item());allm=[None]*16;dist.all_gather_object(allm,local,group=rt.cpu_group);assert allm==quotas
        restored=torch.where(newids==p['dst']*17+16,p['expert'],(newids//17)*16+newids%17)
        assert torch.equal(restored,logical)
        assert new.topk_weights is top.topk_weights
    return new,quotas

def compare(rt,a,b):
    a=a.float();b=b.float();diff=a-b
    err={'max_abs':float(diff.abs().max().item()) if diff.numel() else 0,'nrmse':float((diff.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)).item()) if diff.numel() else 0,'finite':bool(torch.isfinite(b).all())}
    errs=[None]*16;dist.all_gather_object(errs,err,group=rt.cpu_group)
    assert all(e['finite'] and e['nrmse']<=0.01 for e in errs),errs
    return err

def execute(rt,hidden,top,args,kwargs,cfg):
    mode=cfg.get('mode','off');verify=cfg.get('verify',False);profile=cfg.get('profile',False)
    assert mode in ('off','shadow','active_cold','active_warm','replay')
    if mode.startswith('active') or mode=='replay':assert cfg.get('test_only') is True,'No calibrated production cost gate'
    if mode=='off' and not profile and not verify:
        rt.prof=False;return rt.original(hidden,top,*args,**kwargs)
    row={'kind':'call','request':cfg.get('request'),'mode':mode,'local_rows':hidden.shape[0],'profile':profile,'verify':verify,'test_only':True}
    if mode=='replay':return replay(rt,hidden,top,args,kwargs,cfg)
    rt.prof=False
    # Retain an independent reference even if a backend reuses output storage.
    baseline=rt.original(hidden,top,*args,**kwargs).clone() if verify else None
    if profile or verify:torch.npu.synchronize()
    rt.prof=profile;rt.events=[];start=now()
    times={};p=None;newtop=top
    if mode!='off' or verify:
        logical,by_source,totals,p,times=collect(rt,hidden,top,cfg,verify)
        row.update(global_tasks=sum(totals),counts=totals,by_source=by_source,plan=p)
        row['old_slots']=[totals[r*16:(r+1)*16]+[0] for r in range(16)]
        if p and mode.startswith('active'):
            cached=mode=='active_warm' and rt.cache.get(p['dst'])==p['expert'];row['cached']=cached
            if not cached:times.update(rt.copy(p,verify))
            t=now();newtop,quotas=remap(rt,logical,top,p,by_source,verify);times['remap_ms']=elapsed(t);row['q_by_source']=quotas
            slots=[x.copy() for x in row['old_slots']];slots[p['src']][p['expert']%16]-=p['q'];slots[p['dst']][16]=p['q'];row['new_slots']=slots
    out=rt.original(hidden,newtop,*args,**kwargs)
    if profile or verify:
        torch.npu.synchronize();times['target_experts_wall_ms']=elapsed(start)
        for label,s,e in rt.events:times[label+'_device_ms']=s.elapsed_time(e)
        if verify:row['numeric']=compare(rt,baseline,out)
        row['timing']=times;row['hbm_peak_bytes']=torch.npu.max_memory_allocated();rt.log(row)
    # Ordinary E2E requests avoid diagnostic events, logging and final sync.
    rt.prof=False;return out

def replay(rt,hidden,top,args,kwargs,cfg):
    """Live layer-input replay on all16 ranks, blocked paired timings.

    No new routes or weights. Component profiling is a separate diagnostic run.
    Every timed block >=100 ms at EACH rank, 20 warmup calls per variant.
    One sample = entire block; calls are not independent observations.
    """
    rt.prof=False
    logical,by_source,totals,p,_=collect(rt,hidden,top,cfg,True)
    assert p is not None, 'No improving count plan on this input'
    rt.copy(p,True);new,quotas=remap(rt,logical,top,p,by_source,True)
    a=rt.original(hidden,top,*args,**kwargs).clone();b=rt.original(hidden,new,*args,**kwargs);torch.npu.synchronize();err=compare(rt,a,b)
    rt.log({'kind':'replay_input','request':cfg['request'],'counts':totals,'by_source':by_source,'plan':p,'q_by_source':quotas,'numeric':err,'input_rows':hidden.shape[0]})
    def invoke(mode):
        if mode=='off':return rt.original(hidden,top,*args,**kwargs)
        if mode=='prepared':return rt.original(hidden,new,*args,**kwargs)
        lg,src,tot,pp,_=collect(rt,hidden,top,cfg)
        assert pp==p
        if mode=='cold':rt.copy(pp,False)
        nt,_=remap(rt,lg,top,pp,src)
        return rt.original(hidden,nt,*args,**kwargs)
    def block(mode,reps):
        torch.npu.synchronize();dist.barrier(group=rt.group);torch.npu.synchronize()
        t=now()
        for _ in range(reps):invoke(mode)
        torch.npu.synchronize();value=elapsed(t)
        values=[None]*16;dist.all_gather_object(values,value,group=rt.cpu_group)
        return value,min(values),max(values)
    modes=['off','prepared','warm','cold'];repetitions={}
    for mode in modes:
        for _ in range(20):invoke(mode)
        reps=2
        for attempt in range(6):
            value,minimum,maximum=block(mode,reps)
            rt.log({'kind':'calibration','request':cfg['request'],'mode':mode,'calls':reps,'block_ms':value,'min_rank_block_ms':minimum,'max_rank_block_ms':maximum})
            if minimum>=120:break
            reps=max(reps+1,math.ceil(reps*130/max(minimum,0.01)))
        assert minimum>=100
        repetitions[mode]=reps
    for i in range(5):
        # Rotate/reverse A/B order to prevent systematic drift.
        order=modes[i%4:]+modes[:i%4]
        if i%2:order=order[::-1]
        for mode in order:
            reps=repetitions[mode]
            while True:
                value,minimum,maximum=block(mode,reps)
                rt.log({'kind':'timing_block' if minimum>=100 else 'short_block','request':cfg['request'],'block':i,'order':order,'mode':mode,'calls':reps,'block_ms':value,'min_rank_block_ms':minimum,'max_rank_block_ms':maximum,'ms_per_call':value/reps})
                if minimum>=100:break
                reps=math.ceil(reps*120/max(minimum,0.01));repetitions[mode]=reps
    # Return baseline output; this HTTP request is excluded from E2E timings.
    return a
