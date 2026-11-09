"""Concurrent EP16 current-batch experiment, no production serving claim.

Larger workloads repeat live layer49 hidden/top-k locally. They are synthetic
layer replays, not full-context model requests and not independent samples.
"""
import json,time,math,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from planner import plan
from decision import decide,admit,apply
from experiment import local_counts,remap,compare
ROOT=Path('/runexp')

def now():return time.perf_counter_ns()
def ms(t):return (now()-t)/1e6

def collect(rt,top):
    t=now();logical,valid,count=local_counts(top.topk_ids)
    dist.all_gather_into_tensor(rt.count_recv.reshape(-1),count,group=rt.group)
    by_source=rt.count_recv.cpu().tolist()
    totals=list(map(sum,zip(*by_source)))
    return logical,by_source,totals,ms(t)

def group_values(rt,value):
    out=[None]*16;dist.all_gather_object(out,value,group=rt.cpu_group);return out

def block(rt,fn,reps):
    torch.npu.synchronize();dist.barrier(group=rt.group);torch.npu.synchronize();t=now()
    for _ in range(reps):fn()
    torch.npu.synchronize();value=ms(t);values=group_values(rt,value)
    return value,min(values),max(values)

def measure(rt,functions,tag):
    reps_by={}
    for mode,fn in functions.items():
        for _ in range(20):fn()
        reps=2
        for attempt in range(8):
            value,low,high=block(rt,fn,reps)
            rt.log(dict(kind='decision_calibration',mode=mode,calls=reps,block_ms=value,min_rank_block_ms=low,max_rank_block_ms=high,**tag))
            if low>=120:break
            reps=max(reps+1,math.ceil(reps*135/max(low,0.01)))
        assert low>=100
        reps_by[mode]=reps
    modes=list(functions)
    for i in range(5):
        order=modes[i%len(modes):]+modes[:i%len(modes)]
        if i%2:order=order[::-1]
        for mode in order:
            reps=reps_by[mode]
            for attempt in range(5):
                value,low,high=block(rt,functions[mode],reps)
                rt.log(dict(kind='decision_block' if low>=100 else 'decision_short_block',mode=mode,block=i,order=order,calls=reps,block_ms=value,min_rank_block_ms=low,max_rank_block_ms=high,**tag))
                if low>=100:break
                reps=max(reps+1,math.ceil(reps*130/max(low,0.01)));reps_by[mode]=reps
            assert low>=100

def execute(rt,hidden,top,args,kwargs,cfg):
    assert cfg.get('test_only') is True
    rt.prof=False
    mode=cfg['mode'];training=mode=='decision_train'
    assert training or mode=='decision_test'
    # Own independent output before creating synthetic replay work.
    answer=rt.original(hidden,top,*args,**kwargs).clone();torch.npu.synchronize()
    signatures={name:hashlib.sha256((ROOT/'plugin'/name).read_bytes()).hexdigest() for name in ('decision.py','decision_experiment.py','runtime.py','experiment.py')}
    assert len({json.dumps(x,sort_keys=True) for x in group_values(rt,signatures)})==1
    model=None if training else json.loads((ROOT/'artifacts/FROZEN_MODEL.json').read_text())
    model_sha=None if training else hashlib.sha256((ROOT/'artifacts/FROZEN_MODEL.json').read_bytes()).hexdigest()
    rt.log(dict(kind='decision_implementation',request=cfg['request'],signatures=signatures,model_sha256=model_sha))
    for scale in cfg.get('scales',[1,2,4,8]):
        assert scale in (1,2,4,8)
        h=hidden.repeat((scale,1)) if scale!=1 else hidden
        changes={}
        for name in top._fields:
            x=getattr(top,name)
            if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==hidden.shape[0]:
                changes[name]=x.repeat((scale,)+(1,)*(x.ndim-1)) if scale!=1 else x
        tt=top._replace(**changes)
        tag={'request':cfg['request'],'case':cfg['case'],'scale':scale,'global_rows':8192*scale,'split':'development' if training else 'heldout'}
        logical,by_source,counts,_=collect(rt,tt)
        assert h.shape[0]==512*scale and sum(counts)==8192*scale*8
        assert bool((tt.topk_ids>=0).all()) and bool((tt.topk_ids%17<16).all())
        ids_sorted=tt.topk_ids.sort(dim=-1).values
        assert bool((ids_sorted[:,1:]!=ids_sorted[:,:-1]).all())
        old=apply(counts,None);count_plan=plan(counts);assert count_plan is not None
        admitted=True if training else admit(8192*scale,model)
        p,detail=(None,{'reason':'development'}) if training else (decide(counts,model) if admitted else (None,{'reason':'pre_count_compute_bound'}))
        selected=[('count',count_plan)]
        if training:
            half=dict(count_plan);half['q']=counts[half['expert']]//2
            selected.append(('half',half))
        elif p and any(p[k]!=count_plan[k] for k in ('expert','src','dst','q')):
            selected.append(('policy',p))
        baseline=rt.original(h,tt,*args,**kwargs).clone();torch.npu.synchronize()
        prepared={};numerics={}
        for label,pp in selected:
            copy_times=rt.copy(pp,True);nt,quota=remap(rt,logical,tt,pp,by_source,True)
            out=rt.original(h,nt,*args,**kwargs);torch.npu.synchronize();err=compare(rt,baseline,out)
            prepared[label]=(nt,pp);numerics[label]=err
            rt.log(dict(kind='decision_validation',label=label,plan=pp,slots=apply(counts,pp),quotas=quota,numeric=err,copy_timing=copy_times,**tag))
        rt.log(dict(kind='decision_input',counts=counts,by_source=by_source,old_slots=old,count_plan=count_plan,policy_plan=p,policy_detail=detail,admitted=admitted,**tag))
        # Reinstall each variant before its diagnostic call (not inside event window).
        diag=[('off',None)]+[(label,pp) for label,(nt,pp) in prepared.items()]
        for rep in range(5):
            for label,pp in diag:
                if pp:rt.copy(pp,False)
                nt=tt if pp is None else prepared[label][0]
                rt.events=[];rt.prof=True;torch.npu.synchronize()
                t=now();rt.original(h,nt,*args,**kwargs);torch.npu.synchronize();wall=ms(t);rt.prof=False
                events={label+'_device_ms':s.elapsed_time(e) for label,s,e in rt.events}
                rt.log(dict(kind='decision_profile',variant=label,rep=rep,slots=old if pp is None else apply(counts,pp),wall_ms=wall,timing=events,**tag))
        # Active variants always pay a real cold copy, irrespective of cache.
        def native():return rt.original(h,tt,*args,**kwargs)
        def count_cold():
            lg,bs,cs,_=collect(rt,tt);pp=plan(cs);rt.copy(pp,False)
            nt,_=remap(rt,lg,tt,pp,bs);return rt.original(h,nt,*args,**kwargs)
        def policy_cold():
            if not admit(h.shape[0]*16,model):return native()
            lg,bs,cs,_=collect(rt,tt);pp,_=decide(cs,model)
            if pp is None:return native()
            rt.copy(pp,False);nt,_=remap(rt,lg,tt,pp,bs)
            return rt.original(h,nt,*args,**kwargs)
        # All selected plans may have different expert identities: never reuse a
        # prepared map after another expert has overwritten the same bank slot.
        # Prepared timing uses only the count plan, installed immediately here.
        rt.copy(count_plan,False);nt_count=prepared['count'][0]
        def prepared_count():return rt.original(h,nt_count,*args,**kwargs)
        funcs={'off':native,'count_cold':count_cold}
        if training:funcs['prepared_count']=prepared_count
        else:funcs['policy_cold']=policy_cold
        measure(rt,funcs,tag)
        if not training and scale in (1,8):
            measure(rt,{'planner_host_counts':lambda:decide(counts,model)},dict(tag,scope='component'))
        if training:
            measure(rt,{'copy_only':lambda:rt.copy(count_plan,False),'collect_only':lambda:collect(rt,tt),
                        'remap_only':lambda:remap(rt,logical,tt,count_plan,by_source)},dict(tag,scope='component'))
            if scale==1:
                receivers=sorted((r for r in range(16) if r!=count_plan['src']),key=lambda r:sum(old[r]))[:3]
                for dst in receivers:
                    pp=dict(count_plan,dst=dst);ct=rt.copy(pp,True)
                    rt.log(dict(kind='copy_pair_validation',src=pp['src'],dst=dst,timing=ct,**tag))
                    measure(rt,{f'copy_pair_{pp["src"]}_{dst}':lambda pp=pp:rt.copy(pp,False)},dict(tag,scope='component'))
        torch.npu.synchronize()
        rt.log(dict(kind='decision_case_done',hbm_peak_bytes=torch.npu.max_memory_allocated(),**tag))
        del baseline,h,tt
    return answer
