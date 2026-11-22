#!/usr/bin/env python3
"""Functional, owner-state and clean-memory evidence for the repaired K=1 adapter.

All NPU cases enter the same ``UnifiedK1Adapter.execute`` path.  Task identity is
kept in side metadata; hidden tensors are never tagged or rewritten.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, time, uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch, torch_npu, torch.distributed as dist

from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import (
    LAYER_ATTRS, PARTS, ExpertLayer, RuntimeChain, alloc,
    load_raw_state, memory_snapshot, state_layout,
)
from unified_adapter import (
    UnifiedK1Adapter, logical_to_comm, EMPTY_COMM, REPLICA_COMM, plan_tasks,
)

OUT = Path(os.environ.get("OUT_DIR", "/runexp/artifacts"))
LAYERS=(49,50)
RANK_MAIN={0:(0,1,2,3,4,5,43,127),1:(128,129,130,131,132,133,134,255)}
MIXED=(0,43,127,5,128,129,130,255)
ATOL=0.02


def json_default(value):
    if is_dataclass(value): return asdict(value)
    if hasattr(value, "item"): return value.item()
    return str(value)


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_hidden(n, rank, seed):
    torch.manual_seed(seed + rank)
    h=torch.randn((n,6144),dtype=torch.bfloat16,device='npu')
    if n:
        vals=[0,16,199,200,201,256,258]
        for i in range(min(n,len(vals))):
            h[i,0]=vals[i]; h[i,1]=vals[-1-i]
        if n>=2:
            h[1].copy_(h[0])
    return h


def input_rows(hidden):
    if hidden.shape[0]==0: return []
    rows=[]
    for i in range(min(8,int(hidden.shape[0]))):
        rows.append({'row':i,'value0':float(hidden[i,0].float().item()),'value1':float(hidden[i,1].float().item()),'row_sha256':sha_tensor(hidden[i])})
    return rows


def reference(layer, rank, manifests, chains):
    """Independent output reference reads only the original hidden files."""
    case_id=manifests[0]['case_id']
    hs=[torch.load(OUT/f'{case_id}_l{layer}_hidden_rank{i}.pt',map_location='cpu') for i in (0,1)]
    offsets=[]; total=0
    for h in hs: offsets.append(total); total+=h.shape[0]
    gh=torch.cat(hs,dim=0).to('npu') if total else torch.empty((0,6144),dtype=torch.bfloat16,device='npu')
    partial=torch.zeros((total,6144),dtype=torch.float32,device='npu')
    for expert in RANK_MAIN[rank]:
        idx=[]; ws=[]
        for sender,m in enumerate(manifests):
            for token,row in enumerate(m['logical_rows']):
                if expert in row:
                    idx.append(offsets[sender]+token); ws.append(m['weights'][token][row.index(expert)])
        if idx:
            it=torch.tensor(idx,dtype=torch.int64,device='npu')
            wt=torch.tensor(ws,dtype=torch.float32,device='npu').unsqueeze(1)
            out=chains[(layer,expert)](gh.index_select(0,it))
            partial.index_add_(0,it,out.float()*wt)
    dist.all_reduce(partial,op=dist.ReduceOp.SUM,group=dist.group.WORLD)
    return partial[offsets[rank]:offsets[rank]+hs[rank].shape[0]].contiguous()


def expected_tasks(manifests, rank):
    lo,hi=(0,129) if rank==0 else (129,258)
    return sorted((m['rank'],t,s,int(m['logical_rows'][t][s]),int(c)) for m in manifests for t,row in enumerate(m['comm_rows']) for s,c in enumerate(row) if lo<=c<hi)


def actual_tasks(result):
    return sorted((t['sender'],t['token'],t['slot'],t['logical'],t['comm']) for t in result.get('actual_tasks',[]) if not t.get('padding'))


def wait_for_case_files(case_id, layer, timeout_s=20.0):
    manifests=[OUT/f'{case_id}_l{layer}_rank{r}.json' for r in (0,1)]
    hidden=[OUT/f'{case_id}_l{layer}_hidden_rank{r}.pt' for r in (0,1)]
    paths=manifests+hidden
    deadline=time.monotonic()+timeout_s
    while True:
        if all(p.exists() and p.stat().st_size>0 for p in paths):
            # A final JSON parse catches a writer that created but did not close.
            try:
                for path in manifests:
                    json.loads(path.read_text())
                return
            except (json.JSONDecodeError, OSError):
                pass
        if time.monotonic()>=deadline:
            raise TimeoutError(f'case files not visible: {case_id}')
        time.sleep(0.01)


def stage_snapshot(name, fn, records, bank_peak):
    torch.npu.synchronize()
    before=memory_snapshot()
    try: torch.npu.reset_peak_memory_stats()
    except Exception: pass
    started=time.perf_counter_ns()
    value=fn()
    torch.npu.synchronize()
    after=memory_snapshot()
    rec={'stage':name,'started_ns':started,'duration_ms':(time.perf_counter_ns()-started)/1e6,'before':before,'after':after,'peak_allocated':after.get('max_allocated',0),'peak_reserved':after.get('max_reserved',0)}
    records.append(rec)
    bank_peak['peak_allocated']=max(bank_peak.get('peak_allocated',0),after.get('max_allocated',0))
    bank_peak['peak_reserved']=max(bank_peak.get('peak_reserved',0),after.get('max_reserved',0))
    return value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario',choices=('matrix','baseline','candidate'),default='matrix')
    ap.add_argument('--tag',default='functional')
    ap.add_argument('--campaign-id',default=os.environ.get('CAMPAIGN_ID',''))
    ap.add_argument('--subrun-id',default=os.environ.get('SUBRUN_ID',''))
    ap.add_argument('--code-hash',default=os.environ.get('CODE_HASH',''))
    ap.add_argument('--out-dir',default=os.environ.get('OUT_DIR','/runexp/artifacts'))
    args=ap.parse_args()
    global OUT
    OUT=Path(args.out_dir); OUT.mkdir(parents=True,exist_ok=True)
    if not args.campaign_id or not args.subrun_id or not args.code_hash:
        raise SystemExit('campaign-id, subrun-id and code-hash are required')
    run_ctx={'campaign_id':args.campaign_id,'subrun_id':args.subrun_id,'code_hash':args.code_hash,'scenario':args.scenario}

    rank=int(os.environ['RANK']); torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment,destroy_model_parallel,get_tp_group,initialize_model_parallel
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    from sglang.srt.layers.moe.utils import DeepEPMode,initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    init_distributed_environment(world_size=-1,rank=-1,local_rank=-1,backend='hccl')
    sa=ServerArgs(model_path='dummy',tp_size=2,ep_size=2)
    sa.moe_a2a_backend='deepep'; sa.moe_runner_backend='ascend'
    set_global_server_args_for_scheduler(sa); publish(sa,role='scheduler')
    initialize_moe_config(); initialize_model_parallel(tensor_model_parallel_size=2,expert_model_parallel_size=2)
    group=get_tp_group().device_group
    control_group=dist.new_group(ranks=[0,1],backend='gloo')
    dispatcher=DeepEPDispatcher(group=group,router_topk=8,num_experts=258,num_local_experts=129,hidden_size=6144,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False)
    import sglang.srt.layers.quantization
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    method=NPUW8A8Int8MoEMethod(); matmul=GroupedMatmul()
    mem_before=memory_snapshot(); bank={}; slot_hashes_by_layer={}; layout=None; state_bytes=None
    for layer in LAYERS:
        for expert in RANK_MAIN[rank]:
            bank_state,slot_state,offsets,bank_hashes,sh=load_raw_state(layer,expert)
            if layer==49 and expert==43:
                layout,state_bytes=state_layout(slot_state)
            mod=ExpertLayer(bank_state)
            for part in PARTS:
                getattr(mod,LAYER_ATTRS[part]).data=getattr(mod,LAYER_ATTRS[part]).data.npu()
            method.process_weights_after_loading(mod,'w13'); method.process_weights_after_loading(mod,'w2')
            bank[(layer,expert)]=mod
            if expert==43: slot_hashes_by_layer[layer]=sh
    torch.npu.synchronize(); chains={(l,e):RuntimeChain(method,matmul,bank[(l,e)],l) for l,e in bank}
    memory_after_bank=memory_snapshot()
    entries=[]; unique={}
    for (l,e),mod in bank.items():
        for attr in LAYER_ATTRS.values():
            t=getattr(mod,attr,None)
            if t is None: continue
            st=t.untyped_storage(); item={'layer':l,'expert':e,'attr':attr,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'data_ptr':int(t.data_ptr()),'storage_ptr':int(st.data_ptr()),'storage_bytes':int(st.nbytes()),'offset':int(t.storage_offset())}
            entries.append(item); unique[item['storage_ptr']]=item['storage_bytes']
    bank_formats=[]
    for (l,e),mod in bank.items():
        for attr in LAYER_ATTRS.values():
            t=getattr(mod,attr,None)
            if t is None: continue
            try: fs=torch_npu.get_npu_format(t)
            except Exception as ex: fs=f'ERROR:{type(ex).__name__}:{ex}'
            try: fc=int(torch.ops.npu.get_npu_format(t))
            except Exception as ex: fc=f'ERROR:{type(ex).__name__}:{ex}'
            bank_formats.append({'layer':l,'expert':e,'attr':attr,'format_string':fs,'format_code':fc,'stride':list(t.stride()),'storage_ptr':int(t.untyped_storage().data_ptr()),'storage_bytes':int(t.untyped_storage().nbytes()),'data_ptr':int(t.data_ptr())})

    if rank==0:
        meta={'campaign_id':args.campaign_id,'subrun_id':args.subrun_id,'code_hash':args.code_hash,'state_bytes':state_bytes,'layout':[{'name':n,'offset':o,'bytes':b,'dtype':str(d),'shape':list(s)} for n,o,b,d,s in layout],'slot_hashes':slot_hashes_by_layer}
        (OUT/'meta.json').write_text(json.dumps(meta,indent=2))
    dist.barrier(group=control_group)
    if rank==1:
        meta=json.loads((OUT/'meta.json').read_text())
        if meta['campaign_id']!=args.campaign_id or meta['subrun_id']!=args.subrun_id or meta['code_hash']!=args.code_hash: raise RuntimeError('metadata campaign mismatch')
        state_bytes=int(meta['state_bytes'])
        dtype_map={'torch.int8':torch.int8,'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}
        layout=[(x['name'],x['offset'],x['bytes'],dtype_map[x['dtype']],tuple(x['shape'])) for x in meta['layout']]

    create_pool=args.scenario!='baseline'
    def create_resources():
        flat=None; pool=None
        if rank==0 and create_pool: flat=torch.empty(state_bytes,dtype=torch.uint8,device='npu')
        if rank==1 and create_pool: pool=GlobalSlotPool([1],state_bytes,alloc,name='unified-adapter-slot',synchronize=torch.npu.synchronize)
        return flat,pool
    stage_records=[]; cumulative_peak={'peak_allocated':memory_after_bank.get('max_allocated',0),'peak_reserved':memory_after_bank.get('max_reserved',0)}
    flat,pool=stage_snapshot('resources_created',create_resources,stage_records,cumulative_peak)
    adapter=UnifiedK1Adapter(rank=rank,group=group,control_group=control_group,dispatcher=dispatcher,bank_chains=chains,main_experts_by_rank=RANK_MAIN,layers=LAYERS,pool=pool,layout=layout,state_bytes=state_bytes,flat=flat,method=method,matmul=matmul,run_context=run_ctx)

    def run_one(case_id,layer,tokens0,tokens1,experts_set,mode='none',inject_rank=None,slot_requests=None,fixed_hook=None,need_reference=True):
        n=tokens0 if rank==0 else tokens1
        rng=random.Random(70000+layer*17+rank)
        logical=[]; weights=[]
        for _ in range(n):
            row=list(experts_set); rng.shuffle(row); logical.append(row)
            w=[rng.random()+1e-3 for _ in range(8)]; s=sum(w); weights.append([x/s for x in w])
        hidden=make_hidden(n,rank,70000+layer*17)
        comm=[[logical_to_comm(e) for e in row] for row in logical]
        for t,row in enumerate(logical):
            for s,e in enumerate(row):
                if e==43 and (mode=='all' or (mode=='partial' and t%2==0)): comm[t][s]=REPLICA_COMM
        manifest={'rank':rank,'case_id':case_id,'tokens':n,'logical_rows':logical,'comm_rows':comm,'weights':weights,**run_ctx}
        # The manifest path is part of the evidence file naming; the wait helper
        # uses a distinct suffix so it never confuses metadata with hidden state.
        (OUT/f'{case_id}_l{layer}_rank{rank}.json').write_text(json.dumps(manifest))
        torch.save(hidden.detach().cpu(),OUT/f'{case_id}_l{layer}_hidden_rank{rank}.pt')
        dist.barrier(group=control_group)
        wait_for_case_files(case_id,layer)
        manifests=[json.loads((OUT/f'{case_id}_l{layer}_rank{i}.json').read_text()) for i in (0,1)]
        hidden_by_rank=[torch.load(OUT/f'{case_id}_l{layer}_hidden_rank{i}.pt',map_location='cpu').to('npu') for i in (0,1)]
        ref=reference(layer,rank,manifests,chains) if need_reference and (tokens0+tokens1)>0 else None
        before=hidden.clone()
        before_sha=sha_tensor(hidden)
        ret=adapter.execute(case_id=case_id,layer=layer,hidden=hidden,manifests=manifests,hidden_by_rank=hidden_by_rank,mode=mode,slot_requests=slot_requests,inject_invalid_comm=(inject_rank==rank),fixed_lease_hook=fixed_hook)
        if isinstance(ret,tuple): result,out=ret
        else: result,out=ret,None
        after_sha=sha_tensor(hidden)
        result.update({**run_ctx,'tag':args.tag,'input_unchanged':before_sha==after_sha and bool(torch.equal(hidden,before)),'input_sha256_before':before_sha,'input_sha256_after':after_sha,'selected_input_rows':input_rows(hidden),'local_expected_task_count':len(plan_tasks(manifests)[rank]),'global_expected_task_count':sum(int(m['tokens']) for m in manifests)*8})
        exp=expected_tasks(manifests,rank)
        act=actual_tasks(result) if result['status']=='PASS' else []
        if result['status']=='PASS':
            err=float((out.float()-ref).abs().max().item()) if ref is not None and out.numel() else (None if ref is None else 0.0)
            rel=(err/max(float(ref.abs().max().item()),1e-9)) if (err is not None and ref.numel()) else (None if ref is None else 0.0)
            result.update({'reference_max_abs':err,'reference_relative_error':rel,'within_tolerance':None if err is None else bool(err<=ATOL and rel<=ATOL),'expected_task_count':len(exp),'actual_task_count':len(act),'task_multiset_equal':exp==act,'reference_finite':None if ref is None else bool(torch.isfinite(ref).all().item()),'candidate_finite':bool(torch.isfinite(out).all().item())})
        return result

    if args.scenario in ('baseline','candidate'):
        stage_snapshot('common_init',lambda:None,stage_records,cumulative_peak)
        install=None; q0=None; split=None; slot_before_release=None
        if args.scenario=='candidate':
            def do_install():
                return adapter.install(49,OUT,f'{args.tag}_mem_install')
            install=stage_snapshot('install',do_install,stage_records,cumulative_peak)
            si=json.loads((OUT/f'{args.tag}_mem_install_install_rank1.json').read_text())
            req=[si,si]
            q0=stage_snapshot('q0',lambda:run_one('memory_q0',49,2,2,MIXED,'none',None,[None,None],None,False),stage_records,cumulative_peak)
            split=stage_snapshot('split',lambda:run_one('memory_split',49,2,2,MIXED,'partial',None,req,None,False),stage_records,cumulative_peak)
            slot_before_release=adapter.slot_info()
        else:
            q0=stage_snapshot('q0',lambda:run_one('memory_q0',49,2,2,MIXED,'none',None,[None,None],None,False),stage_records,cumulative_peak)
        stage_snapshot('release',adapter.release_slot,stage_records,cumulative_peak)
        final_records=adapter.owner_records() if rank==1 and pool else []
        flat_record={'bytes':int(flat.numel()*flat.element_size()),'storage_ptr':int(flat.untyped_storage().data_ptr())} if flat is not None else None
        rec={
            **run_ctx,'tag':args.tag,'scenario':args.scenario,'rank':rank,'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','performance_status':'NOT_RUN',
            'state_bytes':state_bytes,
            'memory_before':mem_before,'memory_after_bank':memory_after_bank,'memory_final':memory_snapshot(),
            'stage_records':stage_records,'whole_process_peak':cumulative_peak,
            'flat_buffer':flat_record,'resource_objects':{'rank0_flat':flat_record if rank==0 else None,'rank1_pool':pool.stats() if rank==1 and pool else None},
            'install':install,'q0':q0,'split':split,'slot_before_release':slot_before_release,
            'final_slot':{'records':final_records,'pool':pool.stats() if rank==1 and pool else None},
            'main_storage_bytes':sum(unique.values()),'main_unique_storage_count':len(unique),'bank_formats':bank_formats,
            'main_storage_entries':entries,'bank_slot_hashes':slot_hashes_by_layer,
        }
        (OUT/f'memory_{args.scenario}_rank{rank}.json').write_text(json.dumps(rec,indent=2,default=json_default))
        print(json.dumps({'rank':rank,'scenario':args.scenario,'q0_status':q0['status'],'split_status':split['status'] if split else None,'stage_peaks':[(x['stage'],x['peak_allocated']) for x in stage_records]},indent=2))
        dist.barrier(group=control_group); destroy_model_parallel(); destroy_distributed_environment(); return

    records=[]
    def record_case(cid,layer,n0,n1,experts,mode='none',expect='PASS',inject_rank=None,requests=None,hook=None):
        r=run_one(cid,layer,n0,n1,experts,mode,inject_rank,requests,hook,True)
        r['expected_status']=expect
        common=bool(r.get('input_unchanged')) and r.get('actual_expert_inputs_equal') is True and len(r.get('actual_input_mismatches',[]))==0
        if expect=='REJECT': ok=(r['status']=='REJECTED' and r.get('dispatch_delta')==0 and bool(r.get('input_unchanged')))
        elif expect=='EMPTY': ok=(r['status']=='PASS_EMPTY' and r.get('dispatch_delta')==0 and common)
        else: ok=(r['status']=='PASS' and bool(r.get('within_tolerance')) and r.get('task_multiset_equal') is True and common and r.get('empty_legal_tasks',1)==0 and bool(r.get('candidate_finite')))
        r['case_pass']=ok; records.append(r); return r

    record_case('both_mixed_q0_noslot',49,2,2,MIXED,'none','PASS',None,[None,None])
    record_case('no_input_receives_tasks',49,4,0,RANK_MAIN[1],'none','PASS',None,[None,None])
    record_case('input_but_zero_expert_tasks',49,4,4,RANK_MAIN[0],'none','PASS',None,[None,None])
    record_case('both_empty',49,0,0,MIXED,'none','EMPTY',None,[None,None])
    record_case('invalid_rank0',49,2,2,MIXED,'none','REJECT',0,[None,None])
    record_case('valid_after_invalid_rank0',49,2,2,MIXED,'none','PASS',None,[None,None])
    record_case('invalid_rank1',49,2,2,MIXED,'none','REJECT',1,[None,None])
    record_case('valid_after_invalid_rank1',49,2,2,MIXED,'none','PASS',None,[None,None])

    reserved_lease=None; reserved_path=OUT/f'{args.tag}_unready_slot_info.json'
    if rank==1:
        reserved_lease=pool.acquire(request_id=f'{args.tag}-reserved',layer_id=49,expert_id=43)
        info={'pool_id':pool._pool_id,'ready':True,'layer':49,'expert':43,'generation':reserved_lease.generation,'owner':reserved_lease.owner_rank,'local_slot':reserved_lease.local_slot}
        reserved_path.write_text(json.dumps(info))
    dist.barrier(group=control_group)
    reserved_info=json.loads(reserved_path.read_text())
    record_case('reserved_but_claims_ready',49,2,2,MIXED,'partial','REJECT',None,[reserved_info,reserved_info])
    if rank==1: pool.abort(reserved_lease)
    dist.barrier(group=control_group)
    record_case('valid_after_reserved',49,2,2,MIXED,'none','PASS',None,[None,None])

    i49=adapter.install(49,OUT,f'{args.tag}_install49')
    si49=json.loads((OUT/f'{args.tag}_install49_install_rank1.json').read_text()); req49=[si49,si49]
    record_case('partial_slot',49,2,2,MIXED,'partial','PASS',None,req49)
    record_case('all_slot',49,2,2,MIXED,'all','PASS',None,req49)
    large=record_case('large_259_partial',49,259,259,MIXED,'partial','PASS',None,req49)
    bad0=dict(si49); bad0['generation']=int(bad0['generation'])+1
    record_case('one_rank_bad_slot',49,2,2,MIXED,'all','REJECT',None,[bad0,si49])
    record_case('valid_after_one_rank_bad',49,2,2,MIXED,'none','PASS',None,[None,None])

    i50=adapter.install(50,OUT,f'{args.tag}_transition50')
    si50=json.loads((OUT/f'{args.tag}_transition50_install_rank1.json').read_text()); req50=[si50,si50]
    record_case('replay_old_49_after_install50',49,2,2,MIXED,'all','REJECT',None,req49)
    record_case('current_50_after_reinstall',50,2,2,MIXED,'all','PASS',None,req50)
    i49b=adapter.install(49,OUT,f'{args.tag}_reinstall49')
    si49b=json.loads((OUT/f'{args.tag}_reinstall49_install_rank1.json').read_text()); req49b=[si49b,si49b]
    record_case('replay_old_generation_49',49,2,2,MIXED,'all','REJECT',None,req49)
    record_case('current_generation_49',49,2,2,MIXED,'all','PASS',None,req49b)

    def overwrite_hook():
        if rank!=1: return []
        try:
            extra=pool.acquire(request_id=f'{args.tag}-overwrite-attempt',layer_id=49,expert_id=43); pool.abort(extra)
            return ['lease_overwrite_succeeded']
        except Exception as ex:
            return [] if type(ex).__name__=='SlotPoolExhausted' else [f'overwrite_hook_error:{type(ex).__name__}:{ex}']
    record_case('fixed_lease_overwrite_blocked',49,2,2,MIXED,'all','PASS',None,req49b,overwrite_hook)

    reuse=[]
    for cycle in range(3):
        for layer in (50,49):
            inst=adapter.install(layer,OUT,f'{args.tag}_reuse{cycle}_l{layer}')
            si=json.loads((OUT/f'{args.tag}_reuse{cycle}_l{layer}_install_rank1.json').read_text())
            r=run_one(f'reuse_{cycle}_{layer}',layer,2,2,MIXED,'all',None,[si,si],None,True)
            r['case_pass']=r['status']=='PASS' and r.get('within_tolerance') is True and r.get('task_multiset_equal') is True and r.get('input_unchanged') is True and r.get('actual_expert_inputs_equal') is True
            reuse.append({'cycle':cycle,'layer':layer,'install':inst,'result':r})
    adapter.release_slot()
    final_slot={'rank':rank,'lease':None,'records':adapter.owner_records() if rank==1 and pool else [],'pool':pool.stats() if rank==1 and pool else None}
    flat_record={'bytes':int(flat.numel()*flat.element_size()),'storage_ptr':int(flat.untyped_storage().data_ptr())} if flat is not None else None
    result={**run_ctx,'run_id':args.subrun_id,'tag':args.tag,'rank':rank,'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','performance_status':'NOT_RUN','state_bytes':state_bytes,'dispatch_capacity':int(os.environ.get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','0')),'flat_buffer':flat_record,'matrix':records,'reuse':reuse,'main_storage':{'entries':entries,'unique_bytes':sum(unique.values()),'unique_storage_count':len(unique)},'bank_formats':bank_formats,'memory_before':mem_before,'memory_after_bank':memory_after_bank,'memory_final':memory_snapshot(),'stage_records':stage_records,'whole_process_peak':cumulative_peak,'adapter_memory_events':adapter.memory_events,'slot_storage':si49b,'final_slot':final_slot,'dispatch_calls':adapter.dispatch_calls}
    (OUT/f'functional_{rank}.json').write_text(json.dumps(result,indent=2,default=json_default))
    print(json.dumps({'rank':rank,'cases':[(r['case_id'],r['status'],r.get('case_pass')) for r in records],'reuse':len(reuse),'dispatch_calls':adapter.dispatch_calls,'large_global_tasks':large.get('global_expected_task_count')},indent=2))
    dist.barrier(group=control_group); destroy_model_parallel(); destroy_distributed_environment()
    if not all(r.get('case_pass') for r in records) or not all(x['result'].get('case_pass') for x in reuse): raise SystemExit(1)

if __name__=='__main__': main()
