#!/usr/bin/env python3
"""Run the unified adapter functional matrix and real W8A8/format evidence."""
from __future__ import annotations

import argparse, json, os, random, time
from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch, torch_npu, torch.distributed as dist

from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import LAYER_ATTRS, PARTS, ExpertLayer, RuntimeChain, alloc, external_state_from_bank, load_raw_state, memory_snapshot, slot_state_views, state_layout
from unified_adapter import UnifiedK1Adapter, PADDING_TAG_BASE, logical_to_comm, comm_to_logical, EMPTY_COMM, REPLICA_COMM

OUT=Path(os.environ.get('OUT_DIR','/runexp/artifacts'))


def json_default(value):
    if is_dataclass(value): return asdict(value)
    if hasattr(value, 'item'): return value.item()
    return str(value)
LAYERS=(49,50)
RANK_MAIN={0:(0,1,2,3,4,5,43,127),1:(128,129,130,131,132,133,134,255)}
MIXED=(0,43,127,5,128,129,130,255)
ATOL=0.02


def make_hidden(n,rank,seed):
    torch.manual_seed(seed+rank)
    h=torch.randn((n,6144),dtype=torch.bfloat16,device='npu')
    if n:
        h[:,0]=torch.arange(n,dtype=torch.bfloat16,device='npu')+rank*16
        h[:,1]=0
    return h


def make_case(case,layer,rank):
    n0,n1,expert_set=case['tokens'],case['tokens1'] if 'tokens1' in case else case['tokens'],case['experts']
    n=n0 if rank==0 else n1
    rng=random.Random(90000+layer*7+rank)
    logical=[]; weights=[]
    row_experts=list(expert_set)
    for t in range(n):
        row=row_experts.copy(); rng.shuffle(row)
        logical.append(row)
        w=[rng.random()+1e-3 for _ in range(8)]; s=sum(w); weights.append([x/s for x in w])
    hidden=make_hidden(n,rank,90000+layer*7)
    comm=[[logical_to_comm(e) for e in row] for row in logical]
    mode=case.get('mode','none')
    for t,row in enumerate(logical):
        for s,e in enumerate(row):
            if e==43 and (mode=='all' or (mode=='partial' and t%2==0)):
                comm[t][s]=REPLICA_COMM
    return hidden,logical,comm,weights


def write_manifests(case_id,layer,rank,hidden,logical,comm,weights):
    path=OUT/f'{case_id}_l{layer}_rank{rank}.json'
    path.write_text(json.dumps({'rank':rank,'tokens':len(logical),'logical_rows':logical,'comm_rows':comm,'weights':weights}))
    torch.save(hidden.detach().cpu(), OUT/f'{case_id}_l{layer}_hidden_rank{rank}.pt')
    dist.barrier(group=dist.group.WORLD)
    return [json.loads((OUT/f'{case_id}_l{layer}_rank{i}.json').read_text()) for i in (0,1)]


def reference(layer, rank, manifests, chains):
    # Shared files avoid equal-shape all-gather assumptions for zero-input ranks.
    case_id=manifests[0]['case_id']
    hs=[torch.load(OUT/f'{case_id}_l{layer}_hidden_rank{i}.pt',map_location='cpu') for i in (0,1)]
    offsets=[]; total=0
    for h in hs:
        offsets.append(total); total += h.shape[0]
    gh=torch.cat(hs,dim=0).to('npu') if total else torch.empty((0,6144),dtype=torch.bfloat16,device='npu')
    partial=torch.zeros((total,6144),dtype=torch.float32,device='npu')
    for expert in RANK_MAIN[rank]:
        idx=[]; ws=[]
        for sender,m in enumerate(manifests):
            for token,row in enumerate(m['logical_rows']):
                if expert in row:
                    i=offsets[sender]+token; s=row.index(expert)
                    idx.append(i); ws.append(m['weights'][token][s])
        if idx:
            it=torch.tensor(idx,dtype=torch.int64,device='npu'); wt=torch.tensor(ws,dtype=torch.float32,device='npu').unsqueeze(1)
            out=chains[(layer,expert)](gh.index_select(0,it))
            partial.index_add_(0,it,out.float()*wt)
    dist.all_reduce(partial,op=dist.ReduceOp.SUM,group=dist.group.WORLD)
    return partial[offsets[rank]:offsets[rank]+hs[rank].shape[0]].contiguous()


def expected_tasks(manifests,rank):
    lo,hi=(0,129) if rank==0 else (129,258)
    return sorted((m['rank'],t,s,c) for m in manifests for t,row in enumerate(m['comm_rows']) for s,c in enumerate(row) if lo<=c<hi)


def actual_tasks(adapter_result,manifests):
    out=[]
    for sender,token,comm in adapter_result['actual_task_vector']:
        slots=[s for s,c in enumerate(manifests[sender]['comm_rows'][token]) if c==comm]
        if len(slots)!=1: raise RuntimeError(('slot ambiguity',sender,token,comm,slots))
        out.append((sender,token,slots[0],comm))
    return sorted(out)


def determine_case(inputs,ref,result,manifests,rank):
    if result['status']!='PASS':
        return {'correct':False,'error':None,'actual_tasks':[],'expected_tasks':[]}
    out=result
    # execute returns tuple for pass; caller passes result only? handled outside
    return None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scenario',choices=('matrix','baseline','candidate'),default='matrix'); ap.add_argument('--tag',default='functional'); args=ap.parse_args()
    rank=int(os.environ['RANK']); torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment,destroy_model_parallel,get_tp_group,initialize_model_parallel
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    from sglang.srt.layers.moe.utils import DeepEPMode,initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    init_distributed_environment(world_size=-1,rank=-1,local_rank=-1,backend='hccl')
    sa=ServerArgs(model_path='dummy',tp_size=2,ep_size=2); sa.moe_a2a_backend='deepep'; sa.moe_runner_backend='ascend'
    set_global_server_args_for_scheduler(sa); publish(sa,role='scheduler'); initialize_moe_config(); initialize_model_parallel(tensor_model_parallel_size=2,expert_model_parallel_size=2)
    group=get_tp_group().device_group
    dispatcher=DeepEPDispatcher(group=group,router_topk=8,num_experts=258,num_local_experts=129,hidden_size=6144,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False)
    import sglang.srt.layers.quantization
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    method=NPUW8A8Int8MoEMethod(); matmul=GroupedMatmul()
    mem_before=memory_snapshot(); bank={}; slot_hashes_by_layer={}
    for layer in LAYERS:
        for expert in RANK_MAIN[rank]:
            bs,_,offsets,_,sh=load_raw_state(layer,expert)
            mod=ExpertLayer(bs)
            for part in PARTS:
                attr=LAYER_ATTRS[part]; getattr(mod,attr).data=getattr(mod,attr).data.npu()
            method.process_weights_after_loading(mod,'w13'); method.process_weights_after_loading(mod,'w2')
            bank[(layer,expert)]=mod
            if expert==43: slot_hashes_by_layer[layer]=sh
    torch.npu.synchronize(); chains={(l,e):RuntimeChain(method,matmul,bank[(l,e)],l) for l,e in bank}
    entries=[]; unique={}
    for (l,e),mod in bank.items():
        for attr in LAYER_ATTRS.values():
            t=getattr(mod,attr,None)
            if t is None: continue
            st=t.untyped_storage(); item={'layer':l,'expert':e,'attr':attr,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'data_ptr':int(t.data_ptr()),'storage_ptr':int(st.data_ptr()),'storage_bytes':int(st.nbytes()),'offset':int(t.storage_offset())}
            entries.append(item); unique[item['storage_ptr']]=item['storage_bytes']
    # Format basis: query is real, production setting is deliberately unknown.
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
    # One flat source buffer, only on rank0; one pool slot only on rank1.
    if rank==0:
        sample=external_state_from_bank(bank[(49,43)])
        layout,state_bytes=state_layout(sample)
        flat=torch.empty(state_bytes,dtype=torch.uint8,device='npu')
        meta=OUT/f'{args.tag}_meta.json'; meta.write_text(json.dumps({'state_bytes':state_bytes,'layout':[{'name':n,'offset':o,'bytes':b,'dtype':str(d),'shape':list(s)} for n,o,b,d,s in layout],'slot_hashes':slot_hashes_by_layer},indent=2))
        pool=None; lease=None
        conv_alias=[]
        for name,_,_,_,_ in layout:
            src=bank[(49,43)]
            try:
                # conversion storage identity is captured from the actual source tensor.
                src_t=external_state_from_bank(src)[name]
                bank_t=getattr(src,LAYER_ATTRS[name])
                conv_alias.append({'name':name,'bank_storage_ptr':int(bank_t.untyped_storage().data_ptr()),'converted_storage_ptr':int(src_t.untyped_storage().data_ptr()),'alias':int(bank_t.untyped_storage().data_ptr())==int(src_t.untyped_storage().data_ptr())})
            except Exception as ex:
                conv_alias.append({'name':name,'error':f'{type(ex).__name__}:{ex}'})
        del sample
    else:
        meta=json.loads((OUT/f'{args.tag}_meta.json').read_text()) if False else None
    dist.barrier(group=group)
    if rank==1:
        meta=json.loads((OUT/f'{args.tag}_meta.json').read_text()); state_bytes=meta['state_bytes']
        dtype_map={'torch.int8':torch.int8,'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}
        layout=[(x['name'],x['offset'],x['bytes'],dtype_map[x['dtype']],tuple(x['shape'])) for x in meta['layout']]
        pool = None if args.scenario == 'baseline' else GlobalSlotPool([1],state_bytes,alloc,name='unified-adapter-slot',synchronize=torch.npu.synchronize)
        lease=None
        flat=None
        conv_alias=[]
    else:
        pool=lease=None
    adapter=UnifiedK1Adapter(rank=rank,group=group,dispatcher=dispatcher,bank_chains=chains,main_experts_by_rank=RANK_MAIN,layers=LAYERS,pool=pool,layout=layout,state_bytes=state_bytes,flat=flat,method=method,matmul=matmul)

    # Shared slot-info helper reads the target rank's actual install record.
    def shared_slot_info(tag):
        p=OUT/f'{tag}_install_rank1.json'
        dist.barrier(group=group)
        if p.exists(): return json.loads(p.read_text())
        return None

    def run_one(case_id,layer,tokens0,tokens1,experts_set,mode='none',inject_rank=None,slot_info=None,expected_generation=None):
        n=tokens0 if rank==0 else tokens1
        rng=random.Random(70000+layer*17+rank)
        logical=[]; weights=[]
        for t in range(n):
            row=list(experts_set); rng.shuffle(row); logical.append(row)
            w=[rng.random()+1e-3 for _ in range(8)]; s=sum(w); weights.append([x/s for x in w])
        hidden=make_hidden(n,rank,70000+layer*17)
        comm=[[logical_to_comm(e) for e in row] for row in logical]
        for t,row in enumerate(logical):
            for s,e in enumerate(row):
                if e==43 and (mode=='all' or (mode=='partial' and t%2==0)): comm[t][s]=REPLICA_COMM
        manifest={'rank':rank,'case_id':case_id,'tokens':n,'logical_rows':logical,'comm_rows':comm,'weights':weights}
        (OUT/f'{case_id}_l{layer}_rank{rank}.json').write_text(json.dumps(manifest)); torch.save(hidden.detach().cpu(),OUT/f'{case_id}_l{layer}_hidden_rank{rank}.pt')
        dist.barrier(group=group)
        manifests=[json.loads((OUT/f'{case_id}_l{layer}_rank{i}.json').read_text()) for i in (0,1)]
        ref=reference(layer,rank,manifests,chains) if (tokens0+tokens1)>0 else torch.empty((0,6144),dtype=torch.float32,device='npu')
        inject=inject_rank==rank
        ret=adapter.execute(case_id=case_id,layer=layer,hidden=hidden,logical_rows=logical,weights=weights,mode=mode,slot_info=slot_info,inject_invalid_comm=inject,expected_generation=expected_generation)
        if isinstance(ret,tuple): result,out=ret
        else: result,out=ret,None
        exp=expected_tasks(manifests,rank)
        act=actual_tasks(result,manifests) if result['status']=='PASS' else []
        if result['status']=='PASS':
            err=float((out.float()-ref).abs().max().item()) if out.numel() else 0.0
            result.update({'reference_max_abs':err,'within_tolerance':err<=ATOL,'expected_tasks':len(exp),'actual_tasks':len(act),'task_multiset_equal':exp==act,'reference_finite':bool(torch.isfinite(ref).all().item())})
        return result

    # Independent-process baseline/candidate memory modes.
    if args.scenario in ('baseline','candidate'):
        if args.scenario=='candidate':
            inst=adapter.install(49,OUT,f'{args.tag}_mem_install')
            si=json.loads((OUT/f'{args.tag}_mem_install_install_rank1.json').read_text())
            r=run_one('memory_candidate',49,2,2,MIXED,mode='partial',slot_info=si)
        else:
            r=run_one('memory_baseline',49,2,2,MIXED,mode='none',slot_info=None)
        out={'scenario':args.scenario,'rank':rank,'memory_before':mem_before,'memory_after_bank':memory_snapshot(),'memory_final':memory_snapshot(),'result':r,'adapter_memory_events':adapter.memory_events,'main_storage_bytes':sum(unique.values()),'slot_storage':adapter.slot_info() if adapter.lease else None}
        (OUT/f'memory_{args.scenario}_rank{rank}.json').write_text(json.dumps(out,indent=2))
        print(json.dumps(out,indent=2)); dist.barrier(group=group); destroy_model_parallel(); destroy_distributed_environment(); return

    # Matrix: no slot first, then valid q0, negatives, then install.
    cases=[]
    def add(cid,layer,n0,n1,experts,mode='none',expect='PASS',inject_rank=None,slot_kind='none',gen_delta=0):
        cases.append(dict(id=cid,layer=layer,n0=n0,n1=n1,experts=experts,mode=mode,expect=expect,inject_rank=inject_rank,slot_kind=slot_kind,gen_delta=gen_delta))
    add('both_mixed_q0_noslot',49,2,2,MIXED,'none')
    add('rank0_only',49,4,0,RANK_MAIN[0],'none')
    add('rank1_only',49,0,4,RANK_MAIN[1],'none')
    add('both_empty',49,0,0,MIXED,'none')
    add('invalid_rank0',49,2,2,MIXED,'none','REJECT',0)
    add('valid_after_invalid_rank0',49,2,2,MIXED,'none')
    add('invalid_rank1',49,2,2,MIXED,'none','REJECT',1)
    add('valid_after_invalid_rank1',49,2,2,MIXED,'none')
    add('unready_slot',49,2,2,MIXED,'partial','REJECT',None,'none')
    add('valid_after_unready',49,2,2,MIXED,'none')
    # Install valid slot before partial/all.
    inst49=adapter.install(49,OUT,f'{args.tag}_install49')
    si49=json.loads((OUT/f'{args.tag}_install49_install_rank1.json').read_text())
    add('partial_slot',49,2,2,MIXED,'partial','PASS',None,'ready')
    add('all_slot',49,2,2,MIXED,'all','PASS',None,'ready')
    add('wrong_layer',50,2,2,MIXED,'all','REJECT',None,'ready')
    add('valid_after_wrong_layer',50,2,2,MIXED,'none')
    add('stale_generation',49,2,2,MIXED,'all','REJECT',None,'ready',1)
    add('valid_after_stale',49,2,2,MIXED,'none')
    records=[]
    for c in cases:
        si=None
        if c['slot_kind']=='ready':
            si=dict(si49)
            if c['gen_delta']: si['generation']=int(si['generation'])+c['gen_delta']
            if c['id']=='wrong_layer': si['layer']=49
        expected_gen = int(si49['generation']) if c['id']=='stale_generation' else None
        r=run_one(c['id'],c['layer'],c['n0'],c['n1'],c['experts'],c['mode'],c['inject_rank'],si,expected_gen)
        r['expected_status']=c['expect']; r['case_pass']=(r['status']=='REJECTED' if c['expect']=='REJECT' else r['status'] in ('PASS','PASS_EMPTY')) and (c['expect']=='REJECT' or r.get('within_tolerance',True))
        if c['expect']=='REJECT': r['case_pass']=r['status']=='REJECTED' and r.get('dispatch_delta',1)==0
        records.append(r)

    # 49->50->49 cycles, all expert43 tasks to replica; same storage must persist.
    reuse=[]
    for cycle in range(3):
        for layer in (50,49):
            inst=adapter.install(layer,OUT,f'{args.tag}_reuse{cycle}_l{layer}')
            si=json.loads((OUT/f'{args.tag}_reuse{cycle}_l{layer}_install_rank1.json').read_text())
            r=run_one(f'reuse_{cycle}_{layer}',layer,2,2,MIXED,'all',None,si)
            reuse.append({'cycle':cycle,'layer':layer,'install':inst,'result':r})
    adapter.release_slot()
    final_slot={'rank':rank,'lease':None,'pool':pool.stats() if pool else None}
    flat_record = ({'bytes': int(flat.numel()*flat.element_size()), 'storage_ptr': int(flat.untyped_storage().data_ptr())} if rank == 0 else None)
    result={'run_id':args.tag,'rank':rank,'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','flat_buffer':flat_record,'matrix':records,'reuse':reuse,'main_storage':{'entries':entries,'unique_bytes':sum(unique.values()),'unique_storage_count':len(unique)},'bank_formats':bank_formats,'conversion_alias':conv_alias,'memory_before':mem_before,'memory_after_bank':memory_snapshot(),'memory_final':memory_snapshot(),'adapter_memory_events':adapter.memory_events,'slot_storage':adapter.slot_info(),'final_slot':final_slot,'dispatch_calls':adapter.dispatch_calls}
    (OUT/f'functional_{rank}.json').write_text(json.dumps(result,indent=2,default=json_default))
    print(json.dumps({'rank':rank,'cases':[(r['case_id'],r['status'],r['case_pass']) for r in records],'reuse':len(reuse),'dispatch_calls':adapter.dispatch_calls},indent=2))
    dist.barrier(group=group); destroy_model_parallel(); destroy_distributed_environment()
    if not all(r['case_pass'] for r in records) or not all(x['result'].get('within_tolerance',False) for x in reuse): raise SystemExit(1)

if __name__=='__main__': main()
