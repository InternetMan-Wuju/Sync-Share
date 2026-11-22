#!/usr/bin/env python3
"""EP16 layer49 routed-MoE replay: original, native EPLB layout, global K=1.

Inputs are layer49 hidden states and top-k decisions captured from a real EP16
Prefill request. The script keeps native DeepEP dispatch/combine and real W8A8
expert computation, but replays only the routed MoE sublayer.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, time
from pathlib import Path
from types import SimpleNamespace
import torch, torch_npu, torch.distributed as dist

from real_expert_chain_rework import (
    PARTS, LAYER_ATTRS, ExpertLayer, RuntimeChain, external_state_from_bank,
    load_raw_state, memory_snapshot, slot_state_views, state_layout,
)

OUT=Path(os.environ.get('OUT_DIR','/runexp/artifacts'))
LAYER=49
EP=16
TOP_K=8
MAIN_PER_RANK=16
PLACEHOLDER_PER_RANK=17
HOT_COMM=16


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sync(): torch.npu.synchronize()


def rank_unique_bytes(bank):
    seen={}
    for _,chain in bank.items():
        for attr in LAYER_ATTRS.values():
            t=getattr(chain.layer,attr,None)
            if t is not None:
                st=t.untyped_storage(); seen[int(st.data_ptr())]=int(st.nbytes())
    return sum(seen.values()),len(seen)


def load_chain(expert, method, matmul):
    bank_state,slot_state,offsets,bank_hashes,slot_hashes=load_raw_state(LAYER,expert)
    mod=ExpertLayer(bank_state)
    for part in PARTS:
        getattr(mod,LAYER_ATTRS[part]).data=getattr(mod,LAYER_ATTRS[part]).data.npu()
    method.process_weights_after_loading(mod,'w13'); method.process_weights_after_loading(mod,'w2')
    return RuntimeChain(method,matmul,mod,LAYER), slot_state, bank_hashes, slot_hashes


def compute_local(dispatcher, recv, counts, chains_by_local):
    out=torch.zeros_like(recv); pos=0; computed=[]
    for local_idx,count in enumerate(counts):
        count=int(count)
        if count<=0: continue
        chain=chains_by_local.get(local_idx)
        if chain is None:
            raise RuntimeError(f'task arrived for absent local slot {local_idx}')
        out[pos:pos+count].copy_(chain(recv[pos:pos+count])); computed.append((local_idx,count)); pos+=count
    if pos!=int(sum(counts)): raise RuntimeError('row accounting mismatch')
    return out,computed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--observer-dir',type=Path,required=True)
    ap.add_argument('--layout',type=Path,required=True)
    ap.add_argument('--layout-summary',type=Path,required=True)
    ap.add_argument('--size',required=True,choices=('8k','16k','32k'))
    ap.add_argument('--reps',type=int,default=10)
    ap.add_argument('--copy-reps',type=int,default=5)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    rank=int(os.environ['RANK']); local_rank=int(os.environ['LOCAL_RANK']); torch.npu.set_device(local_rank)
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import get_moe_ep_group,initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher,DeepEPNormalCombineInput
    from sglang.srt.layers.moe.utils import DeepEPMode,initialize_moe_config
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    init_distributed_environment(world_size=-1,rank=-1,local_rank=-1,backend='hccl')
    sa=ServerArgs(model_path='dummy',tp_size=EP,ep_size=EP);sa.moe_a2a_backend='deepep';sa.moe_runner_backend='ascend'
    set_global_server_args_for_scheduler(sa);publish(sa,role='scheduler');initialize_moe_config();initialize_model_parallel(tensor_model_parallel_size=EP,expert_model_parallel_size=EP)
    g=get_moe_ep_group(); device_group=g.device_group; cpu_group=g.cpu_group
    method=NPUW8A8Int8MoEMethod(); matmul=GroupedMatmul()
    layout=json.loads(a.layout.read_text())['physical_to_logical_map']; summary=json.loads(a.layout_summary.read_text())
    phy=layout[LAYER]; assert len(phy)==EP*17
    hot=int(summary['top_hot'][0]['logical'])
    hot_source=min(range(EP),key=lambda r:0 if r*16<=hot<(r+1)*16 else 999)
    assert hot_source*16<=hot<(hot_source+1)*16
    seqs={'8k':[0],'16k':[1,2],'32k':[3,4,5,6]}[a.size]

    memory_events=[]
    def mem(stage): memory_events.append({'stage':stage,'rank':rank,'memory':memory_snapshot()})

    mem('start')
    main_bank={}; main_slot_state=None; main_layout=None; main_state_bytes=None; main_hashes={}
    for expert in range(rank*16,(rank+1)*16):
        chain,slot_state,bh,sh=load_chain(expert,method,matmul);main_bank[expert]=chain
        if expert==rank*16: main_slot_state=slot_state;main_layout,main_state_bytes=state_layout(slot_state)
        main_hashes[expert]=bh
    sync();main_bytes,main_storage_count=rank_unique_bytes(main_bank);mem('main_bank_loaded')

    eplb_bank={};eplb_map={}
    for local_idx in range(17):
        expert=int(phy[rank*17+local_idx]); chain,ss,bh,sh=load_chain(expert,method,matmul)
        eplb_bank[local_idx]=chain;eplb_map[local_idx]=expert
    sync();eplb_bytes,eplb_storage_count=rank_unique_bytes(eplb_bank);mem('eplb_bank_loaded')

    # One real global K=1 slot on owner rank0. Other ranks have no slot allocation.
    replica_chain=None;slot=None;slot_bytes=0;copy_timing=None
    if rank==0:
        slot=torch.empty(main_state_bytes,dtype=torch.uint8,device='npu')
        if hot_source==0:
            t=time.perf_counter_ns();src=external_state_from_bank(main_bank[hot].layer)
            for name,off,nbytes,_,_ in main_layout: slot[off:off+nbytes].copy_(src[name].view(torch.uint8).reshape(-1))
            sync();copy_timing={'source_prepare_ms':(time.perf_counter_ns()-t)/1e6,'transfer_ms':0.0,'install_ms':0.0,'source_rank':0,'dst_rank':0,'bytes':main_state_bytes}
        else:
            src_rank=hot_source
        views=slot_state_views(slot,main_layout)
        ext=SimpleNamespace(**{LAYER_ATTRS[p]:views[p] for p in PARTS})
        replica_chain=RuntimeChain(method,matmul,ext,LAYER);slot_bytes=int(slot.numel())
    else:
        src_rank=hot_source
    if rank==0 and hot_source!=0:
        t=time.perf_counter_ns();t1=time.perf_counter_ns();dist.recv(slot,src=hot_source,group=device_group);sync();t2=time.perf_counter_ns()
        views=slot_state_views(slot,main_layout);ext=SimpleNamespace(**{LAYER_ATTRS[p]:views[p] for p in PARTS});replica_chain=RuntimeChain(method,matmul,ext,LAYER)
        copy_timing={'source_prepare_ms':0.0,'transfer_ms':(t2-t1)/1e6,'install_ms':0.0,'source_rank':hot_source,'dst_rank':0,'bytes':main_state_bytes}
    elif rank==hot_source and hot_source!=0:
        t=time.perf_counter_ns();src=external_state_from_bank(main_bank[hot].layer)
        for name,off,nbytes,_,_ in main_layout: slot=torch.empty(main_state_bytes,dtype=torch.uint8,device='npu') if slot is None else slot; slot[off:off+nbytes].copy_(src[name].view(torch.uint8).reshape(-1))
        sync();dist.send(slot,dst=0,group=device_group);sync()
        copy_timing={'source_prepare_ms':(time.perf_counter_ns()-t)/1e6,'transfer_ms':0.0,'install_ms':0.0,'source_rank':hot_source,'dst_rank':0,'bytes':main_state_bytes}
        del slot
    dist.barrier(group=cpu_group)
    mem('k1_slot_ready')

    # Steady-state install diagnostic: repeat the same 37.8 MB hot-expert copy
    # after all one-time HCCL and format-cast setup has happened.
    copy_benchmark=[]
    if hot_source != 0:
        src_buf=torch.empty(main_state_bytes,dtype=torch.uint8,device='npu') if rank==hot_source else None
        for rep in range(a.copy_reps):
            prep_ms=0.0; transfer_ms=0.0; install_ms=0.0
            if rank==hot_source:
                t=time.perf_counter_ns();src=external_state_from_bank(main_bank[hot].layer)
                for name,off,nbytes,_,_ in main_layout: src_buf[off:off+nbytes].copy_(src[name].view(torch.uint8).reshape(-1))
                sync();prep_ms=(time.perf_counter_ns()-t)/1e6
                t=time.perf_counter_ns();dist.send(src_buf,dst=0,group=device_group);sync();transfer_ms=(time.perf_counter_ns()-t)/1e6
            elif rank==0:
                t=time.perf_counter_ns();dist.recv(slot,src=hot_source,group=device_group);sync();transfer_ms=(time.perf_counter_ns()-t)/1e6
            dist.barrier(group=cpu_group)
            row={'rep':rep,'source_rank':hot_source,'dst_rank':0,'bytes':main_state_bytes,'source_prepare_ms':prep_ms,'transfer_ms':transfer_ms,'install_ms':install_ms}
            gathered=[None]*EP;dist.all_gather_object(gathered,row,group=cpu_group)
            if rank==0: copy_benchmark.append(gathered[hot_source])

    d_orig=DeepEPDispatcher(group=device_group,router_topk=8,num_experts=256,num_local_experts=16,hidden_size=6144,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False)
    d_place=DeepEPDispatcher(group=device_group,router_topk=8,num_experts=272,num_local_experts=17,hidden_size=6144,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False)

    # Native DeepSeek EPLB physical dispatch map for this sender rank.
    inv=torch.full((78,256,272),-1,dtype=torch.int64)
    for layer in range(78):
        for e in range(256): inv[layer,e,0]=e
    # Only layer49 has a data-driven layout; other layers are only present so
    # the native mapper sees a complete 78-layer inverse map.
    for e in range(256): inv[LAYER,e,:]=-1
    for p,e in enumerate(phy):
        for slot in range(272):
            if int(inv[LAYER,e,slot])==-1: inv[LAYER,e,slot]=p;break
    from sglang.srt.eplb.expert_location import compute_logical_to_rank_dispatch_physical_map
    native_dispatch_map=compute_logical_to_rank_dispatch_physical_map(inv,ep_size=EP,num_physical_experts=272,ep_rank=rank,seed=42)[LAYER].to('npu')

    def read_chunk(seq):
        base=a.observer_dir
        hidden=torch.load(base/f'rank{rank}_layer49_seq{seq}_hidden.pt',map_location='npu')
        ids=torch.load(base/f'rank{rank}_layer49_seq{seq}_topk_ids.pt',map_location='npu').long()
        weights=torch.load(base/f'rank{rank}_layer49_seq{seq}_topk_weights.pt',map_location='npu').float()
        assert hidden.shape[0]==ids.shape[0]==weights.shape[0]
        return hidden,ids,weights

    def run_arm(arm,hidden,logical_ids,weights):
        if arm=='original':
            dispatcher=d_orig;dispatch_ids=logical_ids;chains={i:main_bank[rank*16+i] for i in range(16)}
        elif arm=='eplb':
            dispatcher=d_place;dispatch_ids=native_dispatch_map[logical_ids.reshape(-1)].reshape_as(logical_ids);chains=eplb_bank
        elif arm=='ours':
            dispatcher=d_place
            flat_logical=logical_ids.reshape(-1);main_comm=torch.where(flat_logical<0,flat_logical,(flat_logical//16)*17+(flat_logical%16))
            mask=flat_logical==hot;ordinal=mask.long().cumsum(0);q=(int(mask.sum())//2)
            move=mask&(ordinal<=q);flat_comm=torch.where(move,torch.full_like(main_comm,HOT_COMM),main_comm);dispatch_ids=flat_comm.reshape_as(logical_ids)
            chains={i:main_bank[rank*16+i] for i in range(16)}
            if rank==0: chains[16]=replica_chain
        else: raise ValueError(arm)
        sync();t0=time.perf_counter_ns()
        dout=dispatcher.dispatch(hidden,StandardTopKOutput(weights,dispatch_ids,None))
        recv=dout.hidden_states[0] if isinstance(dout.hidden_states,tuple) else dout.hidden_states
        counts=[int(x) for x in dout.num_recv_tokens_per_expert]
        expert_out,_=compute_local(dispatcher,recv,counts,chains)
        combined=dispatcher.combine(DeepEPNormalCombineInput(expert_out,None,None));sync()
        wall_ms=(time.perf_counter_ns()-t0)/1e6
        per_rank=[None]*EP;dist.all_gather_object(per_rank,{'rank':rank,'tasks':int(sum(counts)),'counts':counts},group=cpu_group)
        if arm=='eplb':
            extra=sum(counts[i] for i in range(17) if int(summary['replica_count_per_logical'][int(eplb_map[i])])>1)
        elif arm=='ours':
            extra=int(counts[HOT_COMM]) if rank==0 else 0
        else:
            extra=0
        return combined,wall_ms,per_rank,extra,int(dispatch_ids.numel())

    # Warm up dispatch, grouped GMM, combine and the 272-ID path before timing.
    chunks=[read_chunk(s) for s in seqs]
    for _ in range(3):
        for arm in ('original','eplb','ours'):
            for h,i,w in chunks:run_arm(arm,h,i,w)
    records=[];baselines={}
    orders=[('original','eplb','ours'),('eplb','ours','original'),('ours','original','eplb')]
    for rep in range(a.reps):
        rep_rows={}
        outputs={}
        for arm in orders[rep%3]:
            total=0.0;chunk_rows=[];extras=0
            for ci,(h,i,w) in enumerate(chunks):
                out,ms,pr,extra,tasks=run_arm(arm,h,i,w)
                if arm=='original': baselines[ci]=out.clone()
                outputs[(arm,ci)]=out
                total+=ms;extras+=extra
                chunk_rows.append({'chunk':ci,'rows':int(h.shape[0]),'wall_ms':ms,'tasks':tasks,'replica_tasks':extra,'rank_tasks':pr})
            rep_rows[arm]={'total_wall_ms':total,'chunks':chunk_rows,'replica_tasks':extras}
        for arm in ('eplb','ours'):
            errs=[]
            for ci in range(len(chunks)):
                base=baselines[ci].float();got=outputs[(arm,ci)].float();d=(got-base).abs()
                errs.append({'chunk':ci,'max_abs':float(d.max().item()) if d.numel() else 0.0,'nrmse':float((d.square().mean().sqrt()/base.square().mean().sqrt().clamp_min(1e-12)).item()) if d.numel() else 0.0,'finite':bool(torch.isfinite(got).all().item())})
            rep_rows[arm]['error_vs_original']=errs
        records.append({'rep':rep,'order':list(orders[rep%3]),'arms':rep_rows})
    local={'kind':'replay_result','rank':rank,'size':a.size,'seqs':seqs,'hot_expert':hot,'hot_source_rank':hot_source,'layout_ours_local':[int(x) for x in phy[rank*17:(rank+1)*17]],'memory_events':memory_events,'main_bytes':main_bytes,'main_storage_count':main_storage_count,'eplb_bytes':eplb_bytes,'eplb_storage_count':eplb_storage_count,'k1_slot_bytes':slot_bytes,'k1_copy_timing':copy_timing,'k1_copy_benchmark':copy_benchmark,'records':records,'main_hashes':main_hashes,'hidden_rows':sum(int(h.shape[0]) for h,_,_ in chunks)}
    a.out.parent.mkdir(parents=True,exist_ok=True);(a.out/f'replay_rank{rank}.json').write_text(json.dumps(local,indent=2))
    print(json.dumps({'rank':rank,'size':a.size,'rows':local['hidden_rows'],'main_bytes':main_bytes,'eplb_bytes':eplb_bytes,'slot_bytes':slot_bytes},indent=2),flush=True)
    dist.barrier(group=cpu_group);destroy_model_parallel();destroy_distributed_environment()

if __name__=='__main__':main()
