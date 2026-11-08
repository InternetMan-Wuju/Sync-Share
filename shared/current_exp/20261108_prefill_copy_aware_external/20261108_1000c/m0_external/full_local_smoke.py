#!/usr/bin/env python3
"""Full local two-rank expert path driven by the external top-k plugin."""
from __future__ import annotations
import json, os
from pathlib import Path
import torch, torch_npu

OUT=Path('/external/artifacts/full_local_smoke.json')
os.environ.setdefault('SGLANG_PREFILL_COPY_AWARE_MODE','off')

class FullStateTransport:
    def __init__(self, source, destination, src_offsets):
        self.source=source; self.destination=destination; self.src_offsets=src_offsets
        self.calls=0
        self.dest_offsets={k:torch.empty_like(v) for k,v in src_offsets.items()}
    def is_cached(self,*args): return False
    def prepare_copy(self,*args):
        self.destination.w13[1].copy_(self.source.w13[0])
        self.destination.down[1].copy_(self.source.down[0])
        self.destination.w13_scale[1].copy_(self.source.w13_scale[0])
        self.destination.down_scale[1].copy_(self.source.down_scale[0])
        for key,value in self.src_offsets.items(): self.dest_offsets[key].copy_(value)
        self.calls+=1

def make_info(device):
    primary=[]
    for e in range(32):
        r,s=divmod(e,16);primary.append(r*17+s)
    all_map=torch.zeros((32,2),dtype=torch.int64,device=device)
    all_map[:,0]=torch.tensor(primary,device=device);all_map[:,1]=torch.tensor(primary,device=device)
    from types import SimpleNamespace
    return SimpleNamespace(ep_dispatch_algorithm='static',partial_logical_to_all_physical_map=all_map,partial_logical_to_rank_dispatch_physical_map=torch.tensor(primary,device=device))

def make_logits(device):
    rows=512
    logits=torch.full((rows,32),-100.0,dtype=torch.float32,device=device)
    logits[:,0]=10.0
    for i in range(256):
        for e in (1,2,3,16,17,18,19): logits[i,e]=5.0
    for i in range(256,512):
        for e in (1,2,3,4,16,17,18): logits[i,e]=5.0
    return logits

def make_case(x, counts, device):
    n=x.shape[0]
    return {'x':x,'permutation':torch.arange(n,device=device),'inverse':torch.arange(n,device=device),'group_list':torch.tensor(counts,dtype=torch.int64,device=device)}

def run_expert(bank, hidden, idx, counts):
    return bank.chain(make_case(hidden[idx],counts,hidden.device))

def compare(a,b):
    d=a.float()-b.float()
    return {'bitwise':bool(torch.equal(a,b)),'max_abs':float(d.abs().max()),'nrmse':float(torch.linalg.vector_norm(d)/max(torch.linalg.vector_norm(b.float()),1e-12)),'nonfinite':int((~torch.isfinite(a)).sum())}

def main():
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    import m0_external.controller as c
    from m0_external.install import install,uninstall
    from sglang.srt.layers.moe.topk import TopKConfig
    from s2_local_cost import ExpertBank,load_experts
    install();torch.npu.set_device(0);dev=torch.device('npu:0');set_is_extend_in_batch(True)
    from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
    raw=load_experts(Path('/models/glm-q'),[43,40])
    src43=ExpertBank('src43',[43],{43:raw[43]})
    src40=ExpertBank('src40',[40],{40:raw[40]})
    rank1=ExpertBank('rank1_e40_e43',[40,43],{40:raw[40],43:raw[43]})
    # Sentinel: destination group1 starts as e40, so a missing copy is detectable.
    rank1.w13[1].copy_(src40.w13[0]);rank1.down[1].copy_(src40.down[0])
    rank1.w13_scale[1].copy_(src40.w13_scale[0]);rank1.down_scale[1].copy_(src40.down_scale[0])
    src_offsets={p:raw[43][f'{p}.weight_offset'].npu() for p in ('gate_proj','up_proj','down_proj')}
    transport=FullStateTransport(src43,rank1,src_offsets)
    cfg=c.ServiceRuntimeConfig(ep_size=2,local_slots_per_rank=17,destination_rank=1,destination_local_slot=16,cost_model=c.LinearCostModel(per_task_us=1),transport=transport,planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,weight_version=3,layout_version='full-local')
    ctl=c.RuntimeController(c.Mode.ACTIVE,service_config=cfg);c.register_runtime_controller(ctl)
    logits=make_logits(dev);cfg_topk=TopKConfig(top_k=8,use_grouped_topk=False,renormalize=False,scoring_func='softmax')
    hidden=torch.randn((512,6144),generator=torch.Generator().manual_seed(20261108),dtype=torch.float32).to(torch.bfloat16).npu()
    os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='off';off=fused_topk_npu(hidden_states=hidden,router_logits=logits,topk_config=cfg_topk,expert_location_dispatch_info=make_info(dev),layer_id=49)
    # Baseline e43 contribution for every token.
    baseline43=run_expert(src43,hidden,torch.arange(512,device=dev),[512])
    e0mask=off.topk_ids==0
    w0=off.topk_weights[e0mask].flatten()  # one e0 branch per token
    os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='active';active=fused_topk_npu(hidden_states=hidden,router_logits=logits,topk_config=cfg_topk,expert_location_dispatch_info=make_info(dev),layer_id=49)
    plan=next(iter(ctl.last_plan.values()));ctl.slots.commit_epoch(ctl.epoch,[0,1],[0,1]);ctl.slots.validate_dispatch(49,1,16,plan.logical_expert,ctl.epoch)
    idx0=torch.where(active.topk_ids==0)[0];idx33=torch.where(active.topk_ids==33)[0]
    out_active=torch.empty_like(baseline43)
    out_active[idx0]=run_expert(src43,hidden,idx0,[len(idx0)])
    out_active[idx33]=run_expert(rank1,hidden,idx33,[0,len(idx33)])
    base_contrib=baseline43*w0[:,None];active_contrib=out_active*w0[:,None]
    numeric=compare(active_contrib,base_contrib)
    # The copied e43 state must equal the source, including zero offsets.
    state={}
    for name,a,b in [('w13',src43.w13[0],rank1.w13[1]),('down',src43.down[0],rank1.down[1]),('w13_scale',src43.w13_scale[0],rank1.w13_scale[1]),('down_scale',src43.down_scale[0],rank1.down_scale[1])]: state[name]=compare(a,b)
    state['offsets_zero']=all(int(v.abs().sum())==0 for v in transport.dest_offsets.values())
    result={'status':'NPU_FULL_LOCAL_PASS_TEST_ONLY' if numeric['bitwise'] and all(v['bitwise'] for v in state.values() if isinstance(v,dict)) and transport.calls==1 else 'NPU_FULL_LOCAL_FAIL','device_name':torch.npu.get_device_name(0),'topk':'top8','tokens':512,'off_e0_count':int((off.topk_ids==0).sum()),'active_primary_e0':int((active.topk_ids==0).sum()),'active_destination_e0':int((active.topk_ids==33).sum()),'plan':plan.to_dict(),'transport_calls':transport.calls,'copy_bytes':int(src43.w13[0].numel()*src43.w13[0].element_size()+src43.down[0].numel()*src43.down[0].element_size()+src43.w13_scale[0].numel()*src43.w13_scale[0].element_size()+src43.down_scale[0].numel()*src43.down_scale[0].element_size()+sum(v.numel()*v.element_size() for v in transport.src_offsets.values())),'numeric':numeric,'state':state,'service_started':False,'full_model_loaded':False,'scope':'external topk hook + real e43 state copy + full local e43 W8A8 chain; no HCCL/full model'}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
    set_is_extend_in_batch(False);uninstall()
if __name__=='__main__':main()
