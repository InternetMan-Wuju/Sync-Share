#!/usr/bin/env python3
"""Test-only NPU smoke for the external plugin. Never starts the model service."""
from __future__ import annotations
import json, os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_npu
OUT=Path('/external/artifacts/npu_smoke.json')
os.environ.setdefault('SGLANG_PREFILL_COPY_AWARE_MODE','off')
class TestTransport:
    def __init__(self): self.calls=0
    def is_cached(self,*args): return False
    def prepare_copy(self,*args): self.calls+=1
def make_info(device):
    primary=[]
    for e in range(32):
        r,s=divmod(e,16); primary.append(r*17+s)
    all_map=torch.zeros((32,2),dtype=torch.int64,device=device)
    all_map[:,0]=torch.tensor(primary,device=device); all_map[:,1]=torch.tensor(primary,device=device)
    return SimpleNamespace(ep_dispatch_algorithm='static',partial_logical_to_all_physical_map=all_map,partial_logical_to_rank_dispatch_physical_map=torch.tensor(primary,device=device))
def make_logits(device):
    logits=torch.full((512,32),-100.0,dtype=torch.float32,device=device)
    logits[:,0]=10.0; logits[:64,16]=5.0
    for i in range(64,512): logits[i,1+(i-64)%15]=5.0
    return logits
def call_topk(logits,info):
    from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
    from sglang.srt.layers.moe.topk import TopKConfig
    cfg=TopKConfig(top_k=2,use_grouped_topk=False,renormalize=False,scoring_func='softmax')
    hidden=torch.zeros((logits.shape[0],6144),dtype=torch.bfloat16,device=logits.device)
    return fused_topk_npu(hidden_states=hidden,router_logits=logits,topk_config=cfg,expert_location_dispatch_info=info,layer_id=49)
def runtime_config(controller,transport):
    return controller.ServiceRuntimeConfig(ep_size=2,local_slots_per_rank=17,destination_rank=1,destination_local_slot=16,cost_model=controller.LinearCostModel(per_task_us=1),transport=transport,planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,weight_version=3,layout_version='npu-smoke')
def main():
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    import m0_external.controller as controller
    from m0_external.install import install,uninstall
    install(); torch.npu.set_device(0); device=torch.device('npu:0'); info=make_info(device); logits=make_logits(device); set_is_extend_in_batch(True); checks={}
    try:
        os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='off'; off=call_topk(logits,info)
        checks['off']={'shape':list(off.topk_ids.shape),'destination_count':int((off.topk_ids==33).sum().item()),'finite':bool(torch.isfinite(off.topk_weights).all().item())}
        st=TestTransport(); sc=controller.RuntimeController(controller.Mode.SHADOW,service_config=runtime_config(controller,st)); controller.register_runtime_controller(sc); os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='shadow'; shadow=call_topk(logits,info)
        checks['shadow']={'shape':list(shadow.topk_ids.shape),'same_as_off':bool(torch.equal(shadow.topk_ids,off.topk_ids)),'transport_calls':st.calls,'plan':next(iter(sc.last_plan.values())).to_dict() if sc.last_plan else None}
        at=TestTransport(); ac=controller.RuntimeController(controller.Mode.ACTIVE,service_config=runtime_config(controller,at)); controller.register_runtime_controller(ac); os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='active'; start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True); start.record(); active=call_topk(logits,info); end.record(); end.synchronize(); plan=next(iter(ac.last_plan.values())); ac.slots.commit_epoch(ac.epoch,[0,1],[0,1]); ac.slots.validate_dispatch(49,1,16,plan.logical_expert,ac.epoch)
        checks['active']={'shape':list(active.topk_ids.shape),'destination_count':int((active.topk_ids==33).sum().item()),'transport_calls':at.calls,'plan':plan.to_dict(),'event_ms':start.elapsed_time(end),'finite':bool(torch.isfinite(active.topk_weights).all().item()),'slot_validated':True}
        passed=checks['off']['destination_count']==0 and checks['shadow']['same_as_off'] and checks['shadow']['transport_calls']==0 and checks['active']['destination_count']==plan.q_global and checks['active']['transport_calls']==1 and checks['active']['finite']
        result={'status':'NPU_PASS_TEST_ONLY' if passed else 'NPU_FAIL','device':str(torch.npu.current_device()),'device_name':torch.npu.get_device_name(0),'checks':checks,'service_started':False,'full_model_loaded':False,'scope':'external topk hook + fixed test plan on one NPU; no HCCL/full quant state/full model'}
        OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        set_is_extend_in_batch(False); uninstall()
if __name__=='__main__': main()
