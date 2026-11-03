#!/usr/bin/env python3
"""Explicit M1 smoke-test entry. Never starts the model service.

This validates only the M0 controller's plan -> quota -> physical-ID path on a
real NPU using a test-only cost model and a tiny sentinel transport. Full
quantized expert-state copy, DeepEP dispatch and combine remain M1/M2 work.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'patched_tree/sglang/srt/eplb/prefill_copy_aware.py'

def load_module():
    spec=importlib.util.spec_from_file_location('prefill_copy_aware_npu_test',MODULE)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

class TestTransport:
    def __init__(self): self.calls=0
    def is_cached(self,*args): return False
    def prepare_copy(self,*args): self.calls+=1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute',action='store_true',help='actually touch the NPU with a test-only sentinel')
    ap.add_argument('--device',default='npu')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    if not args.execute:
        print(json.dumps({'status':'READY_NOT_RUN','reason':'--execute not supplied','device_touched':False,'service_started':False},indent=2));return
    if os.getenv('SGLANG_PREFILL_COPY_AWARE_TEST_ONLY')!='1':
        raise SystemExit('set SGLANG_PREFILL_COPY_AWARE_TEST_ONLY=1 for test-only execution')
    import torch
    if args.device.startswith('npu'):
        import torch_npu
        if torch.npu.device_count()!=1: raise RuntimeError(f'expected one visible NPU, got {torch.npu.device_count()}')
        device=torch.device('npu:0');torch.npu.set_device(device)
    else:
        device=torch.device('cpu')
    m=load_module();os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='active'
    # 32 logical experts, two virtual ranks, 17 physical slots/rank.
    primary=[]
    for e in range(32):
        r=e//16;s=e%16;primary.append(r*17+s)
    all_map=torch.zeros((32,2),dtype=torch.int64,device=device);all_map[:,0]=torch.tensor(primary,device=device)
    info=SimpleNamespace(ep_dispatch_algorithm='static',partial_logical_to_all_physical_map=all_map,partial_logical_to_rank_dispatch_physical_map=torch.tensor(primary,device=device))
    rows=512;ids=torch.empty((rows,2),dtype=torch.int32,device=device)
    ids[:64]=torch.tensor([0,16],dtype=torch.int32,device=device);ids[64:]=torch.tensor([0,0],dtype=torch.int32,device=device)
    weights=torch.ones_like(ids,dtype=torch.float32)
    transport=TestTransport()
    cfg=m.ServiceRuntimeConfig(ep_size=2,local_slots_per_rank=17,destination_rank=1,destination_local_slot=16,cost_model=m.LinearCostModel(per_task_us=1),transport=transport,planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,weight_version=3,layout_version='npu-smoke')
    ctl=m.RuntimeController(m.Mode.ACTIVE,service_config=cfg);m.register_runtime_controller(ctl)
    out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=49)
    plan=next(iter(ctl.last_plan.values()))
    # Commit and validate the test-only slot lifecycle after planning.
    ctl.slots.commit_epoch(ctl.epoch,[0,1],[0,1]);ctl.slots.validate_dispatch(49,1,16,plan.logical_expert,ctl.epoch)
    result={'status':'PASS_TEST_ONLY','device':str(device),'plan':plan.to_dict(),'applied':applied,
      'dispatch_to_destination':int((out==33).sum().item()),'logical_shape':list(ids.shape),
      'transport_calls':transport.calls,'service_started':False,
      'not_tested':['full quantized expert state transport','NZ/ND conversion','HCCL','DeepEP dispatch','combine','production graph','full model']}
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.out: Path(args.out).write_text(text+'\n')
    print(text)
if __name__=='__main__':main()
