#!/usr/bin/env python3
"""CPU fault-semantics tests for the M0 copy-aware controller."""
from __future__ import annotations
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
MODULE_PATH=ROOT/'patched_tree/sglang/srt/eplb/prefill_copy_aware.py'
spec=importlib.util.spec_from_file_location('prefill_copy_aware_under_test',MODULE_PATH)
import sys
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)


class FakeTransport:
    def __init__(self,cached=False,fail=False):
        self.cached=cached; self.fail=fail; self.prepare_calls=0
    def is_cached(self,layer_id,logical_expert,weight_version,expert_tp_shard): return self.cached
    def prepare_copy(self,layer_id,logical_expert,weight_version,expert_tp_shard):
        self.prepare_calls+=1
        if self.fail: raise RuntimeError('transport failed')


def info_for(num_logical=32,ep_size=2,slots=17):
    # identity primary on rank0 experts 0..15; rank1 experts map to next 16 slots.
    primary=[]
    for e in range(num_logical):
        r=e//16; s=e%16; primary.append(r*slots+s)
    all_map=torch.zeros((num_logical,2),dtype=torch.int64)
    all_map[:,0]=torch.tensor(primary)
    all_map[:,1]=torch.tensor(primary)
    return SimpleNamespace(ep_dispatch_algorithm='static',partial_logical_to_all_physical_map=all_map,partial_logical_to_rank_dispatch_physical_map=torch.tensor(primary))


def make_plan(**kw):
    base=dict(step_id=1,layer_id=49,epoch=1,action='route',logical_expert=0,weight_source_rank=0,
      primary_compute_rank=0,destination_rank=1,destination_local_slot=16,q_global=7,q_by_source=(7,),
      copy_required=True,layout_version='test',weight_version=3,estimated_gain_us=10.0,reason='test',
      primary_physical_id=0,destination_physical_id=33)
    base.update(kw);return m.Plan(**base)


class Tests(unittest.TestCase):
    def setUp(self):
        m.register_runtime_controller(None)
        os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='off'

    def test_invalid_padding_shared_and_empty_conservation(self):
        ids=torch.tensor([[0,1,2,-1,9],[3,4,5,6,7]],dtype=torch.int32)
        st=m.collect_routing_stats([ids],num_logical_experts=16,num_token_non_padded=1,num_fused_shared_experts=1)
        self.assertEqual(st.total_tasks,3)
        self.assertEqual(st.invalid_tasks,1)
        self.assertEqual(st.shared_tasks,1)
        self.assertEqual(st.counts[0],1); self.assertEqual(st.counts[1],1); self.assertEqual(st.counts[2],1); self.assertEqual(st.counts[9],0)
        empty=m.collect_routing_stats([torch.empty((0,4),dtype=torch.int32)],num_logical_experts=16)
        self.assertEqual(empty.total_tasks,0)

    def test_multi_source_quota_prefix(self):
        a=torch.zeros((1,6),dtype=torch.int32)
        b=torch.zeros((1,4),dtype=torch.int32)
        st=m.collect_routing_stats([a,b],num_logical_experts=2)
        self.assertEqual(st.quota_by_source(7),(6,1))
        self.assertEqual(st.quota_by_source(0),(0,0))
        self.assertEqual(st.quota_by_source(10),(6,4))
        swapped=m.collect_routing_stats([b,a],num_logical_experts=2)
        self.assertEqual(swapped.quota_by_source(7),(4,3))

    def test_remap_preserves_logic_and_weights(self):
        ids=torch.tensor([[0,1],[0,2]],dtype=torch.int32)
        weights=torch.tensor([[0.7,0.3],[0.6,0.4]],dtype=torch.float32)
        before=weights.clone()
        plan=make_plan(q_global=1,q_by_source=(1,),primary_physical_id=0,destination_physical_id=33)
        out,=m.remap_logical_to_physical([ids],primary_physical_ids=list(range(8)),plan=plan,logical_expert=0)
        self.assertTrue(torch.equal(weights,before))
        self.assertEqual(int((out==33).sum()),1)
        self.assertEqual(int((out==0).sum()),1)
        self.assertEqual(int((out==1).sum()),1)
        self.assertEqual(int((out==2).sum()),1)

    def test_planner_cold_cached_unknown_and_no_gain(self):
        stats=m.RoutingStats((960,64)+(0,)*14,((960,64)+(0,)*14,),1024,0,0)
        planner=m.Planner()
        args=dict(step_id=1,layer_id=49,epoch=1,rank_expert_counts=[[960],[64]],primary_rank_by_expert=[0,1],primary_local_slot_by_expert=[0,0],primary_physical_by_expert=[0,17],destination_rank=1,destination_local_slot=16,destination_physical_id=33,cost_model=m.LinearCostModel(per_task_us=1),planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,slot_manager=m.SlotManager(),transport=FakeTransport(cached=False))
        cold=planner.plan(stats=stats,**args)
        self.assertEqual(cold.action,'route');self.assertEqual(cold.q_global,448);self.assertTrue(cold.copy_required)
        args_cached=dict(args);transport=FakeTransport(cached=True);args_cached['transport']=transport
        cached=planner.plan(stats=stats,**args_cached)
        self.assertEqual(cached.action,'route');self.assertFalse(cached.copy_required)
        self.assertLess(cold.estimated_gain_us,cached.estimated_gain_us)
        unknown=dict(args);unknown['cost_model']=None
        self.assertEqual(planner.plan(stats=stats,**unknown).reason,'unknown_cost')
        low=m.RoutingStats((100,100)+(0,)*14,((100,100)+(0,)*14,),200,0,0)
        args2=dict(args);args2['rank_expert_counts']=[[100],[100]]
        self.assertNotEqual(planner.plan(stats=low,**args2).action,'route')

    def test_slot_lifecycle_and_epoch(self):
        sm=m.SlotManager(); tr=FakeTransport()
        sm.replace_and_publish(layer_id=49,rank=1,local_slot=16,logical_expert=0,weight_version=3,epoch=1,transport=tr)
        with self.assertRaises(m.SlotError): sm.validate_dispatch(49,1,16,0,1)
        sm.commit_epoch(1,[0,1],[0,1])
        sm.validate_dispatch(49,1,16,0,1)
        sm.begin_read(49,1,16,1)
        with self.assertRaises(m.SlotError): sm.replace_and_publish(layer_id=49,rank=1,local_slot=16,logical_expert=1,weight_version=3,epoch=2,transport=tr)
        sm.end_read(49,1,16)
        sm.replace_and_publish(layer_id=49,rank=1,local_slot=16,logical_expert=1,weight_version=3,epoch=2,transport=tr)
        with self.assertRaises(m.SlotError): sm.validate_dispatch(49,1,16,0,1)

    def test_slot_failure_never_publishes_partial_state(self):
        sm=m.SlotManager(); tr=FakeTransport(fail=True)
        with self.assertRaises(RuntimeError):
            sm.replace_and_publish(layer_id=49,rank=1,local_slot=16,logical_expert=0,weight_version=3,epoch=1,transport=tr)
        rec=sm.slot(49,1,16)
        self.assertEqual(rec.state,m.SlotState.FAILED)
        with self.assertRaises(m.SlotError): sm.validate_dispatch(49,1,16,0,1)
        with self.assertRaises(m.SlotError): sm.commit_epoch(1,[0],[0,1])

    def test_empty_and_no_gain_do_not_route(self):
        empty=m.RoutingStats((0,)*16,((0,)*16,),0,0,0)
        tr=FakeTransport(cached=False); sm=m.SlotManager(); planner=m.Planner()
        common=dict(step_id=9,layer_id=49,epoch=1,rank_expert_counts=[[0],[0]],primary_rank_by_expert=[0,1],primary_local_slot_by_expert=[0,0],primary_physical_by_expert=[0,17],destination_rank=1,destination_local_slot=16,destination_physical_id=33,cost_model=m.LinearCostModel(per_task_us=1),planner_cost_us=1,prepare_cost_us=0,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,slot_manager=sm,transport=tr)
        self.assertEqual(planner.plan(stats=empty,**common).action,'noop')
        hot=m.RoutingStats((100,)+(0,)*15,((100,)+(0,)*15,),100,0,0)
        costly=dict(common);costly['rank_expert_counts']=[[100],[0]];costly['planner_cost_us']=100
        p=planner.plan(stats=hot,**costly)
        self.assertEqual(p.action,'noop');self.assertEqual(p.reason,'below_threshold');self.assertEqual(tr.prepare_calls,0)

    def test_default_off_and_shadow_do_not_remap(self):
        ids=torch.tensor([[0,1],[0,2]],dtype=torch.int32); weights=torch.ones_like(ids,dtype=torch.float32); info=info_for()
        out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=49)
        self.assertFalse(applied); self.assertTrue(torch.equal(out,ids))
        os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='shadow'
        out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=49)
        self.assertFalse(applied); self.assertTrue(torch.equal(out,ids)); self.assertEqual(m.get_last_shadow_stats().total_tasks,4)

    def test_shadow_with_runtime_plans_without_copy(self):
        rows=512;ids=torch.empty((rows,2),dtype=torch.int32)
        ids[:64]=torch.tensor([0,16],dtype=torch.int32);ids[64:]=torch.tensor([0,0],dtype=torch.int32)
        weights=torch.ones_like(ids,dtype=torch.float32);info=info_for();tr=FakeTransport(cached=False)
        cfg=m.ServiceRuntimeConfig(ep_size=2,local_slots_per_rank=17,destination_rank=1,destination_local_slot=16,cost_model=m.LinearCostModel(per_task_us=1),transport=tr,planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,weight_version=3,layout_version='test')
        ctl=m.RuntimeController(m.Mode.SHADOW,service_config=cfg);m.register_runtime_controller(ctl);os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='shadow'
        out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=49)
        plan=next(iter(ctl.last_plan.values()))
        self.assertFalse(applied);self.assertTrue(torch.equal(out,ids));self.assertEqual(plan.action,'route');self.assertEqual(plan.reason,'shadow_plan');self.assertEqual(tr.prepare_calls,0)

    def test_active_plan_drives_copy_and_physical_ids(self):
        rows=512
        ids=torch.empty((rows,2),dtype=torch.int32)
        ids[:64]=torch.tensor([0,16],dtype=torch.int32);ids[64:]=torch.tensor([0,0],dtype=torch.int32)
        weights=torch.ones_like(ids,dtype=torch.float32)
        info=info_for()
        transport=FakeTransport(cached=False)
        cfg=m.ServiceRuntimeConfig(ep_size=2,local_slots_per_rank=17,destination_rank=1,destination_local_slot=16,cost_model=m.LinearCostModel(per_task_us=1),transport=transport,planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,weight_version=3,layout_version='test')
        ctl=m.RuntimeController(m.Mode.ACTIVE,service_config=cfg)
        m.register_runtime_controller(ctl); os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='active'
        out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=49)
        plan=next(iter(ctl.last_plan.values()))
        self.assertTrue(applied);self.assertEqual(plan.action,'route');self.assertEqual(plan.q_global,448)
        self.assertEqual(int((out==33).sum()),448)
        self.assertTrue(torch.equal(weights,weights))
        self.assertEqual(transport.prepare_calls,1)

    def test_unsupported_layer_returns_original(self):
        ids=torch.tensor([[0,1]],dtype=torch.int32);weights=torch.ones_like(ids,dtype=torch.float32);info=info_for()
        m.register_runtime_controller(m.RuntimeController(m.Mode.ACTIVE,service_config=None));os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='active'
        out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=48)
        self.assertFalse(applied);self.assertTrue(torch.equal(out,ids))

    def test_noop_is_consistent_across_multiple_rank_controllers(self):
        ids=torch.arange(32,dtype=torch.int32).reshape(32,1)  # uniform one task per logical expert
        weights=torch.ones_like(ids,dtype=torch.float32);info=info_for()
        results=[]
        for _ in range(2):
            cfg=m.ServiceRuntimeConfig(ep_size=2,local_slots_per_rank=17,destination_rank=1,destination_local_slot=16,cost_model=m.LinearCostModel(per_task_us=1),transport=FakeTransport(),planner_cost_us=1,prepare_cost_us=0,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,weight_version=3,layout_version='test')
            ctl=m.RuntimeController(m.Mode.ACTIVE,service_config=cfg);m.register_runtime_controller(ctl);os.environ['SGLANG_PREFILL_COPY_AWARE_MODE']='active'
            out,applied=m.maybe_apply_prefill_copy_aware(topk_ids=ids,topk_weights=weights,expert_location_dispatch_info=info,layer_id=49)
            results.append((applied,out.tolist(),next(iter(ctl.last_plan.values())).action))
        self.assertEqual(results[0],results[1]);self.assertFalse(results[0][0]);self.assertEqual(results[0][2],'noop')


if __name__=='__main__': unittest.main(verbosity=2)
