#!/usr/bin/env python3
import importlib.util,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'patched_tree/sglang/srt/eplb/prefill_copy_aware.py'
spec=importlib.util.spec_from_file_location('pcp_examples',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class T:
 def is_cached(self,*a): return False
 def prepare_copy(self,*a): pass
common=dict(step_id=1,layer_id=49,epoch=1,primary_rank_by_expert=[0,1],primary_local_slot_by_expert=[0,0],primary_physical_by_expert=[0,17],destination_rank=1,destination_local_slot=16,destination_physical_id=33,cost_model=m.LinearCostModel(per_task_us=1),planner_cost_us=1,prepare_cost_us=10,dispatch_extra_us=0,min_gain_us=0,epsilon_us=0,transport=T(),weight_version=3,layout_version='cpu-example')
planner=m.Planner()
hot=m.RoutingStats((960,64)+(0,)*14,((960,64)+(0,)*14,),1024,0,0)
p1=planner.plan(stats=hot,rank_expert_counts=[[960],[64]],slot_manager=m.SlotManager(),**common)
uniform=m.RoutingStats((512,512)+(0,)*14,((512,512)+(0,)*14,),1024,0,0)
p2=planner.plan(stats=uniform,rank_expert_counts=[[512],[512]],slot_manager=m.SlotManager(),**common)
out={'evidence_type':'synthetic_cpu_plan','cost_model':'test-only linear 1us/task','examples':[p1.to_dict(),p2.to_dict()]}
(ROOT/'PLAN_EXAMPLES.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
