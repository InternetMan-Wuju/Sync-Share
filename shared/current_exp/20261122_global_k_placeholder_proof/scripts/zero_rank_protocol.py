#!/usr/bin/env python3
"""258 native all-to-all with one zero-legal-task rank and explicit empty-id padding."""
import json, os
from pathlib import Path
import torch, torch_npu, torch.distributed as dist
OUT=Path(os.environ.get('OUT_DIR','/runexp/artifacts'))
rank=int(os.environ['RANK']); torch.npu.set_device(int(os.environ['LOCAL_RANK']))
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import destroy_distributed_environment,destroy_model_parallel,get_tp_group,initialize_model_parallel
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher,DeepEPNormalCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode,initialize_moe_config
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
init_distributed_environment(world_size=-1,rank=-1,local_rank=-1,backend='hccl')
a=ServerArgs(model_path='dummy',tp_size=2,ep_size=2); a.moe_a2a_backend='deepep'; a.moe_runner_backend='ascend'
set_global_server_args_for_scheduler(a); publish(a,role='scheduler'); initialize_moe_config(); initialize_model_parallel(tensor_model_parallel_size=2,expert_model_parallel_size=2)
d=DeepEPDispatcher(group=get_tp_group().device_group,router_topk=8,num_experts=258,num_local_experts=129,hidden_size=64,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False)
# rank0 sends 8 legal tokens; rank1 sends 1 explicit padding token to empty comm 128.
rows=8 if rank==0 else 1
hidden=torch.randn((rows,64),dtype=torch.bfloat16,device='npu'); hidden[:,0]=torch.arange(rows,dtype=torch.bfloat16,device='npu')+rank*16
if rank==0:
 ids=torch.arange(8,dtype=torch.int32,device='npu').repeat(8,1); w=torch.full((8,8),1/8,dtype=torch.float32,device='npu')
else:
 ids=torch.full((1,8),128,dtype=torch.int32,device='npu'); w=torch.zeros((1,8),dtype=torch.float32,device='npu')
out=d.dispatch(hidden,StandardTopKOutput(w,ids,None)); h=out.hidden_states[0] if isinstance(out.hidden_states,tuple) else out.hidden_states
counts=[int(x) for x in out.num_recv_tokens_per_expert]; base=0 if rank==0 else 129
row_comm=[]
for i,c in enumerate(counts): row_comm += [base+i]*c
eo=torch.zeros_like(h)
for i,comm in enumerate(row_comm):
 if comm==128:
  continue  # empty id is padding: no expert computation/state
 eo[i,0]=float(comm%8); eo[i,1]=h[i,0]
combined=d.combine(DeepEPNormalCombineInput(eo,None,None))
# rank0 real tokens expect mean of experts 0..7 = 3.5.
err=0.0 if rank==1 else float((combined[:,0].float()-3.5).abs().max().item())
empty_padding_rows=counts[128] if rank==0 else 0
rec={
 'rank':rank,'legal_tokens_sent':8 if rank==0 else 0,'padding_tokens_sent':0 if rank==0 else 1,
 'empty_padding_comm':128,'empty_padding_rows_received':empty_padding_rows,
 'legal_recv_tasks_rank':sum(counts)-empty_padding_rows,'combine_shape':list(combined.shape),
 'label_error':err,'correct':err<=0.05,
 'note':'rank1 is a zero-legal-task participant; one zero-weight padding token uses the empty comm id and is never treated as an expert task.'
}
print(json.dumps(rec)); OUT.joinpath(f'zero_rank_258_rank{rank}.json').write_text(json.dumps(rec,indent=2))
dist.barrier(); destroy_model_parallel(); destroy_distributed_environment()
if not rec['correct']: raise SystemExit(1)
