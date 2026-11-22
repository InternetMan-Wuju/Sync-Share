import os,json
import torch,torch_npu,torch.distributed as dist
rank=int(os.environ['RANK']); torch.npu.set_device(int(os.environ['LOCAL_RANK']))
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import get_tp_group,initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment
from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
from sglang.srt.runtime_context import publish
from sglang.srt.layers.moe.utils import DeepEPMode,initialize_moe_config
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher,DeepEPNormalCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput
init_distributed_environment(world_size=-1,rank=-1,local_rank=-1,backend='hccl')
a=ServerArgs(model_path='dummy',tp_size=2,ep_size=2);a.moe_a2a_backend='deepep';a.moe_runner_backend='ascend'
set_global_server_args_for_scheduler(a);publish(a,role='scheduler');initialize_moe_config();initialize_model_parallel(tensor_model_parallel_size=2,expert_model_parallel_size=2)
d=DeepEPDispatcher(group=get_tp_group().device_group,router_topk=8,num_experts=258,num_local_experts=129,hidden_size=64,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False)
x=torch.randn((2,64),dtype=torch.bfloat16,device='npu');x[:,0]=torch.tensor([rank*10,rank*10+1],dtype=torch.bfloat16,device='npu');x[:,1]=rank
ids=torch.tensor([[0,1,2,3,129,130,131,132],[0,1,2,3,129,130,131,132]],dtype=torch.int32,device='npu') if rank==0 else torch.tensor([[0,1,2,3,129,130,131,132],[0,1,2,3,129,130,131,132]],dtype=torch.int32,device='npu')
w=torch.full((2,8),0.125,dtype=torch.float32,device='npu')
out=d.dispatch(x,StandardTopKOutput(w,ids,None));h=out.hidden_states if not isinstance(out.hidden_states,tuple) else out.hidden_states[0]
handle=d._normal_dispatcher.handle
print('RANK',rank,'counts',[int(v) for v in out.num_recv_tokens_per_expert])
print('hshape',h.shape,'tags',h[:,0].tolist(),'col1',h[:,1].tolist())
for k,v in handle.items():
 if isinstance(v,torch.Tensor):
  print('H',k,v.shape,v.dtype,v.detach().cpu().tolist())
 else: print('H',k,type(v),v if not isinstance(v,list) or len(v)<30 else (len(v),v[:20]))
# combine zeros to reset
eo=torch.zeros_like(h);d.combine(DeepEPNormalCombineInput(eo,None,None))
dist.barrier();destroy_model_parallel();destroy_distributed_environment()
