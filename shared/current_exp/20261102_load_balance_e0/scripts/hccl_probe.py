import os
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(local_rank)
dist.init_process_group(backend='hccl')
value = torch.ones(1, dtype=torch.float32, device=f'npu:{local_rank}') * (rank + 1)
dist.all_reduce(value, op=dist.ReduceOp.SUM)
got = float(value.cpu().item())
expected = 136.0
if abs(got - expected) > 1e-6:
    raise RuntimeError(f'rank={rank} local_rank={local_rank} all_reduce={got} expected={expected}')
if rank == 0:
    print(f'HCCL_ALLREDUCE_OK world_size={dist.get_world_size()} sum={got}', flush=True)
dist.destroy_process_group()
