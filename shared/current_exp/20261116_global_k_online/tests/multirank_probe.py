#!/usr/bin/env python3
import os, json, torch, torch_npu, torch.distributed as dist
from global_slot_pool import GlobalSlotPool


def alloc(capacity, nbytes):
    return torch.empty((capacity, nbytes), dtype=torch.uint8, device='npu')


def main():
    local_rank=int(os.environ['LOCAL_RANK']); rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend='hccl', init_method='env://', rank=rank, world_size=world)
    owners=[2,5,7]
    pool=GlobalSlotPool([rank], 1024, alloc) if rank in owners else None
    if rank==0:
        for i,dst in enumerate(owners):
            t=torch.full((1024,), 0x40+i, dtype=torch.uint8, device='npu')
            torch.npu.synchronize(); dist.send(t,dst=dst)
        result={'rank':rank,'role':'source','sent_to':owners}
    elif rank in owners:
        t=torch.empty((1024,),dtype=torch.uint8,device='npu'); dist.recv(t,src=0)
        torch.npu.synchronize()
        lease=pool.acquire(request_id='multi',layer_id=49,expert_id=owners.index(rank))
        slot=pool.slot_buffer(lease,allow_reserved=True); slot.copy_(t); torch.npu.synchronize(); pool.mark_ready(lease)
        with pool.use(lease) as used:
            assert int(used[0].item())==0x40+owners.index(rank)
            assert int(used[-1].item())==0x40+owners.index(rank)
        pool.release(lease); result={'rank':rank,'role':'owner','slot_bytes':pool.stats()['committed_bytes'],'per_rank_capacity':pool.per_rank_capacity()}
    else:
        result={'rank':rank,'role':'no-slot'}
    dist.barrier()
    if rank==0: print('MULTIRANK_RESULT '+json.dumps({'world_size':world,'K':3,'owners':owners,'ranks':[result]}),flush=True)
    elif rank in owners: print('MULTIRANK_OWNER '+json.dumps(result),flush=True)
    dist.destroy_process_group()
    if pool is not None: del pool


if __name__=='__main__': main()
