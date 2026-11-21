#!/usr/bin/env python3
"""Two-card real expert43: warm cold-copy and existing-replica timing."""
import os,json,time,statistics
import torch,torch_npu,torch.distributed as dist
from real_expert_chain_probe import load_actual,expert_chain,alloc
from global_slot_pool import GlobalSlotPool
LAYER=49; EXPERT=43
SIZES=[('w13',25165824),('w2',12582912),('s13',8192),('s2',12288),('o13',16384),('o2',24576)]
SHAPES={'w13':(1,6144,4096),'w2':(1,2048,6144),'s13':(1,4096),'s2':(1,6144),'o13':(1,4096),'o2':(1,6144)}
DT={'w13':torch.int8,'w2':torch.int8,'s13':torch.bfloat16,'s2':torch.bfloat16,'o13':torch.float32,'o2':torch.float32}
TOTAL=sum(n for _,n in SIZES)

def slot_views(slot):
    out={}; pos=0
    for name,n in SIZES:
        out[name]=slot[pos:pos+n].view(DT[name]).reshape(SHAPES[name]); pos+=n
    return out

def pack(src,flat):
    parts=[src[0],src[1],src[2],src[3],src[4],src[5]]; pos=0
    for t in parts:
        n=t.numel()*t.element_size(); flat[pos:pos+n].copy_(t.view(torch.uint8).reshape(-1)); pos+=n
    assert pos==TOTAL

def main():
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local); dist.init_process_group(backend='hccl',init_method='env://',rank=rank,world_size=2)
    torch.manual_seed(123); x=torch.randn((6,6144),dtype=torch.bfloat16,device='npu'); dist.broadcast(x,src=0)
    mi=torch.tensor([0,2,4],device='npu',dtype=torch.long); ei=torch.tensor([1,3,5],device='npu',dtype=torch.long)
    src=[t.npu() for t in load_actual(LAYER,EXPERT)] if rank==0 else None
    flat=torch.empty(TOTAL,dtype=torch.uint8,device='npu')
    pool=GlobalSlotPool([1],TOTAL,alloc,name='timing-two-rank',synchronize=torch.npu.synchronize) if rank==1 else None
    result={'rank':rank,'role':'source' if rank==0 else 'dest','state_bytes':TOTAL,'cold_ms':[],'existing_ms':[],'cold_copy_ms':[],'existing_compute_ms':[]}
    # Warm compile on both ranks.
    for _ in range(2):
        if rank==0:
            pack(src,flat); dist.send(flat,dst=1)
        else:
            lease=pool.acquire(request_id='warm',layer_id=LAYER,expert_id=EXPERT)
            slot=pool.slot_buffer(lease,allow_reserved=True); dist.recv(slot,src=0); torch.npu.synchronize(); pool.mark_ready(lease)
            expert_chain(x.index_select(0,ei),*[slot_views(slot)[k] for k in ('w13','w2','s13','s2')])
            pool.release(lease)
    # Formal cold copies: every iteration repacks, sends and installs.
    for rep in range(10):
        if rank==0:
            t=time.perf_counter(); pack(src,flat); torch.npu.synchronize(); dist.send(flat,dst=1)
            main_out=expert_chain(x.index_select(0,mi),*src[:4]); torch.npu.synchronize(); result['cold_copy_ms'].append((time.perf_counter()-t)*1000)
        else:
            lease=pool.acquire(request_id=f'cold-{rep}',layer_id=LAYER,expert_id=EXPERT)
            slot=pool.slot_buffer(lease,allow_reserved=True); t=time.perf_counter(); dist.recv(slot,src=0); torch.npu.synchronize(); pool.mark_ready(lease)
            v=slot_views(slot); ext=expert_chain(x.index_select(0,ei),v['w13'],v['w2'],v['s13'],v['s2']); torch.npu.synchronize(); result['cold_copy_ms'].append((time.perf_counter()-t)*1000); result['cold_ms'].append((time.perf_counter()-t)*1000); pool.release(lease)
    # One prepared copy, then repeated existing-replica compute without copy.
    if rank==0:
        pack(src,flat); dist.send(flat,dst=1)
    else:
        lease=pool.acquire(request_id='existing',layer_id=LAYER,expert_id=EXPERT)
        slot=pool.slot_buffer(lease,allow_reserved=True); dist.recv(slot,src=0); torch.npu.synchronize(); pool.mark_ready(lease)
    for rep in range(10):
        t=time.perf_counter()
        if rank==0: expert_chain(x.index_select(0,mi),*src[:4])
        else:
            v=slot_views(slot); expert_chain(x.index_select(0,ei),v['w13'],v['w2'],v['s13'],v['s2'])
        torch.npu.synchronize(); result['existing_ms'].append((time.perf_counter()-t)*1000)
    if rank==1: pool.release(lease)
    for k in ('cold_ms','existing_ms','cold_copy_ms'):
        if result[k]: result[k+'_median']=statistics.median(result[k]); result[k+'_min']=min(result[k]); result[k+'_max']=max(result[k])
    dist.barrier()
    if rank==0:
        result['remote']='rank1 (timing not merged for numerical correctness)'
        print('TWO_RANK_TIMING '+json.dumps(result),flush=True)
        open('/runexp/artifacts/two_rank_timing.json','w').write(json.dumps(result,indent=2))
    else:
        print('TWO_RANK_DEST_TIMING '+json.dumps(result),flush=True)
    dist.destroy_process_group()
    if rank==1: del pool

if __name__=='__main__': main()
