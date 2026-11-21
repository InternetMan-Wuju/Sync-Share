#!/usr/bin/env python3
"""Two-card real GLM expert43: copy full 37.8MB state, split tokens, compare."""
import os,json,time
import torch, torch_npu, torch.distributed as dist
from real_expert_chain_probe import load_actual, expert_chain, alloc

LAYER=49; EXPERT=43

def pack(src, flat):
    pos=0; offs=[]; parts=[('w13',src[0]),('w2',src[1]),('s13',src[2]),('s2',src[3]),('o13',src[4]),('o2',src[5])]
    for name,t in parts:
        n=t.numel()*t.element_size(); flat[pos:pos+n].copy_(t.view(torch.uint8).reshape(-1)); offs.append((name,pos,n,t.dtype,t.shape)); pos+=n
    assert pos==flat.numel(); return offs

def views(flat,offs):
    return {name:flat[off:off+n].view(dt).reshape(shape) for name,off,n,dt,shape in offs}

def main():
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    dist.init_process_group(backend='hccl',init_method='env://',rank=rank,world_size=world)
    # Both ranks generate the same input; broadcast is used to remove RNG ambiguity.
    torch.manual_seed(123)
    x=torch.randn((6,6144),dtype=torch.bfloat16,device='npu')
    dist.broadcast(x,src=0)
    main_idx=torch.tensor([0,2,4],device='npu',dtype=torch.long); ext_idx=torch.tensor([1,3,5],device='npu',dtype=torch.long)
    state=load_actual(LAYER,EXPERT) if rank==0 else None
    if rank==0:
        src=[t.npu() for t in state]
        ref=expert_chain(x,*src[:4])
        flat=torch.empty(37810176,dtype=torch.uint8,device='npu')
        offs=pack(src,flat)
        torch.npu.synchronize()
        result={'rank':0,'role':'source','state_bytes':flat.numel()}
    else:
        flat=torch.empty(37810176,dtype=torch.uint8,device='npu')
        offs=None
    # Cold copy: source packs (done above), destination receives into a real slot.
    pool=None; lease=None
    if rank==1:
        from global_slot_pool import GlobalSlotPool
        pool=GlobalSlotPool([1],flat.numel(),alloc,name='two-rank-real-slot',synchronize=torch.npu.synchronize)
        lease=pool.acquire(request_id='two-rank-cold',layer_id=LAYER,expert_id=EXPERT)
        slot=pool.slot_buffer(lease,allow_reserved=True)
        torch.npu.synchronize()
    if rank==0:
        t=time.perf_counter(); dist.send(flat,dst=1); torch.npu.synchronize(); send_ms=(time.perf_counter()-t)*1000
        main_out=expert_chain(x.index_select(0,main_idx),*src[:4])
        ext_out=torch.empty((3,6144),dtype=torch.bfloat16,device='npu'); dist.recv(ext_out,src=1)
        split=torch.empty_like(ref); split.index_copy_(0,main_idx,main_out); split.index_copy_(0,ext_idx,ext_out)
        torch.npu.synchronize(); cold_total_ms=(time.perf_counter()-t)*1000
        cold_diff=float((ref.float()-split.float()).abs().max().item())
    else:
        t=time.perf_counter(); dist.recv(slot,src=0); torch.npu.synchronize(); recv_ms=(time.perf_counter()-t)*1000
        pool.mark_ready(lease)
        v=views(slot,offs) if offs is not None else None
        # offs are not available on rank1; reconstruct from known shapes/offsets by packing order.
        # The sender and receiver use the same serialization order fixed below.
        sizes=[25165824,12582912,8192,12288,16384,24576]
        pos=0; v={}
        for name,n in zip(('w13','w2','s13','s2','o13','o2'),sizes):
            dt=torch.int8 if name in ('w13','w2') else (torch.bfloat16 if name in ('s13','s2') else torch.float32)
            shape={'w13':(1,6144,4096),'w2':(1,2048,6144),'s13':(1,4096),'s2':(1,6144),'o13':(1,4096),'o2':(1,6144)}[name]
            v[name]=slot[pos:pos+n].view(dt).reshape(shape); pos+=n
        t2=time.perf_counter(); ext_out=expert_chain(x.index_select(0,ext_idx),v['w13'],v['w2'],v['s13'],v['s2']); torch.npu.synchronize(); ext_compute_ms=(time.perf_counter()-t2)*1000
        dist.send(ext_out,dst=0)
        result={'rank':1,'role':'destination','recv_ms':recv_ms,'ext_compute_ms':ext_compute_ms,'state_bytes':slot.numel(),'pool_committed_bytes':pool.stats()['committed_bytes']}
    dist.barrier()
    if rank==0:
        result.update({'cold_send_ms':send_ms,'cold_total_ms':cold_total_ms,'cold_diff':cold_diff})
        print('TWO_RANK_RESULT '+json.dumps(result),flush=True)
        open('/runexp/artifacts/two_rank_result.json','w').write(json.dumps(result,indent=2))
    else:
        pool.release(lease); print('TWO_RANK_DEST '+json.dumps(result),flush=True)
    dist.barrier()
    if rank==1: del pool
    dist.destroy_process_group()

if __name__=='__main__': main()
