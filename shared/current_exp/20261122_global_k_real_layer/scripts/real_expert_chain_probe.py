#!/usr/bin/env python3
"""Single-card real GLM W8A8 expert-chain probe.

This is an expert-level replay, not a full layer/model service. It loads the
actual layer49 expert43 weights/scales from the model shards, runs the W8A8
gate/up -> SwiGLU -> dynamic requant -> down chain, and compares a normal bank
against the same bytes viewed from a GlobalSlotPool slot.
"""
import json, time, hashlib
import torch
import torch_npu
from safetensors import safe_open
from global_slot_pool import GlobalSlotPool

MODEL='/models/glm-q'
LAYER=49
EXPERT=43
IDX=MODEL+'/quant_model_weights.safetensors.index.json'
OUT='/runexp/artifacts/real_chain_result.json'

def load_actual(layer=LAYER, expert=EXPERT):
    import json as _json
    idx=_json.load(open(IDX))['weight_map']
    prefix=f'model.layers.{layer}.mlp.experts.{expert}.'
    names=['gate_proj.weight','gate_proj.weight_scale','gate_proj.weight_offset',
           'up_proj.weight','up_proj.weight_scale','up_proj.weight_offset',
           'down_proj.weight','down_proj.weight_scale','down_proj.weight_offset']
    names=[prefix+n for n in names]
    shard=idx[names[0]]
    out={}
    with safe_open(MODEL+'/'+shard,framework='pt',device='cpu') as f:
        for n in names:
            out[n]=f.get_tensor(n)
    gate=out[prefix+'gate_proj.weight']; up=out[prefix+'up_proj.weight']; down=out[prefix+'down_proj.weight']
    gate_s=out[prefix+'gate_proj.weight_scale']; up_s=out[prefix+'up_proj.weight_scale']; down_s=out[prefix+'down_proj.weight_scale']
    gate_o=out[prefix+'gate_proj.weight_offset']; up_o=out[prefix+'up_proj.weight_offset']; down_o=out[prefix+'down_proj.weight_offset']
    # ModelSlim stores W8A8 weights as [out,in], then FusedMoE processes them
    # to [experts,in,out]. The 3D banks below follow that layout.
    w13=torch.cat([gate,up],dim=0).unsqueeze(0).transpose(1,2).contiguous()
    w2=down.unsqueeze(0).transpose(1,2).contiguous()
    s13=torch.cat([gate_s.squeeze(-1),up_s.squeeze(-1)],dim=0).to(torch.bfloat16).unsqueeze(0)
    s2=down_s.squeeze(-1).to(torch.bfloat16).unsqueeze(0)
    o13=torch.cat([gate_o.squeeze(-1),up_o.squeeze(-1)],dim=0).unsqueeze(0)
    o2=down_o.squeeze(-1).unsqueeze(0)
    return w13,w2,s13,s2,o13,o2

def quant(x):
    return torch.ops.npu.npu_dynamic_quant(x,dst_type=torch.int8)

def gmm(xq, xs, w, scale):
    gl=torch.tensor([xq.shape[0]],dtype=torch.int64,device=xq.device)
    return torch.ops.npu.npu_grouped_matmul(
        x=[xq], weight=[w], scale=[scale], per_token_scale=[xs],
        group_list=gl, split_item=2, group_type=0, group_list_type=1,
        output_dtype=torch.bfloat16)[0]

def expert_chain(x,w13,w2,s13,s2):
    xq,xs=quant(x)
    up=gmm(xq,xs,w13,s13)
    act=torch.ops.npu.npu_swiglu(up)
    aq,as_=quant(act)
    out=gmm(aq,as_,w2,s2)
    return out

def alloc(capacity,nbytes):
    return torch.empty((capacity,nbytes),dtype=torch.uint8,device='npu')

def timeit(fn, warm=5, reps=10):
    for _ in range(warm): fn()
    torch.npu.synchronize()
    vals=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); torch.npu.synchronize(); vals.append((time.perf_counter()-t)*1000)
    return {'n':len(vals),'median_ms':float(torch.tensor(vals).median().item()),'min_ms':min(vals),'max_ms':max(vals)}

def main():
    torch.npu.set_device(0)
    w13,w2,s13,s2,o13,o2=load_actual()
    # Move source state to HBM.
    w13=w13.npu(); w2=w2.npu(); s13=s13.npu(); s2=s2.npu(); o13=o13.npu(); o2=o2.npu()
    x=torch.randn((6,6144),dtype=torch.bfloat16,device='npu')
    torch.npu.synchronize()
    baseline=expert_chain(x,w13,w2,s13,s2)
    # Pack the exact weight/scale bytes into one actual-size K=1 external slot.
    parts=[('w13',w13),('w2',w2),('s13',s13),('s2',s2),('o13',o13),('o2',o2)]
    offsets=[]; pos=0
    for name,t in parts:
        n=t.numel()*t.element_size(); offsets.append((name,pos,n,t.dtype)); pos+=n
    pool=GlobalSlotPool([1],pos,alloc,name='real-expert43-slot',synchronize=torch.npu.synchronize)
    lease=pool.acquire(request_id='real-probe',layer_id=LAYER,expert_id=EXPERT)
    slot=pool.slot_buffer(lease,allow_reserved=True)
    t=time.perf_counter()
    for name,off,n,dt in offsets:
        src=dict(parts)[name]
        slot[off:off+n].copy_(src.view(torch.uint8).reshape(-1))
    torch.npu.synchronize(); copy_ms=(time.perf_counter()-t)*1000
    pool.mark_ready(lease)
    sl=pool.slot_buffer(lease)
    state_equal={}
    for name,off,n,dt in offsets:
        state_equal[name]=bool(torch.equal(sl[off:off+n], dict(parts)[name].view(torch.uint8).reshape(-1)))
    views={name:sl[off:off+n].view(dt).reshape(dict(parts)[name].shape) for name,off,n,dt in offsets if name in ('w13','w2','s13','s2')}
    external=expert_chain(x,views['w13'],views['w2'],views['s13'],views['s2'])
    torch.npu.synchronize()
    diff=float((baseline.float()-external.float()).abs().max().item())
    # Manual task split: positions [0,2,4] original, [1,3,5] external.
    main_idx=torch.tensor([0,2,4],device='npu',dtype=torch.long); ext_idx=torch.tensor([1,3,5],device='npu',dtype=torch.long)
    main_out=expert_chain(x.index_select(0,main_idx),w13,w2,s13,s2)
    ext_out=expert_chain(x.index_select(0,ext_idx),views['w13'],views['w2'],views['s13'],views['s2'])
    split=torch.empty_like(baseline); split.index_copy_(0,main_idx,main_out); split.index_copy_(0,ext_idx,ext_out)
    torch.npu.synchronize(); split_diff=float((baseline.float()-split.float()).abs().max().item())
    # Cross-layer reuse: release the same K=1 slot and install layer50/expert43.
    pool.release(lease)
    w13b,w2b,s13b,s2b,o13b,o2b=load_actual(50, EXPERT)
    w13b=w13b.npu(); w2b=w2b.npu(); s13b=s13b.npu(); s2b=s2b.npu(); o13b=o13b.npu(); o2b=o2b.npu()
    parts2=[('w13',w13b),('w2',w2b),('s13',s13b),('s2',s2b),('o13',o13b),('o2',o2b)]
    assert [t.shape for _,t in parts2] == [t.shape for _,t in parts]
    lease2=pool.acquire(request_id='reuse-layer50',layer_id=50,expert_id=EXPERT)
    slot2=pool.slot_buffer(lease2,allow_reserved=True)
    t2=time.perf_counter()
    for name,off,n,dt in offsets:
        src2=dict(parts2)[name]
        slot2[off:off+n].copy_(src2.view(torch.uint8).reshape(-1))
    torch.npu.synchronize(); copy_ms2=(time.perf_counter()-t2)*1000
    pool.mark_ready(lease2)
    sl2=pool.slot_buffer(lease2)
    state_equal2={name:bool(torch.equal(sl2[off:off+n],dict(parts2)[name].view(torch.uint8).reshape(-1))) for name,off,n,dt in offsets}
    views2={name:sl2[off:off+n].view(dt).reshape(dict(parts2)[name].shape) for name,off,n,dt in offsets if name in ('w13','w2','s13','s2')}
    baseline2=expert_chain(x,w13b,w2b,s13b,s2b)
    external2=expert_chain(x,views2['w13'],views2['w2'],views2['s13'],views2['s2'])
    torch.npu.synchronize(); diff2=float((baseline2.float()-external2.float()).abs().max().item())
    pool.release(lease2)
    timings={
        'baseline':timeit(lambda: expert_chain(x,w13,w2,s13,s2)),
        'external':timeit(lambda: expert_chain(x,views['w13'],views['w2'],views['s13'],views['s2'])),
        'split':timeit(lambda: (expert_chain(x.index_select(0,main_idx),w13,w2,s13,s2),expert_chain(x.index_select(0,ext_idx),views['w13'],views['w2'],views['s13'],views['s2']))),
    }
    s=pool.stats()
    result={
        'status':'PASS' if diff==0.0 and split_diff==0.0 else 'FAIL',
        'layer':LAYER,'expert':EXPERT,'input':'synthetic bfloat16 [6,6144]',
        'actual_model_state':True,'w8a8':True,
        'slot_bytes':pos,'committed_bytes':s['committed_bytes'],'state_bytes_equal_all':state_equal,
        'per_rank_capacity':s['per_rank_capacity'],
        'external_vs_bank_max_abs':diff,'split_vs_bank_max_abs':split_diff,
        'layer50_external_vs_bank_max_abs':diff2,'layer50_state_bytes_equal_all':state_equal2,
        'layer50_copy_ms':copy_ms2,'cross_layer_slot_reused':True,
        'copy_ms':copy_ms,'timings_ms':timings,
        'memory_allocated':int(torch.npu.memory_allocated()),'memory_reserved':int(torch.npu.memory_reserved()),'max_memory_allocated':int(torch.npu.max_memory_allocated()),
    }
    open(OUT,'w').write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result['status']!='PASS': raise SystemExit(1)
    del pool
    torch.npu.synchronize()

if __name__=='__main__': main()
