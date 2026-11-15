"""Single-node EP16 experimental runtime, default control=off.

Maintains all original kernels and dispatch/combine. Only routing at the target
layer and the reserved slot's complete state are changed. CPU counts/planning,
ND pack, real HCCL P2P, destination install and ready commit are synchronous.
"""
import json, os, time, functools
from pathlib import Path
import torch, torch_npu
import torch.distributed as dist
from planner import plan, source_quotas
ROOT=Path(os.environ.get('CYT_RUN','/runexp'))
STATE_NAMES=('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_offset','w2_weight_offset','w13_scale_bias','w2_scale_bias','w13_weight_bias','w2_weight_bias')

def stamp(): return time.perf_counter_ns()
def ms(t): return (stamp()-t)/1e6

def primary_map(ids, info, *a, **kw):
    if info is None: return ids
    assert info.num_physical_experts==272
    return torch.where(ids>=0,ids+ids//16,ids)

class Runtime:
    def __init__(self, moe):
        self.moe=moe;self.layer=moe.experts;self.original=self.layer.forward
        self.ready=False;self.cache={};self.seq=0;self.events=[];self.prof=False
        self.layer.forward=self.forward
        for obj,name,label in ((self.layer.dispatcher,'dispatch','dispatch'),(self.layer,'run_moe_core','compute'),(self.layer.dispatcher,'combine','combine')):
            old=getattr(obj,name)
            def timed(*a,_old=old,_label=label,**kw):
                if not self.prof:return _old(*a,**kw)
                s=torch.npu.Event(enable_timing=True);e=torch.npu.Event(enable_timing=True);s.record();out=_old(*a,**kw);e.record();self.events.append((_label,s,e));return out
            setattr(obj,name,timed)
    def log(self,row):
        row.update(rank=self.rank,layer=self.moe.layer_id,seq=self.seq)
        with (ROOT/'local'/f'rank{self.rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def init(self):
        from sglang.srt.distributed import get_moe_ep_group
        g=get_moe_ep_group();self.rank=g.rank_in_group;self.group=g.device_group;self.cpu_group=g.cpu_group
        assert g.world_size==16 and self.layer.num_local_experts==17
        assert list(g.ranks)==list(range(16)), "P2P peer mapping must be global"
        self.names=[n for n in STATE_NAMES if getattr(self.layer,n,None) is not None]
        self.spec=[];offset=0
        for n in self.names:
            x=getattr(self.layer,n)
            assert x.shape[0]==17
            size=x[0].numel()*x.element_size()
            self.spec.append((n,tuple(x.shape[1:]),x.dtype,offset,size));offset+=size
        self.buffer=torch.empty(offset,dtype=torch.uint8,device=self.layer.w13_weight.device)
        self.count_recv=torch.empty((16,256),dtype=torch.int64,device=self.buffer.device)
        warm=torch.ones(1,device=self.buffer.device);dist.all_reduce(warm,group=self.group);torch.npu.synchronize()
        self.log({'kind':'state','copy_bytes':offset,'moe_tp_size':self.layer.moe_tp_size,'spec':[{'name':n,'shape':shape,'dtype':str(dtype),'bytes':sz,'format':torch_npu.get_npu_format(getattr(self.layer,n))} for n,shape,dtype,off,sz in self.spec]})
        assert self.layer.moe_tp_size==1
        self.ready=True
    def copy(self,p,verify):
        src,dst=p['src'],p['dst'];times={};t=stamp()
        if self.rank==src:
            for n,shape,dtype,off,sz in self.spec:
                x=getattr(self.layer,n)
                # Cast the expert view to ND before packing. No raw NZ bytes.
                nd=torch_npu.npu_format_cast(x[p['expert']%16],2).contiguous()
                self.buffer[off:off+sz].copy_(nd.view(torch.uint8).reshape(-1))
        torch.npu.synchronize();times['source_prepare_ms']=ms(t);t=stamp()
        if self.rank==src:dist.send(self.buffer,dst=dst,group=self.group)
        elif self.rank==dst:dist.recv(self.buffer,src=src,group=self.group)
        torch.npu.synchronize();times['transfer_ms']=ms(t);t=stamp()
        if self.rank==dst:
            for n,shape,dtype,off,sz in self.spec:
                nd=self.buffer[off:off+sz].view(dtype).reshape(shape)
                getattr(self.layer,n)[16].copy_(nd)
        torch.npu.synchronize();times['destination_install_ms']=ms(t);t=stamp()
        dist.barrier(group=self.group);torch.npu.synchronize();times['ready_wait_ms']=ms(t)
        if verify:
            # Verify bytes of all runtime tensors after installing into actual bank.
            import hashlib
            local=None
            if self.rank in (src,dst):
                slot=p['expert']%16 if self.rank==src else 16
                h=hashlib.sha256()
                for n in self.names:
                    x=torch_npu.npu_format_cast(getattr(self.layer,n)[slot],2).contiguous()
                    h.update(x.view(torch.uint8).cpu().numpy().tobytes())
                local=h.hexdigest()
            all_hashes=[None]*16;dist.all_gather_object(all_hashes,local,group=self.cpu_group)
            assert all_hashes[src]==all_hashes[dst],all_hashes
            times['state_bytes_equal']=True
        self.cache[dst]=p['expert'];return times
    def forward(self,hidden_states,topk_output,*args,**kwargs):
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch
        if not get_is_extend_in_batch():return self.original(hidden_states,topk_output,*args,**kwargs)
        if not self.ready:self.init()
        self.seq+=1
        cfg=json.loads((ROOT/'local/control.json').read_text())
        # Explicit revision is for this isolated, sequential experiment only.
        # It allows testing revised external code without reloading 750GB weights.
        revision=cfg.get('implementation_revision',1)
        if getattr(self,'revision',None)!=revision:
            import experiment,importlib,hashlib
            self.experiment=importlib.reload(experiment);self.revision=revision
            digest=hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest()
            hashes=[None]*16;dist.all_gather_object(hashes,digest,group=self.cpu_group)
            assert len(set(hashes))==1
            self.log({'kind':'implementation','revision':revision,'sha256':digest})
        return self.experiment.execute(self,hidden_states,topk_output,args,kwargs,cfg)

def install(module):
    import sglang.srt.hardware_backend.npu.moe.topk as npu_topk
    import sglang.srt.layers.moe.topk as core_topk
    npu_topk.topk_ids_logical_to_physical=primary_map
    core_topk.topk_ids_logical_to_physical=primary_map
    orig=module.DeepseekV2MoE.__init__
    @functools.wraps(orig)
    def init(self,*a,**kw):
        # Initialize the ACL compiler before loading hundreds of weight shards;
        # compiler subprocesses must not inherit the weight loader's open files.
        if not getattr(init, '_acl_preinitialized', False):
            import resource
            from sglang.srt.distributed import get_moe_ep_group
            g=get_moe_ep_group()
            for owner in range(g.world_size):
                if g.rank_in_group==owner:
                    before=len(os.listdir('/proc/self/fd'))
                    probe=torch.ones((64,64),dtype=torch.int8,device='npu')
                    probe=torch_npu.npu_format_cast(probe,29)
                    torch.npu.synchronize()
                    print('CYT_ACL_SERIAL_INIT',torch.npu.current_device(),before,
                          resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)
                dist.barrier(group=g.cpu_group)
            init._acl_preinitialized=True
        orig(self,*a,**kw)
        if self.layer_id==49:self._cyt_runtime=Runtime(self)
    module.DeepseekV2MoE.__init__=init
