"""Configurable global-K service adapter for selected routed-MoE layers.

First service prototype uses weight-masked local backfill:
- logical top-k and router weights stay unchanged;
- a fixed quota of selected hot-expert tasks is assigned to a real external slot;
- the original path keeps those tasks but masks their original weight to zero;
- the replica computes the task from the real slot and is added to the same
  final per-token output after the original routed MoE combine;
- the slot owner, identity, generation and storage are tracked independently
  from logical/communication expert ids.
"""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch,torch_npu,torch.distributed as dist
PARTS=("w13","w2","s13","s2","o13","o2")
_GLOBAL_SLOTS={"buffers":{},"chains":{},"gen":{},"meta":{}}
ATTR={"w13":"w13_weight","w2":"w2_weight","s13":"w13_weight_scale","s2":"w2_weight_scale","o13":"w13_weight_offset","o2":"w2_weight_offset"}

def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def state_layout(state):
 out=[];off=0
 for n in PARTS:
  t=state[n];b=int(t.numel()*t.element_size());out.append((n,off,b,t.dtype,tuple(t.shape)));off+=b
 return out,off
def slot_views(slot,layout):return {n:slot[o:o+b].view(d).reshape(s) for n,o,b,d,s in layout}
def external_state(exp,expert):
 d={}
 for name,attr in ATTR.items():
  src=getattr(exp,attr);part=src[expert:expert+1]
  if name in ('w13','w2'):part=torch_npu.npu_format_cast(part,2).contiguous()
  else:part=part.contiguous()
  d[name]=part
 return d
class Chain:
 def __init__(self,method,matmul,views,layer):self.method=method;self.matmul=matmul;self.layer=views;self.layer_id=layer
 def __call__(self,x):
  xq,xs=self.method.hidden_states_quantizer(x);n=torch.tensor([xq.shape[0]],dtype=torch.int64,device=xq.device)
  up=self.matmul.forward(self.layer,'w13',xq,n,torch.bfloat16,group_list_type=1,transposed=True,scale=[self.layer.w13_weight_scale],per_token_scale=[xs])
  act=torch.ops.npu.npu_swiglu(up);aq,a_s=self.method.hidden_states_quantizer(act);n2=torch.tensor([aq.shape[0]],dtype=torch.int64,device=aq.device)
  return self.matmul.forward(self.layer,'w2',aq,n2,torch.bfloat16,group_list_type=1,transposed=True,scale=[self.layer.w2_weight_scale],per_token_scale=[a_s])
class GlobalKService:
 def __init__(self,experts,layer_id,cfg):
  self.experts=experts;self.layer_id=layer_id;self.cfg=cfg;self.orig=experts.forward;self.ready=False;self.seq=0
  self.logdir=Path(cfg.get('log_dir','/exp-logs/k_service'));self.logdir.mkdir(parents=True,exist_ok=True)
 def log(self,r):
  r.update(rank=self.rank,layer=self.layer_id,seq=self.seq)
  with (self.logdir/f'rank{self.rank}.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
 def init(self):
  from sglang.srt.distributed import get_moe_ep_group
  from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
  from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
  g=get_moe_ep_group();self.rank=g.rank_in_group;self.cpu_group=g.cpu_group;self.device_group=g.device_group;self.ep=int(self.cfg.get('ep',16));assert g.world_size==self.ep
  self.num_main=int(self.cfg.get('main_per_rank',256//self.ep));self.method=NPUW8A8Int8MoEMethod();self.matmul=GroupedMatmul()
  sample=external_state(self.experts,0);self.layout,self.state_bytes=state_layout(sample);del sample
  self.slot_buffers=_GLOBAL_SLOTS['buffers'];self.slot_chains=_GLOBAL_SLOTS['chains'];self.slot_gen=_GLOBAL_SLOTS['gen'];self.slot_meta=_GLOBAL_SLOTS['meta']
  owners=[int(x) for x in self.cfg['owners']];hots=[int(x) for x in self.cfg['hot_experts']]
  assert len(owners)==len(hots)==int(self.cfg['K'])
  for owner,hot in zip(owners,hots):
   assert 0<=owner<self.ep and 0<=hot<256
   if self.rank==owner and owner not in self.slot_buffers:self.slot_buffers[owner]=torch.empty(self.state_bytes,dtype=torch.uint8,device='npu')
   self.slot_gen.setdefault(owner,0);self.slot_meta.setdefault(owner,None)
  self.log({'kind':'init','K':int(self.cfg['K']),'owners':owners,'hots':hots,'slot_bytes_per_owner':self.state_bytes,'main_num':self.num_main,'mode':'weight_masked_local_backfill'})
  self.ready=True
 def ensure_slots(self):
  owners=[int(x) for x in self.cfg['owners']];hots=[int(x) for x in self.cfg['hot_experts']]
  for owner,hot in zip(owners,hots):
   meta=self.slot_meta.get(owner)
   if meta and meta['layer']==self.layer_id and meta['expert']==hot and meta['ready']:continue
   src_rank=hot//self.num_main;t=time.perf_counter_ns();buf=None;state=None
   torch.npu.synchronize()
   if self.rank==src_rank:
    state=external_state(self.experts,hot%self.num_main);buf=torch.empty(self.state_bytes,dtype=torch.uint8,device='npu')
    for name,off,nbytes,_,_ in self.layout:buf[off:off+nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
    torch.npu.synchronize()
   if self.rank==owner and self.rank==src_rank:self.slot_buffers[owner].copy_(buf)
   elif self.rank==src_rank:dist.send(buf,dst=owner,group=self.device_group)
   elif self.rank==owner:dist.recv(self.slot_buffers[owner],src=src_rank,group=self.device_group)
   torch.npu.synchronize();dist.barrier(group=self.cpu_group);torch.npu.synchronize()
   if self.rank==owner:
    views=slot_views(self.slot_buffers[owner],self.layout);lay=SimpleNamespace(**{ATTR[p]:views[p] for p in PARTS})
    self.slot_chains[owner]=Chain(self.method,self.matmul,lay,self.layer_id);self.slot_gen[owner]+=1
    self.slot_meta[owner]={'layer':self.layer_id,'expert':hot,'generation':self.slot_gen[owner],'ready':True,'storage_ptr':int(self.slot_buffers[owner].untyped_storage().data_ptr()),'bytes':self.state_bytes}
   meta_out=[None]*self.ep;dist.all_gather_object(meta_out,self.slot_meta.get(owner),group=self.cpu_group)
   self.log({'kind':'install','slot_owner':owner,'expert':hot,'source_rank':src_rank,'elapsed_ms':(time.perf_counter_ns()-t)/1e6,'owner_meta':meta_out[owner]})
   del state,buf
 def forward(self,hidden_states,topk_output,*args,**kwargs):
  from sglang.srt.distributed import get_moe_ep_group
  g=get_moe_ep_group();rank=g.rank_in_group
  logical=topk_output.topk_ids.long();flat=logical.reshape(-1);owners=[int(x) for x in self.cfg['owners']];hots=[int(x) for x in self.cfg['hot_experts']];num=int(self.cfg.get('split_num',1));den=int(self.cfg.get('split_den',2))
  candidate=[]
  for owner,hot in zip(owners,hots):
   mask=flat==hot;cnt=int(mask.sum());q=((cnt*num)//den) if rank==owner else 0;candidate.append((owner,hot,mask,q))
  local_q=sum(q for _,_,_,q in candidate);flag=torch.tensor([local_q],dtype=torch.int64);dist.all_reduce(flag,op=dist.ReduceOp.SUM,group=g.cpu_group);global_q=int(flag.item())
  if global_q==0:
   self.rank=rank;self.seq+=1;self.log({'kind':'call_noop','rows':int(hidden_states.shape[0]),'tasks':int(logical.numel()),'replica_tasks':0})
   return self.orig(hidden_states,topk_output,*args,**kwargs)
  if not self.ready:self.init()
  self.seq+=1;self.ensure_slots();orig_w=topk_output.topk_weights;masked=orig_w.clone();contrib=torch.zeros_like(hidden_states,dtype=torch.float32);moved={}
  for owner,hot,mask,q in candidate:
   if q:
    ordinal=mask.long().cumsum(0);sel=mask&(ordinal<=q);rows=torch.nonzero(sel,as_tuple=False).flatten()//logical.shape[1]
    x=hidden_states.index_select(0,rows);y=self.slot_chains[owner](x);w=orig_w.reshape(-1)[sel].unsqueeze(1)
    contrib.index_add_(0,rows,y.float()*w.float());masked.reshape(-1)[sel]=0.0;moved[owner]=q
  t=time.perf_counter_ns();base=self.orig(hidden_states,topk_output._replace(topk_weights=masked),*args,**kwargs);out=base+contrib.to(base.dtype);torch.npu.synchronize();verify={}
  if self.cfg.get('verify'):
   ref=self.orig(hidden_states,topk_output,*args,**kwargs);torch.npu.synchronize();d=(out.float()-ref.float()).abs();verify={'max_abs':float(d.max().item()) if d.numel() else 0.0,'nrmse':float((d.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)).item()) if d.numel() else 0.0,'finite':bool(torch.isfinite(out).all().item())}
  self.log({'kind':'call','rows':int(hidden_states.shape[0]),'tasks':int(logical.numel()),'moved_by_owner':moved,'replica_tasks':sum(moved.values()),'zeroed_weight_tasks':sum(moved.values()),'wall_ms':(time.perf_counter_ns()-t)/1e6,'input_sha':sha(hidden_states),'output_sha':sha(out),'verify':verify})
  return out


def install(module):
 cfg_path=os.environ.get('CYT_GLOBAL_K_CONFIG')
 if not cfg_path:return
 cfg=json.loads(Path(cfg_path).read_text())
 if not cfg.get('enabled',True):return
 targets=set(int(x) for x in cfg.get('layers',[49]))
 orig=module.DeepseekV2MoE.__init__
 def patched(self,*a,**kw):
  orig(self,*a,**kw)
  if self.layer_id in targets:
   rt=GlobalKService(self.experts,self.layer_id,cfg);self._cyt_global_k=rt;self.experts.forward=rt.forward
 module.DeepseekV2MoE.__init__=patched
