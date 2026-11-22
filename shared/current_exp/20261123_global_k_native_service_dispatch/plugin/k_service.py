"""Configurable global-K service adapter with an opt-in native-dispatch attempt.

The production matrix for 2026-11-23 uses the local-backfill path: real owner
slots, real W8A8 expert chains, main-path weight masking, and result backfill.
The opt-in native extra-communication-id path is retained only as an experiment;
target-image combine rejected it with Invalid attr=601 and it must not be
reported as a working native replica route.
"""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch,torch_npu,torch.distributed as dist
PARTS=("w13","w2","s13","s2","o13","o2")
ATTR={"w13":"w13_weight","w2":"w2_weight","s13":"w13_weight_scale","s2":"w2_weight_scale","o13":"w13_weight_offset","o2":"w2_weight_offset"}
_GLOBAL_SLOTS={"buffers":{},"chains":{},"gen":{},"meta":{}}
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def state_layout(state):
 out=[];off=0
 for n in PARTS:
  t=state[n];b=int(t.numel()*t.element_size());out.append((n,off,b,t.dtype,tuple(t.shape)));off+=b
 return out,off
def slot_views(slot,layout):return {n:slot[o:o+b].view(d).reshape(s) for n,o,b,d,s in layout}
def external_state(exp,expert):
 out={}
 for name,attr in ATTR.items():
  part=getattr(exp,attr)[expert:expert+1]
  out[name]=torch_npu.npu_format_cast(part,2).contiguous() if name in ('w13','w2') else part.contiguous()
 return out
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
  from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher,DeepEPNormalCombineInput
  from sglang.srt.layers.moe.topk import StandardTopKOutput
  from sglang.srt.layers.moe.utils import DeepEPMode
  from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
  from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
  g=get_moe_ep_group();self.rank=g.rank_in_group;self.cpu_group=g.cpu_group;self.device_group=g.device_group;self.ep=int(self.cfg.get('ep',16));assert g.world_size==self.ep
  self.num_main=int(self.cfg.get('main_per_rank',256//self.ep));self.local_comm=self.num_main+1;self.num_comm=self.ep*self.local_comm
  owners=[int(x) for x in self.cfg['owners']];hots=[int(x) for x in self.cfg['hot_experts']]
  assert len(owners)==len(hots)==int(self.cfg['K']);assert len(owners)==len(set(owners)),'first service version requires one slot per owner'
  self.owners=owners;self.hots=hots;self.placeholder_rank=next(r for r in range(self.ep) if r not in owners);self.placeholder_comm=self.placeholder_rank*self.local_comm+self.num_main
  self.StandardTopKOutput=StandardTopKOutput;self.DeepEPNormalCombineInput=DeepEPNormalCombineInput
  self.native_dispatch=bool(self.cfg.get('native_dispatch',False))
  self.replica_dispatcher=DeepEPDispatcher(group=self.device_group,router_topk=8,num_experts=self.num_comm,num_local_experts=self.local_comm,hidden_size=6144,params_dtype=torch.bfloat16,deepep_mode=DeepEPMode.NORMAL,async_finish=False) if self.native_dispatch else None
  self.method=NPUW8A8Int8MoEMethod();self.matmul=GroupedMatmul();sample=external_state(self.experts,0);self.layout,self.state_bytes=state_layout(sample);del sample
  self.slot_buffers=_GLOBAL_SLOTS['buffers'];self.slot_chains=_GLOBAL_SLOTS['chains'];self.slot_gen=_GLOBAL_SLOTS['gen'];self.slot_meta=_GLOBAL_SLOTS['meta']
  for owner in owners:
   assert 0<=owner<self.ep
   if self.rank==owner and owner not in self.slot_buffers:self.slot_buffers[owner]=torch.empty(self.state_bytes,dtype=torch.uint8,device='npu')
   self.slot_gen.setdefault(owner,0);self.slot_meta.setdefault(owner,None)
  self.log({'kind':'init','K':int(self.cfg['K']),'owners':owners,'hots':hots,'num_comm':self.num_comm,'local_comm':self.local_comm,'placeholder_rank':self.placeholder_rank,'placeholder_comm':self.placeholder_comm,'state_bytes':self.state_bytes,'mode':'native_deepep_topk1_reroute'})
  self.ready=True
 def ensure_slots(self):
  for owner,hot in zip(self.owners,self.hots):
   local=self.slot_meta.get(owner) if self.rank==owner else None;gathered=[None]*self.ep;dist.all_gather_object(gathered,local,group=self.cpu_group);meta=gathered[owner]
   reuse=bool(meta and meta.get('ready') and meta.get('layer')==self.layer_id and meta.get('expert')==hot and meta.get('owner')==owner and meta.get('local_slot')==0)
   if self.rank==owner:
    ptr=int(self.slot_buffers[owner].untyped_storage().data_ptr())
    reuse=reuse and meta.get('storage_ptr')==ptr
   flag=torch.tensor([1 if reuse else 0],dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=self.cpu_group);reuse=int(flag.item())==1
   if reuse:
    if self.rank==owner:self.slot_chains[owner]=_GLOBAL_SLOTS['chains'].get(owner)
    self.log({'kind':'reuse','slot_owner':owner,'local_slot':0,'expert':hot,'owner_meta':meta});continue
   source=hot//self.num_main;t=time.perf_counter_ns();state=None;buf=None
   torch.npu.synchronize()
   if self.rank==source:
    state=external_state(self.experts,hot%self.num_main);buf=torch.empty(self.state_bytes,dtype=torch.uint8,device='npu')
    for name,off,nbytes,_,_ in self.layout:buf[off:off+nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
    torch.npu.synchronize()
   if self.rank==owner and self.rank==source:self.slot_buffers[owner].copy_(buf)
   elif self.rank==source:dist.send(buf,dst=owner,group=self.device_group)
   elif self.rank==owner:dist.recv(self.slot_buffers[owner],src=source,group=self.device_group)
   torch.npu.synchronize();dist.barrier(group=self.cpu_group);torch.npu.synchronize()
   if self.rank==owner:
    views=slot_views(self.slot_buffers[owner],self.layout);lay=SimpleNamespace(**{ATTR[p]:views[p] for p in PARTS});self.slot_chains[owner]=Chain(self.method,self.matmul,lay,self.layer_id);self.slot_gen[owner]+=1
    self.slot_meta[owner]={'owner':owner,'local_slot':0,'layer':self.layer_id,'expert':hot,'generation':self.slot_gen[owner],'ready':True,'storage_ptr':int(self.slot_buffers[owner].untyped_storage().data_ptr()),'bytes':self.state_bytes}
   del state,buf
   updated=[None]*self.ep;dist.all_gather_object(updated,self.slot_meta.get(owner) if self.rank==owner else None,group=self.cpu_group)
   self.log({'kind':'install','slot_owner':owner,'local_slot':0,'expert':hot,'source_rank':source,'elapsed_ms':(time.perf_counter_ns()-t)/1e6,'owner_meta':updated[owner]})
 def forward(self,hidden_states,topk_output,*args,**kwargs):
  from sglang.srt.layers.dp_attention import get_is_extend_in_batch
  if not get_is_extend_in_batch():return self.orig(hidden_states,topk_output,*args,**kwargs)
  if not self.ready:self.init()
  self.seq+=1;logical=topk_output.topk_ids.long();flat=logical.reshape(-1);orig_w=topk_output.topk_weights;masked=orig_w.clone();tasks=[];num=int(self.cfg.get('split_num',1));den=int(self.cfg.get('split_den',2))
  for owner,hot in zip(self.owners,self.hots):
   mask=flat==hot;cnt=int(mask.sum());q=(cnt*num)//den
   if q:
    ordinal=mask.long().cumsum(0);sel=mask&(ordinal<=q);positions=torch.nonzero(sel,as_tuple=False).flatten();rows=positions//logical.shape[1];task_ids=torch.full((positions.numel(),8),self.placeholder_comm,dtype=torch.int32,device=logical.device);task_ids[:,0]=owner*self.local_comm+self.num_main;task_w=torch.zeros((positions.numel(),8),dtype=orig_w.dtype,device=logical.device);task_w[:,0]=orig_w.reshape(-1)[positions]
    for r,p in zip(rows.cpu().tolist(),positions.cpu().tolist()):tasks.append({'sender':self.rank,'token':r//logical.shape[1],'topk_pos':r%logical.shape[1],'logical':hot,'owner':owner,'local_slot':0,'comm':owner*self.local_comm+self.num_main})
    masked.reshape(-1)[positions]=0.0
    if 'task_hidden' not in locals():task_hidden=[];task_ids_all=[];task_weights=[]
    task_hidden.append(hidden_states.index_select(0,rows));task_ids_all.append(task_ids);task_weights.append(task_w)
  local_q=sum(len(x['logical']) for x in []) if False else len(tasks)
  flag=torch.tensor([local_q],dtype=torch.int64);dist.all_reduce(flag,op=dist.ReduceOp.SUM,group=self.cpu_group);global_q=int(flag.item())
  if global_q==0:
   self.log({'kind':'call_noop','rows':int(hidden_states.shape[0]),'tasks':int(logical.numel()),'replica_tasks':0})
   return self.orig(hidden_states,topk_output,*args,**kwargs)
  self.ensure_slots();t=time.perf_counter_ns()
  if not self.native_dispatch:
   # First service fallback: only owner-local tasks, real slot compute and local backfill.
   local_tasks=[];local_rows=[];local_weights=[]
   masked=orig_w.clone();contrib=torch.zeros_like(hidden_states,dtype=torch.float32)
   for owner,hot in zip(self.owners,self.hots):
    if self.rank!=owner:continue
    mask=flat==hot;cnt=int(mask.sum());q=(cnt*num)//den
    if q:
     ordinal=mask.long().cumsum(0);sel=mask&(ordinal<=q);pos=torch.nonzero(sel,as_tuple=False).flatten();rows=pos//logical.shape[1]
     y=self.slot_chains[owner](hidden_states.index_select(0,rows));w=orig_w.reshape(-1)[pos].unsqueeze(1);contrib.index_add_(0,rows,y.float()*w.float());masked.reshape(-1)[pos]=0.0;local_tasks.append((owner,q))
   base=self.orig(hidden_states,topk_output._replace(topk_weights=masked),*args,**kwargs);out=base+contrib.to(base.dtype);torch.npu.synchronize();verify={}
   if self.cfg.get('verify'):
    ref=self.orig(hidden_states,topk_output,*args,**kwargs);torch.npu.synchronize();d=(out.float()-ref.float()).abs();verify={'max_abs':float(d.max().item()) if d.numel() else 0.0,'nrmse':float((d.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)).item()) if d.numel() else 0.0,'finite':bool(torch.isfinite(out).all().item())}
   self.log({'kind':'call_local_fallback','rows':int(hidden_states.shape[0]),'tasks':int(logical.numel()),'replica_tasks':sum(q for _,q in local_tasks),'per_owner':{str(o):q for o,q in local_tasks},'verify':verify,'wall_ms':(time.perf_counter_ns()-t)/1e6,'output_sha':sha(out)})
   return out
  # Every rank enters the replica dispatch. Rankings without a real moved task send one explicit zero-weight padding row.
  if local_q:
   rh=torch.cat(task_hidden,dim=0);rids=torch.cat(task_ids_all,dim=0);rw=torch.cat(task_weights,dim=0);real_rows=torch.cat([torch.arange(x.shape[0],device=x.device) for x in task_hidden]).cpu().tolist()
   # row positions are retained again after cat via a second index vector
   row_pos=[];posoff=0
   for owner,hot in zip(self.owners,self.hots):
    mask=flat==hot;cnt=int(mask.sum());q=(cnt*num)//den
    if q:
     ordinal=mask.long().cumsum(0);sel=mask&(ordinal<=q);poss=torch.nonzero(sel,as_tuple=False).flatten();row_pos.extend((poss//logical.shape[1]).cpu().tolist())
   row_t=torch.tensor(row_pos,dtype=torch.int64,device=hidden_states.device)
  else:
   rh=torch.zeros((1,hidden_states.shape[1]),dtype=hidden_states.dtype,device=hidden_states.device);rids=torch.full((1,8),self.placeholder_comm,dtype=torch.int32,device=hidden_states.device);rw=torch.zeros((1,8),dtype=orig_w.dtype,device=hidden_states.device);row_t=torch.empty((0,),dtype=torch.int64,device=hidden_states.device)
  dout=self.replica_dispatcher.dispatch(rh,self.StandardTopKOutput(rw,rids,None));recv=dout.hidden_states[0] if isinstance(dout.hidden_states,tuple) else dout.hidden_states;counts=[int(x) for x in dout.num_recv_tokens_per_expert]
  rout=torch.zeros_like(recv);pos=0;real_received=0;padding_received=0
  for li,c in enumerate(counts):
   if c<=0:continue
   comm=self.rank*self.local_comm+li
   if li<self.num_main:raise RuntimeError(f'unexpected replica dispatch to main comm {comm}')
   if comm==self.placeholder_comm:padding_received+=c
   elif self.rank in self.owners and li==self.num_main:
    rout[pos:pos+c].copy_(self.slot_chains[self.rank](recv[pos:pos+c]));real_received+=c
   else:raise RuntimeError(f'non-owner got replica task comm={comm} rank={self.rank}')
   pos+=c
  combined=self.replica_dispatcher.combine(self.DeepEPNormalCombineInput(rout,dout.topk_ids,dout.topk_weights));torch.npu.synchronize()
  contrib=torch.zeros_like(hidden_states,dtype=torch.float32)
  if local_q:contrib.index_add_(0,row_t,combined[:local_q].float())
  base=self.orig(hidden_states,topk_output._replace(topk_weights=masked),*args,**kwargs);out=base+contrib.to(base.dtype);torch.npu.synchronize();verify={}
  if self.cfg.get('verify'):
   ref=self.orig(hidden_states,topk_output,*args,**kwargs);torch.npu.synchronize();d=(out.float()-ref.float()).abs();verify={'max_abs':float(d.max().item()) if d.numel() else 0.0,'nrmse':float((d.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)).item()) if d.numel() else 0.0,'finite':bool(torch.isfinite(out).all().item())}
  self.log({'kind':'call','rows':int(hidden_states.shape[0]),'tasks':int(logical.numel()),'replica_tasks':local_q,'global_replica_tasks':global_q,'counts':counts,'real_received':real_received,'padding_received':padding_received,'dispatch_tasks':tasks,'verify':verify,'wall_ms':(time.perf_counter_ns()-t)/1e6,'output_sha':sha(out)})
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
