import json,os
from pathlib import Path
import torch,torch_npu

def install(module):
    orig=module.DeepseekV2MoE.__init__
    def patched(self,*a,**kw):
        orig(self,*a,**kw)
        if self.layer_id!=49:return
        old=self.experts.forward
        self._cyt_inspected=False
        def wrapped(hidden_states,topk_output=None,*fa,**fk):
            if not self._cyt_inspected:
                self._cyt_inspected=True
                row={'layer':self.layer_id,'rank':torch.distributed.get_rank(),'hidden_shape':list(hidden_states.shape),'topk_ids_shape':list(topk_output.topk_ids.shape),'experts_class':type(self.experts).__name__,'quant_method':type(getattr(self.experts,'quant_method',None)).__name__,'num_local_experts':getattr(self.experts,'num_local_experts',None),'disabled_expert_ids':list(self.experts.disabled_expert_ids)[:20] if hasattr(self.experts,'disabled_expert_ids') else None,'attrs':{}}
                for name in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_weight_offset','w2_weight_offset','w13_weight_bias','w2_weight_bias'):
                    x=getattr(self.experts,name,None)
                    if x is not None:
                        try:fmt=torch_npu.get_npu_format(x)
                        except Exception as e:fmt=f'ERR:{e}'
                        row['attrs'][name]={'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'storage_offset':int(x.storage_offset()),'storage_bytes':int(x.untyped_storage().nbytes()),'format':fmt}
                for name in ('dispatcher','top_k','num_experts','num_local_experts','ep_size','moe_tp_size'):
                    x=getattr(self.experts,name,None)
                    if hasattr(x,'__dict__'):
                        row[name]={'class':type(x).__name__,'dict':{k:str(v)[:200] for k,v in x.__dict__.items() if k in ('num_experts','num_local_experts','num_experts_per_rank','group','async_finish','deepep_mode')}}
                    elif x is not None:row[name]=str(x)[:200]
                out=Path(os.environ.get('CYT_INSPECT_DIR','/exp-logs/inspect'));out.mkdir(parents=True,exist_ok=True)
                (out/f'inspect_rank{row["rank"]}.json').write_text(json.dumps(row,indent=2,default=str))
            return old(hidden_states,topk_output,*fa,**fk)
        self.experts.forward=wrapped
    module.DeepseekV2MoE.__init__=patched
