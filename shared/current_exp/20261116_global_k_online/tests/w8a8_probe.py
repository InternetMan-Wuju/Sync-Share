#!/usr/bin/env python3
import json
import torch, torch_npu

def call(x,w,scale,pts,gl,mode):
    kw=dict(x=[x], weight=w, scale=[scale], per_token_scale=[pts], group_list=gl,
            split_item=2, group_type=0, group_list_type=1, output_dtype=torch.bfloat16)
    if mode=='list': kw['weight']=list(w)
    return torch.ops.npu.npu_grouped_matmul(**kw)[0]

def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)
    x=torch.randint(-8,8,(6,8),dtype=torch.int8,device='npu')
    w=torch.randint(-8,8,(3,4,8),dtype=torch.int8,device='npu') # [expert,out,in]
    s=torch.ones((3,4),dtype=torch.bfloat16,device='npu')
    pts=torch.ones((6,),dtype=torch.float32,device='npu')
    gl=torch.tensor([2,2,2],dtype=torch.int64,device='npu')
    out={}
    for mode,ww in [('bank',[w]),('multi',[w[i] for i in range(3)])]:
      for scale_mode,ss in [('3d',s),('list',[s[i] for i in range(3)])]:
       try:
        y=torch.ops.npu.npu_grouped_matmul(x=[x],weight=ww,scale=([ss] if scale_mode=='3d' else ss),per_token_scale=[pts],group_list=gl,split_item=2,group_type=0,group_list_type=1,output_dtype=torch.bfloat16)[0]
        torch.npu.synchronize(); out[mode+'/'+scale_mode]={'ok':True,'shape':list(y.shape),'sum':float(y.float().sum().item())}
       except Exception as e:
        out[mode+'/'+scale_mode]={'ok':False,'error':str(e).splitlines()[0]}
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
