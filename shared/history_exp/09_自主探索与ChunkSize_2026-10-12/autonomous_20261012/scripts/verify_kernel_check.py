#!/usr/bin/env python3
"""Exercise the installed NPU greedy verifier against an independent CPU oracle.

This checks linear topk=1 token comparison, not the model logits or attention KV.
Run inside our node23 container only, with tiny manufactured tensors.
"""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import torch
import torch_npu

SOURCE=Path('/usr/local/python3.11.15/lib/python3.11/site-packages/sgl_kernel_npu/sample/verify_tree_greedy.py')
spec=importlib.util.spec_from_file_location('actual_verifier',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.npu.set_device(0)
results=[]
for dtype in (torch.int32,torch.int64):
    for bs in (1,2,4,8,16):
        rng=random.Random(812+bs);candidates=[[rng.randrange(100,150000) for _ in range(4)] for _ in range(bs)]
        targets=[];counts=[]
        for i,row in enumerate(candidates):
            count=(i+bs)%4;counts.append(count)
            target=[row[j+1] if j<count else (row[min(j+1,3)]+37)%150000 for j in range(4)]
            targets.append(target)
        cand=torch.tensor(candidates,dtype=dtype,device='npu:0')
        target=torch.tensor(targets,dtype=dtype,device='npu:0')
        retrieve=torch.arange(bs*4,dtype=dtype,device='npu:0').reshape(bs,4)
        predictions=torch.full((bs*4,),-1,dtype=dtype,device='npu:0')
        index=torch.full((bs,4),-1,dtype=dtype,device='npu:0')
        accepted=torch.full((bs,),-1,dtype=dtype,device='npu:0')
        module.verify_tree_greedy(predictions,index,accepted,cand,retrieve,retrieve,retrieve,target)
        got=accepted.cpu().tolist();actual_index=index.cpu().tolist();pred=predictions.cpu().tolist()
        assert got==counts,(dtype,bs,got,counts)
        for i,n in enumerate(counts):
            assert actual_index[i][:n+1]==list(range(i*4,i*4+n+1))
            assert pred[i*4:i*4+n+1]==targets[i][:n+1]
        results.append({'dtype':str(dtype),'batch':bs,'accepted':got,'passed':True})
        print('verifier oracle passed',dtype,bs,got,flush=True)
out=Path('/exp-logs/verifier_kernel_check.json')
out.write_text(json.dumps({'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                            'passed':True,'scope':'linear topk=1 comparison and bonus token; no model-forward equivalence claim','results':results},indent=2))
