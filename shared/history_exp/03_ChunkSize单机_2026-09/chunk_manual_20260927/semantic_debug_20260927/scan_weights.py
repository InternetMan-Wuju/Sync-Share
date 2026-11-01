#!/usr/bin/env python3
"""Inspect safetensors with read-only mmap, never write or load on NPU."""
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--model',default='/mnt/raid/user/data/models/glm-qq')
p.add_argument('--out',required=True)
p.add_argument('--names',nargs='*',default=['lm_head.weight','model.embed_tokens.weight','rot.weight'])
a=p.parse_args()
base=Path(a.model)
wm=json.loads((base/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
results=[]
for name in a.names:
 start=time.perf_counter()
 path=base/wm[name]
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n))
 info=header[name]; dtype=info['dtype']; begin,end=info['data_offsets']
 dt={'BF16':'<u2','F16':'<f2','F32':'<f4','F64':'<f8'}[dtype]
 arr=np.memmap(path,mode='r',offset=8+n+begin,dtype=dt,shape=((end-begin)//np.dtype(dt).itemsize,))
 sha=hashlib.sha256(); bad_count=0; examples=[]; finite_min=None; finite_max=None
 for offset in range(0,arr.size,4*1024*1024):
  block=arr[offset:offset+4*1024*1024]
  sha.update(block.tobytes())
  if dtype=='BF16':
   values=(block.astype(np.uint32)<<16).view(np.float32)
  else: values=block
  mask=np.isfinite(values)
  cnt=int(np.count_nonzero(~mask)); bad_count+=cnt
  if cnt and len(examples)<12:
   for i in np.flatnonzero(~mask)[:12-len(examples)]:
    index=offset+int(i)
    examples.append({'flat_index':index,'coordinates':list(map(int,np.unravel_index(index,info['shape']))),'value':str(values[i])})
  finite=values[mask]
  if finite.size:
   lo=float(finite.min()); hi=float(finite.max())
   finite_min=lo if finite_min is None else min(finite_min,lo)
   finite_max=hi if finite_max is None else max(finite_max,hi)
 result={'name':name,'file':str(path),'dtype':dtype,'shape':info['shape'],'elements':int(arr.size),'nonfinite':bad_count,'examples':examples,'finite_min':finite_min,'finite_max':finite_max,'sha256_tensor_bytes':sha.hexdigest(),'elapsed_s':time.perf_counter()-start,'file_mtime_ns':path.stat().st_mtime_ns}
 results.append(result)
 (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False),flush=True)
 del arr
