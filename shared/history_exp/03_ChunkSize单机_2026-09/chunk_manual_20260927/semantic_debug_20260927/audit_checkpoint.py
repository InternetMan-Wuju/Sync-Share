#!/usr/bin/env python3
"""Hash original shards and inspect floating tensors, strictly read-only."""
import argparse, hashlib, json, struct, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--model',default='/mnt/raid/user/data/models/glm-qq')
p.add_argument('--out',required=True)
p.add_argument('--workers',type=int,default=3)
a=p.parse_args()
model=Path(a.model); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
DT={'BF16':'<u2','F16':'<f2','F32':'<f4','F64':'<f8'}
start=time.perf_counter()
def check(path):
 stamp=path.stat(); sha=hashlib.sha256()
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n)); f.seek(0)
  while True:
   block=f.read(16*1024*1024)
   if not block: break
   sha.update(block)
 bad=[]; floats=0; maxabs=0.0
 for name,info in header.items():
  if name=='__metadata__' or info['dtype'] not in DT: continue
  floats+=1; begin,end=info['data_offsets']; dt=np.dtype(DT[info['dtype']])
  arr=np.memmap(path,mode='r',offset=8+n+begin,dtype=dt,shape=((end-begin)//dt.itemsize,))
  nonfinite=0; extreme=0; peak=0.0
  for offset in range(0,arr.size,4*1024*1024):
   block=arr[offset:offset+4*1024*1024]
   values=(block.astype(np.uint32)<<16).view(np.float32) if info['dtype']=='BF16' else block
   finite=np.isfinite(values); nonfinite+=int(np.count_nonzero(~finite))
   vals=values[finite]
   if vals.size:
    vabs=np.abs(vals); peak=max(peak,float(vabs.max()))
    extreme+=int(np.count_nonzero(vabs>1e10))
  del arr
  maxabs=max(maxabs,peak)
  if nonfinite or extreme:
   bad.append({'name':name,'dtype':info['dtype'],'shape':info['shape'],'nonfinite':nonfinite,'abs_gt_1e10':extreme,'finite_abs_max':peak})
 finish=path.stat()
 return {'file':path.name,'bytes':stamp.st_size,'mtime_ns':stamp.st_mtime_ns,'sha256':sha.hexdigest(),'floating_tensors':floats,'max_floating_abs':maxabs,'bad_tensors':bad,'unchanged_during_audit':stamp.st_size==finish.st_size and stamp.st_mtime_ns==finish.st_mtime_ns}
paths=sorted(model.glob('*.safetensors'))
results=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
 futures={pool.submit(check,path):path for path in paths}
 for fut in as_completed(futures):
  result=fut.result(); results.append(result)
  (out/(result['file']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  if result['bad_tensors'] or len(results)%10==0 or len(results)==len(paths):
   print(json.dumps({'files_done':len(results),'files_total':len(paths),'file':result['file'],'bad_tensors':len(result['bad_tensors']),'elapsed_s':round(time.perf_counter()-start,1)}),flush=True)
summary={'model':str(model),'files':sorted(results,key=lambda x:x['file']),'elapsed_s':time.perf_counter()-start,'bad_tensors':sum(len(r['bad_tensors']) for r in results),'nonfinite_values':sum(t['nonfinite'] for r in results for t in r['bad_tensors'])}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='files'}),flush=True)
