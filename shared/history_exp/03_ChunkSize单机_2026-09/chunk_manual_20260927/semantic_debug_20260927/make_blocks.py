#!/usr/bin/env python3
"""Create read-only byte-block manifests for an existing checkpoint."""
import argparse,hashlib,json,struct,time
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--audit',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--accept-new-snapshot',action='store_true');p.add_argument('--files',nargs='*')
a=p.parse_args();model=Path(a.model);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);audit=Path(a.audit)
block_size=32*1024*1024;start=time.perf_counter()
def check(path):
 stat=path.stat();previous=json.loads((audit/(path.name+'.json')).read_text())
 sha=hashlib.sha256();blocks=[];bad_values=0;extreme_values=0
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));f.seek(0);offset=0
  floats=sorted([(8+n+t['data_offsets'][0],8+n+t['data_offsets'][1],t['dtype']) for name,t in header.items() if name!='__metadata__' and t['dtype'] in ('BF16','F16','F32','F64')]);cursor=0
  while True:
   data=f.read(block_size)
   if not data:break
   sha.update(data);blocks.append({'offset':offset,'length':len(data),'sha256':hashlib.sha256(data).hexdigest()})
   while cursor<len(floats) and floats[cursor][1]<=offset:cursor+=1
   j=cursor
   while j<len(floats) and floats[j][0]<offset+len(data):
    begin,end,dtype=floats[j];left=max(begin,offset)-offset;right=min(end,offset+len(data))-offset
    dt={'BF16':'<u2','F16':'<f2','F32':'<f4','F64':'<f8'}[dtype]
    vals=np.frombuffer(data,dtype=dt,offset=left,count=(right-left)//np.dtype(dt).itemsize)
    if dtype=='BF16':vals=(vals.astype(np.uint32)<<16).view(np.float32)
    finite=np.isfinite(vals);bad_values+=int(np.count_nonzero(~finite));extreme_values+=int(np.count_nonzero(np.abs(vals[finite])>1e10));j+=1
   offset+=len(data)
 if sha.hexdigest()!=previous['sha256'] and not a.accept_new_snapshot:raise RuntimeError('checkpoint changed since audit: '+str(path))
 end=path.stat()
 if stat.st_mtime_ns!=end.st_mtime_ns:raise RuntimeError('file changed during scan')
 result={'file':path.name,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino,'sha256':sha.hexdigest(),'header_bytes':8+n,'header_sha256':hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),'tensors':header,'blocks':blocks,'previous_audit_sha256':previous['sha256'],'audit_digest_drift':sha.hexdigest()!=previous['sha256'],'streamed_nonfinite':bad_values,'streamed_abs_gt_1e10':extreme_values}
 (out/(path.name+'.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
 return result
results=[];paths=[model/name for name in a.files] if a.files else sorted(model.glob('*.safetensors'))
with ThreadPoolExecutor(max_workers=a.workers) as pool:
 for future in as_completed([pool.submit(check,p) for p in paths]):
  result=future.result();results.append(result)
  if len(results)%10==0 or len(results)==len(paths):print(json.dumps({'files':len(results),'total':len(paths),'elapsed_s':round(time.perf_counter()-start,1)}),flush=True)
summary={'model':str(model),'block_size':block_size,'file_count':len(results),'elapsed_s':time.perf_counter()-start,'digest_drift_files':sum(r['audit_digest_drift'] for r in results),'streamed_nonfinite':sum(r['streamed_nonfinite'] for r in results),'streamed_abs_gt_1e10':sum(r['streamed_abs_gt_1e10'] for r in results)}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
