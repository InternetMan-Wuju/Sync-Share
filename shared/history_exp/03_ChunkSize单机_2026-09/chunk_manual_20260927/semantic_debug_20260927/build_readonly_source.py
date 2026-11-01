#!/usr/bin/env python3
"""Copy only differing reference blocks into a workspace source directory."""
import argparse,datetime,hashlib,json,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--reference-host',default='10.120.72.31')
a=p.parse_args();base=Path(a.base);reference=Path(a.reference);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
basestat=json.loads((base/'summary.json').read_text());refstat=json.loads((reference/'summary.json').read_text())
if basestat['file_count']!=182 or refstat['file_count']!=182:raise RuntimeError('incomplete checkpoint snapshot')
if refstat.get('streamed_nonfinite') or refstat.get('streamed_abs_gt_1e10'):raise RuntimeError('reference snapshot has abnormal floating values')
files={};fetch=[];total_blocks=0;bytes_external=0
for path in sorted(base.glob('*.safetensors.json')):
 b=json.loads(path.read_text());r=json.loads((reference/path.name).read_text())
 if b['bytes']!=r['bytes'] or b['header_sha256']!=r['header_sha256']:raise RuntimeError('checkpoint layouts differ: '+b['file'])
 entry={k:b[k] for k in ['file','bytes','mtime_ns','inode','header_bytes','header_sha256','tensors']}
 entry.update({'original_sha256':b['sha256'],'reference_sha256':r['sha256'],'external_blocks':[]})
 if len(b['blocks'])!=len(r['blocks']):raise RuntimeError('block counts differ')
 for orig,ref in zip(b['blocks'],r['blocks']):
  total_blocks+=1
  if orig['offset']!=ref['offset'] or orig['length']!=ref['length']:raise RuntimeError('block layouts differ')
  if orig['sha256']==ref['sha256']:continue
  relative=f"payloads/{b['file']}/{ref['offset']:012d}.bin"
  block={**ref,'payload':relative}
  entry['external_blocks'].append(block);bytes_external+=block['length']
  target=out/relative
  if target.exists() and target.stat().st_size==block['length'] and hashlib.sha256(target.read_bytes()).hexdigest()==block['sha256']:continue
  fetch.append({'file':b['file'],**block,'reference_mtime_ns':r['mtime_ns'],'reference_inode':r['inode']})
 files[b['file']]=entry
print(json.dumps({'files':len(files),'changed_files':sum(bool(f['external_blocks']) for f in files.values()),'external_blocks':sum(len(f['external_blocks']) for f in files.values()),'external_gib':bytes_external/2**30,'fetch_blocks':len(fetch)}),flush=True)
program='''import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin)
root=Path('/mnt/raid/user/data/models/glm-qq')
for block in request:
 path=root/block['file'];stat=path.stat()
 if stat.st_mtime_ns!=block['reference_mtime_ns'] or stat.st_ino!=block['reference_inode']:
  raise RuntimeError('reference file identity changed: '+str(path))
 with path.open('rb') as f:f.seek(block['offset']);data=f.read(block['length'])
 if len(data)!=block['length'] or hashlib.sha256(data).hexdigest()!=block['sha256']:
  raise RuntimeError('reference block changed: '+str(path)+' '+str(block['offset']))
 sys.stdout.buffer.write(data)
 sys.stdout.buffer.flush()
'''
start=time.perf_counter()
if fetch:
 cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','LogLevel=ERROR',a.reference_host,shlex.join(['python3','-c',program])]
 # A regular file avoids stdin pipe backpressure while stdout carries binary data.
 request_path=out/'fetch_request.json';request_path.write_text(json.dumps(fetch))
 with request_path.open('rb') as request, (out/'fetch_stderr.log').open('w') as error:
  proc=subprocess.Popen(cmd,stdin=request,stdout=subprocess.PIPE,stderr=error)
  try:
   for index,block in enumerate(fetch,1):
    remain=block['length'];parts=[]
    while remain:
     part=proc.stdout.read(min(remain,4*1024*1024))
     if not part:raise RuntimeError('reference transfer ended before expected bytes; see fetch_stderr.log')
     parts.append(part);remain-=len(part)
    data=b''.join(parts)
    if hashlib.sha256(data).hexdigest()!=block['sha256']:raise RuntimeError('transferred block digest mismatch')
    target=out/block['payload'];target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.part');temp.write_bytes(data);temp.replace(target)
    if index%20==0 or index==len(fetch):print(json.dumps({'received':index,'total':len(fetch),'elapsed_s':round(time.perf_counter()-start,1)}),flush=True)
   if proc.stdout.read(1):raise RuntimeError('unexpected trailing transfer bytes')
   if proc.wait()!=0:raise RuntimeError('reference transfer failed; see fetch_stderr.log')
  except BaseException:
   proc.kill();proc.wait();raise
manifest={'created_at':datetime.datetime.now().astimezone().isoformat(),'verification':'all_blocks_match_audited_reference','reference_id':'node31_glm-qq_snapshot_'+reference.name,'reference_snapshot_summary':refstat,'original_snapshot_summary':basestat,'total_block_count':total_blocks,'external_block_count':sum(len(f['external_blocks']) for f in files.values()),'external_bytes':bytes_external,'files':files}
(out/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
summary={k:v for k,v in manifest.items() if k!='files'};summary['manifest_sha256']=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'verified':True,'elapsed_s':time.perf_counter()-start,'record':str(out)}),flush=True)
