#!/usr/bin/env python3
import argparse,json,time,urllib.request,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18023');p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-new-tokens',type=int,default=2);a=p.parse_args();d=json.loads(a.input.read_text());ids=d['input_ids']
payload={'input_ids':ids,'sampling_params':{'max_new_tokens':a.max_new_tokens,'temperature':0,'ignore_eos':True},'stream':True}
req=urllib.request.Request(a.url+'/generate',data=json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'},method='POST')
t0=time.perf_counter();first=None;out_ids=[];text='';meta=None;events=0
with urllib.request.urlopen(req,timeout=1800) as r:
 for raw in r:
  if not raw.startswith(b'data:'):continue
  body=raw[5:].strip()
  if body==b'[DONE]':break
  o=json.loads(body);events+=1;meta=o.get('meta_info') or meta
  x=o.get('output_ids') or []
  if x:
   if first is None:first=time.perf_counter()
   out_ids.extend(x)
  if o.get('text'):text=o['text']
end=time.perf_counter();a.out.parent.mkdir(parents=True,exist_ok=True)
rec={'input_file':str(a.input),'input_tokens':len(ids),'input_sha256':d['input_sha256'],'max_new_tokens':a.max_new_tokens,'ttft_s':None if first is None else first-t0,'total_s':end-t0,'output_ids':out_ids,'output_ids_sha256':hashlib.sha256(json.dumps(out_ids,separators=(',',':')).encode()).hexdigest(),'text':text,'meta_info':meta,'events':events}
a.out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:rec[k] for k in ('input_tokens','ttft_s','total_s','output_ids_sha256','events')},indent=2))
