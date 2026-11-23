#!/usr/bin/env python3
import argparse,json,time,urllib.request,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18023');p.add_argument('--prompt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-tokens',type=int,default=32);a=p.parse_args();text=a.prompt.read_text()
payload={'model':'glm-5.2','messages':[{'role':'user','content':text}],'temperature':0.0,'max_tokens':a.max_tokens,'stream':True}
req=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'},method='POST')
t0=time.perf_counter();pieces=[];first=None;finish=None;meta=None
with urllib.request.urlopen(req,timeout=1800) as r:
 for raw in r:
  if not raw.startswith(b'data:'):continue
  body=raw[5:].strip()
  if body==b'[DONE]':break
  o=json.loads(body);meta=o.get('meta_info') or meta
  for ch in o.get('choices',[]):
   finish=ch.get('finish_reason') or finish;c=(ch.get('delta') or {}).get('content')
   if c:
    if first is None:first=time.perf_counter()
    pieces.append(c)
end=time.perf_counter();out=''.join(pieces)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'prompt':str(a.prompt),'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens_hint':json.loads(a.prompt.with_suffix('.meta.json').read_text())['actual_tokens'] if a.prompt.with_suffix('.meta.json').exists() else None,'ttft_s':None if first is None else first-t0,'total_s':end-t0,'output_text':out,'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'output_chars':len(out),'finish_reason':finish,'meta_info':meta},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'ok','ttft_s':None if first is None else first-t0,'total_s':end-t0,'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'chars':len(out)},indent=2))
