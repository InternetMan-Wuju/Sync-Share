#!/usr/bin/env python3
import argparse, json, time, urllib.request, hashlib
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:18000')
p.add_argument('--prompt',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--warmups',type=int,default=1)
p.add_argument('--reps',type=int,default=3)
p.add_argument('--max-tokens',type=int,default=32)
a=p.parse_args()
text=a.prompt.read_text()

def one():
    payload={'model':'glm-5.2','messages':[{'role':'user','content':text}], 'temperature':0.0, 'max_tokens':a.max_tokens, 'stream':True}
    req=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'},method='POST')
    t0=time.perf_counter(); first_event=None; first_content=None; pieces=[]; usage=None; finish=None
    with urllib.request.urlopen(req,timeout=1800) as resp:
        for raw in resp:
            if not raw.startswith(b'data:'): continue
            body=raw[5:].strip()
            if body==b'[DONE]': break
            try: obj=json.loads(body)
            except Exception: continue
            if first_event is None: first_event=time.perf_counter()
            if obj.get('usage'): usage=obj['usage']
            for ch in obj.get('choices',[]):
                finish=ch.get('finish_reason') or finish
                content=(ch.get('delta') or {}).get('content')
                if content:
                    if first_content is None: first_content=time.perf_counter()
                    pieces.append(content)
    end=time.perf_counter()
    out_text=''.join(pieces)
    return {'ttft_s':None if first_content is None else first_content-t0,
            'ttfb_s':None if first_event is None else first_event-t0,
            'total_s':end-t0,
            'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'input_chars':len(text), 'output_chars':len(out_text),
            'output_tail':out_text[-500:], 'usage':usage, 'finish_reason':finish}
rows=[]
for i in range(a.warmups):
    r=one(); r.update(phase='warmup', rep=i); print(json.dumps(r,ensure_ascii=False),flush=True)
for i in range(a.reps):
    r=one(); r.update(phase='measured', rep=i); rows.append(r); print(json.dumps(r,ensure_ascii=False),flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'url':a.url,'prompt':str(a.prompt),'prompt_tokens_hint':json.loads(a.prompt.with_suffix('.meta.json').read_text())['actual_tokens'] if a.prompt.with_suffix('.meta.json').exists() else None,'rows':rows},ensure_ascii=False,indent=2)+'\n')
