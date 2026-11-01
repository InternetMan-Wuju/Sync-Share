#!/usr/bin/env python3
"""Read-only single-request diagnostic; saves complete request/response."""
import argparse, datetime, json, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--host', default='23')
p.add_argument('--api', choices=['chat','generate'], default='chat')
p.add_argument('--max-tokens', type=int, default=16)
p.add_argument('--prompt', default='请用中文回答：1加1等于几？只回复一句话。')
a=p.parse_args()
base=Path(__file__).resolve().parent/'probes'
base.mkdir(exist_ok=True)
out=base/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_node{a.host}_{a.api}')
out.mkdir()
if a.api=='chat':
 path='/v1/chat/completions'
 body={'model':'glm-5.2','messages':[{'role':'user','content':a.prompt}], 'temperature':0,'max_tokens':a.max_tokens,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
else:
 path='/generate'
 body={'text':'[gMASK]<sop><|user|>'+a.prompt+'<|assistant|><think></think>', 'sampling_params':{'temperature':0,'max_new_tokens':a.max_tokens}, 'stream':False, 'return_logprob':True, 'top_logprobs_num':5}
url=f'http://10.120.72.{a.host}:30080'+path
(out/'request.json').write_text(json.dumps({'url':url,'body':body},ensure_ascii=False,indent=2)+'\n')
start=time.perf_counter()
try:
 req=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:
  status=r.status
  raw=r.read().decode()
 data=json.loads(raw)
 (out/'response.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 summary={'status':status,'e2e_s':time.perf_counter()-start,'text':data['choices'][0]['message'] if a.api=='chat' else data.get('text'),'record':str(out)}
except Exception as e:
 summary={'error':str(e),'e2e_s':time.perf_counter()-start,'record':str(out)}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
