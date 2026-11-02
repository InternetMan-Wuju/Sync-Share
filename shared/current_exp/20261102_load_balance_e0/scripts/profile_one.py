#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time,urllib.request,urllib.error
BASE=os.environ.get('SERVICE_BASE_URL','http://127.0.0.1:30080')
def post(path,payload=None,timeout=900):
 data=json.dumps(payload or {}).encode()
 req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json','Accept':'application/json'},method='POST')
 start=time.perf_counter()
 try:
  with urllib.request.urlopen(req,timeout=timeout) as r:
   raw=r.read(); return {'status':r.status,'body':json.loads(raw) if raw else None,'elapsed_s':time.perf_counter()-start}
 except urllib.error.HTTPError as e:
  return {'status':e.code,'body':e.read().decode('utf-8','replace'),'elapsed_s':time.perf_counter()-start}
 except Exception as e:
  return {'status':0,'body':repr(e),'elapsed_s':time.perf_counter()-start}
p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--output-dir',default='/exp-logs/profiler/e1_prefill'); p.add_argument('--input-len',type=int,default=8192); p.add_argument('--max-new-tokens',type=int,default=1); a=p.parse_args()
os.makedirs(a.output_dir,exist_ok=True)
res={'start_profile':post('/start_profile',{'output_dir':a.output_dir,'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'profile_id':'e1_prefill_'+str(int(time.time())),'merge_profiles':True}),'request':None,'stop_profile':None}
ids=[1000+((9199+i*37)%150000) for i in range(a.input_len)]
res['request']=post('/generate',{'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':a.max_new_tokens,'ignore_eos':True}},timeout=900)
res['stop_profile']=post('/stop_profile',{},timeout=900)
open(a.out,'w').write(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps({'start':res['start_profile']['status'],'request':res['request']['status'],'request_elapsed_s':res['request']['elapsed_s'],'stop':res['stop_profile']['status']}))
