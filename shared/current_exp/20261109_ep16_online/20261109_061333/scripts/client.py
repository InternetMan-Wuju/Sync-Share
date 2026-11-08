"""Sequential fixed-input requests, exact 8192 -> 32, streamed TTFT."""
import argparse,json,time,hashlib,urllib.request
from pathlib import Path
ROOT=Path('/runexp')
p=argparse.ArgumentParser();p.add_argument('--mode',default='off');p.add_argument('--case',type=int,default=0);p.add_argument('--profile',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--name',required=True);p.add_argument('--revision',type=int,default=2);p.add_argument('--force-quota',type=int);args=p.parse_args()
verify=args.verify or (args.name.startswith('warmup_') and args.mode=='active_cold')
cfg={'mode':args.mode,'profile':args.profile,'verify':verify,'request':args.name,'test_only':True,'implementation_revision':args.revision,'expected_global_rows':8192}
if args.force_quota is not None:cfg['force_quota']=args.force_quota
(ROOT/'local/control.tmp').write_text(json.dumps(cfg));(ROOT/'local/control.tmp').replace(ROOT/'local/control.json')
case=json.loads((ROOT/'local/inputs.json').read_text())['cases'][args.case]
payload={'input_ids':case['input_ids'],'sampling_params':{'temperature':0,'max_new_tokens':32,'ignore_eos':True},'stream':True,'return_logprob':True,'logprob_start_len':-1}
req=urllib.request.Request('http://127.0.0.1:30080/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
s=time.perf_counter();first=None;events=[];last=None
with urllib.request.urlopen(req,timeout=900) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        raw=line[6:].strip()
        if raw==b'[DONE]':continue
        body=json.loads(raw);t=time.perf_counter()-s
        if first is None and body.get('meta_info',{}).get('completion_tokens',0)>0:first=t
        events.append(t);last=body
elapsed=time.perf_counter()-s
meta=last['meta_info'];ids=[x[1] for x in meta.get('output_token_logprobs',[])]
result={'request':args.name,'case_id':args.case,**cfg,'input_sha256':case['input_sha256'],'client_ttft_ms':first*1000 if first else None,'client_e2e_ms':elapsed*1000,'client_decode_ms':(elapsed-first)*1000 if first else None,'client_tpot_ms':(elapsed-first)*1000/31 if first else None,'meta':meta,'output_ids':ids,'output_text_sha256':hashlib.sha256(last.get('text','').encode()).hexdigest(),'stream_events':len(events)}
(ROOT/'artifacts'/f'{args.name}.json').write_text(json.dumps(result,indent=2))
assert meta['prompt_tokens']==8192 and meta['completion_tokens']==32,meta
assert meta.get('cached_tokens',0)==0,meta
print(json.dumps({k:v for k,v in result.items() if k not in ('meta','output_ids')},indent=2))
