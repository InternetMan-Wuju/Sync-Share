"""One real 8K/32-token request containing isolated layer49 experiments."""
import json,time,urllib.request,argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--split',choices=['train','test'],required=True);p.add_argument('--scales',default='1,2,4,8');p.add_argument('--name',required=True);a=p.parse_args()
root=Path('/runexp');cfg={'mode':'decision_'+a.split,'test_only':True,'case':a.case,'request':a.name,'scales':list(map(int,a.scales.split(','))),'implementation_revision':10}
(root/'local/control.tmp').write_text(json.dumps(cfg));(root/'local/control.tmp').replace(root/'local/control.json')
case=json.loads((root/'local/inputs.json').read_text())['cases'][a.case]
payload={'input_ids':case['input_ids'],'sampling_params':{'temperature':0,'max_new_tokens':32,'ignore_eos':True},'stream':True}
req=urllib.request.Request('http://127.0.0.1:30080/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter();last=None
with urllib.request.urlopen(req,timeout=1500) as resp:
    for line in resp:
        if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':last=json.loads(line[6:])
out={'config':cfg,'elapsed_s':time.perf_counter()-start,'response':last,'scope':'experiment request; NOT E2E performance sample'}
(root/'artifacts'/f'{a.name}_request.json').write_text(json.dumps(out,indent=2));print(json.dumps({'elapsed_s':out['elapsed_s'],'meta':last.get('meta_info') if last else None}),flush=True)
assert last['meta_info']['prompt_tokens']==8192 and last['meta_info']['completion_tokens']==32
