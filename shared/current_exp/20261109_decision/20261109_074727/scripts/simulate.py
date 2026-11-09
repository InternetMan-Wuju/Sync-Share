"""PRE-NPU exploratory simulation; old 8K fit extrapolations are not evidence."""
import sys,json,gzip,time,statistics
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plugin'))
from decision import features,decide,apply
OLD=ROOT.parent.parent/'20261109_ep16_online/20261109_061333'
rows=[]
with gzip.open(OLD/'artifacts/rank_records.jsonl.gz','rt') as f:
    for line in f:
        x=json.loads(line)
        if x.get('request','').startswith('profile_') and x.get('mode') in ('shadow','active_warm'):
            v=x['new_slots' if x['mode']=='active_warm' else 'old_slots'][x['rank']]
            rows.append((v,x['timing']['compute_device_ms']))
fits=[]
for tile in (1,16,32,64,128,256):
    X=np.array([features(v,tile) for v,y in rows]);Y=np.array([y for v,y in rows])
    coef=nnls(X,Y)[0];err=np.abs(X@coef-Y)
    fits.append({'tile':tile,'coef':coef.tolist(),'mae_ms':float(err.mean()),'max_error_ms':float(err.max())})
fit=min(fits,key=lambda x:x['mae_ms'])
model={**fit,'costs_known':True,'copy_ms':3.787,'control_ms':1.32,'remap_ms':0.27,'planner_budget_ms':0.5,'extra_comm_ms':0.0,'margin_ms':0.5,'max_global_tasks':None,'status':'EXPLORATORY_OLD_8K_FIT_EXTRAPOLATION_NOT_VALIDATED'}
inputs=json.loads((OLD/'INPUTS_PLANS.json').read_text());records=[]
for case in range(3):
    c=inputs[f'validate_cloned_cold_{case}']['counts']
    for scale in (1,2,4,8):
        counts=[n*scale for n in c];start=time.perf_counter_ns();p,d=decide(counts,model)
        records.append({'case':case,'scale':scale,'effective_rows':8192*scale,'plan':p,'detail':d,'cpu_single_ms':(time.perf_counter_ns()-start)/1e6,'slots':apply(counts,p)})
result={'status':'CPU_SIMULATION_ONLY','source':'previous run profile diagnostics, only 8K measured; all larger inputs extrapolated','fits':fits,'provisional_model':model,'cases':records}
(ROOT/'artifacts/SIMULATION_INITIAL.json').write_text(json.dumps(result,indent=2))
for x in records:
    b=x['detail'].get('best_candidate',{});print(x['case'],x['effective_rows'],x['detail']['reason'],round(b.get('predicted_net_ms',0),3),round(x['cpu_single_ms'],3))
