"""Fit on case0 only; freeze before any held-out NPU measurements."""
import sys,json,statistics,collections,datetime,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'plugin'))
from decision import features,rank_ms,decide,admit
rows=[]
for p in sorted((ROOT/'local').glob('rank*.jsonl')):
    rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
dev=[r for r in rows if r.get('split')=='development']
assert dev and all(r.get('case')==0 for r in dev)
data=[r for r in dev if r['kind']=='decision_profile' and r['variant'] in ('off','count')]
def xfeatures(r,tile):return features(r['slots'][r['rank']],tile,max(1,r['scale']//2))
fits=[]
for tile in (1,16,32,64,128,256):
    train=[r for r in data if r['rep']<3];valid=[r for r in data if r['rep']>=3]
    X=np.array([xfeatures(r,tile) for r in train]);Y=np.array([r['timing']['compute_device_ms'] for r in train])
    coef=nnls(X,Y)[0]
    err=np.abs(np.array([xfeatures(r,tile) for r in valid])@coef-np.array([r['timing']['compute_device_ms'] for r in valid]))
    fits.append({'tile':tile,'coef':coef.tolist(),'validation_mae_ms':float(err.mean()),'validation_abs_p90_ms':float(np.quantile(err,.9)),'validation_max_ms':float(err.max())})
fit=min(fits,key=lambda f:f['validation_mae_ms'])
groups=collections.defaultdict(list)
for r in dev:
    if r['kind']=='decision_block':groups[(r['scale'],r['mode'],r['block'])].append(r)
blocks=collections.defaultdict(list)
for (s,m,b),rr in groups.items():
    assert len(rr)==16 and len({r['calls'] for r in rr})==1
    blocks[(s,m)].append(max(r['block_ms'] for r in rr)/rr[0]['calls'])
med={k:statistics.median(v) for k,v in blocks.items()}
copy_pairs={m:statistics.median(v) for (s,m),v in blocks.items() if m.startswith('copy_pair_')}
assert copy_pairs
model={**fit,'costs_known':True,'copy_ms':max(v for (s,m),v in med.items() if m=='copy_only' or m.startswith('copy_pair_')),
       'control_ms':max(v for (s,m),v in med.items() if m=='collect_only'),
       'remap_ms':max(v for (s,m),v in med.items() if m=='remap_only'),
       'planner_budget_ms':0.8,'extra_comm_ms':0.3,'margin_ms':max(.5,2*fit['validation_abs_p90_ms']),
       'max_global_tasks':max(r['global_rows']*8 for r in data),'chunk_rows':16384,
       'known_copy_pairs':[[int(x) for x in m.split('_')[-2:]] for m in copy_pairs],
       'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'training_case':0,'training_scales':sorted({r['scale'] for r in data}),
       'status':'EXPERIMENTAL_CURRENT_BATCH_MODEL_NOT_PRODUCTION','assumptions':['linear nonnegative bank-time proxy','zero communication savings credited','0.3ms extra-communication safety allowance, NOT a calibrated topology model','same per-rank input rows, distinct top8 verified','cold copy on every selected action']}
# Check new-topology work has not yet been performed under the held-out split.
assert not any(r.get('split')=='heldout' for r in rows)
(ROOT/'artifacts/FROZEN_MODEL.json').write_text(json.dumps(model,indent=2))
(ROOT/'artifacts/MODEL_FIT.json').write_text(json.dumps({'fits':fits,'selected':fit,'calibration_blocks':[{'scale':s,'mode':m,'ms_per_call':v} for (s,m),v in blocks.items()],'n_rank_diagnostics':len(data)},indent=2))
inputs=[r for r in dev if r['kind']=='decision_input' and r['rank']==0]
pred=[]
for r in inputs:
    allowed=admit(r['global_rows'],model);p,d=decide(r['counts'],model)
    pred.append({'scale':r['scale'],'admitted':allowed,'plan':p,'detail':d})
(ROOT/'artifacts/DEVELOPMENT_PREDICTIONS.json').write_text(json.dumps(pred,indent=2))
print(json.dumps(model,indent=2))
for r in pred:print('SCALE',r['scale'],'ADMIT',r['admitted'],'PLAN',r['plan'])
