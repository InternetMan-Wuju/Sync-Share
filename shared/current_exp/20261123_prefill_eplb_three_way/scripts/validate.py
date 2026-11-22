#!/usr/bin/env python3
from __future__ import annotations
import glob,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS={'8k':ROOT/'artifacts/replay16_8k_final_8k','16k':ROOT/'artifacts/replay16_16k_final_16k','32k':ROOT/'artifacts/replay16_32k_final_32k'}
errors=[];checks={}
for size,d in RUNS.items():
 rows=[json.loads(Path(p).read_text()) for p in glob.glob(str(d/'replay_rank*.json'))]
 if len(rows)!=16:errors.append(f'{size}:rank_count={len(rows)}')
 for r in rows:
  if len(r.get('records',[]))!=10:errors.append(f'{size}:rank{r["rank"]}:reps')
  for rec in r['records']:
   for arm in ('original','eplb','ours'):
    x=rec['arms'][arm]
    if any(c['tasks']!=c['rows']*8 for c in x['chunks']):errors.append(f'{size}:rank{r["rank"]}:rep{rec["rep"]}:{arm}:task_count')
  if r['rank']==0:
   if r.get('k1_slot_bytes')!=37810176:errors.append(f'{size}:rank0:k1_slot')
  elif r.get('k1_slot_bytes')!=0:errors.append(f'{size}:rank{r["rank"]}:extra_slot')
 for arm in ('eplb','ours'):
  if sum(rows[i]['records'][0]['arms'][arm]['replica_tasks'] for i in range(16))<=0:errors.append(f'{size}:{arm}:no_replica_tasks')
  errs=[e for r in rows for e in r['records'][0]['arms'][arm].get('error_vs_original',[])]
  if any(e['max_abs']!=0 or e['nrmse']!=0 or not e['finite'] for e in errs):errors.append(f'{size}:{arm}:correctness')
 checks[size]={'ranks':len(rows),'reps':10,'rows_per_rank':rows[0]['hidden_rows'],'k1_slot_owner':0,'status':'PASS' if not any(x.startswith(size+':') for x in errors) else 'FAIL'}
layout=json.loads((ROOT/'layouts/eplb16_static.json').read_text())['physical_to_logical_map']
if len(layout)!=78 or len(layout[49])!=272:errors.append('layout_shape')
summary=json.loads((ROOT/'artifacts/RESULTS_SUMMARY.json').read_text())
if summary['blockers']['ours_service']['status']!='FAIL':errors.append('missing_k1_service_blocker')
if summary['blockers']['native_eplb_online_rebalance']['status']!='FAIL':errors.append('missing_online_rebalance_blocker')
out={'status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,'scope':'EP16 routed-MoE replay, 3 sizes, 16 ranks, 10 paired repetitions; service and blocker evidence summarized separately'}
(ROOT/'artifacts/VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if not errors else 1)
