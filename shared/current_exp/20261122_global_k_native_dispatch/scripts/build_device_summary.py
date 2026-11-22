#!/usr/bin/env python3
import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'
rank_data=[json.loads((ART/f'device_timeline_rank{i}.json').read_text()) for i in (0,1)]

def category(name):
 n=name.lower()
 if n.startswith('hcom_') or 'hccl' in n: return 'communication'
 if 'groupedmatmul' in n or 'matmul' in n or 'dynamicquant' in n or 'swiglu' in n: return 'compute'
 if 'send' in n or 'receive' in n: return 'send_recv_control'
 return 'other'

summary={'scope':'torch_npu profiler diagnostic; profiler overhead means values are not formal timings','n':8192,'q':2048,'clock_alignment':'not calibrated across devices; no cross-device overlap percentage computed','ranks':{}}
for r in rank_data:
 rank=r['rank']; modes={}
 for mode in ('original','existing'):
  runs=[x for x in r['runs'] if x['mode']==mode]
  cats={}
  for cat in ('communication','compute','send_recv_control','other'):
   vals=[]; waits=[]
   for x in runs:
    for k in x['trace'].get('kernels',[]):
     if category(k['name'])==cat:
      vals.append(k['duration_us']); waits.append(k['wait_us'])
   cats[cat]={'duration_us_total':sum(vals),'wait_us_total':sum(waits),'kernel_count':len(vals)}
  modes[mode]={'runs':runs,'category_totals':cats}
 summary['ranks'][str(rank)]=modes
(ART/'DEVICE_TIMELINE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
