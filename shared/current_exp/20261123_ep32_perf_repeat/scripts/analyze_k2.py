#!/usr/bin/env python3
"""Aggregate rank-local single-dispatch logs into task-accounting evidence."""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--logs',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--ep',type=int,required=True)

ap.add_argument('--main-per-rank',type=int,default=8)
ap.add_argument('--owners',default='0,1')
ap.add_argument('--hots',default='163,133')
ap.add_argument('--artifacts',type=Path,default=None)
a=ap.parse_args()
owners=[int(x) for x in a.owners.split(',') if x!='']
hots=[int(x) for x in a.hots.split(',') if x!='']
rows=[]
for p in glob.glob(str(a.logs/'**/rank*.jsonl'),recursive=True):
 for line in Path(p).read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line)
  if r.get('kind')=='call_single_dispatch':rows.append(r)
by=defaultdict(list)
for r in rows:by[(int(r['layer']),int(r['seq']))].append(r)
chunks=[]
for (layer,seq),rs in sorted(by.items()):
 logical=[0]*256
 for r in rs:
  lc=r.get('logical_counts',[])
  for i,v in enumerate(lc[:256]):logical[i]+=int(v)
 moved=sum(int(r.get('local_replica_tasks',0)) for r in rs)
 owner_recv=[sum(int((r.get('replica_slot_counts') or [0])[-1]) for r in rs if int(r.get('rank',-1))==o) for o in owners]
 replica_total=sum(owner_recv)
 hot_counts={str(x):logical[x] for x in hots}
 original_after={str(x):logical[x]-(owner_recv[i] if i<len(owner_recv) else 0) for i,x in enumerate(hots)}
 main_hot_counts={str(x):sum(int((r.get('main_slot_counts') or [])[x%a.main_per_rank]) for r in rs if int(r.get('rank',-1))==x//a.main_per_rank) for x in hots}
 ver=[r.get('verify',{}) for r in rs]
 chunks.append({
  'layer':layer,'seq':seq,'active_ranks':len(rs),'tasks':sum(int(r.get('tasks',0)) for r in rs),
  'moved_tasks':moved,'owner_received_tasks':owner_recv,'replica_tasks_in_logs':replica_total,
  'hot_logical_tasks':hot_counts,'hot_original_after_move':original_after,'hot_main_computed_tasks':main_hot_counts,
  'max_abs':max((float(x.get('max_abs',0)) for x in ver),default=0.0),
  'max_nrmse':max((float(x.get('nrmse',0)) for x in ver),default=0.0),
  'finite_all':all(bool(x.get('finite',False)) for x in ver) if ver else False,
  'dispatch_num_experts':sorted(set(int(x.get('dispatch_num_experts',-1)) for x in rs)),
  'dispatch_num_local_experts':sorted(set(int(x.get('dispatch_num_local_experts',-1)) for x in rs)),
  'execution':'native single DeepEP NORMAL dispatch/combine + W8A8 main/replica chains'
 })
out={'ep':a.ep,'owners':owners,'hots':hots,'call_rows':len(rows),'chunks':chunks,
     'task_conservation':all(x['moved_tasks']==x['replica_tasks_in_logs'] for x in chunks),
     'all_finite':all(x['finite_all'] for x in chunks) if chunks else False,
     'provenance':'single-dispatch rank JSONL: logical counts, sender-side moved counts, owner-side receive counts, and per-rank verify errors'}
if a.artifacts:
 # Optional output identity table; only compares matching request files.
 out['request_artifacts']={}
 for p in sorted(a.artifacts.glob('ep32_*.json')):
  try:
   d=json.loads(p.read_text())
  except Exception:continue
  if 'output_ids_sha256' in d:out['request_artifacts'][p.name]={'input_tokens':d.get('input_tokens'),'output_ids':d.get('output_ids'),'output_ids_sha256':d['output_ids_sha256'],'ttft_s':d.get('ttft_s')}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
