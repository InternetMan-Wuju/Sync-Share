#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RE_LAYER=re.compile(r'^layer=49:\s+(.*)$')
RE_EP=re.compile(r'ep\d+=\[([0-9, ]+)\]')
def parse(p):
 lines=p.read_text(errors='replace').splitlines();maps=[];starts=[];ends=[]
 for i,line in enumerate(lines):
  if '[EPLBManager] rebalance start' in line:starts.append(line)
  if '[EPLBManager] rebalance end' in line:ends.append(line)
  if 'rebalance layout after:' not in line:continue
  for row in lines[i+1:i+120]:
   m=RE_LAYER.match(row)
   if not m:continue
   vals=[]
   for raw in RE_EP.findall(m.group(1)):vals.extend(int(x) for x in raw.split(',') if x.strip())
   if len(vals)==288:maps.append(vals)
   break
 return starts,ends,maps
def sha(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
rounds={}
for r in (1,2,3):
 p=ROOT/'logs'/f'r{r}_eplb'/'node23'/'server.log';starts,ends,maps=parse(p)
 rounds[str(r)]={'log':str(p.relative_to(ROOT)),'rebalance_start_rank_events':len(starts),'rebalance_end_rank_events':len(ends),'layout_snapshots':len(maps),'layout_changed':bool(maps and maps[0]!=maps[-1]),'first_layer49_sha256':sha(maps[0]) if maps else None,'last_layer49_sha256':sha(maps[-1]) if maps else None,'last_layer49_physical_to_logical':maps[-1] if maps else None}
patches={}
for p in sorted((ROOT/'patches').glob('*.py')):patches[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
out={'rounds':rounds,'patches':patches,'observer_in_formal':'disabled; actual task accounting reused from verbose validation and previous route evidence','status':'PASS' if all(v['rebalance_end_rank_events']>0 and v['layout_changed'] for v in rounds.values()) else 'FAIL'}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='last_layer49_physical_to_logical'} for k,v in rounds.items()},indent=2))
