#!/usr/bin/env python3
"""Analyze the 1000-iteration online EPLB retest against bf94982 baselines."""
from __future__ import annotations
import csv,json,re,statistics
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';RAW=ART/'raw';PREV=ROOT.parent/'20261123_ep32_perf_repeat'
SIZES=('8192','16384','32768');ARMS=('original','eplb','k2');RE_START=re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0 EP0\] \[EPLBManager\] rebalance start');RE_END=re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0 EP0\] \[EPLBManager\] rebalance end time=([0-9.]+)s');RE_LAYER=re.compile(r'^layer=49:\s+(.*)$');RE_EP=re.compile(r'ep\d+=\[([0-9, ]+)\]')
def load(p):return json.loads(Path(p).read_text())
def iso(s):return datetime.fromisoformat(s).astimezone(timezone.utc)
def logtime(s):return datetime.strptime(s,'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
def stats(xs):
 xs=[float(x) for x in xs if x is not None]
 if not xs:return {'n':0}
 mean=statistics.mean(xs);return {'n':len(xs),'median':statistics.median(xs),'mean':mean,'std':statistics.pstdev(xs) if len(xs)>1 else 0.0,'min':min(xs),'max':max(xs)}
def pct(v,b):return None if v is None or not b else 100.0*(v-b)/b
def parse_timeline():
 out={}
 for r in (1,2,3):
  p=ROOT/'logs'/f'r{r}_eplb'/'node23'/'server.log';events=[];opens=[]
  if p.exists():
   for line in p.read_text(errors='replace').splitlines():
    m=RE_START.match(line)
    if m:opens.append({'start':logtime(m.group(1)).isoformat()});continue
    m=RE_END.match(line)
    if m and opens:
     e=opens.pop(0);e.update({'end':logtime(m.group(1)).isoformat(),'duration_s':float(m.group(2))});events.append(e)
  before=[];after=[]
  if p.exists():
   lines=p.read_text(errors='replace').splitlines()
   for i,line in enumerate(lines):
    kind=None
    if 'rebalance layout before:' in line:kind='before'
    elif 'rebalance layout after:' in line:kind='after'
    if kind is None:continue
    for row in lines[i+1:i+120]:
     m=RE_LAYER.match(row)
     if not m:continue
     vals=[]
     for raw in RE_EP.findall(m.group(1)):vals.extend(int(x) for x in raw.split(',') if x.strip())
     if len(vals)==288:(before if kind=='before' else after).append(vals)
     break
  changed=bool(before and after and before[-1]!=after[-1])
  out[str(r)]={'events':events,'layout_before_snapshots':len(before),'layout_after_snapshots':len(after),'layout_changed':changed,'before_layer49_sha256':__import__('hashlib').sha256(json.dumps(before[-1],separators=(',',':')).encode()).hexdigest() if before else None,'last_layer49_sha256':__import__('hashlib').sha256(json.dumps(after[-1],separators=(',',':')).encode()).hexdigest() if after else None}
 return out
def overlap(row,events):
 a=iso(row['started_at']);b=iso(row['finished_at']);hits=[]
 for i,e in enumerate(events):
  if a<=iso(e['end']) and b>=iso(e['start']):hits.append(i)
 return hits
def load_dir(p):
 out=[]
 for x in sorted(list(p.glob('*.json'))+list(p.glob('*.jsonl'))):
  try: rel=str(x.relative_to(ROOT))
  except ValueError: rel=str(x)
  if x.suffix=='.jsonl':
   for line in x.read_text().splitlines():
    if line.strip():out.append(json.loads(line)|{'_file':rel})
  else:out.append(load(x)|{'_file':rel})
 return out
def main():
 timeline=parse_timeline();formal=load_dir(RAW/'formal');warm=load_dir(RAW/'warmup');cal=load_dir(RAW/'calibration')
 for r in formal+warm+cal:
  hits=overlap(r,timeline[str(r['round'])]['events']) if r['arm']=='eplb' else []
  r['rebalance_overlap']=bool(hits);r['overlap_events']=hits
 # Prior original/K2 samples are reused by exact round/repeat/input identity.
 prev_formal=load_dir(PREV/'artifacts/raw/formal')
 prev={(r['arm'],r['size'],r['round'],r['repeat']):r for r in prev_formal}
 by=defaultdict(list)
 for r in formal:by[r['size']].append(r)
 cell={};paired={};outchecks={}
 for s in SIZES:
  rs=by[s];cell[s]={'n':len(rs),'ttft_s':stats([r['ttft_s'] for r in rs]),'total_s':stats([r['total_s'] for r in rs]),'overlap_n':sum(r['rebalance_overlap'] for r in rs),'no_overlap_n':sum(not r['rebalance_overlap'] for r in rs),'overlap_ttft_s':stats([r['ttft_s'] for r in rs if r['rebalance_overlap']]),'no_overlap_ttft_s':stats([r['ttft_s'] for r in rs if not r['rebalance_overlap']]),'failures':sum(bool(r.get('error')) for r in rs)}
  ds=[];ps=[];dts=[];psts=[]
  for r in rs:
   b=prev.get(('original',s,r['round'],r['repeat']))
   if b:
    ds.append(r['ttft_s']-b['ttft_s']);ps.append(pct(r['ttft_s'],b['ttft_s']));dts.append(r['total_s']-b['total_s']);psts.append(pct(r['total_s'],b['total_s']))
  paired[s]={'n':len(ds),'delta_ttft_s':stats(ds),'delta_ttft_pct':stats(ps),'delta_total_s':stats(dts),'delta_total_pct':stats(psts)}
  orig_ids=[prev[('original',s,r['round'],r['repeat'])]['output_ids'] for r in rs if ('original',s,r['round'],r['repeat']) in prev]
  outchecks[s]={'n':len(rs),'same_as_prior_original':bool(rs) and all(r['output_ids']==o for r,o in zip(rs,orig_ids)),'sample_output_ids':rs[0]['output_ids'] if rs else None}
 # Prior table values for direct citation, not remeasured.
 prevsum=load(PREV/'artifacts/PERFORMANCE_SUMMARY.json');prior_cells={k:{'ttft_median':v['ttft_s']['median'],'total_median':v['total_s']['median'],'n':v['n']} for k,v in prevsum['cell_stats'].items() if k.startswith(('original/','k2/'))}
 calibration_per_round={str(r0):sum(1 for r in cal if int(r['round'])==r0) for r0 in (1,2,3)}
 checks={'formal_10_per_size':all(cell[s]['n']==10 for s in SIZES),'calibration_every_round':all(calibration_per_round[str(r0)]>0 for r0 in (1,2,3)),'rebalance_end_every_round':all(len(timeline[str(r0)]['events'])>0 for r0 in (1,2,3)),'layout_changed_every_round':all(timeline[str(r0)]['layout_changed'] for r0 in (1,2,3)),'all_formal_output_matches_prior_original':all(outchecks[s]['same_as_prior_original'] for s in SIZES),'paired_refs_10_per_size':all(paired[s]['n']==10 for s in SIZES),'no_formal_failures':all(cell[s]['failures']==0 for s in SIZES),'warmup_9':len(warm)==9}
 errors=[k for k,v in checks.items() if not v]
 summary={'run_id':'eplb1000_20261123','base_commit':'bf94982d1fcc5cad7aafa4d8b2c9f3f74b02d744','retest_arm':'online EPLB +32','setup':{'rebalance_num_iterations':1000,'redundant_experts':32,'observer':'disabled','prefix_cache':'disabled','concurrency':1,'output_tokens':2,'formal_counts_per_round':[4,3,3],'calibration_input':'agentx_64_cal','input_fingerprints':{'8192':'86240a227b4f194e7f0e3ab5661942b09c0aa5a44cdcc610f503f54d9e3f3d84','16384':'ef1b6259ebc4a272df50b9e905887909e9acab82924231e72cff73c981b217c9','32768':'cd222160c6815f5ce8db984894f40d40880333dfed62139a23b4ffa0464fafe1'}},'cell':cell,'paired_deltas':paired,'output_checks':outchecks,'timeline':timeline,'prior_bf94982_cells':prior_cells,'calibration_request_count':len(cal),'calibration_per_round':calibration_per_round,'warmup':[{'round':r['round'],'size':r['size'],'ttft_s':r['ttft_s'],'total_s':r['total_s']} for r in warm],'checks':checks,'errors':errors,'status':'PASS_EPLB_1000' if not errors else 'FAIL'}
 ART.mkdir(parents=True,exist_ok=True);(ART/'EPLB1000_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (ART/'EPLB1000_RAW_REQUESTS.csv').open('w',newline='') as f:
  fields=['run_phase','arm','round','repeat','size','input_tokens','started_at','finished_at','ttft_s','total_s','output_token_count','output_ids_sha256','rebalance_overlap','overlap_events','error','file'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in formal+warm+cal:w.writerow({**{k:r.get(k) for k in fields if k not in ('output_token_count','overlap_events','file')},'output_token_count':len(r.get('output_ids',[])),'overlap_events':json.dumps(r.get('overlap_events',[])),'file':r.get('_file')})
 lines=['# EPLB +32, 1000-iteration retest','','| arm/source | size | n | TTFT median | total median | Δ TTFT vs original |','|---|---:|---:|---:|---:|---:|']
 for arm in ARMS:
  for s in SIZES:
   key=f'{arm}/{s}';v=prior_cells[key] if key in prior_cells else {'ttft_median':cell[s]['ttft_s']['median'],'total_median':cell[s]['total_s']['median'],'n':cell[s]['n']}
   p=paired[s]['delta_ttft_pct']['median'] if arm=='eplb' else 0.0;src='prior bf94982' if arm!='eplb' else 'this retest'
   lines.append(f'| {arm} ({src}) | {int(s)//1024}K | {v["n"]} | {v["ttft_median"]:.6f} s | {v["total_median"]:.6f} s | {p:.2f}% |')
 lines+=['','EPLB rows are this retest; original and K=2 rows are reused from bf94982 and are not rerun.']
 (ART/'THREE_ARM_TABLE.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':summary['status'],'formal':len(formal),'warmup':len(warm),'calibration':len(cal),'checks':checks,'errors':errors,'overlap':{s:cell[s]['overlap_n'] for s in SIZES}},indent=2));raise SystemExit(0 if not errors else 1)
if __name__=='__main__':main()
