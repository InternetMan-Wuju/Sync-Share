#!/usr/bin/env python3
"""Stop only the active owned service after a confirmed scheduler crash.

Retains its container and logs; the client can finish saving failed streams.
No restart, no resource use outside node23, no attempt to interrupt loading.
"""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
seen=set()
while True:
 try:
  s=json.loads((ROOT/'overnight_status.json').read_text())
  if s.get('completed') or s['phase']=='needs_conservative_review':break
  name=s.get('container','')
  if not name.startswith('cyt-auto-') or name in seen:time.sleep(10);continue
  log=ROOT/'logs'/name/'server.log'
  if log.exists():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-262144));tail=f.read()
   crashed=b'crashed with exit code' in tail or b'scheduler died during initialization' in tail
   if crashed and s['phase']=='command_running':
    p=subprocess.run(['sudo','-n','docker','inspect',name],capture_output=True,text=True,timeout=20)
    if p.returncode==0:
     info=json.loads(p.stdout)[0]
     if info['State']['Running'] and info['Config']['Labels'].get('owner')=='chengyetai':
      ev={'unix':time.time(),'container':name,'reason':'confirmed scheduler crash while client active'}
      with (ROOT/'logs/scheduler_watch.jsonl').open('a') as f:f.write(json.dumps(ev)+'\n')
      print(ev,flush=True)
      subprocess.run(['python3','-B',str(ROOT/'scripts/service.py'),'stop','--name',name],check=True,timeout=70)
      seen.add(name)
 except Exception as e:print('watch error',repr(e),flush=True)
 time.sleep(10)
