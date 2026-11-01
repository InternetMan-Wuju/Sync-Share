#!/usr/bin/env python3
"""Archive final service state after all formal measurements finish."""
import concurrent.futures
import datetime
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

root=Path(sys.argv[1]);dest=root/'final_state';dest.mkdir(exist_ok=True)
for scene in ('decode','mixed','prefix'):
    if not (root/scene/'summaries.json').exists():raise SystemExit('Refuse final collection: incomplete '+scene)


def capture(node,mode):
    folder=dest/f'node{node}_{mode}';folder.mkdir(exist_ok=True)
    url=f'http://10.120.72.{node}:30080'
    rows={}
    for name in ('get_server_info','metrics','health'):
        try:
            with urllib.request.urlopen(url+'/'+name,timeout=45) as r:
                body=r.read();rows[name]={'http_status':r.status,'bytes':len(body)}
                (folder/(name+('.json' if name=='get_server_info' else '.txt'))).write_bytes(body)
        except Exception as exc:rows[name]={'error':repr(exc)}
    cmds={'npu_smi':['npu-smi','info'],
          'docker_inspect':['sudo','-n','docker','inspect',f'mtp-full{node}-{mode}'],
          'docker_logs':['sudo','-n','docker','logs','--timestamps',f'mtp-full{node}-{mode}']}
    for key,cmd in cmds.items():
        if node==25:
            # All arguments are fixed and contain no user text.
            cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','LogLevel=ERROR',
                 '10.120.72.25',' '.join(cmd)]
        try:
            result=subprocess.run(cmd,capture_output=True,timeout=60)
            (folder/(key+'.stdout')).write_bytes(result.stdout)
            (folder/(key+'.stderr')).write_bytes(result.stderr)
            rows[key]={'returncode':result.returncode,'stdout_bytes':len(result.stdout),'stderr_bytes':len(result.stderr)}
        except Exception as exc:rows[key]={'error':repr(exc)}
    (folder/'capture.json').write_text(json.dumps({'captured_at_local':datetime.datetime.now().astimezone().isoformat(),
          'endpoint':url,'records':rows},ensure_ascii=False,indent=2)+'\n')
    return node,rows

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    print(json.dumps(dict(pool.map(lambda item:capture(*item),[(23,'off'),(25,'on')])),ensure_ascii=False))
