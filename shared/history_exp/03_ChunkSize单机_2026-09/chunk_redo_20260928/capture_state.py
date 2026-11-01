#!/usr/bin/env python3
"""Preserve server and hardware snapshots around one archived benchmark group."""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('directory',type=Path)
ap.add_argument('--chunk',required=True,type=int)
ap.add_argument('--label',required=True)
args=ap.parse_args()
root=args.directory/args.label
root.mkdir(parents=True,exist_ok=False)
(root/'captured_at.json').write_text(json.dumps({'local_ns':time.time_ns(),
    'monotonic_ns':time.perf_counter_ns()},indent=2)+'\n')
for route,name in [('/health','health.txt'),('/get_server_info','server_info.json'),('/metrics','metrics.prom')]:
    try:
        with urllib.request.urlopen('http://127.0.0.1:30080'+route,timeout=15) as res:
            (root/name).write_bytes(res.read())
            (root/(name+'.http_status')).write_text(str(res.status)+'\n')
    except Exception as exc:
        (root/(name+'.error')).write_text(repr(exc)+'\n')
for name,command in [('docker_inspect.json',['sudo','-n','docker','inspect',f'chunk-redo23-{args.chunk}']),
                     ('npu_smi.txt',['npu-smi','info']),
                     ('docker_ps.txt',['sudo','-n','docker','ps','--no-trunc'])]:
    try:
        proc=subprocess.run(command,text=True,capture_output=True,timeout=35)
        (root/name).write_text(proc.stdout)
        (root/(name+'.stderr')).write_text(proc.stderr)
        (root/(name+'.exit_code')).write_text(str(proc.returncode)+'\n')
    except Exception as exc:
        (root/(name+'.error')).write_text(repr(exc)+'\n')
print(root)
