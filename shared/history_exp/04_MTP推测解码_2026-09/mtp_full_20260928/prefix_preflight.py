#!/usr/bin/env python3
"""Check restored services and exact 32K total / ~60% cached inputs before timing."""
import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import bench

root=Path(sys.argv[1]);base=Path(__file__).resolve().parent
folder=root/'prefix_recovery_preflight';folder.mkdir(exist_ok=True)
dataset=bench.data(root)

def run(mode):
    url=bench.URLS[mode];dest=folder/mode;dest.mkdir(exist_ok=True)
    while True:
        try:
            with urllib.request.urlopen(url+'/get_server_info',timeout=3) as r:info=json.loads(r.read())
            bench.save(dest/'server_info.json',info);break
        except Exception:time.sleep(2)
    check=subprocess.run([sys.executable,str(base/'chat_once.py'),'请用三个简短句子解释推测解码是什么，并说明接受长度的含义。',
        '--url',url,'--max-tokens','160','--out-root',str(dest/'semantic')],capture_output=True,text=True,timeout=180)
    (dest/'semantic_stdout.txt').write_text(check.stdout)
    (dest/'semantic_stderr.txt').write_text(check.stderr)
    if check.returncode:raise RuntimeError('semantic request failed '+mode)
    bench.flush(url,dest/'flush_cache.json')
    ids=dataset['scenarios']['prefix']['cases'][0]['input_ids']
    warm=bench.request(url,ids[:19661],1,dest/'warmup',role='prefix_recovery_warmup')
    if warm['error'] or warm['actual_output']!=1:raise RuntimeError('warmup failed '+mode)
    measured=bench.request(url,ids,20,dest/'measurement',role='prefix_recovery_probe')
    valid=(measured['http_status']==200 and not measured['error'] and measured['actual_output']==20
           and measured['prompt_tokens']==32768 and .585<=measured['cached_tokens']/32768<=.615
           and (mode=='off' or measured['meta_info_last'].get('spec_verify_ct',0)>0))
    bench.save(dest/'verdict.json',{'passed':valid,'ratio':measured['cached_tokens']/32768,'result':measured})
    if not valid:raise RuntimeError('probe failed '+mode)
    return mode,True

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    verdict=dict(pool.map(run,['off','on']))
bench.save(folder/'verdict.json',{'passed':all(verdict.values()),'modes':verdict,'purpose':'not a formal45min sample'})
print('PREFIX RECOVERY PREFLIGHT PASSED',flush=True)
