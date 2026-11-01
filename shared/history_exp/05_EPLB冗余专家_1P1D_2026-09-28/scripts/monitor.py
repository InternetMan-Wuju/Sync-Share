#!/usr/bin/env python3
"""Snapshot every exposed PD endpoint and keep responses, including errors."""
import argparse,datetime,json,time,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--interval',type=int,default=15);a=p.parse_args()
urls={'prefill23':'http://10.120.72.23:18423','decode25':'http://10.120.72.25:18423','router':'http://10.120.72.23:18723'}
a.out.mkdir(parents=True,exist_ok=True)
def call(name,endpoint,method='GET',tag='metrics'):
    now=datetime.datetime.now().astimezone().isoformat()
    path=a.out/f'{tag}_{name}_{now.replace(":", "-")}.txt'
    try:
        req=urllib.request.Request(urls[name]+endpoint,method=method)
        with urllib.request.urlopen(req,timeout=8) as resp:
            body=resp.read(30_000_000);status=resp.status
        path.write_bytes(body)
        return {'at':now,'name':name,'endpoint':endpoint,'status':status,'path':str(path),'size':len(body)}
    except Exception as e:
        return {'at':now,'name':name,'endpoint':endpoint,'error':repr(e)}
with (a.out/'events.jsonl').open('w',buffering=1) as stream:
    def emit(result):stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    for name in urls:
        if name!='router':emit(call(name,'/start_expert_distribution_record','POST','start'))
    while not a.stop_file.exists():
        t=time.monotonic()
        for name in urls:emit(call(name,'/metrics'))
        time.sleep(max(1,a.interval-(time.monotonic()-t)))
    for name in urls:
        if name!='router':
            emit(call(name,'/stop_expert_distribution_record','POST','stop'))
            emit(call(name,'/dump_expert_distribution_record','POST','dump'))
