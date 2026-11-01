#!/usr/bin/env python3
"""Request terminal SSE just before each active 45-minute client cutoff.

Intermediate SSE lacks cumulative speculation counters in this fixed image.
Abort only the active experiment request, identified by its archived SSE rid,
so the open client can receive a terminal frame containing final counters.
Any tiny last request submitted in the remaining boundary interval stays in
raw data and is reported separately when it lacks terminal counters.
"""
import concurrent.futures
import json
import time
import urllib.request
import zlib
from pathlib import Path
import sys

root=Path(sys.argv[1])
urls={'off':'http://10.120.72.23:30080','on':'http://10.120.72.25:30080'}


def active(mode,scene):
    dirs=sorted((root/scene/mode).glob('request_*/measurement'),reverse=True)
    for path in dirs:
        if (path/'result.json').exists():continue
        request_path=path/'request.json'
        if request_path.exists():
            payload=json.loads(request_path.read_text()).get('payload',{})
            if payload.get('rid'):
                return {'rid':payload['rid'],'path':str(path),'source':'archived client-assigned request ID'}
        file=path/'sse_raw.jsonl.gz'
        if not file.exists() or file.stat().st_size==0:continue
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        with file.open('rb') as source: prefix=source.read(65536)
        text=decoder.decompress(prefix,4<<20).decode('utf-8','replace')
        for line in text.splitlines():
            try:
                raw=json.loads(line).get('raw_line','')
                if not raw.startswith('data:') or raw[5:].strip()=='[DONE]':continue
                meta=json.loads(raw[5:]).get('meta_info',{})
                if meta.get('id'):return {'rid':meta['id'],'completion_tokens_in_first_persisted_frame':meta.get('completion_tokens'),'path':str(path)}
            except Exception:continue
    return None


def end(mode,scene):
    item=active(mode,scene)
    row={'scene':scene,'mode':mode,'requested_wall_ns':time.time_ns(),'requested_perf_s':time.monotonic(),'active':item}
    if item:
        try:
            payload={'rid':item['rid'],'abort_message':'45-minute experiment boundary; retain final counters'}
            req=urllib.request.Request(urls[mode]+'/abort_request',json.dumps(payload).encode(),{'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=10) as response:row['http_status']=response.status;row['body']=response.read(200).decode('utf-8','replace')
        except Exception as exc:row['error']=repr(exc)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    return row


scenes=tuple(sys.argv[2:]) or ('decode','mixed','prefix')
for scene in scenes:
    schedule=root/scene/'schedule.json'
    while not schedule.exists():time.sleep(1)
    details=json.loads(schedule.read_text())
    while time.monotonic()<details['stop_perf_s']-2:
        time.sleep(min(1,max(.01,details['stop_perf_s']-2-time.monotonic())))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda mode:end(mode,scene),('off','on')))
    (root/scene/'terminal_counter_cutoff.json').write_text(json.dumps({'client_deadline_perf_s':details['stop_perf_s'],'helper_policy':'send exact-rid abort about 2 seconds before client deadline to retain terminal cumulative counters','responses':rows},ensure_ascii=False,indent=2)+'\n')
print('all cutoff actions saved',flush=True)
