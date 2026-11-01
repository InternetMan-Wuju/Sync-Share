#!/usr/bin/env python3
"""Time-bounded closed-loop AgentX-derived workload; raw SSE and request records retained.

Trace has only hash IDs and counts, not original text nor model token IDs.
"""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--url',default='http://10.120.72.23:18623')
ap.add_argument('--dataset',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--duration',type=int,default=1800)
ap.add_argument('--concurrency',type=int,default=20)
ap.add_argument('--timeout',type=int,default=1200)
ap.add_argument('--input-tokens',type=int,default=16384)
ap.add_argument('--output-tokens',type=int,default=512)
args=ap.parse_args()
assert args.concurrency>0 and args.duration>0 and args.input_tokens%64==0 and 0<args.input_tokens<=16384 and 0<args.output_tokens<=512
source=json.loads(args.dataset.read_text()); rows=source['requests']; assert len(rows)>=128
args.out.mkdir(parents=True,exist_ok=False)
(args.out/'sse').mkdir()
(args.out/'config.json').write_text(json.dumps({'start_at':datetime.datetime.now().astimezone().isoformat(),
  'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),'dataset':str(args.dataset),
  'duration_s':args.duration,'concurrency':args.concurrency,'timeout_s':args.timeout,
  'input_tokens':args.input_tokens,'output_tokens':args.output_tokens,
  'url':args.url,'policy':'closed loop: next trace request sent when a worker finishes; sequence repeated only if all rows consumed; no trace arrival-time replay','sampling':{'temperature':0,'ignore_eos':True,'max_new_tokens':args.output_tokens}},indent=2))
state_lock=threading.Lock(); write_lock=threading.Lock(); cursor=0; started=time.monotonic(); deadline=started+args.duration
record_f=(args.out/'requests.jsonl').open('w',buffering=1)
block_cache={}

def ids_for(row):
    # Seeds are stable across groups, and a shared trace hash denotes the same
    # 64-token prefix block. This is synthetic token materialization, not text.
    arr=[]
    for block in row['block_ids'][:args.input_tokens//64]:
        key=(row['trace_id'],str(block))
        if key not in block_cache:
            digest=hashlib.sha256((str(key[0])+'\0'+key[1]).encode()).digest()
            rng=random.Random(int.from_bytes(digest,'big'))
            block_cache[key]=[rng.randrange(1,154880) for _ in range(64)]
        arr.extend(block_cache[key])
    assert len(arr)==args.input_tokens
    return arr

def do_request(serial,row):
    input_ids=ids_for(row)
    record={'serial':serial,'sequence_index':row['sequence_index'],'trace_id':row['trace_id'],
      'trace_request_index':row['trace_request_index'],'source_t':row['t'],
      'input_sha256':hashlib.sha256(json.dumps(input_ids,separators=(',',':')).encode()).hexdigest(),
      'sent_at':datetime.datetime.now().astimezone().isoformat(),'sent_rel_s':time.monotonic()-started}
    payload={'input_ids':input_ids,'sampling_params':{'max_new_tokens':args.output_tokens,'temperature':0,'ignore_eos':True},'stream':True}
    req=urllib.request.Request(args.url+'/generate',data=json.dumps(payload,separators=(',',':')).encode(),
        headers={'Content-Type':'application/json'},method='POST')
    first_token=None; last_token=None; last_meta={}; first_event=None; event_count=0; final_ids=[]; text=''; finish_reason=None
    raw=args.out/'sse'/f'{serial:06d}.jsonl.gz'
    try:
        with gzip.open(raw,'wt') as stream:
            with urllib.request.urlopen(req,timeout=args.timeout) as response:
                record['http_status']=response.status
                for line in response:
                    if not line.startswith(b'data:'):continue
                    now=time.monotonic(); body=line[5:].strip()
                    stream.write(json.dumps({'rel_s':now-started,'data':body.decode('utf8',errors='replace')},ensure_ascii=False)+'\n')
                    event_count+=1
                    if first_event is None:first_event=now
                    if body==b'[DONE]':continue
                    obj=json.loads(body)
                    out=obj.get('output_ids') or []
                    if out:
                        if first_token is None:first_token=now
                        last_token=now
                        final_ids=out if len(out)>=len(final_ids) else final_ids+out
                    if obj.get('text'):text=obj['text']
                    meta=obj.get('meta_info') or {}
                    if meta:last_meta=meta
                    finish_reason=obj.get('finish_reason') or finish_reason
        record['finished_rel_s']=time.monotonic()-started
        record['ttft_s']=first_token-(started+record['sent_rel_s']) if first_token else None
        record['e2e_s']=record['finished_rel_s']-record['sent_rel_s']
        record['output_tokens']=last_meta.get('completion_tokens',len(final_ids))
        record['prompt_tokens']=last_meta.get('prompt_tokens')
        record['cached_tokens']=last_meta.get('cached_tokens')
        record['tpot_s']=(record['e2e_s']-record['ttft_s'])/(record['output_tokens']-1) if first_token and record['output_tokens']>1 else None
        record['last_token_rel_s']=last_token-started if last_token else None
        record['first_event_rel_s']=first_event-started if first_event else None
        record['event_count']=event_count
        record['finish_reason']=finish_reason
        record['text_tail']=text[-300:] if text else ''
        record['output_ids_tail']=final_ids[-24:]
        record['meta_info']=last_meta
        record['status']='ok' if first_token and record['output_tokens']==args.output_tokens else 'incomplete'
    except Exception as exc:
        record.update(status='error',error=repr(exc),finished_rel_s=time.monotonic()-started,event_count=event_count)
    record['sse_file']=str(raw.relative_to(args.out))
    return record

def worker():
    global cursor
    while True:
        with state_lock:
            if time.monotonic()>=deadline: return
            serial=cursor; cursor+=1; row=rows[serial%len(rows)]
        record=do_request(serial,row)
        with write_lock:
            record_f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
            if serial%25==0 or record['status']!='ok':
                print(f"{datetime.datetime.now().isoformat()} serial={serial} status={record['status']} output={record.get('output_tokens')} ttft={record.get('ttft_s')} elapsed={time.monotonic()-started:.1f}",flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    futures=[pool.submit(worker) for _ in range(args.concurrency)]
    for f in concurrent.futures.as_completed(futures):f.result()
record_f.close()
records=[json.loads(line) for line in (args.out/'requests.jsonl').read_text().splitlines()]
ok=[r for r in records if r['status']=='ok']
def measure(name):
    a=sorted(r[name] for r in ok if isinstance(r.get(name),(int,float)))
    return {'mean':statistics.mean(a),'p50':a[int((len(a)-1)*.5)],'p95':a[int((len(a)-1)*.95)],'p99':a[int((len(a)-1)*.99)]} if a else None
wall=time.monotonic()-started
summary={'at':datetime.datetime.now().astimezone().isoformat(),'requested_run_duration_s':args.duration,'input_tokens':args.input_tokens,'output_tokens_target':args.output_tokens,
  'actual_wall_s':wall,'requests_sent':len(records),'completed':len(ok),'incomplete':sum(r['status']=='incomplete' for r in records),
  'errors':sum(r['status']=='error' for r in records),'output_tokens':sum(r['output_tokens'] for r in ok),
  'output_tokens_per_s':sum(r['output_tokens'] for r in ok)/wall,
  'ttft_s':measure('ttft_s'),'tpot_s':measure('tpot_s'),'e2e_s':measure('e2e_s')}
(args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if summary["completed"]==0 or summary["errors"] or summary["incomplete"]:
    sys.exit(2)
