#!/usr/bin/env python3
"""Paired, 45-minute-per-scenario MTP ON/OFF experiment; archive original SSE.

Each host has an independent single-request queue. A scenario measures both
hosts concurrently for 2700 seconds. At its deadline, mark a still generating
request partial and close its stream; never claim partial 32K output completed.
"""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import random
import re
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path

URLS={'off':'http://10.120.72.23:30080','on':'http://10.120.72.25:30080'}
SCENARIOS={'decode':(1024,32768,0),'mixed':(8192,1024,0),'prefix':(32768,300,19661)}
COUNT={'decode':4,'mixed':64,'prefix':80}
SEED=2026092802
VOCAB=154880


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def digest(obj):
    return hashlib.sha256(json.dumps(obj,separators=(',',':')).encode()).hexdigest()


def data(root):
    path=root/'dataset.json'
    if path.exists():return json.loads(path.read_text())
    rng=random.Random(SEED)
    prefix=[rng.randrange(1,VOCAB) for _ in range(19661)]
    out={'seed':SEED,'vocab_id_range':[1,VOCAB-1],'shared_prefix_ids':prefix,
        'shared_prefix_sha256':digest(prefix),'scenarios':{}}
    for name,(inp,num,pref) in SCENARIOS.items():
        cases=[]
        for i in range(COUNT[name]):
            ids=(prefix if pref else [])+[rng.randrange(1,VOCAB) for _ in range(inp-pref)]
            cases.append({'index':i,'input_ids':ids,'input_sha256':digest(ids)})
        out['scenarios'][name]={'input_tokens':inp,'target_output_tokens':num,
                               'shared_prefix_tokens':pref,'cases':cases}
    save(path,out)
    return out


def request(url,ids,output,directory,deadline=None,role='measurement'):
    directory.mkdir(parents=True,exist_ok=True)
    payload={'rid':uuid.uuid4().hex,'input_ids':ids,'sampling_params':{'max_new_tokens':output,'temperature':0,'ignore_eos':True},'stream':True}
    save(directory/'request.json',{'endpoint':url+'/generate','role':role,'input_sha256':digest(ids),'payload':payload})
    req=urllib.request.Request(url+'/generate',json.dumps(payload).encode(),{'Content-Type':'application/json'},method='POST')
    start=time.perf_counter_ns(); wall=time.time_ns()
    first=None;last=None;count=0;meta={};last_ids=[];end_reason=None;error=None;http=None;events=0;done=0;partial=False
    response=None
    try:
        with urllib.request.urlopen(req,timeout=60) as response, gzip.open(directory/'sse_raw.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as f:
            http=response.status
            for raw in response:
                now=time.perf_counter_ns(); wall_now=time.time_ns()
                text=raw.decode('utf-8','replace')
                f.write(json.dumps({'recv_perf_ns':now,'recv_wall_ns':wall_now,'raw_line':text},ensure_ascii=False)+'\n')
                if events==0 and raw.startswith(b'data:'):
                    f.flush()
                    f.buffer.fileobj.flush()
                if raw.startswith(b'data:'):
                    if raw[5:].strip()==b'[DONE]':
                        done+=1
                        break
                    obj=json.loads(raw[5:]);events+=1
                    if obj.get('error'):raise RuntimeError('server error: '+repr(obj['error']))
                    server=obj.get('meta_info') or {}
                    meta.update(server)
                    ids_event=obj.get('output_ids')
                    if isinstance(ids_event,list):last_ids=ids_event
                    c=server.get('completion_tokens')
                    if isinstance(c,int):count=max(count,c)
                    elif isinstance(ids_event,list):count=max(count,len(ids_event))
                    if first is None and count>0:first=now
                    if count>0:last=now
                    if server.get('finish_reason'):end_reason=server['finish_reason']
                if deadline is not None and time.monotonic()>=deadline and done==0:
                    partial=True
                    break
    except Exception as e:
        error=traceback.format_exc()
        if deadline is not None and time.monotonic()>=deadline and count>0:
            partial=True
        else:
            print(f'ERROR {role}: {e!r}',flush=True)
    abort_result=None
    if partial and meta.get('id'):
        try:
            abort_req=urllib.request.Request(url+'/abort_request',
                json.dumps({'rid':meta['id']}).encode(),{'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(abort_req,timeout=15) as aborted:
                abort_result={'http_status':aborted.status,'rid':meta['id'],'body':aborted.read(1000).decode('utf-8','replace')}
        except Exception as exc:abort_result={'rid':meta['id'],'error':repr(exc)}
        save(directory/'abort_after_deadline.json',abort_result)
    end=time.perf_counter_ns();end_wall=time.time_ns()
    result={'role':role,'http_status':http,'error':error,'partial_at_45m_deadline':partial,
            'done_events':done,'events':events,'abort_after_deadline':abort_result,'requested_input':len(ids),'input_sha256':digest(ids),
            'prompt_tokens':meta.get('prompt_tokens'),'cached_tokens':meta.get('cached_tokens'),
            'requested_output':output,'actual_output':count,'actual_output_ids':len(last_ids),
            'actual_output_sha256':digest(last_ids),'finish_reason':end_reason,
            'started_perf_ns':start,'started_wall_ns':wall,'first_perf_ns':first,
            'last_perf_ns':last,'ended_perf_ns':end,'ended_wall_ns':end_wall,
            'ttft_s':(first-start)/1e9 if first else None,
            'observed_tpot_ms':(last-first)/1e6/(count-1) if first and last and count>1 else None,
            'e2e_s':(end-start)/1e9,'meta_info_last':meta}
    save(directory/'output_ids_final_or_partial.json',last_ids)
    save(directory/'result.json',result)
    print(f'{role} {url} {len(ids)}/{count}/{output} ttft={result["ttft_s"]} accept={meta.get("spec_accept_length")} partial={partial} error={bool(error)}',flush=True)
    return result


def flush(url,path):
    # Abort completion and scheduler cleanup are asynchronous. Metrics may show
    # idle before tokenizer state is removed; retain HTTP bodies and retry 400.
    deadline=time.monotonic()+30
    attempts=[]
    while True:
        entry={'at_perf_ns':time.perf_counter_ns(),'at_wall_ns':time.time_ns()}
        try:
            req=urllib.request.Request(url+'/flush_cache',b'',method='POST')
            with urllib.request.urlopen(req,timeout=10) as resp:
                entry.update(http_status=resp.status,body=resp.read().decode('utf-8','replace'))
        except urllib.error.HTTPError as exc:
            entry.update(http_status=exc.code,body=exc.read().decode('utf-8','replace'))
        except Exception:entry['error']=traceback.format_exc()
        attempts.append(entry)
        save(path,{'attempts':attempts,**entry})
        if entry.get('http_status')==200:return
        if entry.get('http_status')!=400 or time.monotonic()>=deadline:
            raise RuntimeError('cache flush failed: '+repr(entry))
        time.sleep(.5)


def sampler(url,mode,directory,stop):
    with gzip.open(directory/'metrics_raw.jsonl.gz','wt',encoding='utf-8',compresslevel=2) as out:
        tick=0
        while not stop.is_set():
            obj={'perf_ns':time.perf_counter_ns(),'wall_ns':time.time_ns()}
            try:
                with urllib.request.urlopen(url+'/metrics',timeout=7) as r:obj['http_status']=r.status;obj['raw_prometheus']=r.read().decode('utf-8','replace')
            except Exception as e:obj['metrics_error']=repr(e)
            if tick%3==0:
                cmd=['npu-smi','info'] if mode=='off' else ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','LogLevel=ERROR','10.120.72.25','npu-smi info']
                try:
                    p=subprocess.run(cmd,capture_output=True,text=True,timeout=16)
                    obj['npu_smi']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
                except Exception as e:obj['npu_smi_error']=repr(e)
            out.write(json.dumps(obj,ensure_ascii=False)+'\n');out.flush()
            tick+=1;stop.wait(10)


def wait_until_idle(url,path,timeout_s=90):
    """Prevent an aborted 32K request from contaminating the next stage."""
    deadline=time.monotonic()+timeout_s
    samples=[]
    while time.monotonic()<deadline:
        try:
            with urllib.request.urlopen(url+'/metrics',timeout=8) as response:
                text=response.read().decode('utf-8','replace')
            counts=[float(x) for x in re.findall(r'^sglang:num_running_reqs\{[^\n]*\}\s+([0-9.]+)',text,re.M)]
            running=max(counts) if counts else None
            samples.append({'at_wall_ns':time.time_ns(),'running':running})
            if running==0:
                save(path,{'idle':True,'samples':samples})
                return
        except Exception as exc:samples.append({'at_wall_ns':time.time_ns(),'error':repr(exc)})
        time.sleep(4)
    save(path,{'idle':False,'samples':samples})
    raise RuntimeError('previous request still running; refuse next workload to protect cache validity')


def worker(mode,scenario,scene_data,root,stop_at):
    url=URLS[mode]; dest=root/scenario/mode;dest.mkdir(parents=True,exist_ok=False)
    save(dest/'config.json',{'mode':mode,'url':url,'scenario':scenario,'start_local':datetime.datetime.now().astimezone().isoformat(),
         'deadline_perf_s':stop_at,'seconds':int(round(stop_at-time.monotonic())),'input':scene_data['input_tokens'],
         'output':scene_data['target_output_tokens'],'prefix':scene_data['shared_prefix_tokens'],
         'temperature':0,'ignore_eos':True,'max_concurrent_requests':1})
    wait_until_idle(url,dest/'idle_before_scenario.json')
    stop=threading.Event();thread=threading.Thread(target=sampler,args=(url,mode,dest,stop),daemon=True);thread.start()
    out=[]
    try:
        for case in scene_data['cases']:
            if time.monotonic()>=stop_at-2.5:break
            idx=case['index'];folder=dest/f'request_{idx:04d}'
            flush(url,folder/'flush_cache.json')
            if scenario=='prefix':
                warm=request(url,scene_data['cases'][0]['input_ids'][:19661],1,folder/'prefix_warmup',role='prefix_warmup')
                if warm['error'] or warm['actual_output']!=1:raise RuntimeError('prefix warmup failed')
            if time.monotonic()>=stop_at:break
            r=request(url,case['input_ids'],scene_data['target_output_tokens'],folder/'measurement',deadline=stop_at)
            r['case_index']=idx;r['expected_input_sha256']=case['input_sha256']
            save(folder/'measurement/result.json',r)
            out.append(r)
            save(dest/'completed_or_partial_results.json',out)
            if r['partial_at_45m_deadline']:break
            if r['error']:break
        stop.wait(max(0,stop_at-time.monotonic()))
        summary={'mode':mode,'scenario':scenario,'measurement_start_local':json.loads((dest/'config.json').read_text())['start_local'],
             'measurement_end_local':datetime.datetime.now().astimezone().isoformat(),
             'request_count':len(out),'complete_count':sum(x['done_events']==1 and x['actual_output']==x['requested_output'] for x in out),
             'partial_count':sum(x['partial_at_45m_deadline'] for x in out),'error_count':sum(bool(x['error']) for x in out)}
        save(dest/'summary.json',summary)
        return summary
    finally:
        stop.set();thread.join(timeout=23)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_root',type=Path)
    parser.add_argument('--duration',type=int,default=2700)
    parser.add_argument('--scenario',choices=tuple(SCENARIOS)+('all','remaining'),default='all')
    args=parser.parse_args();root=args.result_root
    cases=data(root)
    scenarios=list(SCENARIOS) if args.scenario=='all' else (['mixed','prefix'] if args.scenario=='remaining' else [args.scenario])
    for scenario in scenarios:
        stage_start=time.monotonic(); deadline=stage_start+args.duration
        (root/scenario).mkdir(exist_ok=True)
        save(root/scenario/'schedule.json',{'start_perf_s':stage_start,'start_local':datetime.datetime.now().astimezone().isoformat(),
            'duration_s':args.duration,'stop_perf_s':deadline,'modes':['off','on']})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={m:pool.submit(worker,m,scenario,cases['scenarios'][scenario],root,deadline) for m in ('off','on')}
            summaries={m:f.result() for m,f in futures.items()}
        save(root/scenario/'summaries.json',summaries)
        print('SCENARIO COMPLETED',scenario,json.dumps(summaries,ensure_ascii=False),flush=True)
    print('ALL REQUESTED SCENARIOS DONE',flush=True)


if __name__=='__main__':main()
