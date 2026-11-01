#!/usr/bin/env python3
"""Collect fixed-concurrency ChunkSize cells with raw SSE and exact timestamps."""
import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
sys.path.insert(0,str(ROOT.parent/'Q1MTPExp/scripts'))
from bench_official_mtp_full import capture, percentile, write_json


def http(url,path,data=None,timeout=30):
    req=urllib.request.Request(url+path,data=json.dumps(data).encode() if data is not None else None,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()


def summarize(records,wall,target,prefix):
    ok=[r for r in records if not r['error'] and r['stream_done'] and
        r['completion_tokens']==target==r['meta'].get('completion_tokens') and
        r['input_tokens_sent']==r['meta'].get('prompt_tokens') and
        (r['meta'].get('finish_reason') or {}).get('type')=='length']
    result={'requests':len(records),'requests_ok':len(ok),'complete':len(ok)==len(records),
            'wall_s':wall,'output_tokens':sum(r['completion_tokens'] for r in ok),
            'total_output_tok_s':sum(r['completion_tokens'] for r in ok)/wall}
    for k in ('ttft_s','tpot_ms','e2e_s','decode_s','client_finalization_s'):
        values=[r[k] for r in ok if r[k] is not None]
        result.update({k+'_mean':statistics.mean(values) if values else None,
                       k+'_p50':percentile(values,.5),k+'_p95':percentile(values,.95)})
    prompt=sum(r['meta'].get('prompt_tokens',0) for r in ok)
    result['cache_hit_ratio']=sum(r['meta'].get('cached_tokens',0) for r in ok)/prompt if prompt else None
    result['cache_condition_valid']=result['cache_hit_ratio'] is not None and (
        .585<=result['cache_hit_ratio']<=.615 if prefix else result['cache_hit_ratio']<.005)
    components=[]
    for r in ok:
        m=r['meta'];recv=m.get('request_received_ts');entry=m.get('forward_entry_time');pref=m.get('prefill_finished_time');finish=m.get('request_finished_ts')
        if all(x is not None for x in (recv,entry,pref,finish)):
            server_total=finish-recv
            components.append({'server_before_forward_s':entry-recv,'server_prefill_span_s':pref-entry,
                               'server_decode_and_finish_s':finish-pref,'server_e2e_s':server_total,
                               'client_minus_server_e2e_s':r['e2e_s']-server_total,
                               'reported_queue_s':m.get('queue_time')})
    result['timestamp_breakdown_mean_s']={k:statistics.mean([r[k] for r in components if r[k] is not None])
                                           for k in components[0]} if components else None
    result['breakdown_scope']='Server wall spans include scheduler gaps; not pure operator time. Client/server difference is a residual, not assigned to an invented cause.'
    result['periodic_suffix_requests']=sum(bool(r['output_shape']['periodic_suffix']) for r in ok)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--concurrency',type=int,required=True);ap.add_argument('--chunk',type=int,required=True)
    ap.add_argument('--url',default='http://127.0.0.1:30080');ap.add_argument('--repeat',type=int,default=1)
    ap.add_argument('--mtp-on',action='store_true')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ds=json.loads(Path(a.dataset).read_text());n=a.concurrency*(2 if ds['name'] in ('w1_8k_1k','w4_prefix32k_300') else 1)
    assert len(ds['cases'])>=n
    cases=ds['cases'][:n];prefix=ds.get('shared_prefix_tokens',0);output=ds['target_output_tokens']
    state={'stage':'initializing','done':False,'chunk':a.chunk,'concurrency':a.concurrency,'requests':n};lock=threading.Lock()
    def status(**kw):
        with lock:state.update(updated_unix=time.time(),**kw);write_json(out/'status.json',state)
    status(stage='flush_cache');http(a.url,'/flush_cache',{},timeout=60)
    info=json.loads(http(a.url,'/get_server_info'));write_json(out/'server_info.json',info)
    assert info['chunked_prefill_size']==a.chunk,('requested/effective chunk differs',a.chunk,info['chunked_prefill_size'])
    assert bool(info.get('speculative_algorithm'))==a.mtp_on,'MTP state differs from experiment plan'
    assert all(len(c['input_ids'])==ds['target_input_tokens'] for c in cases) if 'target_input_tokens' in ds else True
    # Warm the same prompt shapes/concurrency before timing. Keep these raw
    # streams separate, then flush so cold-cache measurements remain cold.
    status(stage='shape_warmup')
    def shape_warm(c):
        payload={'input_ids':c['input_ids'],'stream':True,
                 'sampling_params':{'max_new_tokens':16,'temperature':0,'ignore_eos':True}}
        return capture(a.url,payload,out/f'shape_warmup_request_{c["index"]}.json',c,timeout=7200)
    with ThreadPoolExecutor(max_workers=a.concurrency) as executor:
        warmups=list(executor.map(shape_warm,cases[:a.concurrency]))
    warmup_valid=all(not r['error'] and r['stream_done'] and r['completion_tokens']==16
                     and r['meta'].get('completion_tokens')==16
                     and r['input_tokens_sent']==r['meta'].get('prompt_tokens') for r in warmups)
    write_json(out/'shape_warmup.json',{'protocol':'First C fixed dataset prompts, concurrency C, output16; excluded from measured wall time; cache flushed afterwards',
                                      'valid':warmup_valid,'requests':warmups})
    if not warmup_valid:raise RuntimeError('shape warmup failed; no measured cell')
    http(a.url,'/flush_cache',{},timeout=60)
    if prefix:
        c={'index':-1,'input_ids':cases[0]['input_ids'][:prefix]}
        payload={'input_ids':c['input_ids'],'stream':True,'sampling_params':{'max_new_tokens':1,'temperature':0,'ignore_eos':True}}
        warm=capture(a.url,payload,out/'warmup.json',c,timeout=900)
        if warm['error'] or not warm['stream_done'] or warm['completion_tokens']!=1:raise RuntimeError('prefix warmup failed: '+str(warm.get('error')))
    (out/'metrics_before.prom').write_bytes(http(a.url,'/metrics'))
    stop=threading.Event()
    def monitor():
        directory=out/'monitor';directory.mkdir(exist_ok=True)
        while not stop.is_set():
            try:(directory/(str(int(time.time()))+'.prom')).write_bytes(http(a.url,'/metrics',timeout=10))
            except Exception as e:(directory/'error.txt').write_text(str(e))
            stop.wait(10)
    t=threading.Thread(target=monitor,daemon=True);t.start();status(stage='running')
    def one(c):
        payload={'input_ids':c['input_ids'],'stream':True,'sampling_params':{'max_new_tokens':output,'temperature':0,'ignore_eos':True}}
        def progress(i,detail):
            with lock:state.setdefault('requests_progress',{})[str(i)]=detail;write_json(out/'status.json',state)
        return capture(a.url,payload,out/f'request_{c["index"]}.json',c,progress,timeout=5400)
    started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:records=list(ex.map(one,cases))
    finally:stop.set();t.join(timeout=12)
    wall=time.monotonic()-started
    (out/'metrics_after.prom').write_bytes(http(a.url,'/metrics'))
    agg=summarize(records,wall,output,prefix)
    result={'name':ds['name'],'dataset':a.dataset,'dataset_sha256':hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
            'chunk':a.chunk,'concurrency':a.concurrency,'requests':records,'aggregate':agg,'mtp_on':a.mtp_on,
            'repeat':a.repeat,'payload_sampling':{'temperature':0,'ignore_eos':True}}
    result['warmup_protocol']='First C fixed dataset prompts at concurrency C, output16, then flush_cache; shared prefix separately warmed only in prefix scenes. All warmup streams saved and excluded from measured time.'
    write_json(out/'result.json',result);status(stage='completed',done=True,aggregate=agg)
    print('CHUNK RESULT',json.dumps(agg),flush=True)
    if not agg['complete'] or not agg['cache_condition_valid']:raise SystemExit(2)


if __name__=='__main__':main()
