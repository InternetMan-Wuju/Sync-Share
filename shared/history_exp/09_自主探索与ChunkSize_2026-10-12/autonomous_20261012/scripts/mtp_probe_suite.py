#!/usr/bin/env python3
"""Durable short causal probes using saved inputs, with complete raw outputs."""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
Q1=ROOT.parent/'Q1MTPExp'
sys.path.insert(0,str(Q1/'scripts'))
from bench_official_mtp_full import aggregate, capture, histogram, write_json


def http(url,path,data=None,timeout=30):
    req=urllib.request.Request(url+path,data=json.dumps(data).encode() if data is not None else None,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--container',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--url',default='http://127.0.0.1:30080');ap.add_argument('--full-short-groups',action='store_true')
    ap.add_argument('--only',default='');ap.add_argument('--eos-only',action='store_true')
    ap.add_argument('--business',action='store_true')
    ap.add_argument('--framing-pair',action='store_true')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('*/result.json')):raise RuntimeError('output directory already has results; choose a new directory')
    state={'container':a.container,'started_unix':time.time(),'runner_pid':__import__('os').getpid(),
           'stage':'waiting_health','completed':False,'groups':{}}
    lock=threading.Lock()
    def status(stage,**kw):
        with lock:
            state.update(stage=stage,updated_unix=time.time(),**kw);write_json(out/'status.json',state)
    status('waiting_health')
    for _ in range(720):
        inspect=subprocess.run(['sudo','-n','docker','inspect',a.container,'--format','{{json .State}}'],capture_output=True,text=True)
        if inspect.returncode or not json.loads(inspect.stdout)['Running']:
            status('failed',error='service exited before healthy');raise RuntimeError(state['error'])
        try:http(a.url,'/health',timeout=3);break
        except Exception:time.sleep(5)
    else:status('failed',error='health timeout');raise RuntimeError(state['error'])
    server_info=json.loads(http(a.url,'/get_server_info'))
    write_json(out/'server_info.json',server_info)
    latest=json.loads((Q1/'results/latest_native_full.json').read_text());previous=Path(latest['out'])
    short=Q1/'results/native_randomids_20261012_1844/datasets/native_exact_ids.json'
    short_ds=json.loads(short.read_text())
    native_short=short_ds['cases']
    long_ds=json.loads((previous/'datasets/w3_1k_32k.json').read_text())
    sanity=json.loads((previous/'datasets/sanity_chat.json').read_text())
    groups=[('sanity_chat',sanity['cases'],128,False,1,0),
            ('native_exact_1k256',native_short,256,True,1,0),
            ('native_exact_eos_1k256',native_short,256,False,1,0),
            ('native_longinput_1k1024',long_ds['cases'],1024,True,1,0)]
    audit=json.loads((Q1/'results/native_randomids_20261012_1844/text_roundtrip_audit.json').read_text())
    roundtrip=[{'index':c['index'],'input_ids':c['text_reencoded_ids'],
                'description':'Saved actual native default decode-text/re-tokenize result; not the original1024 IDs'} for c in audit]
    groups.append(('native_default_text_roundtrip_256',roundtrip,256,True,1,0))
    # Same raw bodies, now inside the exact role framing used by the saved
    # GLM sanity template. This changes the task; it is a diagnostic, not a
    # fixed-length benchmark replacement or an attempt to tune the random seed.
    framed=[]
    for i,c in enumerate(native_short+long_ds['cases']):
        framed.append({'index':i,'input_ids':[154822,154824,154827]+c['input_ids']+[154828,154841,154842],
                       'raw_body_index':c['index'],'raw_body_group':'short' if i<len(native_short) else 'longoutput',
                       'description':'Raw random ID body preserved, exact saved GLM chat role framing added'})
    groups.append(('native_random_body_chatframed_256',framed,256,False,1,0))
    if a.framing_pair:
        # Match output cap and EOS policy as well as the raw body. These are
        # paired input-format diagnostics, not a replacement standard workload.
        groups.extend([('framing_pair_raw_eos256',long_ds['cases'],256,False,1,0),
                       ('framing_pair_chat_eos256',framed[len(native_short):],256,False,1,0)])
    if a.full_short_groups:
        for name in ('w1_8k_1k','w2_32k_1k','w4_prefix32k_300'):
            d=json.loads((previous/'datasets'/f'{name}.json').read_text())
            groups.append((name,d['cases'],d['target_output_tokens'],True,d['concurrency'],d.get('shared_prefix_tokens',0)))
    if a.business:
        business=json.loads((ROOT/'real_chat_dataset.json').read_text())['cases']
        groups=[('sharegpt_chat_eos',business,256,False,4,0),
                ('sharegpt_chat_forced',business,256,True,4,0),
                # First two intact records, chosen before measuring acceptance.
                # This separates a length-capped prefix from a complete reply.
                ('sharegpt_chat_complete',business[:2],2048,False,2,0)]
    if a.only:
        names=a.only.split(',')
        if 'native_exact_1k256' in names:
            names+=['native_default_text_roundtrip_256','native_random_body_chatframed_256']
        if a.framing_pair:
            names+=['framing_pair_raw_eos256','framing_pair_chat_eos256']
        groups=[g for g in groups if g[0] in names]
    if a.eos_only:groups=[g for g in groups if not g[3]]
    eos=set(json.loads(Path('/data_lib/data/models/glm-q/config.json').read_text())['eos_token_id'])
    stop=threading.Event()
    def monitor():
        d=out/'monitor';d.mkdir(exist_ok=True)
        while not stop.is_set():
            stamp=time.strftime('%Y%m%d_%H%M%S')
            try:(d/(stamp+'.prom')).write_bytes(http(a.url,'/metrics',timeout=10))
            except Exception as e:(d/(stamp+'.error')).write_text(str(e))
            stop.wait(60)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        for name,cases,maxout,ignore,conc,prefix in groups:
            target=out/name;target.mkdir(exist_ok=True);status('running',current_group=name,request_progress={})
            print('GROUP',name,'requests',len(cases),'output',maxout,'ignore_eos',ignore,flush=True)
            http(a.url,'/flush_cache',{},timeout=60)
            if prefix:
                case={'index':-1,'input_ids':cases[0]['input_ids'][:prefix]}
                payload={'input_ids':case['input_ids'],'stream':True,'sampling_params':{'max_new_tokens':1,'temperature':0,'ignore_eos':True}}
                warm=capture(a.url,payload,target/'prefix_warmup.json',case,timeout=900)
                if warm['error']:raise RuntimeError(warm['error'])
            def one(c):
                payload={'input_ids':c['input_ids'],'stream':True,
                         'sampling_params':{'max_new_tokens':maxout,'temperature':0,'ignore_eos':ignore}}
                def progress(index,detail):
                    with lock:
                        state.setdefault('request_progress',{})[str(index)]=detail
                        write_json(out/'status.json',state)
                rec=capture(a.url,payload,target/f'request_{c["index"]}.json',c,progress,timeout=1800)
                ids=rec['output_ids'];pos=next((i for i,t in enumerate(ids) if t in eos),None)
                rec.update(first_eos_position=pos,tokens_after_first_eos=len(ids)-pos-1 if pos is not None else 0,
                           input_ids_sha256=hashlib.sha256(json.dumps(c['input_ids'],separators=(',',':')).encode()).hexdigest())
                h=histogram(rec['meta'])
                if h and rec['meta'].get('spec_verify_ct'):
                    assert sum(h)==rec['meta']['spec_verify_ct']
                    assert sum(i*n for i,n in enumerate(h))==rec['meta']['spec_num_correct_drafts']
                    assert 3*sum(h)==rec['meta']['spec_num_proposed_drafts']
                write_json(target/f'request_{c["index"]}.json',rec);return rec
            started=time.monotonic()
            with ThreadPoolExecutor(max_workers=conc) as ex:records=list(ex.map(one,cases))
            wall=time.monotonic()-started
            agg=aggregate(records,wall)
            validation=[]
            for rec in records:
                checks={'no_error':rec['error'] is None,'stream_done':rec['stream_done'],
                        'input_length':rec['input_tokens_sent']==rec['meta'].get('prompt_tokens'),
                        'output_count':rec['completion_tokens']==rec['meta'].get('completion_tokens')}
                if ignore:
                    checks['fixed_output_length']=rec['completion_tokens']==maxout
                if prefix:
                    checks['cache_hit_60pct']=.585<=rec['meta'].get('cached_tokens',0)/rec['input_tokens_sent']<=.615
                if name=='sanity_chat':
                    visible=rec['text'].strip()
                    checks['readable_answer']=('你好' in visible if rec['index']==0 else visible=='2')
                validation.append({'index':rec['index'],'checks':checks,
                                   'failures':[key for key,value in checks.items() if not value]})
            agg['successful_requests']=sum(not v['failures'] for v in validation)
            if not agg['requests_ok']:
                ok=[r for r in records if not r['error']]
                agg.update(requests_ok=len(ok),output_tokens_per_s=sum(r['completion_tokens'] for r in ok)/wall,
                           ttft_s_mean=statistics.mean([r['ttft_s'] for r in ok if r['ttft_s'] is not None]) if ok else None)
            result={'name':name,'container':a.container,'source_inputs':str(short if name.startswith('native_exact') else previous/'datasets'),
                    'concurrency':conc,'ignore_eos':ignore,'target_output_tokens':maxout,'requests':records,
                    'aggregate':agg,'validation':validation,
                    'input_provenance':[{k:v for k,v in c.items() if k!='input_ids'} for c in cases]}
            write_json(target/'result.json',result)
            if agg['successful_requests']!=len(cases):
                raise RuntimeError('request validation failed: '+str(validation))
            with lock:state['groups'][name]={'completed':True,'result':str(target/'result.json'),'aggregate':agg}
            status('group_done',current_group=name)
            print('RESULT',name,json.dumps(agg),flush=True)
        if not a.business:
            # Keep the target-decision check sequential with requests/cache flushes.
            source=out if server_info.get('speculative_algorithm') else ROOT/'results/publisher_v19'
            diagnostic=[sys.executable,'-B',str(ROOT/'scripts/trajectory_check.py'),
                        '--url',a.url,'--source',str(source),'--out',str(out/'target_prefill_audit')]
            if not server_info.get('speculative_algorithm'):
                diagnostic+=['--no-windows']
            if (source/'native_exact_1k256/result.json').exists():
                status('target_prefill_audit')
                state['target_prefill_audit_rc']=subprocess.run(diagnostic,timeout=2400).returncode
        status('completed',completed=True)
    except Exception as e:status('failed',error=str(e));raise
    finally:stop.set();thread.join(timeout=12)


if __name__=='__main__':main()
