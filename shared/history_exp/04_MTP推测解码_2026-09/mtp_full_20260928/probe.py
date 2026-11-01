#!/usr/bin/env python3
"""Small 1K / 64-output ON/OFF probe; verify per-request acceptance counters."""
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from bench import URLS, data, flush, request, save

root=Path(sys.argv[1]);out=root/'probe';out.mkdir(exist_ok=True)
prompt=data(root)['scenarios']['decode']['cases'][0]['input_ids']

def run(mode):
    path=out/mode;path.mkdir(exist_ok=True)
    flush(URLS[mode],path/'flush_cache.json')
    return request(URLS[mode],prompt,64,path/'measurement',time.monotonic()+240,role='acceptance_probe')

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures={m:pool.submit(run,m) for m in ('off','on')}
    results={m:f.result() for m,f in futures.items()}
on=results['on']; meta=on['meta_info_last']
verdict={'requests':{mode:{k:r[k] for k in ('http_status','actual_output','requested_output','error','done_events','ttft_s','observed_tpot_ms')} for mode,r in results.items()},
    'spec_accept_length':meta.get('spec_accept_length'),
    'spec_accept_rate':meta.get('spec_accept_rate'),
    'spec_verify_ct':meta.get('spec_verify_ct'),
    'spec_num_correct_drafts':meta.get('spec_num_correct_drafts'),
    'spec_num_proposed_drafts':meta.get('spec_num_proposed_drafts'),
    'spec_correct_drafts_histogram':meta.get('spec_correct_drafts_histogram')}
verdict['passed']=all(r['http_status']==200 and r['actual_output']==64 and r['done_events']==1 and r['error'] is None for r in results.values()) and (
    isinstance(verdict['spec_verify_ct'],int) and verdict['spec_verify_ct']>0 and
    abs(verdict['spec_accept_length']-64/verdict['spec_verify_ct'])<1e-8 and
    sum(verdict['spec_correct_drafts_histogram'])==verdict['spec_verify_ct'])
save(out/'verdict.json',verdict)
print(json.dumps(verdict,ensure_ascii=False,indent=2),flush=True)
if not verdict['passed']:raise SystemExit('acceptance counters preflight failed')
