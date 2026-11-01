#!/usr/bin/env python3
"""Audit saved experimental streams, token counts, acceptance and file hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def main(root):
    dataset=json.loads((root/'dataset.json').read_text())
    report={'scenarios':{},'errors':[]}
    for scene in ('decode','mixed','prefix'):
        report['scenarios'][scene]={}
        cutoff=json.loads((root/scene/'terminal_counter_cutoff.json').read_text())
        controlled={item['active']['rid'] for item in cutoff['responses'] if item.get('active')}
        for mode in ('off','on'):
            folder=root/scene/mode
            rows=json.loads((folder/'completed_or_partial_results.json').read_text())
            stage=[]
            for r in rows:
                ix=r['case_index']
                path=folder/f'request_{ix:04d}/measurement'
                payload=json.loads((path/'request.json').read_text())['payload']
                expected=dataset['scenarios'][scene]['cases'][ix]['input_ids']
                last=None;events=0;done=0;prev=0;first_time=None;last_time=None
                with gzip.open(path/'sse_raw.jsonl.gz','rt',encoding='utf-8') as source:
                    for line in source:
                        item=json.loads(line);raw=item.get('raw_line','')
                        if raw.startswith('data:') and raw[5:].strip()=='[DONE]':done+=1;continue
                        if not raw.startswith('data:'):continue
                        event=json.loads(raw[5:]);events+=1
                        meta=event.get('meta_info') or {}
                        c=meta.get('completion_tokens',0)
                        if c<prev:report['errors'].append(f'{scene}/{mode}/{ix}: token count fell at {events}')
                        prev=c
                        if c>0 and first_time is None:first_time=item['recv_perf_ns']
                        if c>0:last_time=item['recv_perf_ns']
                        if isinstance(event.get('output_ids'),list):last=event['output_ids']
                saved=json.loads((path/'output_ids_final_or_partial.json').read_text())
                valid=(payload['input_ids']==expected and digest(expected)==r['input_sha256'] and
                       payload['sampling_params']['max_new_tokens']==r['requested_output'] and
                       events==r['events'] and done==r['done_events'] and
                       last==saved and len(saved)==r['actual_output'] and
                       digest(saved)==r['actual_output_sha256'] and prev==r['actual_output'])
                m=r['meta_info_last'];verify=m.get('spec_verify_ct')
                if mode=='on' and isinstance(verify,int) and verify>0:
                    if abs(m['spec_accept_length']-r['actual_output']/verify)>1e-6:valid=False
                    hist=m.get('spec_correct_drafts_histogram')
                    if hist is not None and (sum(hist)!=verify or sum(i*n for i,n in enumerate(hist))!=m.get("spec_num_correct_drafts")):valid=False
                if scene=='prefix' and not (r['prompt_tokens']==32768 and r['cached_tokens'] is not None and .585<=r['cached_tokens']/32768<=.615):valid=False
                if scene!='prefix' and r['cached_tokens']!=0:valid=False
                flush=json.loads((path.parent/'flush_cache.json').read_text())
                if flush.get('http_status')!=200:valid=False
                if scene=='prefix':
                    warm=json.loads((path.parent/'prefix_warmup/result.json').read_text())
                    if not (warm['http_status']==200 and not warm['error'] and warm['actual_output']==1 and
                            warm['prompt_tokens']==19661 and warm['cached_tokens']==0):valid=False
                if mode=='off' and verify is not None:valid=False
                if not valid:report['errors'].append(f'{scene}/{mode}/{ix}: count/input/cache/acceptance mismatch')
                stage.append({'case_index':ix,'valid':valid,'sse_events':events,'done':done,'requested_output':r['requested_output'],
                              'actual_output':r['actual_output'],'partial_client_flag':r['partial_at_45m_deadline'],'partial':r['partial_at_45m_deadline'] or (r['actual_output']<r['requested_output'] and m.get('id') in controlled),
                              'spec_verify_ct':verify,'prompt_tokens':r['prompt_tokens'],'cached_tokens':r['cached_tokens']})
            report['scenarios'][scene][mode]=stage
    report['all_valid']=not report['errors']
    out=root/'analysis';out.mkdir(exist_ok=True)
    (out/'integrity_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (root/'file_inventory.sha256.tsv').open('w') as target:
        target.write('sha256\tbytes\trelative_path\n')
        skip={'file_inventory.sha256.tsv','node_23/chunk_16384/server.log','node_25/chunk_16384/server.log'}
        for item in sorted(root.rglob('*')):
            if not item.is_file():continue
            rel=item.relative_to(root).as_posix()
            if rel in skip:continue
            h=hashlib.sha256()
            with item.open('rb') as source:
                for chunk in iter(lambda:source.read(1<<20),b''):h.update(chunk)
            target.write(f'{h.hexdigest()}\t{item.stat().st_size}\t{rel}\n')
    print('audited',sum(len(x) for group in report['scenarios'].values() for x in group.values()),
          'requests; errors:',report['errors'],'inventory saved')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);main(ap.parse_args().root)
