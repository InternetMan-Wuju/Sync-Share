#!/usr/bin/env python3
"""Check saved MTP trajectories against independent target prefill decisions.

This is an equivalence diagnostic, not a benchmark. No speculative OFF grid is
needed: max_new_tokens=1 invokes target prefill on a fixed supplied prefix.
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
sys.path.insert(0, str(ROOT.parent / 'Q1MTPExp/scripts'))
from bench_official_mtp_full import capture, write_json


def http(url, path, body=None, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data=r.read()
        if path=='/flush_cache':
            return {'status':r.status,'body':data.decode(errors='replace')}
        return json.loads(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--url', default='http://127.0.0.1:30080')
    ap.add_argument('--no-windows', action='store_true')
    a = ap.parse_args()
    source, out = Path(a.source), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    server_info=http(a.url, '/get_server_info');write_json(out / 'server_info.json', server_info)
    summary = {'source': str(source), 'started_unix': time.time(), 'prefill_checks': [], 'windows': [], 'mismatch_prefill_rechecks': []}
    groups = ('sanity_chat', 'native_exact_1k256', 'native_longinput_1k1024', 'sharegpt_chat_eos')
    for group in groups:
        p = source / group / 'result.json'
        if group=='sharegpt_chat_eos' and not p.exists():
            p=source.with_name(source.name+'_business')/group/'result.json'
        if not p.exists():
            continue
        for record in json.loads(p.read_text())['requests']:
            if record.get('error') or not record['output_ids']:
                continue
            case_index = record['index']
            payload_path = Path(record['payload_file'])
            if not payload_path.is_absolute():
                payload_path = Path('/home/chengyetai') / payload_path
            prompt = json.loads(payload_path.read_text())['input_ids']
            output = record['output_ids']
            # Every selected trajectory is checked; no acceptance-based selection.
            supplied = output[:min(128, len(output))]
            body = {'input_ids': prompt + supplied, 'stream': False,
                    'return_logprob': True, 'top_logprobs_num': 2,
                    # Native API prepends one undefined score at the requested
                    # start even when nonzero. Include the final prompt token so
                    # every supplied output token has a real preceding score.
                    'logprob_start_len': len(prompt)-1,
                    'sampling_params': {'max_new_tokens': 1, 'temperature': 0, 'ignore_eos': True}}
            tag = f'{group}_{case_index}'
            write_json(out / f'{tag}.prefill.payload.json', body)
            entry = {'group': group, 'index': case_index, 'supplied_output_tokens': len(supplied)}
            try:
                http(a.url, '/flush_cache', {})
                response = http(a.url, '/generate', body)
                write_json(out / f'{tag}.prefill.response.json', response)
                meta = response.get('meta_info', {})
                logprobs = meta.get('input_token_logprobs', [])
                tops = meta.get('input_top_logprobs', [])
                # Native logprob rows pair the score with the actual input token.
                # Only compare when alignment is proven by every supplied ID.
                actual = [row[1] for row in logprobs]
                expected=[prompt[-1]]+supplied
                entry.update(logprob_rows=len(logprobs), top_rows=len(tops),
                             alignment_valid=actual == expected and len(tops) == len(expected),
                             undefined_anchor_rows=1)
                if entry['alignment_valid']:
                    comparisons = []
                    for position, (token, row, top) in enumerate(zip(supplied, logprobs[1:], tops[1:])):
                        values = [v[0] for v in top if v[0] is not None]
                        if not top or not values:
                            raise ValueError('empty top-logprob row')
                        best = max(top, key=lambda item: item[0])
                        gap = sorted(values, reverse=True)
                        comparisons.append({'position': position, 'emitted_token': token,
                                            'prefill_argmax': best[1], 'match': token == best[1],
                                            'supplied_logprob': row[0],
                                            'top1_minus_top2': gap[0] - gap[1] if len(gap) > 1 else None,
                                            'reported_values_finite': all(math.isfinite(v) for v in values)
                                            and row[0] is not None and math.isfinite(row[0])})
                    entry.update(comparisons=comparisons,
                                 mismatches=sum(not c['match'] for c in comparisons),
                                 reported_nonfinite=sum(not c['reported_values_finite'] for c in comparisons))
                else:
                    entry['error'] = 'logprob rows do not align with supplied output; comparison withheld'
            except Exception as e:
                entry['error'] = repr(e)
            summary['prefill_checks'].append(entry)
            write_json(out / 'summary.json', summary)
            print('PREFILL CHECK', tag, json.dumps({k: v for k, v in entry.items() if k != 'comparisons'}), flush=True)
            # If a bulk-prefill score disagrees, recheck the first and the
            # largest-gap discrepancy using exactly that causal prefix. These
            # locations are diagnostic selections, not an acceptance estimate.
            wrong=[c for c in entry.get('comparisons',[]) if not c['match']]
            if wrong:
                selected={wrong[0]['position'],max(wrong,key=lambda c:c['top1_minus_top2'] or 0)['position']}
                for position in sorted(selected):
                    routes=[None]+list(range(min(2,server_info.get('dp_size',1)))) if server_info.get('dp_size',1)>1 else [None]
                    for dp_route in routes:
                        for trial in (1,2):
                            causal={'input_ids':prompt+output[:position],'stream':False,
                                    'return_logprob':True,'top_logprobs_num':2,'logprob_start_len':-1,
                                    'sampling_params':{'max_new_tokens':1,'temperature':0,'ignore_eos':True}}
                            if dp_route is not None:causal['routed_dp_rank']=dp_route
                            tag2=f'{tag}.mismatch{position}.dp{dp_route}.trial{trial}'
                            write_json(out/(tag2+'.payload.json'),causal)
                            row={'group':group,'index':case_index,'position':position,'trial':trial,'routed_dp_rank':dp_route,
                                 'saved_token':output[position],
                                 'bulk_prefill_token':next(c['prefill_argmax'] for c in wrong if c['position']==position)}
                            try:
                                http(a.url,'/flush_cache',{})
                                response=http(a.url,'/generate',causal)
                                write_json(out/(tag2+'.response.json'),response)
                                tokens=response.get('output_ids')
                                if tokens is None:
                                    scores=response.get('meta_info',{}).get('output_token_logprobs',[])
                                    tokens=[r[1] for r in scores]
                                if len(tokens)!=1:raise ValueError('one-token recheck returned unexpected IDs')
                                row.update(token=tokens[0],matches_saved=tokens[0]==row['saved_token'],
                                           matches_bulk=tokens[0]==row['bulk_prefill_token'])
                            except Exception as e:row['error']=repr(e)
                            summary['mismatch_prefill_rechecks'].append(row)
                            write_json(out/'summary.json',summary)
                            print('DIRECT PREFILL',tag2,json.dumps(row),flush=True)
            # Independent continuations at fixed, predeclared positions. They are
            # only comparable to the saved trajectory when output IDs match.
            if group in ('sanity_chat','sharegpt_chat_eos') or a.no_windows:
                continue
            for position in (0, 32, 128, 512):
                if position + 128 > len(output):
                    continue
                http(a.url, '/flush_cache', {})
                payload = {'input_ids': prompt + output[:position], 'stream': True,
                           'sampling_params': {'max_new_tokens': 128, 'temperature': 0, 'ignore_eos': True}}
                case = {'index': position}
                rec = capture(a.url, payload, out / f'{tag}.window{position}.json', case, timeout=900)
                expected = output[position:position + 128]
                difference = next((i for i, (x, y) in enumerate(zip(expected, rec['output_ids'])) if x != y), None)
                summary['windows'].append({'group': group, 'index': case_index, 'position': position,
                                           'same_output_ids': expected == rec['output_ids'],
                                           'first_difference': difference, 'error': rec['error'],
                                           'AL': rec['meta'].get('spec_accept_length'),
                                           'P1_P2_P3': rec['pos_rates'], 'output_shape': rec['output_shape']})
                write_json(out / 'summary.json', summary)
    summary.update(completed=True, ended_unix=time.time(),
                   scope='Prefill-vs-MTP target decisions on fixed prefixes; finite top2 does not prove all logits finite. Window acceptance belongs to independent continuations, not reconstructed original step counts.')
    summary['aligned_checks']=sum(bool(x.get('alignment_valid')) for x in summary['prefill_checks'])
    summary['interface_errors']=sum(bool(x.get('error')) for x in summary['prefill_checks'])
    summary['recheck_interface_errors']=sum(bool(x.get('error')) for x in summary['mismatch_prefill_rechecks'])
    summary['valid_comparison']=summary['recheck_interface_errors']==0 and bool(summary['prefill_checks']) and summary['interface_errors']==0 and summary['aligned_checks']==len(summary['prefill_checks'])
    write_json(out / 'summary.json', summary)
    if not summary['valid_comparison']:raise SystemExit(2)


if __name__ == '__main__':
    main()
