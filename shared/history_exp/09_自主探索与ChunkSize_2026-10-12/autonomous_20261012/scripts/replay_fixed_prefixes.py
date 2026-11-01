#!/usr/bin/env python3
"""Replay exact saved baseline prefixes and routes on the isolation service."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def http(url, path, body=None):
    req = urllib.request.Request(url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        data = response.read()
        return json.loads(data) if path != '/flush_cache' else {'status': response.status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:30080')
    args = parser.parse_args()
    source = ROOT / 'results/mtp-v19-dp2-no-plan-lp-safe/target_prefill_audit'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((source / 'summary.json').read_text())
    state = {'source': str(source), 'started_unix': time.time(), 'completed': False, 'probes': []}
    info = http(args.url, '/get_server_info')
    write(out / 'server_info.json', info)
    assert info['dp_size'] == 2
    for row in original['mismatch_prefill_rechecks']:
        route = row.get('routed_dp_rank')
        if route is None or row.get('error'):
            continue
        tag = f"{row['group']}_{row['index']}.mismatch{row['position']}.dp{route}.trial{row['trial']}"
        source_payload = source / (tag + '.payload.json')
        body = json.loads(source_payload.read_text())
        assert body['routed_dp_rank'] == route
        assert body['sampling_params'] == {'max_new_tokens': 1, 'temperature': 0, 'ignore_eos': True}
        assert body['logprob_start_len'] == -1
        write(out / (tag + '.payload.json'), body)
        http(args.url, '/flush_cache', {})
        response = http(args.url, '/generate', body)
        write(out / (tag + '.response.json'), response)
        meta = response['meta_info']
        tokens = response.get('output_ids')
        if tokens is None:
            tokens = [item[1] for item in meta.get('output_token_logprobs', [])]
        assert len(tokens) == 1 and meta.get('dp_rank') == route
        state['probes'].append({
            'group': row['group'], 'index': row['index'], 'position': row['position'],
            'trial': row['trial'], 'routed_dp_rank': route, 'actual_dp_rank': meta['dp_rank'],
            'baseline_token': row['token'], 'token': tokens[0],
            'same_as_baseline': tokens[0] == row['token'],
            'prefix_sha256': hashlib.sha256(json.dumps(body['input_ids'], separators=(',', ':')).encode()).hexdigest(),
            'source_payload': str(source_payload), 'response_source': str(out / (tag + '.response.json')),
            'output_top_logprobs': meta.get('output_top_logprobs')})
        write(out / 'summary.json', state)
        print(tag, state['probes'][-1]['baseline_token'], '->', tokens[0], flush=True)
    assert len(state['probes']) == 16, 'baseline pinned probe set changed; inspect before proceeding'
    state.update(completed=True, ended_unix=time.time(),
        scope='Exact baseline prefixes and DP routes replayed twice; diagnostic positions selected in the baseline, not a random accuracy sample.')
    write(out / 'summary.json', state)


if __name__ == '__main__':
    main()
