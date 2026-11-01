#!/usr/bin/env python3
"""Short target-only generation check; not an MTP OFF performance grid."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
OUT = ROOT / 'review_20261013'
NAME = 'cyt-auto-review-target-only-20261013'
URL = 'http://127.0.0.1:30080'
sys.path.insert(0, str(ROOT.parent / 'Q1MTPExp/scripts'))
from bench_official_mtp_full import capture, write_json

STATE = {'pid': os.getpid(), 'started_unix': time.time(), 'completed': False, 'container': NAME}


def read(path):
    return json.loads(path.read_text())


def status(phase, **fields):
    STATE.update(phase=phase, updated_unix=time.time(), **fields)
    write_json(OUT / 'target_only_status.json', STATE)
    print(time.strftime('%H:%M:%S'), phase, fields, flush=True)


def http(path, body=None, timeout=20):
    request = urllib.request.Request(URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        return {'status': response.status} if path in ('/health', '/flush_cache') else json.loads(data)


def main():
    output = OUT / 'target_only'
    output.mkdir(exist_ok=False)
    command = [sys.executable, '-B', str(ROOT / 'scripts/service.py'), 'start', '--name', NAME,
        '--repaired', '--slim-draft', '--mtp', 'off', '--dp', '2', '--dp-attention',
        '--context', '180000', '--mrr', '8', '--diagnostics',
        '--env', 'SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0', '--env', 'SGLANG_NPU_USE_MULTI_STREAM=0']
    status('starting', command=command)
    subprocess.run(command, check=True, timeout=120)
    deadline = time.monotonic() + 3600
    status('waiting_health')
    while time.monotonic() < deadline:
        result = subprocess.run(['sudo', '-n', 'docker', 'inspect', NAME, '--format', '{{json .State}}'],
            capture_output=True, text=True, timeout=20, check=True)
        if not json.loads(result.stdout)['Running']:
            raise RuntimeError('target-only service exited before health')
        log = ROOT / 'logs' / NAME / 'server.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 262144))
                tail = stream.read()
            if b'crashed with exit code' in tail or b'scheduler died during initialization' in tail:
                raise RuntimeError('target-only scheduler failed')
        try:
            http('/health', timeout=3)
            break
        except Exception:
            time.sleep(20)
    else:
        raise RuntimeError('target-only health timeout')
    info = http('/get_server_info')
    write_json(output / 'server_info.json', info)
    assert not info.get('speculative_algorithm') and info['dp_size'] == 2
    before = read(ROOT / 'logs/cyt-auto-mtp-v19-dp2-no-multistream/config.json')
    after = read(ROOT / 'logs' / NAME / 'config.json')
    config_diff = {key: [before.get(key), after.get(key)] for key in set(before) | set(after)
        if key not in ('name', 'started_at', 'server_command') and before.get(key) != after.get(key)}
    assert config_diff == {'mtp': ['on', 'off']}, config_diff
    write_json(output / 'config_comparison.json', {'differences': config_diff,
        'before': str(ROOT / 'logs/cyt-auto-mtp-v19-dp2-no-multistream/config.json'),
        'after': str(ROOT / 'logs' / NAME / 'config.json'),
        'scope': 'Same explicit service configuration except MTP. Removing the draft also changes actual memory allocations and execution shapes.'})

    base = ROOT / 'results/mtp-v19-dp2-no-multistream'
    cases = []
    for group in ('sanity_chat', 'native_exact_1k256', 'native_longinput_1k1024'):
        result_path = base / group / 'result.json'
        for record in read(result_path)['requests']:
            body = read(Path(record['payload_file']))
            route = record['meta']['dp_rank']
            body['routed_dp_rank'] = route
            cases.append({'name': f"{group}_{record['index']}", 'body': body,
                'expected_ids': record['output_ids'], 'baseline_AL': record['meta'].get('spec_accept_length'),
                'baseline_P1_P2_P3': record['pos_rates'], 'baseline_source': str(result_path),
                'baseline_index': record['index'], 'route': route})
    cycle = read(base / 'native_longinput_1k1024/result.json')['requests'][0]
    cycle_body = read(Path(cycle['payload_file']))
    cases.append({'name': 'cycle_fixed_prefix32_window128',
        'body': {'input_ids': cycle_body['input_ids'] + cycle['output_ids'][:32], 'stream': True,
            'routed_dp_rank': cycle['meta']['dp_rank'],
            'sampling_params': {'max_new_tokens': 128, 'temperature': 0, 'ignore_eos': True}},
        'expected_ids': cycle['output_ids'][32:160], 'baseline_source': str(base / 'native_longinput_1k1024/result.json'),
        'baseline_index': 0, 'route': cycle['meta']['dp_rank'],
        'comparison_scope': 'Expected continuation of saved cycle, not a newly measured MTP acceptance rate.'})
    summary = {'started_unix': time.time(), 'completed': False, 'comparisons': [],
        'scope': 'Fixed inputs and DP routes, target-only generation; no speed claim or formal OFF workload grid.'}
    eos = set(read(Path('/data_lib/data/models/glm-q/config.json'))['eos_token_id'])
    for index, case in enumerate(cases):
        status('running', case=case['name'], case_index=index, cases=len(cases), progress={})
        def progress(_, fields):
            STATE['progress'] = fields
            STATE['updated_unix'] = time.time()
            write_json(OUT / 'target_only_status.json', STATE)
        http('/flush_cache', {}, timeout=60)
        record = capture(URL, case['body'], output / (case['name'] + '.json'),
                         {'index': index}, progress, timeout=900)
        assert not record['error'] and record['stream_done'], record['error']
        assert not record['meta'].get('spec_verify_ct'), 'target-only request performed speculative verification'
        assert record['meta'].get('dp_rank') == case['route'], 'actual DP route drifted'
        assert record['meta']['prompt_tokens'] == len(case['body']['input_ids'])
        assert record['meta']['completion_tokens'] == record['completion_tokens']
        expected = case['expected_ids']
        first_difference = next((i for i, (a, b) in enumerate(zip(expected, record['output_ids'])) if a != b), None)
        if first_difference is None and len(expected) != len(record['output_ids']):
            first_difference = min(len(expected), len(record['output_ids']))
        row = {key: value for key, value in case.items() if key not in ('body', 'expected_ids')}
        row.update(output_source=str(output / (case['name'] + '.json')),
            input_sha256=hashlib.sha256(json.dumps(case['body']['input_ids'], separators=(',', ':')).encode()).hexdigest(),
            completion_tokens=record['completion_tokens'], same_output_ids=expected == record['output_ids'],
            first_difference=first_difference, output_shape=record['output_shape'],
            finish_reason=record['meta'].get('finish_reason'),
            actual_dp_rank=record['meta']['dp_rank'], eos_positions=[i for i, token in enumerate(record['output_ids']) if token in eos])
        summary['comparisons'].append(row)
        write_json(output / 'summary.json', summary)
        print('COMPARISON', case['name'], 'same', row['same_output_ids'],
              'first difference', first_difference, 'shape', record['output_shape'], flush=True)
    summary.update(completed=True, ended_unix=time.time())
    write_json(output / 'summary.json', summary)
    status('generated', results=str(output / 'summary.json'))


if __name__ == '__main__':
    failure = None
    try:
        main()
    except Exception as error:
        failure = error
        status('needs_conservative_review', error=repr(error))
    finally:
        result = subprocess.run(['sudo', '-n', 'docker', 'inspect', NAME], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            container = json.loads(result.stdout)[0]
            assert container['Config']['Labels'].get('owner') == 'chengyetai'
            assert container['Config']['Labels'].get('experiment') == 'autonomous_mtp_chunk'
            if container['State']['Running']:
                subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/service.py'), 'stop', '--name', NAME], check=True, timeout=60)
        if failure is None:
            status('completed', completed=True, stopped=True)
    if failure is not None:
        raise failure
