#!/usr/bin/env python3
"""Offline checks of saved counts, inputs, timestamps and comparison conditions."""
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK = ROOT.parent / 'Qchunksize'


def read(path):
    return json.loads(path.read_text())


def close(a, b, atol=1e-6):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=atol)


def request_check(record, mtp, expected_ids=None):
    meta = record['meta']
    errors = []
    checks = {
        'no_error': not record.get('error'),
        'stream_done': record.get('stream_done'),
        'output_count': len(record['output_ids']) == record['completion_tokens'] == meta.get('completion_tokens'),
        'input_count': record['input_tokens_sent'] == meta.get('prompt_tokens'),
    }
    payload_path = Path(record['payload_file'])
    if not payload_path.is_absolute():
        payload_path = Path('/home/chengyetai') / payload_path
    payload = read(payload_path)
    if expected_ids is not None:
        checks['same_dataset_ids'] = payload.get('input_ids') == expected_ids
    if payload.get('input_ids') is not None:
        ids = payload['input_ids']
        checks['payload_length'] = len(ids) == record['input_tokens_sent']
        digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        declared = record.get('input_ids_sha256') or record.get('input_sha256')
        if declared:
            checks['input_sha256'] = digest == declared
    steps = meta.get('spec_verify_ct')
    if mtp and steps:
        hist = meta.get('spec_correct_drafts_histogram')
        if hist is None:
            errors.append('missing acceptance histogram')
        else:
            checks['histogram_verifications'] = sum(hist) == steps
            correct = sum(i * count for i, count in enumerate(hist))
            checks['histogram_correct_drafts'] = correct == meta.get('spec_num_correct_drafts')
            proposed = meta.get('spec_num_proposed_drafts')
            checks['proposed_drafts'] = proposed == 3 * steps
            checks['reported_AL'] = close(meta.get('spec_accept_length'), record['completion_tokens'] / steps)
            checks['reported_rate'] = close(meta.get('spec_accept_rate'), correct / proposed)
            if record.get('pos_rates'):
                checks['reported_positions'] = all(close(a, sum(hist[i:]) / steps)
                                                    for i, a in enumerate(record['pos_rates'], 1))
    elif not mtp:
        checks['no_mtp_verifications'] = not steps
    spans = None
    names = ('request_received_ts', 'forward_entry_time', 'prefill_finished_time', 'request_finished_ts')
    stamps = [meta.get(name) for name in names]
    if all(value is not None for value in stamps):
        recv, entry, prefill, finish = stamps
        checks['timestamp_order'] = all(a <= b + 1e-6 for a, b in zip(stamps, stamps[1:]))
        server = finish - recv
        parts = [entry - recv, prefill - entry, finish - prefill]
        residual = record['e2e_s'] - server
        checks['server_e2e_conservation'] = close(sum(parts), server)
        checks['client_e2e_conservation'] = close(sum(parts) + residual, record['e2e_s'])
        spans = {'before_forward_s': parts[0], 'prefill_wall_s': parts[1], 'decode_and_finish_s': parts[2],
                 'server_e2e_s': server, 'client_residual_s': residual,
                 'residual_fraction': residual / record['e2e_s'] if record['e2e_s'] else None,
                 'reported_queue_s': meta.get('queue_time'), 'retractions': meta.get('num_retractions')}
    else:
        errors.append('missing stage timestamps')
    errors.extend(name for name, passed in checks.items() if not passed)
    return {'index': record['index'], 'checks': checks, 'errors': errors, 'spans': spans}


def main():
    mtp_groups = []
    for path in sorted((ROOT / 'results').glob('*/*/result.json')):
        data = read(path)
        info_path = path.parent.parent / 'server_info.json'
        if not info_path.exists() or not read(info_path).get('speculative_algorithm'):
            continue
        if any(v.get('failures') for v in data.get('validation', [])):
            continue  # Failed groups are separately retained in the result table.
        if any(r.get('error') or not r.get('stream_done') for r in data['requests']):
            continue
        requests = [request_check(record, True) for record in data['requests']]
        mtp_groups.append({'source': str(path), 'requests': requests,
                           'passed': all(not row['errors'] for row in requests)})
    chunk_cells = []
    for path in sorted((CHUNK / 'results/autonomous_20261012').glob('*/result.json')):
        data = read(path)
        dataset = read(Path(data['dataset']))
        info = read(path.parent / 'server_info.json')
        inputs = {row['index']: row['input_ids'] for row in dataset['cases']}
        requests = [request_check(row, data['mtp_on'], inputs[row['index']]) for row in data['requests']]
        agg = data['aggregate']
        checks = {'effective_chunk': info.get('chunked_prefill_size') == data['chunk'],
                  'mtp_off': not data['mtp_on'] and not info.get('speculative_algorithm'),
                  'all_requests_complete': agg.get('complete'),
                  'cache_condition': agg.get('cache_condition_valid'),
                  'dataset_sha256': hashlib.sha256(Path(data['dataset']).read_bytes()).hexdigest() == data['dataset_sha256'],
                  'aggregate_tokens': sum(r['completion_tokens'] for r in data['requests']) == agg['output_tokens'],
                  'aggregate_throughput': close(agg['total_output_tok_s'], agg['output_tokens'] / agg['wall_s'])}
        config_keys = ('version', 'tp_size', 'dp_size', 'ep_size', 'mem_fraction_static', 'max_running_requests',
                       'context_length', 'deepep_mode', 'enable_mixed_chunk', 'disable_cuda_graph',
                       'disable_shared_experts_fusion', 'max_prefill_tokens', 'speculative_algorithm')
        chunk_cells.append({'source': str(path), 'scene': data['name'], 'concurrency': data['concurrency'],
                            'chunk': data['chunk'], 'repeat': data.get('repeat', 1), 'checks': checks,
                            'requests': requests, 'dataset_sha256': data['dataset_sha256'],
                            'fixed_server_config': {k: info.get(k) for k in config_keys},
                            'passed': all(checks.values()) and all(not row['errors'] for row in requests)})
    consistency = []
    for scene, concurrency in sorted({(r['scene'], r['concurrency']) for r in chunk_cells}):
        rows = [r for r in chunk_cells if (r['scene'], r['concurrency']) == (scene, concurrency)]
        consistency.append({'scene': scene, 'concurrency': concurrency,
                            'same_inputs': len({r['dataset_sha256'] for r in rows}) == 1,
                            'same_service_except_chunk': len({json.dumps(r['fixed_server_config'], sort_keys=True) for r in rows}) == 1})
    report = {'updated_unix': time.time(), 'mtp_groups': mtp_groups, 'chunk_cells': chunk_cells,
              'comparison_consistency': consistency,
              'scope': 'Completed saved records only. Timestamp spans include scheduler gaps; conservation does not attribute residuals to an operator. No full-model accuracy claim.'}
    (ROOT / '数据验收与时间戳核对.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    invalid = [r['source'] for r in mtp_groups + chunk_cells if not r['passed']]
    invalid.extend(str(r) for r in consistency if not r['same_inputs'] or not r['same_service_except_chunk'])
    print(json.dumps({'mtp_groups': len(mtp_groups), 'chunk_cells': len(chunk_cells), 'invalid': invalid}, ensure_ascii=False))
    if invalid:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
