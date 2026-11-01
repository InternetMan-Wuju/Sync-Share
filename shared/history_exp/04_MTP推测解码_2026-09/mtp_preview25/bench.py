#!/usr/bin/env python3
"""Matched ON/OFF NEXTN preview: deterministic token IDs, streamed TTFT/TPOT, acceptance metrics."""
import argparse
import concurrent.futures
import json
import math
import random
import re
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'chunk_single23'))
from bench import request, flush, ids


def fetch_metrics(url):
    with urllib.request.urlopen(url + '/metrics', timeout=8) as r:
        return r.read().decode('utf-8', 'replace')


def filtered_metrics(text):
    result = {}
    for line in text.splitlines():
        if not line.startswith('sglang:') or 'spec' not in line.lower():
            continue
        m = re.match(r'^(sglang:[^\s]+)\s+([-+\d.eE]+)$', line)
        if not m or ('tp_rank=' in m[1] and 'tp_rank="0"' not in m[1]):
            continue
        result[m[1]] = float(m[2])
    return result


class Sampler:
    def __init__(self, url, path):
        self.url, self.path, self.stop = url, path, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        with self.path.open('a', encoding='utf-8', buffering=1) as f:
            while not self.stop.is_set():
                try:
                    data = filtered_metrics(fetch_metrics(self.url))
                    f.write(json.dumps({'at': time.time(), 'spec': data}) + '\n')
                except Exception as exc:
                    f.write(json.dumps({'at': time.time(), 'error': repr(exc)}) + '\n')
                self.stop.wait(3)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)


def measure(values):
    values = sorted(v for v in values if isinstance(v, (float, int)))
    if not values:
        return None
    return {'mean': statistics.mean(values), 'p50': values[len(values)//2],
            'p95': values[math.ceil(.95*len(values))-1], 'min': min(values), 'max': max(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://10.120.72.25:30081')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--mode', choices=['on', 'off'], required=True)
    p.add_argument('--plan', choices=['preview', 'full'], default='preview')
    p.add_argument('--reps', type=int, default=2)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--scenario', choices=['all', 'decode', 'mixed', 'long'], default='all')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out_dir/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    rng = random.Random(20260926)
    prefix = ids(rng, 19661)  # round(32768 * .60); flushed and primed per request.
    cases = [
        ('decode', 1024, 32768 if args.plan == 'full' else 2048, 0),
        ('mixed', 8192, 1024, 0),
        ('long', 32768, 300, len(prefix)),
    ]
    logpath = args.out_dir/'requests.jsonl'
    def save(stage, row):
        item = {'scenario': stage, 'mode': args.mode, 'plan': args.plan, 'timestamp': time.time(), **row}
        with logpath.open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"{stage:7} {row.get('role')} out={row.get('output_tokens')} ttft={row.get('ttft_s')} tpot_ms={row.get('tpot_ms')} cache={row.get('cached_tokens')} err={row.get('error')}", flush=True)
        return item

    all_results = []
    with Sampler(args.url, args.out_dir/'spec_metrics_samples.jsonl'):
        for name, inp, out, pref in cases:
            if args.scenario != 'all' and name != args.scenario:
                continue
            stage_results = []
            for r in range(args.reps):
                stage = f'{name}_{r}'
                save(stage, {'role': 'flush', **flush(args.url, args.timeout)})
                if pref:
                    warm = save(stage, request(args.url, prefix, 1, args.timeout, 'prefix_prime'))
                    if warm.get('error'):
                        raise RuntimeError('prefix prime failed')
                prompt = (prefix + ids(rng, inp-pref)) if pref else ids(rng, inp)
                before = fetch_metrics(args.url)
                (args.out_dir/f'{stage}_metrics_before.txt').write_text(before)
                start = time.perf_counter()
                row = save(stage, request(args.url, prompt, out, args.timeout, 'measurement'))
                wall = time.perf_counter() - start
                after = fetch_metrics(args.url)
                (args.out_dir/f'{stage}_metrics_after.txt').write_text(after)
                row['wall_s'] = wall
                row['throughput_output_tok_s'] = row.get('output_tokens', 0) / wall if not row.get('error') else None
                row['spec_metrics_after'] = filtered_metrics(after)
                row['spec_metrics_before'] = filtered_metrics(before)
                stage_results.append(row)
                all_results.append(row)
                if row.get('error'):
                    break
            summary = {'scenario': name, 'input_tokens': inp, 'output_tokens': out,
                       'shared_prefix_tokens': pref, 'target_shared_ratio': pref / inp,
                       'n': len(stage_results),
                       'errors': [row['error'] for row in stage_results if row.get('error')],
                       'ttft_s': measure([row.get('ttft_s') for row in stage_results]),
                       'tpot_ms': measure([row.get('tpot_ms') for row in stage_results]),
                       'output_tok_s': measure([row.get('throughput_output_tok_s') for row in stage_results]),
                       'spec_accept_length': measure([row.get('meta_info', {}).get('spec_accept_length') for row in stage_results]),
                       'spec_accept_rate': measure([row.get('meta_info', {}).get('spec_accept_rate') for row in stage_results]),
                       'spec_verify_ct': sum(row.get('meta_info', {}).get('spec_verify_ct', 0) for row in stage_results),
                       'spec_num_correct_drafts': sum(row.get('meta_info', {}).get('spec_num_correct_drafts', 0) for row in stage_results),
                       'spec_num_proposed_drafts': sum(row.get('meta_info', {}).get('spec_num_proposed_drafts', 0) for row in stage_results),
                       'spec_histograms': [row.get('meta_info', {}).get('spec_correct_drafts_histogram') for row in stage_results],
                       'actual_cache_ratio': measure([row.get('cached_tokens', 0) / row['prompt_tokens'] for row in stage_results
                         if isinstance(row.get('cached_tokens'), int) and row.get('prompt_tokens')]),
                       'spec_metrics_after': [row.get('spec_metrics_after') for row in stage_results]}
            (args.out_dir/f'{name}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    print('complete', flush=True)

if __name__ == '__main__':
    main()
