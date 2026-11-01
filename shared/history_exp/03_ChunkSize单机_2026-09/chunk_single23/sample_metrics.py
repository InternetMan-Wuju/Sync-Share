#!/usr/bin/env python3
"""Record rank-0 KV/cache and decode counters during mixed load at low sampling rate."""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

NAMES = {
    'max_total_num_tokens', 'kv_available_tokens', 'kv_evictable_tokens',
    'kv_used_tokens', 'num_running_reqs', 'num_queue_reqs', 'gen_throughput',
    'cache_hit_rate', 'token_usage', 'prefill_effective_tokens_total',
    'realtime_tokens_total', 'prompt_tokens_total', 'generation_tokens_total',
}
PATTERN = re.compile(r'^sglang:(\w+)(\{[^}]*\})? ([\d.eE+\-]+)$')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://10.120.72.23:30080')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--interval', type=float, default=3)
    ap.add_argument('--duration', type=int, default=3600)
    ap.add_argument('--stop-file', type=Path)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    since = time.monotonic()
    misses = 0
    with a.out.open('a', encoding='utf-8', buffering=1) as f:
        while time.monotonic() - since < a.duration:
            if a.stop_file and a.stop_file.exists():
                break
            row = {'at': time.time()}
            try:
                with urllib.request.urlopen(a.url + '/metrics', timeout=5) as resp:
                    for line in resp:
                        if not line.startswith(b'sglang:'):
                            continue
                        text = line.decode('utf-8', 'replace').strip()
                        match = PATTERN.match(text)
                        if not match or match.group(1) not in NAMES:
                            continue
                        labels = match.group(2) or ''
                        if 'tp_rank=' in labels and 'tp_rank="0"' not in labels:
                            continue
                        name = match.group(1)
                        mode = re.search(r'mode="([^"]+)"', labels)
                        if mode:
                            name += '_' + mode.group(1)
                        row[name] = float(match.group(3))
                misses = 0
            except Exception as exc:
                misses += 1
                row['error'] = repr(exc)
            f.write(json.dumps(row) + '\n')
            if misses >= 5:
                break
            time.sleep(a.interval)
    print(f'metrics_samples={a.out} duration_s={time.monotonic()-since:.1f}', flush=True)

if __name__ == '__main__':
    main()
