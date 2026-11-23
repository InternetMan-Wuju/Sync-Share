#!/usr/bin/env python3
"""One timed request for the EP32 performance matrix."""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:18023')
ap.add_argument('--input', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--phase', required=True, choices=('warmup', 'formal', 'calibration', 'validation'))
ap.add_argument('--round', type=int, default=0)
ap.add_argument('--repeat', type=int, default=0)
ap.add_argument('--size', required=True)
ap.add_argument('--max-new-tokens', type=int, default=2)
a = ap.parse_args()
d = json.loads(a.input.read_text())
ids = d['input_ids']
payload = {
    'input_ids': ids,
    'sampling_params': {'max_new_tokens': a.max_new_tokens, 'temperature': 0, 'ignore_eos': True},
    'stream': True,
}
req = urllib.request.Request(
    a.url + '/generate',
    data=json.dumps(payload, separators=(',', ':')).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec='milliseconds')
t0 = time.perf_counter()
first = None
out_ids = []
text = ''
meta = None
events = 0
error = None
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b'data:'):
                continue
            body = raw[5:].strip()
            if body == b'[DONE]':
                break
            o = json.loads(body)
            events += 1
            meta = o.get('meta_info') or meta
            x = o.get('output_ids') or []
            if x:
                if first is None:
                    first = time.perf_counter()
                out_ids.extend(x)
            if o.get('text'):
                text = o['text']
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'
end = time.perf_counter()
finished_at = datetime.now(timezone.utc).astimezone().isoformat(timespec='milliseconds')
rec = {
    'run_phase': a.phase,
    'arm': a.arm,
    'round': a.round,
    'repeat': a.repeat,
    'size': a.size,
    'input_file': str(a.input),
    'input_tokens': len(ids),
    'input_sha256': d.get('input_sha256'),
    'max_new_tokens': a.max_new_tokens,
    'started_at': started_at,
    'finished_at': finished_at,
    'ttft_s': None if first is None else first - t0,
    'total_s': end - t0,
    'output_ids': out_ids,
    'output_ids_sha256': hashlib.sha256(json.dumps(out_ids, separators=(',', ':')).encode()).hexdigest(),
    'text': text,
    'events': events,
    'meta_info': meta,
    'error': error,
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: rec[k] for k in ('arm', 'run_phase', 'round', 'repeat', 'size', 'ttft_s', 'total_s', 'output_ids_sha256', 'error')}, indent=2))
if error:
    raise SystemExit(1)
