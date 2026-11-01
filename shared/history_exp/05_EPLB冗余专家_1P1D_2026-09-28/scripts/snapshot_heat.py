#!/usr/bin/env python3
"""Preserve six in-flight expert heat snapshots and reproducible PNGs.

EPLB-on: read the latest automatic rebalance window without touching the recorder.
EPLB-off: stop/dump/restart the recorder for independent five-minute windows.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
import urllib.request

ROOT = Path('/home/chengyetai/workspace/current_exp/eplb_pd_agentx_20260928')
p = argparse.ArgumentParser()
p.add_argument('--session', required=True)
p.add_argument('--group', required=True)
p.add_argument('--decode-source-group', default=None)
p.add_argument('--mode', choices=['on', 'off'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--stop-file', type=Path, required=True)
p.add_argument('--marks', type=int, nargs='+', default=[60, 150, 240, 330, 420, 510, 570])
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

start = time.monotonic()
log = (a.out / 'snapshots.jsonl').open('a', buffering=1)
sources = {23: set(), 25: set()}

def emit(**record):
    log.write(json.dumps({'at': dt.datetime.now().astimezone().isoformat(), **record}, ensure_ascii=False) + '\n')

def url_call(node, action):
    req = urllib.request.Request(f'http://10.120.72.{node}:18423/{action}', method='POST')
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read().decode('utf-8', errors='replace').strip()

def latest_pt(node, source_group, filename_glob, dest):
    sub = ROOT / 'results' / a.session / source_group / f'node_{node}' / 'eplb_windows'
    if node == 23:
        files = sorted(sub.glob(filename_glob), key=lambda f: f.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f'{sub}/{filename_glob}')
        source = files[-1]
        shutil.copy2(source, dest)
        return str(source)
    # 25 has a different local disk; fetch only the newest full recorder dump.
    cmd = f"find {sub} -maxdepth 1 -type f -name '{filename_glob}' -printf '%T@ %p\\n' | sort -nr | head -1"
    found = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=10', '10.120.72.25', cmd], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    if not found:
        raise FileNotFoundError(f'{sub}/{filename_glob}')
    source = found.split(' ', 1)[1]
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o LogLevel=ERROR', f'10.120.72.25:{source}', str(dest)], timeout=60, check=True, capture_output=True)
    return source

def render(pt, png, node, minute):
    record = torch.load(pt, map_location='cpu', weights_only=False)
    data = record['logical_count']
    if not isinstance(data, torch.Tensor):
        raise ValueError('logical_count is not a tensor')
    if data.numel() == 0 or not bool(data.any()):
        raise ValueError('empty expert heat window; refusing zero heat snapshot')
    # steps x layers x logical experts: both rank placement and the raw count
    # tensor remain in .pt; the image shows logical-expert traffic per layer.
    counts = data.numpy()
    if counts.ndim == 3:
        counts = counts.sum(axis=0)
    if counts.ndim != 2:
        raise ValueError(f'unexpected logical_count shape: {tuple(data.shape)}')
    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(np.log1p(counts), aspect='auto', interpolation='nearest', cmap='inferno', origin='lower')
    fig.colorbar(im, ax=ax, label='log(1 + routed token count)')
    ax.set(xlabel='Logical expert ID', ylabel='MoE layer index', title=f'{a.group} / node {node} / t≈{minute} min / {counts.shape[0]} layers × {counts.shape[1]} experts')
    fig.tight_layout()
    fig.savefig(png, dpi=145)
    plt.close(fig)
    return {'steps': int(data.shape[0]) if data.ndim == 3 else None, 'layers': int(counts.shape[0]), 'logical_experts': int(counts.shape[1]), 'tokens_routed_total': int(counts.sum()), 'window_kind': record.get('window_kind'), 'dump_timestamp': record.get('dump_timestamp'), 'rebalance_index': record.get('full_window_dump_index')}

for at_s in a.marks:
    while time.monotonic() - start < at_s:
        if a.stop_file.exists():
            break
        time.sleep(min(5, at_s - (time.monotonic() - start)))
    if a.stop_file.exists():
        emit(mark_seconds=at_s, status='stopped_before_mark')
        break
    for node in (23, 25):
        role = 'prefill' if node == 23 else 'decode'
        source_group = a.group if node == 23 else (a.decode_source_group or a.group)
        tag = f'{at_s//60:02d}m_{role}{node}'
        pt, png = a.out / f'{tag}.pt', a.out / f'{tag}.png'
        try:
            actions = []
            if a.mode == 'off':
                try:
                    actions.append(['stop', url_call(node, 'stop_expert_distribution_record')])
                    actions.append(['dump', url_call(node, 'dump_expert_distribution_record')])
                finally:
                    actions.append(['start', url_call(node, 'start_expert_distribution_record')])
            source = latest_pt(node, source_group,
                               'expert_distribution_window_*.pt' if a.mode == 'on' else 'expert_distribution_recorder_*.pt', pt)
            stats = render(pt, png, node, at_s // 60)
            sources[node].add(source)
            emit(mark_seconds=at_s, node=node, mode=a.mode, source=source, raw=str(pt), image=str(png), raw_bytes=pt.stat().st_size, actions=actions, status='ok', **stats)
        except Exception as exc:
            emit(mark_seconds=at_s, node=node, mode=a.mode, status='error', error=repr(exc))
emit(status="final_coverage", unique_sources={str(node): len(found) for node, found in sources.items()})
log.close()
if min(map(len, sources.values())) < 2:
    print("fewer than two distinct in-flight heat snapshots for a role", file=sys.stderr)
    sys.exit(2)
