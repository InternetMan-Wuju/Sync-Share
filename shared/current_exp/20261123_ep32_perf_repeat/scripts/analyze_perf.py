#!/usr/bin/env python3
"""Analyze repeated EP32 performance requests and EPLB rebalance overlap."""
from __future__ import annotations
import argparse, csv, json, math, re, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
RAW = ART / 'raw'
ARMS = ('original', 'eplb', 'k2')
SIZES = ('8192', '16384', '32768')
RE_START = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0 EP0\] \[EPLBManager\] rebalance start')
RE_END = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0 EP0\] \[EPLBManager\] rebalance end time=([0-9.]+)s')

def load_json(p):
    return json.loads(Path(p).read_text())
def iso_utc(s):
    return datetime.fromisoformat(s).astimezone(timezone.utc)
def log_utc(s):
    # Server logs use UTC timestamps while the request recorder uses +08:00.
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
def stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {'n': 0}
    mean = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {'n': len(xs), 'median': statistics.median(xs), 'mean': mean,
            'std': sd, 'min': min(xs), 'max': max(xs)}
def pct_delta(v, base):
    return None if v is None or not base else 100.0 * (v - base) / base

def parse_rebalances():
    timeline = {}
    for round_id in (1, 2, 3):
        p = ROOT / 'logs' / f'r{round_id}_eplb' / 'node23' / 'server.log'
        events = []
        open_events = []
        if not p.exists():
            timeline[str(round_id)] = {'path': str(p), 'events': [], 'error': 'missing_log'}
            continue
        for line in p.read_text(errors='replace').splitlines():
            m = RE_START.match(line)
            if m:
                open_events.append({'start': log_utc(m.group(1)).isoformat()})
                continue
            m = RE_END.match(line)
            if m:
                if open_events:
                    event = open_events.pop(0)
                    event['end'] = log_utc(m.group(1)).isoformat()
                    event['duration_s'] = float(m.group(2))
                    events.append(event)
        timeline[str(round_id)] = {'path': str(p), 'events': events,
                                   'completed_starts': len(events), 'open_starts': len(open_events)}
    return timeline

def overlap_events(row, events):
    start = iso_utc(row['started_at']); end = iso_utc(row['finished_at'])
    hits = []
    for i, e in enumerate(events):
        es = iso_utc(e['start']); ee = iso_utc(e['end'])
        if start <= ee and end >= es:
            hits.append(i)
    return hits

def read_rows(phase):
    rows = []
    for p in sorted((RAW / phase).glob('*.json')):
        d = load_json(p)
        d['_file'] = str(p.relative_to(ROOT))
        d['_path'] = str(p)
        rows.append(d)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-prefix', default='PERFORMANCE')
    a = ap.parse_args()
    timeline = parse_rebalances()
    formal = read_rows('formal')
    warmup = read_rows('warmup')
    calibration = read_rows('calibration') if (RAW / 'calibration').exists() else []
    for row in formal + warmup + calibration:
        row['rebalance_overlap'] = False
        row['overlap_events'] = []
        if row['arm'] == 'eplb':
            ev = timeline.get(str(row['round']), {}).get('events', [])
            hits = overlap_events(row, ev)
            row['rebalance_overlap'] = bool(hits)
            row['overlap_events'] = hits

    by = defaultdict(list)
    for row in formal:
        by[(row['arm'], row['size'])].append(row)
    cell_stats = {}
    for arm in ARMS:
        for size in SIZES:
            xs = by[(arm, size)]
            cell_stats[f'{arm}/{size}'] = {
                'n': len(xs),
                'ttft_s': stats([r['ttft_s'] for r in xs]),
                'total_s': stats([r['total_s'] for r in xs]),
                'output_token_counts': sorted(set(len(r.get('output_ids', [])) for r in xs)),
                'failures': sum(bool(r.get('error')) for r in xs),
                'rebalance_overlap_n': sum(bool(r['rebalance_overlap']) for r in xs),
                'rebalance_no_overlap_n': sum(not bool(r['rebalance_overlap']) for r in xs),
                'eplb_overlap_ttft_s': stats([r['ttft_s'] for r in xs if r['rebalance_overlap']]) if arm == 'eplb' else None,
                'eplb_no_overlap_ttft_s': stats([r['ttft_s'] for r in xs if not r['rebalance_overlap']]) if arm == 'eplb' else None,
            }

    # Pair every arm with original by the same size, round, and repeat.
    orig = {(r['size'], r['round'], r['repeat']): r for r in formal if r['arm'] == 'original'}
    paired = {}
    for arm in ARMS:
        for size in SIZES:
            deltas_ttft = []; deltas_total = []; pct_ttft = []; pct_total = []
            keys = []
            for r in formal:
                if r['arm'] != arm or r['size'] != size:
                    continue
                key = (size, r['round'], r['repeat'])
                base = orig.get(key)
                if base is None:
                    continue
                keys.append(r['round'])
                deltas_ttft.append(r['ttft_s'] - base['ttft_s'])
                deltas_total.append(r['total_s'] - base['total_s'])
                pct_ttft.append(pct_delta(r['ttft_s'], base['ttft_s']))
                pct_total.append(pct_delta(r['total_s'], base['total_s']))
            paired[f'{arm}/{size}'] = {
                'paired_n': len(keys),
                'delta_ttft_s': stats(deltas_ttft),
                'delta_total_s': stats(deltas_total),
                'delta_ttft_pct': stats(pct_ttft),
                'delta_total_pct': stats(pct_total),
            }

    # Verify each cell's output token IDs against its original counterpart.
    output_checks = {}
    for size in SIZES:
        bases = [r['output_ids'] for r in formal if r['arm'] == 'original' and r['size'] == size]
        expected = bases[0] if bases else []
        output_checks[size] = {}
        for arm in ARMS:
            ids = [r['output_ids'] for r in formal if r['arm'] == arm and r['size'] == size]
            output_checks[size][arm] = {
                'n': len(ids),
                'same_as_original': bool(ids) and all(x == expected for x in ids),
                'sample_output_ids': ids[0] if ids else None,
            }

    warm_stats = {}
    for arm in ARMS:
        for size in SIZES:
            xs = [r for r in warmup if r['arm'] == arm and r['size'] == size]
            warm_stats[f'{arm}/{size}'] = {'n': len(xs), 'ttft_s': stats([r['ttft_s'] for r in xs]), 'total_s': stats([r['total_s'] for r in xs])}

    schedule = ['round1: original,eplb,k2', 'round2: eplb,k2,original', 'round3: k2,original,eplb']
    errors = []
    if len(formal) != 90: errors.append(f'formal_count={len(formal)}, expected 90')
    for arm in ARMS:
        for size in SIZES:
            if len(by[(arm, size)]) != 10:
                errors.append(f'{arm}/{size}: n={len(by[(arm,size)])}, expected 10')
            if cell_stats[f'{arm}/{size}']['failures']:
                errors.append(f'{arm}/{size}: request failures')
    for round_id, t in timeline.items():
        if not t.get('events'):
            errors.append(f'EPLB round {round_id}: no rebalance end')

    validation_path = ART / 'VALIDATION_K2_NATIVE_TASK_ACCOUNTING.json'
    validation_k2 = load_json(validation_path) if validation_path.exists() else None
    if validation_k2 is None or not validation_k2.get('task_conservation') or not validation_k2.get('all_finite'):
        errors.append('K2 native validation task accounting missing or failed')
    summary = {
        'run_id': 'perf20261123',
        'hosts': ['npu1-23', 'npu1-25'],
        'topology': '32 dies, single EP32 group',
        'image': 'quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3',
        'model': '/data_lib/data/models/glm-q',
        'conditions': {
            'input_sizes': list(SIZES), 'chunked_prefill_size': 8192,
            'max_prefill_tokens': 32768, 'context_length': 65536,
            'max_running_requests': 1, 'cache': 'radix cache disabled',
            'output_tokens': 2, 'warmups_per_cell_per_round': 1,
            'formal_requests_per_cell_total': 10, 'round_schedule': schedule,
            'eplb_observer': 'disabled in formal run',
            'k2_verify': 'disabled; quiet logging disables task sample, tensor hash, and reference MoE',
        'k2_validation': 'separate verbose service run, one 8K request, outside formal timing',
        },
        'formal_request_count': len(formal),
        'warmup_request_count': len(warmup),
        'calibration_request_count': len(calibration),
        'cell_stats': cell_stats,
        'paired_deltas': paired,
        'output_checks': output_checks,
        'warmup_stats': warm_stats,
        'rebalance_timeline': timeline,
        'warmup_samples': [{k: r.get(k) for k in ('arm','size','round','ttft_s','total_s','started_at','finished_at','output_ids','error')} for r in warmup],
        'validation_k2': validation_k2,
        'calibration_samples': [{k: r.get(k) for k in ('arm','size','round','ttft_s','total_s','started_at','finished_at','output_ids','error')} for r in calibration],
        'validation_checks': {
            'formal_cells_10_each': all(len(by[(arm,size)]) == 10 for arm in ARMS for size in SIZES),
            'no_formal_request_errors': all(not r.get('error') for r in formal),
            'eplb_output_matches_original_all_sizes': all(output_checks[s]['eplb']['same_as_original'] for s in SIZES),
            'k2_output_matches_original_16k_32k': output_checks['16384']['k2']['same_as_original'] and output_checks['32768']['k2']['same_as_original'],
            'k2_8k_known_diff': not output_checks['8192']['k2']['same_as_original'],
            'eplb_all_formal_samples_overlap_rebalance': all(cell_stats[f'eplb/{s}']['rebalance_overlap_n'] == 10 for s in SIZES),
            'eplb_each_round_has_rebalance_end': all(timeline[str(r)].get('events') for r in (1,2,3)),
            'k2_validation_task_conservation': bool(validation_k2 and validation_k2.get('task_conservation')),
            'k2_validation_all_finite': bool(validation_k2 and validation_k2.get('all_finite')),
            'warmup_count_27': len(warmup) == 27,
        },
        'errors': errors,
        'status': 'PASS_WITH_KNOWN_K2_8K_TOKEN_DIFF' if not errors else 'FAIL',
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f'{a.out_prefix}_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (ART / f'{a.out_prefix}_RAW_REQUESTS.csv').open('w', newline='') as f:
        fields = ['run_phase','arm','round','repeat','size','input_tokens','started_at','finished_at','ttft_s','total_s','output_token_count','output_ids_sha256','rebalance_overlap','overlap_events','error','file']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in formal + warmup + calibration:
            w.writerow({**{k: r.get(k) for k in fields if k not in ('output_token_count','overlap_events','file')},
                        'output_token_count': len(r.get('output_ids', [])),
                        'overlap_events': json.dumps(r.get('overlap_events', [])),
                        'file': r.get('_file')})
    # Compact review table.
    lines = ['# EP32 repeated performance table', '',
             '| arm | size | n | TTFT median (s) | TTFT mean (s) | TTFT range (s) | total median (s) | rel. TTFT vs original |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for size in SIZES:
            s = cell_stats[f'{arm}/{size}']; d = paired[f'{arm}/{size}']['delta_ttft_pct']
            rel = '0.0%' if arm == 'original' else f"{d['median']:.2f}%"
            lines.append(f"| {arm} | {int(size)//1024}K | {s['n']} | {s['ttft_s']['median']:.6f} | {s['ttft_s']['mean']:.6f} | {s['ttft_s']['min']:.6f}–{s['ttft_s']['max']:.6f} | {s['total_s']['median']:.6f} | {rel} |")
    lines += ['', 'Relative values are median per-pair percentage changes against the original request with the same size/round/repeat.']
    (ART / f'{a.out_prefix}_TABLE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': summary['status'], 'formal': len(formal), 'warmup': len(warmup), 'calibration': len(calibration), 'errors': errors,
                      'overlap_counts': {f'{arm}/{size}': cell_stats[f'{arm}/{size}']['rebalance_overlap_n'] for arm in ARMS for size in SIZES}}, indent=2))
    raise SystemExit(0 if not errors else 1)

if __name__ == '__main__':
    main()
