#!/usr/bin/env python3
"""Summarize logged TP0 prefill shapes within saved measurement windows."""
from report_labels import humanize_report
import collections
import datetime
import json
import math
import re
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK = ROOT.parent / 'Qchunksize'
HEADER = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0 EP0\] (Prefill|Decode) batch, (.*)')
FIELD = re.compile(r'#([a-z-]+): ([0-9]+)')


def main():
    owners = {}
    for run in json.loads((ROOT / 'overnight_status.json').read_text()).get('runs', []):
        for cell in run.get('cells', []):
            owners[cell['name']] = run['container']
    repeat = ROOT / 'chunk_repeat_status.json'
    if repeat.exists():
        for run in json.loads(repeat.read_text()).get('runs', []):
            owners[run['cell']] = run['container']
    parsed = {}
    rows = []
    for path in sorted((CHUNK / 'results/autonomous_20261012').glob('*/result.json')):
        data = json.loads(path.read_text())
        name = path.parent.name
        container = owners.get(name)
        if not container:
            continue
        if container not in parsed:
            events = []
            for line in (ROOT / 'logs' / container / 'server.log').read_text(errors='replace').splitlines():
                match = HEADER.search(line)
                if match:
                    stamp = datetime.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()
                    events.append({'unix': stamp, 'phase': match[2], 'fields': {k: int(v) for k,v in FIELD.findall(match[3])},
                                   'graph': 'npu graph: True' in match[3]})
            parsed[container] = events
        started = min(r['started_unix'] for r in data['requests'])
        ended = max(r['ended_unix'] for r in data['requests'])
        events = [r for r in parsed[container] if math.floor(started) <= r['unix'] <= math.ceil(ended)]
        prefills = [r for r in events if r['phase'] == 'Prefill']
        decodes = [r for r in events if r['phase'] == 'Decode']
        interior = [r for r in prefills if math.ceil(started) <= r['unix'] < math.floor(ended)]
        boundary = [r for r in prefills if r not in interior]
        sizes = collections.Counter(r['fields']['new-token'] for r in prefills)
        expected = sum(r['meta']['prompt_tokens'] - r['meta'].get('cached_tokens',0) for r in data['requests'])
        observed = sum(r['fields']['new-token'] for r in prefills)
        lower = sum(r['fields']['new-token'] for r in interior)
        rows.append({'cell': name, 'container': container, 'source': str(path),
                     'measurement_start_unix': started, 'measurement_end_unix': ended,
                     'prefill_log_records': len(prefills), 'new_token_size_counts': dict(sorted(sizes.items())),
                     'interior_prefill_records': len(interior), 'boundary_prefill_records': len(boundary),
                     'logged_new_token_lower_bound': lower,
                     'logged_new_token_sum': observed, 'request_uncached_prompt_tokens': expected,
                     'sum_matches_request_counts': observed == expected,
                     'request_count_within_log_bounds': lower <= expected <= observed,
                     'prefill_logs_with_running_decode_reqs': sum(r['fields'].get('running-req',0)>0 for r in prefills),
                     'sampled_decode_logs': len(decodes),
                     'sampled_decode_graph_true': sum(r['graph'] for r in decodes)})
    scope = ('Only TP0/EP0 log records in the measured request wall window; timestamps have one-second resolution. '
             'Boundary records may be ambiguous; an equal token sum is a consistency check, not an operator trace. '
             'Decode logs are sampled, and running-req describes scheduling state rather than concurrent kernels.')
    (CHUNK / 'ChunkSize批次形状证据.json').write_text(json.dumps({'scope': scope, 'cells': rows}, ensure_ascii=False, indent=2))
    lines = ['# ChunkSize实际批次形状', '', scope, '',
             '| 单元 | Prefill日志内区+边界候选 | 新token数:候选出现次数 | 日志新token下界～上界 / 请求未缓存输入 | 请求输入在日志界内 | Decode采样Graph为真/总数 |',
             '|---|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['cell']} | {row['interior_prefill_records']}+{row['boundary_prefill_records']} | {row['new_token_size_counts']} | {row['logged_new_token_lower_bound']}～{row['logged_new_token_sum']} / {row['request_uncached_prompt_tokens']} | {row['request_count_within_log_bounds']} | {row['sampled_decode_graph_true']}/{row['sampled_decode_logs']} |")
    lines += ['', '日志只有秒级时间，不能精确剔除与正式请求同秒的前缀预热或下一组形状预热。内区不含首尾有歧义秒，边界另列候选；不把候选总数叫正式总数。与阶段跨度结合使用：更多小前向不必然更慢，少量大前向也不必然更快。不凭此表反推算子耗时。', '']
    (CHUNK / 'ChunkSize批次形状证据.md').write_text(humanize_report('\n'.join(lines)))
    print('chunk batch-log evidence', len(rows), 'cells;', sum(r['sum_matches_request_counts'] for r in rows), 'token sums equal')


if __name__ == '__main__':
    main()
