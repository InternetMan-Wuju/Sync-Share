#!/usr/bin/env python3
"""Read saved pinned-DP prefix probes; never send new inference requests."""
from report_labels import humanize_report
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')


def main():
    runs = []
    for path in sorted((ROOT / 'results').rglob('summary.json')):
        data = json.loads(path.read_text())
        probes = data.get('mismatch_prefill_rechecks')
        if not probes:
            continue
        grouped = defaultdict(list)
        for record in probes:
            row = dict(record)
            name = f"{row['group']}_{row['index']}.mismatch{row['position']}.dp{row.get('routed_dp_rank')}.trial{row['trial']}.response.json"
            response_path = path.parent / name
            if response_path.exists():
                meta = json.loads(response_path.read_text()).get('meta_info', {})
                row['actual_dp_rank'] = meta.get('dp_rank')
                row['output_top_logprobs'] = meta.get('output_top_logprobs')
                row['response_source'] = str(response_path)
            grouped[(row['group'], row['index'], row['position'])].append(row)
        points = []
        for key, rows in grouped.items():
            routes = {}
            for route in sorted({r['routed_dp_rank'] for r in rows if r.get('routed_dp_rank') is not None}):
                records = [r for r in rows if r.get('routed_dp_rank') == route and not r.get('error')]
                ids = [r['token'] for r in records]
                routes[str(route)] = {'tokens': ids, 'stable': len(ids) >= 2 and len(set(ids)) == 1,
                                      'route_honored': all(r.get('actual_dp_rank') == route for r in records)}
            stable = [r['tokens'][0] for r in routes.values() if r['stable'] and r['route_honored']]
            points.append({'group': key[0], 'index': key[1], 'position': key[2], 'routes': routes,
                           'stable_dp_comparison_valid': len(stable) >= 2,
                           'stable_dp_predictions_differ': len(set(stable)) > 1 if len(stable) >= 2 else None,
                           'probes': rows})
        runs.append({'source': str(path), 'completed': data.get('completed'), 'points': points})
    report = {'runs': runs, 'scope': 'Diagnostic positions were selected at bulk-prefill mismatches. This is not a random acceptance sample or full-model accuracy test. Stable differing DP predictions establish a route-associated difference, not its kernel cause.'}
    (ROOT / '固定DP前缀核对.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ['# 固定DP的相同前缀核对', '', report['scope'], '',
             '| 轮次 | 输入/位置 | DP0两次token | DP1两次token | 固定路由预测不同 |',
             '|---|---|---|---|---|']
    for run in runs:
        for point in run['points']:
            routes = point['routes']
            outcome=point['stable_dp_predictions_differ'] if point['stable_dp_comparison_valid'] else '未具备两个稳定DP的对照'
            lines.append(f"| {Path(run['source']).parents[1].name} | {point['group']} #{point['index']} pos{point['position']} | {routes.get('0', {}).get('tokens', [])} | {routes.get('1', {}).get('tokens', [])} | {outcome} |")
    lines += ['', '每次请求都先清缓存，温度0，max_new_tokens=1；实际返回dp_rank与要求的路由一起验收。所有top2、payload及原始响应均保存。',
              'DP间不同不自动等于权重字节不同。量化、attention TP/归约、通信与前向路径仍需独立隔离，不能从这里指定某个kernel为根因。', '']
    (ROOT / '固定DP前缀核对.md').write_text(humanize_report('\n'.join(lines)))
    print('prefix audit runs', len(runs), 'stable differing points', sum(p['stable_dp_predictions_differ'] is True for r in runs for p in r['points']))


if __name__ == '__main__':
    main()
