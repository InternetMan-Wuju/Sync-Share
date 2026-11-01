#!/usr/bin/env python3
"""Audit only autonomous results against the four AboutModels workloads."""
from report_labels import humanize_report
import csv
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
REQUIREMENTS = ROOT.parent / 'docs/AboutModels.md'
OUTPUT = ROOT / 'MTP自主探索四组覆盖'
WORKLOADS = (
    ('w1_8k_1k', '8K / 1K', 8192, 1024, 2, None),
    ('w2_32k_1k', '32K / 1K', 32768, 1024, 1, None),
    ('w3_1k_32k', '1K / 32K', 1024, 32768, 1, None),
    ('w4_prefix32k_300', '32K / 300，缓存60%', 32768, 300, 2, .6),
)


def read(path):
    return json.loads(path.read_text())


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def overview_lines(report):
    lines = [
        f"**四组覆盖：自主探索的标准长度ON覆盖{report['on_workload_shapes_covered']}/4，OFF性能基线覆盖{report['off_performance_workloads_covered']}/4。四组ON/OFF记录{'齐全' if report['all_four_on_off_complete'] else '尚未齐全'}；配置对齐也需单独核验。**", '',
        '| AboutModels要求 | 自主探索ON | OFF性能基线 | 实际并发/请求 |',
        '|---|---|---|---|',
    ]
    for workload in report['workloads']:
        runs = workload['on_results']
        status = ('规定长度已跑完，服务配置有偏离' if runs else
                  '未跑；1K/1024只作短诊断' if workload['name'] == 'w3_1k_32k' else '未跑')
        counts = '、'.join(f"{r['concurrency']}/{r['requests']}" for r in runs) or '—'
        lines.append(f"| {workload['label']} | {status} | {'已跑' if workload['off_results'] else '未跑'} | {counts} |")
    lines += ['', '三组标准长度测试使用0.5.19版和公开源恢复权重：数据并行1路，关闭DP attention/LM head，DeepEP常规通信，关闭计算图；不能写成AboutModels默认配置四组已完成。',
              '1K/1024短诊断的输出长度仅为要求32K的1/32。后续6次OFF用于输出正确性检查，未形成四组性能基线；ChunkSize的OFF矩阵也未作为MTP开关配对。',
              '45组有效新增记录包含重复配置和诊断变体，不能用记录数代表四组场景覆盖。完整统计与原始结果见[自主探索四组覆盖核对](MTP自主探索四组覆盖.md)。', '']
    return lines


def main():
    records = []
    for path in sorted((ROOT / 'results').glob('*/*/result.json')):
        data = read(path)
        requests = data.get('requests', [])
        if not requests or any(r.get('error') or not r.get('stream_done') for r in requests):
            continue
        if any(v.get('failures') for v in data.get('validation', [])):
            continue
        info = read(path.parent.parent / 'server_info.json')
        records.append((path, data, info))
    workloads = []
    for name, label, input_length, output_length, multiplier, target_cache in WORKLOADS:
        workload = dict(name=name, label=label, input_tokens=input_length, output_tokens=output_length,
                        request_multiplier=multiplier, target_cache_ratio=target_cache,
                        on_results=[], off_results=[])
        for path, data, info in records:
            if path.parent.name != name:
                continue
            requests = data['requests']
            assert len(requests) == multiplier * data['concurrency'], path
            assert data['target_output_tokens'] == output_length and data['ignore_eos'], path
            for row in requests:
                assert row['input_tokens_sent'] == row['meta']['prompt_tokens'] == input_length, path
                assert row['completion_tokens'] == len(row['output_ids']) == output_length, path
                assert row['sampling_params']['max_new_tokens'] == output_length, path
            total_prompt = sum(r['meta']['prompt_tokens'] for r in requests)
            cache_ratio = sum(r['meta'].get('cached_tokens', 0) for r in requests) / total_prompt
            if target_cache is not None:
                assert abs(cache_ratio - target_cache) < .01, path
            verify = sum(r['meta'].get('spec_verify_ct', 0) for r in requests)
            hist = [0] * 4
            for row in requests:
                for index, count in enumerate(row['meta'].get('spec_correct_drafts_histogram') or []):
                    hist[index] += count
            positions = [sum(hist[i:]) / verify for i in (1, 2, 3)] if verify else None
            aggregate = data['aggregate']
            on = bool(info.get('speculative_algorithm'))
            result = dict(run=path.parent.parent.name, source=str(path), mode='ON' if on else 'OFF',
                          concurrency=data['concurrency'], requests=len(requests), cache_hit_ratio=cache_ratio,
                          AL=sum(r['completion_tokens'] for r in requests) / verify if verify else None,
                          P1_P2_P3=positions,
                          output_tok_s=sum(r['completion_tokens'] for r in requests) / aggregate['wall_s'],
                          TTFT_s=aggregate.get('ttft_s_mean'), TPOT_ms=aggregate.get('tpot_ms_mean'),
                          E2E_s=aggregate.get('e2e_s_mean'),
                          server_config={k: info.get(k) for k in (
                              'version', 'tp_size', 'ep_size', 'dp_size', 'enable_dp_attention',
                              'enable_dp_lm_head', 'deepep_mode', 'disable_cuda_graph',
                              'context_length', 'max_running_requests', 'chunked_prefill_size')})
            workload['on_results' if on else 'off_results'].append(result)
        workloads.append(workload)
    state_path = ROOT / 'review_20261013/target_only_status.json'
    diagnostic_state = read(state_path) if state_path.exists() else None
    report = dict(updated=datetime.datetime.now().isoformat(timespec='seconds'),
                  scope='Only results created under autonomous_20261012; no historical result read or counted.',
                  requirements=str(REQUIREMENTS), requirements_sha256=hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
                  workloads=workloads, on_workload_shapes_covered=sum(bool(w['on_results']) for w in workloads),
                  off_performance_workloads_covered=sum(bool(w['off_results']) for w in workloads),
                  all_four_on_complete=all(w['on_results'] for w in workloads),
                  all_four_on_off_complete=all(w['on_results'] and w['off_results'] for w in workloads),
                  target_only_diagnostic_status=diagnostic_state,
                  cache_tolerance='60% checked within one percentage point; actual measured ratio retained.')
    OUTPUT.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ['# MTP自主探索：四组要求覆盖核对', '', f"更新时间：{report['updated']}。只统计本自主探索目录。", '']
    lines += overview_lines(report)
    lines += ['## 三组标准长度ON的现有数值', '',
              '吞吐为整套服务的总输出/整组墙钟；TTFT、TPOT和E2E列请求均值。单次小样本描述，不是ON/OFF速度收益。', '',
              '| 场景 | AL | P1/P2/P3 | 总输出tok/s | TTFT s | TPOT ms | E2E s | 实测缓存命中率 | 原始结果 |',
              '|---|---|---|---|---|---|---|---|---|']
    flat = []
    for workload in workloads:
        runs = workload['on_results']
        if not runs:
            lines.append(f"| {workload['label']} | 未测 | — | — | — | — | — | — | — |")
        for result in runs:
            positions = '/'.join(number(p, 4) for p in result['P1_P2_P3'] or []) or '—'
            lines.append(f"| {workload['label']} | {number(result['AL'])} | {positions} | {number(result['output_tok_s'])} | {number(result['TTFT_s'])} | {number(result['TPOT_ms'])} | {number(result['E2E_s'])} | {result['cache_hit_ratio']:.4%} | [原始结果]({result['source']}) |")
            flat.append({'workload': workload['name'], **result})
    lines += ['', '## 当前缺项', '',
              '1. 第3组完整1K/32K的MTP ON记录。',
              '2. 四组在共同可运行配置下的MTP OFF性能基线；已有三组ON若更换配置，也需重跑ON才能配对。',
              '3. 服务配置与AboutModels默认值的明确偏离说明。当前三组ON使用DP1；后续DP2短诊断不能回填为这三组的DP2性能。', '',
              '原有总控completed只说明当时编排的阶段已结束，不证明用户要求的四组全部完成。权重/接受率探索已有结论，四组性能任务仍未完成。', '']
    OUTPUT.with_suffix('.md').write_text(humanize_report('\n'.join(lines)))
    if flat:
        with OUTPUT.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in flat)
    # Refresh the coverage notice without rereading historical weight comparisons.
    for table, heading in (
        (ROOT / 'MTP自主探索实测.md', '# 自主MTP探索：阶段实测结果（四组要求未完成）'),
        (ROOT / '自主实验最终总结.md', '# MTP与ChunkSize自主探索阶段总结（MTP四组要求未完成）'),
    ):
        if not table.exists():
            continue
        text = table.read_text()
        begin, end = '<!-- mtp-workload-coverage:start -->', '<!-- mtp-workload-coverage:end -->'
        if begin in text:
            start, finish = text.index(begin), text.index(end) + len(end)
            text = text[:start] + text[finish:].lstrip('\n')
        title, _, body = text.partition('\n')
        text = heading + '\n\n' + begin + '\n'
        text += '\n'.join(overview_lines(report)) + '\n' + end + '\n' + body.lstrip('\n')
        table.write_text(humanize_report(text))
    summary_path = ROOT / '自主实验最终总结.json'
    if summary_path.exists():
        summary = read(summary_path)
        summary['mtp_workload_coverage'] = {
            key: report[key] for key in ('on_workload_shapes_covered', 'off_performance_workloads_covered',
                                        'all_four_on_complete', 'all_four_on_off_complete')}
        summary['mtp_workload_coverage']['report'] = str(OUTPUT.with_suffix('.md'))
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('autonomous workload coverage:', report['on_workload_shapes_covered'], '/4 ON;',
          report['off_performance_workloads_covered'], '/4 OFF performance baselines')


if __name__ == '__main__':
    main()
