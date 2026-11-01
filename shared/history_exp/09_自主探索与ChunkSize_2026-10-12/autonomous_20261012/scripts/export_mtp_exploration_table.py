#!/usr/bin/env python3
"""Export completed MTP exploration groups with performance and provenance."""
import csv
import datetime
import hashlib
import json
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

from report_labels import service_label, workload_label

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'MTP自主探索全量性能表'
CONFIGS = {
    'publisher_v19': '0.5.19单路',
    'mtp-v16-auto-cap64-mem87-hccl650': '旧版双路87%',
    'mtp-v16-auto-cap64-mem90-hccl650': '旧版双路90%',
    'mtp-v19-dp2-normal': '0.5.19双路原断言',
    'mtp-v19-dp2-ordered-probe': '0.5.19双路调整检查',
    'mtp-v19-dp2-no-plan-lp-safe': '0.5.19双路关调度重叠',
    'mtp-v19-dp2-no-multistream': '0.5.19双路再关多流',
    'w3_supplement_20261013': '0.5.19单路（W3补跑）',
}
TASKS = {
    'native_exact_1k256': '随机ID直发，强制输出',
    'native_exact_eos_1k256': '随机ID直发，允许结束',
    'native_default_text_roundtrip_256': '随机ID转文本再分词',
    'native_longinput_1k1024': '随机ID，长输出短诊断',
    'native_random_body_chatframed_256': '随机正文加聊天模板',
    'framing_pair_raw_eos256': '配对随机正文，直接续写',
    'framing_pair_chat_eos256': '配对随机正文，加聊天边界',
    'sharegpt_chat_eos': '真实聊天短窗口，允许结束',
    'sharegpt_chat_forced': '真实聊天短窗口，强制输出',
    'sharegpt_chat_complete': '真实聊天完整回复',
    'sanity_chat': '问候与算术功能检查',
}


def read(path):
    return json.loads(path.read_text())


def mean(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return statistics.mean(values) if values else None


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def percentage(value):
    return '—' if value is None else f'{value:.2%}'


def render_table(rows):
    lines = [
        '| 服务配置 | 测试内容 | 输入 → 输出上限 | 实际输出（逐请求） | 并发 / 请求 | AL | 草稿接受率 | P1 / P2 / P3 | 总吞吐token/s | TTFT秒 | TPOT毫秒 | E2E秒 | 周期尾段请求数 | 来源 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for row in rows:
        inputs = '、'.join(str(v) for v in row['input_lengths'])
        outputs = ('全部' + str(row['output_lengths'][0]) if len(set(row['output_lengths'])) == 1
                   else '、'.join(str(v) for v in row['output_lengths']))
        task = TASKS.get(row['group'], workload_label(row['group']))
        positions = ' / '.join(percentage(v) for v in row['positions']) or '—'
        lines.append(
            f"| {row['configuration']} | {task} | {inputs} → {row['output_target']} | {outputs} | "
            f"{row['concurrency']} / {row['requests']} | {number(row['AL'])} | {percentage(row['draft_rate'])} | "
            f"{positions} | {number(row['output_tok_s'])} | {number(row['TTFT_s'])} | "
            f"{number(row['TPOT_ms'])} | {number(row['E2E_s'])} | {row['periodic_requests']} | "
            f"[原始结果]({row['source']}) |"
        )
    return lines


def main():
    rows, sanity, failed = [], [], []
    raw_hashes = {}
    for path in sorted((ROOT / 'results').glob('*/*/result.json')):
        info_path = path.parent.parent / 'server_info.json'
        info = read(info_path)
        if not info.get('speculative_algorithm'):
            continue
        data = read(path)
        raw_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = data['requests']
        if (not records or any(r.get('error') or not r.get('stream_done') for r in records)
                or any(v.get('failures') for v in data.get('validation') or [])):
            failed.append({'source': str(path), 'run': path.parent.parent.name,
                           'group': path.parent.name, 'validation': data.get('validation')})
            continue
        verified = [r for r in records if r['meta'].get('spec_verify_ct', 0) > 0]
        verify = sum(r['meta']['spec_verify_ct'] for r in verified)
        correct = sum(r['meta']['spec_num_correct_drafts'] for r in verified)
        proposed = sum(r['meta']['spec_num_proposed_drafts'] for r in verified)
        hist = [0] * 4
        for record in verified:
            for index, count in enumerate(record['meta']['spec_correct_drafts_histogram']):
                hist[index] += count
        assert sum(hist) == verify and sum(i * n for i, n in enumerate(hist)) == correct, path
        assert proposed == 3 * verify, path
        for record in records:
            assert record['completion_tokens'] == len(record['output_ids']) == record['meta']['completion_tokens'], path
            assert record['input_tokens_sent'] == record['meta']['prompt_tokens'], path
        run = path.parent.parent.name
        base_run = run.removesuffix('_business')
        row = {
            'run': run, 'group': path.parent.name,
            'configuration': CONFIGS.get(base_run, service_label(base_run)),
            'configuration_full': service_label(base_run),
            'input_lengths': sorted({r['input_tokens_sent'] for r in records}),
            'output_target': data['target_output_tokens'],
            'output_lengths': [r['completion_tokens'] for r in records],
            'ignore_eos': data['ignore_eos'], 'concurrency': data['concurrency'],
            'requests': len(records), 'acceptance_requests': len(verified),
            'AL': sum(r['completion_tokens'] for r in verified) / verify if verify else None,
            'draft_rate': correct / proposed if proposed else None,
            'positions': [sum(hist[i:]) / verify for i in (1, 2, 3)] if verify else [],
            'verify_count': verify, 'accepted_drafts': correct, 'proposed_drafts': proposed,
            'histogram': hist,
            'output_tok_s': sum(r['completion_tokens'] for r in records) / data['aggregate']['wall_s'],
            'TTFT_s': mean(records, 'ttft_s'), 'TPOT_ms': mean(records, 'tpot_ms'),
            'E2E_s': mean(records, 'e2e_s'),
            'cache_hit_ratio': sum(r['meta'].get('cached_tokens', 0) for r in records) /
                               sum(r['meta']['prompt_tokens'] for r in records),
            'periodic_requests': sum(bool(r.get('output_shape', {}).get('periodic_suffix')) for r in records),
            'source': str(path), 'source_sha256': raw_hashes[str(path)],
        }
        (sanity if path.parent.name == 'sanity_chat' else rows).append(row)
    report = {
        'updated': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'scope': 'Completed autonomous MTP ON groups, including full W3 supplement; original data are read only.',
        'rows': rows, 'sanity': sanity, 'failed_groups': failed,
    }
    DEST.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with DEST.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    lines = ['# MTP自主探索全量性能表（含完整W3补跑）', '',
             f"更新时间：{report['updated']}。共{len(rows)}组有效探索/标准场景结果，另列{len(sanity)}组功能检查和{len(failed)}组不完整结果。", '',
             '长度单位为token；总吞吐＝全部完成请求的总输出/整组墙钟；TTFT、TPOT、E2E为请求均值，TPOT只平均有定义的请求。AL与草稿接受率按有验证计数的请求加权，prefill即结束而无验证的请求不补零。允许结束的输出上限和实际输出分列。', '',
             '配置中文简称对应关系如下；旧版实际自报0.5.17开发版（镜像标签0.5.16）。旧版双路使用自动通信与小批次计算图，0.5.19单/双路使用常规通信且关闭计算图。', '',
             '| 表中配置简称 | 具体说明 |', '|---|---|']
    for name, short in CONFIGS.items():
        full = ('沿用0.5.19单路配置，完整长输出补跑' if name == 'w3_supplement_20261013'
                else service_label(name))
        lines.append(f'| {short} | {full} |')
    lines += ['', '## 四类标准场景', '']
    standards = sorted([r for r in rows if r['group'].startswith('w')], key=lambda r: r['group'])
    lines += render_table(standards)
    lines += ['', '## 随机输入与输入格式诊断', '']
    probes = [r for r in rows if not r['group'].startswith(('w', 'sharegpt'))]
    lines += render_table(probes)
    lines += ['', '## 真实聊天', '']
    chats = [r for r in rows if r['group'].startswith('sharegpt')]
    lines += render_table(chats)
    lines += ['', '完整聊天测试是在三套配置下重复同两条问题；短窗口使用同一批8条问题，不能相加当作更多独立样本。配置、输入长度、并发、输出是否自然结束和预热会影响性能，各行不能直接用作单变量收益或MTP开关加速比。', '',
              'W3完整补跑的两token循环尾段占输出66.77%；输入1024/输出1024的短诊断经常接近AL4且出现循环。接受率需要和输出形状一起看。W4实测缓存命中59.77%。', '',
              '## 功能检查（不计入46组探索结果）', '']
    lines += render_table(sanity)
    lines += ['', '## 不完整请求组', '']
    for row in failed:
        lines.append(f"- {service_label(row['run'])}：{workload_label(row['group'])}，一条请求未收到完整流结束标记；"
                     f"[原始结果]({row['source']})，不计入有效性能与接受率表。")
    lines += ['', '## 未形成有效结果的MTP服务启动', '',
              '| 配置尝试 | 结果 | 来源 |', '|---|---|---|']
    translations = {
        'container exited before health': '服务达到可用状态前退出，未产生该配置的有效性能结果',
        'scheduler crashed while HTTP router remained alive': '调度器退出，HTTP入口仍在；未形成有效性能结果',
        'stopped during loading before requests: avoid proven fused input-logprob compiler crash; safe scoring restart':
        '加载期间停止以避开已确认的融合评分编译错误，改用常规评分重启；没有推理请求结果',
    }
    for attempt in read(ROOT / 'overnight_status.json').get('runs', []):
        if attempt['tag'].startswith('mtp-') and not attempt.get('healthy'):
            reason = translations.get(attempt.get('failure'), attempt.get('failure', '无有效性能结果'))
            log = ROOT / 'logs' / attempt['container'] / 'server.log'
            lines.append(f"| {service_label(attempt['tag'])} | {reason} | [服务日志]({log}) |")
    lines += ['', '另外6次MTP关闭正确性诊断及16次固定前缀重放分别见'
              '[补充诊断](review_20261013/MTP一般情况检索与补充诊断.md)和[固定前缀核对](固定DP前缀核对.md)。'
              '这些正确性检查不充当MTP关闭性能基线。', '']
    DEST.with_suffix('.md').write_text('\n'.join(lines))
    assert len(rows) == len(standards) + len(probes) + len(chats)
    for source, digest in raw_hashes.items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
    print(f'Exported {len(rows)} groups: {len(standards)} standard, {len(probes)} random diagnostics, {len(chats)} chat; '
          f'{len(sanity)} sanity and {len(failed)} incomplete separately. Raw result hashes unchanged.')


if __name__ == '__main__':
    main()
