#!/usr/bin/env python3
"""Record the W1 C32 gap using only autonomous ChunkSize result files."""
from report_labels import humanize_report
import datetime
import json
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK = ROOT.parent / 'Qchunksize'


def main():
    cells = []
    for path in sorted((CHUNK / 'results/autonomous_20261012').glob('*/result.json')):
        data = json.loads(path.read_text())
        cells.append(dict(scene=data['name'], concurrency=data['concurrency'], chunk=data['chunk'],
                          requests=len(data['requests']), repeat=data.get('repeat', 1),
                          complete=data['aggregate'].get('complete'), source=str(path)))
    w1 = [r for r in cells if r['scene'] == 'w1_8k_1k']
    c32 = [r for r in w1 if r['concurrency'] == 32 and r['requests'] == 64 and r['complete']]
    report = dict(updated=datetime.datetime.now().isoformat(timespec='seconds'),
                  scope='Only Qchunksize/results/autonomous_20261012 is counted; no historical results.',
                  w1_required_concurrency=32, w1_required_requests=64,
                  w1_observed_concurrencies=sorted({r['concurrency'] for r in w1}),
                  w1_c32_completed_cells=c32, w1_c32_tested=bool(c32), cells=cells,
                  source_requirements=str(ROOT.parent / 'docs/AboutModels.md'),
                  w1_baseline=dict(tp=16, dp=2, ep=1, dp_attention=True, deepep=False, chunk=8192),
                  autonomous_service=dict(tp=16, dp=1, ep=16, dp_attention=False, deepep=True,
                                          deepep_mode='auto', max_running_requests=16, kv_tokens=77824),
                  interpretation='Low concurrency evidence only. No claim that C32 fails or shares the same optimal chunk.')
    (CHUNK / 'ChunkSize要求与自主探索偏差.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ['# ChunkSize要求与自主探索偏差', '',
             f"更新时间：{report['updated']}。只核对本次自主探索。", '',
             '**W1要求客户端并发32、正式请求64。自主探索只跑了并发2/4，未完成W1规定并发的ChunkSize对照。**', '',
             '| 项目 | AboutModels W1基线/当前要求 | 自主探索实际 |', '|---|---|---|',
             '| 客户端并发 | 32 | 2、4 |',
             '| 正式请求数 | 64，即2×并发 | 4、8 |',
             '| 并行布局 | TP16 / DP2 / EP1 | TP16 / DP1 / EP16 |',
             '| DP attention / LM head | 开 | 关 |',
             '| DeepEP | 关 | auto，开启 |',
             '| Chunk基线 | 8192 | 扫描1024/4096/8192/16384，用16384作表内比较基线 |', '',
             '当时因KV容量和服务启动限制选择了小并发预实验，随后止步于该网格。没有实际测并发32，不得把容量估算或其他配置的启动失败写成C32无法运行的证据。', '',
             '客户端并发、服务器running batch和KV驻留量是不同指标。客户端发出32个请求时，服务器可以只处理其中一部分，其余排队；排队时间应计入TTFT。文档W1同时列Concurrency=32与Running Batch=14，正说明二者不能混为一个数。', '',
             '现有34单元是低并发预实验：20个基础单元及14个复测。它们可描述已测配置、已测并发下的Chunk差异，不能替代C32结果，也不能据此为W1榜单宣布4K/8K最优。', '',
             '后续W1正式对照应固定C32、64请求和一套明确的基线配置，再只改ChunkSize。W1榜单的NoEP基线与通用默认脚本也要区分；DP/EP/DeepEP等同时改变时，不能把总收益归给ChunkSize。', '',
             '## 自主探索实际覆盖', '',
             '| 场景 | 客户端并发 | 每单元请求数 | 已测Chunk | 单元数（含复测） |', '|---|---|---|---|---|']
    for scene, concurrency in sorted({(r['scene'], r['concurrency']) for r in cells}):
        group = [r for r in cells if (r['scene'], r['concurrency']) == (scene, concurrency)]
        lines.append(f"| {scene} | {concurrency} | {sorted({r['requests'] for r in group})} | {sorted({r['chunk'] for r in group})} | {len(group)} |")
    lines += ['', '依据：[AboutModels](../docs/AboutModels.md)、[实际配置](RUN_CONFIG_20261013.md)、'
              '[原始实测表](ChunkSize自主实验实测.md)、[编排脚本](../autonomous_20261012/scripts/overnight_runner.py)。', '']
    (CHUNK / 'ChunkSize要求与自主探索偏差.md').write_text(humanize_report('\n'.join(lines)))
    notice = ['**ChunkSize正式要求未完成：W1应并发32、64请求，当前只测并发2/4。**', '',
              '现有34单元仅作低并发预实验；4K/8K候选结论限于已测配置和并发，不能推广到W1并发32，也不能当作榜单的单项优化结果。', '',
              '客户端并发不等于服务器running batch，KV容量估算不能直接证明并发32无法运行。'
              '差异与证据见[ChunkSize要求与自主探索偏差]({link})。', '']
    begin, end = '<!-- chunk-requirement-coverage:start -->', '<!-- chunk-requirement-coverage:end -->'
    for path, heading, link in (
        (CHUNK / 'ChunkSize自主实验实测.md', '# ChunkSize低并发预实验表（W1并发32未测）', 'ChunkSize要求与自主探索偏差.md'),
        (ROOT / '自主实验最终总结.md', '# MTP与ChunkSize自主探索阶段总结（正式任务未完成）', '../Qchunksize/ChunkSize要求与自主探索偏差.md'),
    ):
        if not path.exists():
            continue
        text = path.read_text()
        if begin in text:
            first, last = text.index(begin), text.index(end) + len(end)
            text = text[:first] + text[last:].lstrip('\n')
        _, _, body = text.partition('\n')
        section = '\n'.join(line.format(link=link) for line in notice)
        path.write_text(humanize_report(heading + '\n\n' + begin + '\n' + section + '\n' + end + '\n' + body.lstrip('\n')))
    final = ROOT / '自主实验最终总结.json'
    if final.exists():
        summary = json.loads(final.read_text())
        summary['chunk_requirement_coverage'] = {k: report[k] for k in (
            'w1_required_concurrency', 'w1_required_requests', 'w1_observed_concurrencies', 'w1_c32_tested')}
        summary['chunk_requirement_coverage']['report'] = str(CHUNK / 'ChunkSize要求与自主探索偏差.md')
        final.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('autonomous Chunk cells:', len(cells), '; W1 concurrency:', report['w1_observed_concurrencies'], '; C32 tested:', bool(c32))


if __name__ == '__main__':
    main()
