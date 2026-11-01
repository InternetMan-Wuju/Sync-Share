#!/usr/bin/env python3
"""Create the Chunk 8K/1K Prefill report from archived SSE; never contact a server.

All latency quantiles use nearest-rank, ceil(n*q)-1. Request latency n=4;
pooled ITL n=4*299=1196. Speed bars use the arithmetic mean of per-request
rates, not the inverse mean duration. Stage stacks are an accounting identity,
not an independently instrumented hardware timeline.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
import csv
import gzip
import hashlib
import json
import math
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / 'chunk_redo_20260928/results/20260928_043035'
DOC = ROOT / 'doc/ChunkSize_8K_vs_1K_Prefill图表分析_2026-09-28.md'
CHUNKS = (8192, 1024)
NAMES = 'ABCD'
COLORS = {8192: '#2563A6', 1024: '#DF7D24'}
STAGES = [
    ('queue_s', 'Queue：服务端报告', '#94A3B8'),
    ('prefill_s', 'Prefill：服务端阶段', '#128C88'),
    ('residual_s', '首 token 前未归因（非 IPC 实测）', '#E4AB42'),
    ('after_first_s', '首 token 后至结束（Decode＋其他）', '#5D78BD'),
]
INPUT_FILES = set()
FIGURES = []


def load(path):
    INPUT_FILES.add(path)
    return json.loads(path.read_text())


def nearest(values, q):
    assert values and 0 < q <= 1
    return sorted(values)[math.ceil(len(values) * q) - 1]


def load_request(task):
    chunk, name = task
    folder = RUN / f'node_23/chunk_{chunk}/bench/requests/{name}'
    result = load(folder / 'result.json')
    ids = load(folder / 'output_ids_final.json')
    stream = folder / 'sse_raw.jsonl.gz'
    INPUT_FILES.add(stream)
    frames = []
    previous = 0
    last = None
    done = 0
    with gzip.open(stream, 'rt', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            raw = entry['raw_line']
            if not raw.startswith('data:'):
                continue
            body = raw[5:].strip()
            if body == '[DONE]':
                done += 1
                continue
            last = json.loads(body)
            count = last['meta_info']['completion_tokens']
            if count > previous:
                # Essential before treating SSE event gaps as single-token ITL.
                assert count == previous + 1, (chunk, name, previous, count)
                frames.append((count, entry['recv_perf_ns']))
                previous = count
    assert result['http_status'] == 200 and result['error'] is None
    assert done == 1 and previous == result['output_tokens'] == len(ids) == 300
    assert last['output_ids'] == ids
    assert result['prompt_tokens'] == 32768 and result['cached_tokens'] == 19584
    assert result['meta_info']['num_retractions'] == 0
    meta = last['meta_info']
    for key in ('queue_time', 'forward_entry_time', 'prefill_finished_time'):
        assert meta[key] == result['meta_info'][key]
    times = [frame[1] for frame in frames]
    assert times[0] == result['first_token_perf_ns']
    assert times[-1] == result['last_token_perf_ns']
    ttft = (times[0] - result['started_perf_ns']) / 1e9
    e2e = (result['ended_perf_ns'] - result['started_perf_ns']) / 1e9
    queue = meta['queue_time']
    prefill = meta['prefill_finished_time'] - meta['forward_entry_time']
    residual = ttft - queue - prefill
    after = e2e - ttft
    tpot = after / (previous - 1) * 1000
    gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    assert len(gaps) == 299 and min(gaps) >= 0 and residual >= 0
    assert abs(ttft - result['ttft_s']) < 1e-8
    assert abs(e2e - result['e2e_s']) < 1e-8
    assert abs(tpot - result['tpot_ms']) < 1e-8
    assert abs(queue + prefill + residual + after - e2e) < 1e-8
    row = dict(chunk=chunk, request=name, input_sha256=result['input_sha256'],
               prompt_tokens=32768, cached_tokens=19584, uncached_tokens=13184,
               output_tokens=300, queue_s=queue, prefill_s=prefill,
               residual_s=residual, after_first_s=after, ttft_s=ttft,
               tpot_ms=tpot, e2e_s=e2e,
               itl_mean_ms=mean(gaps), itl_p50_ms=nearest(gaps, .5),
               itl_p95_ms=nearest(gaps, .95), itl_p99_ms=nearest(gaps, .99),
               itl_max_ms=max(gaps),
               prefill_speed_tok_s=13184 / prefill,
               decode_speed_tok_s=(previous - 1) / after,
               tail_after_last_output_ms=(result['ended_perf_ns']-times[-1])/1e6,
               output_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    return row, times, gaps


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_fonts():
    candidates = [Path('/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf')]
    candidates += [Path(p) for p in font_manager.findSystemFonts()
                   if 'NotoSansCJK' in p or 'SourceHanSans' in p]
    font_file = next((p for p in candidates if p.exists()), None)
    if font_file is None:
        raise RuntimeError('A CJK font is required for the Chinese figure labels.')
    font_manager.fontManager.addfont(str(font_file))
    cjk = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams.update({
        'font.family': ['DejaVu Sans', cjk], 'font.size': 10.5,
        'axes.unicode_minus': False, 'axes.titleweight': 'bold',
        'axes.titlesize': 12, 'axes.labelsize': 10,
        'axes.edgecolor': '#D3DBE4', 'axes.labelcolor': '#334155',
        'text.color': '#172B42', 'xtick.color': '#475569',
        'ytick.color': '#475569', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'svg.fonttype': 'path',
        'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })
    warnings.filterwarnings('error', message='Glyph .* missing from font')
    return str(font_file)


def style(ax, axis='x'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#E3E9EF', linewidth=.7)


def save(fig, stem):
    fig.savefig(OUT / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight')
    FIGURES.append((stem, fig))


def figure_e2e(summaries):
    fig = plt.figure(figsize=(13.3, 10.3))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .9],
                          left=.09, right=.95, bottom=.15, top=.83,
                          hspace=.84, wspace=.32)
    fig.suptitle('ChunkSize 8K / 1K：E2E 分解与 Prefill 放大',
                 fontsize=19, fontweight='bold', y=.978)
    fig.text(.5, .936, '32K 总输入 · 19,584 token 命中 · 300 输出 · 每档 4 请求算术均值',
             ha='center', fontsize=11.5, color='#536579')
    handles = [Patch(facecolor=color, label=label) for _, label, color in STAGES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .908),
               ncol=2, frameon=False, fontsize=10, columnspacing=2.2)
    means = {c: summaries[c]['means'] for c in CHUNKS}
    for row_index, is_ttft in enumerate((False, True)):
        ax = fig.add_subplot(grid[row_index, :])
        keys = STAGES[:3] if is_ttft else STAGES
        for y, chunk in enumerate(CHUNKS):
            left = 0
            for key, _, color in keys:
                width = means[chunk][key]
                ax.barh(y, width, left=left, height=.5, color=color, edgecolor='none')
                # Tiny queue/residual contributions remain at their true scale.
                # Separate panels below make these values legible.
                if (is_ttft and key == 'prefill_s') or (not is_ttft and key == 'after_first_s'):
                    ax.text(left + width / 2, y, f'{width:.3f} s',
                            ha='center', va='center', color='white', fontweight='bold', fontsize=11)
                left += width
            ax.text(left + (.12 if is_ttft else 1.6), y, f'{left:.3f} s',
                    va='center', fontsize=11.5, fontweight='bold', color=COLORS[chunk])
        ax.set_yticks([0, 1], ['Chunk 8K', 'Chunk 1K'])
        ax.set_ylim(1.55, -.55)
        ax.set_xlim(0, 11.6 if is_ttft else 184)
        ax.set_xlabel('客户端首 token 前时间（s）' if is_ttft else '客户端 E2E（s）')
        ax.set_title('B. 首 token 前放大：主要差距位于 Prefill 阶段' if is_ttft
                     else 'A. 完整 E2E：四段按账面相加，保留真实比例', loc='left', pad=12)
        style(ax)
    for col, key, title in [(0, 'queue_s', 'C. Queue 单独放大（服务端字段）'),
                            (1, 'residual_s', 'D. 首 token 前未归因（不能命名为 IPC）')]:
        ax = fig.add_subplot(grid[2, col])
        values = [means[c][key] * 1000 for c in CHUNKS]
        ax.barh([0, 1], values, height=.5, color=[COLORS[c] for c in CHUNKS])
        for y, (c, value) in enumerate(zip(CHUNKS, values)):
            ax.text(value + max(values) * .025, y, f'{value:.3f} ms',
                    va='center', color=COLORS[c], fontweight='bold')
        ax.set_yticks([0, 1], ['Chunk 8K', 'Chunk 1K'])
        ax.set_ylim(1.55, -.55)
        ax.set_xlim(0, max(values) * 1.34)
        ax.set_xlabel('时间（ms）；两面板横轴量程不同')
        ax.set_title(title, loc='left', pad=12, fontsize=11)
        style(ax)
    fig.text(.09, .025,
             '分解：Queue + Prefill + (TTFT − Queue − Prefill) + (E2E − TTFT)。\n'
             '后两段含调度、传输及计时口径差等；这里不是 IPC / 纯 NPU Decode 的独立计时。',
             fontsize=10, color='#607286', linespacing=1.6)
    save(fig, '01_e2e_breakdown')


def grouped_barh(ax, labels, by_chunk, decimals=1, limit=None):
    ys = np.arange(len(labels))
    h = .31
    maximum = max(max(values) for values in by_chunk.values())
    for j, c in enumerate(CHUNKS):
        positions = ys + (-.18 if j == 0 else .18)
        bars = ax.barh(positions, by_chunk[c], height=h,
                       color=COLORS[c], label=f'Chunk {c // 1024}K')
        for rect, val in zip(bars, by_chunk[c]):
            ax.text(val + maximum * .016, rect.get_y() + rect.get_height()/2,
                    f'{val:.{decimals}f}', va='center', fontsize=9.5,
                    color=COLORS[c], fontweight='bold')
    ax.set_yticks(ys, labels)
    ax.set_ylim(len(labels) - .4, -.65)
    ax.set_xlim(0, limit if limit else maximum * 1.2)
    style(ax)


def figure_itl(summaries, records, effects):
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 13.2))
    fig.subplots_adjust(left=.13, right=.94, top=.86, bottom=.09, hspace=.58)
    fig.suptitle('ChunkSize 8K / 1K：ITL 与长 Prefill 注入停顿',
                 fontsize=19, fontweight='bold', y=.98)
    fig.text(.5, .946, '每条输出 300 token；每个 SSE 事件新增 1 token；ITL 为客户端到达间隔',
             ha='center', fontsize=11, color='#536579')
    fig.legend(handles=[Patch(facecolor=COLORS[c], label=f'Chunk {c//1024}K') for c in CHUNKS],
               loc='upper center', bbox_to_anchor=(.5, .929), ncol=2, frameon=False)
    keys = ['mean_ms', 'p50_ms', 'p95_ms', 'p99_ms']
    grouped_barh(axes[0], ['均值', 'P50', 'P95', 'P99'],
                 {c: [summaries[c]['pooled_itl'][k] for k in keys] for c in CHUNKS}, 1, 720)
    axes[0].set_title('A. 全部 ITL 汇合：每档 1,196 个间隔（4 条请求）', loc='left', pad=12)
    axes[0].set_xlabel('客户端 ITL（ms）；越低越好')
    grouped_barh(axes[1], [f'请求 {n}' for n in NAMES],
                 {c: [r['itl_max_ms']/1000 for r in records if r['chunk']==c] for c in CHUNKS}, 3, 11.5)
    axes[1].set_title('B. 每条请求最长停顿：常见间隔的分位数会掩盖稀有长停顿', loc='left', pad=12)
    axes[1].set_xlabel('全请求最大 ITL（s）；越低越好')
    grouped_barh(axes[2], ['A 遇到 B', 'B 遇到 C', 'C 遇到 D'],
                 {c: [e['overlap_max_ms']/1000 for e in effects if e['chunk']==c] for c in CHUNKS}, 3, 11.5)
    axes[2].set_title('C. 后来 Prefill 注入窗口内：已有 Decode 请求的最长停顿', loc='left', pad=12)
    axes[2].set_xlabel('与注入窗口相交的最大 ITL（s）；两档使用相同坐标尺度')
    fig.text(.13, .025,
             'P95/P99 是间隔级最近秩分位数，不是 1,196 条独立请求的尾延迟。\n'
             '注入窗口 = 后来请求发起至首 token 到达；包含排队、Prefill 和交付，不是纯 Prefill 区间。',
             fontsize=10, color='#607286', linespacing=1.6)
    save(fig, '02_itl_bars')


def figure_speed(records, summaries, prefill):
    key = 'prefill_speed_tok_s' if prefill else 'decode_speed_tok_s'
    ratio = summaries[8192]['means'][key] / summaries[1024]['means'][key]
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    fig.subplots_adjust(left=.085, right=.965, top=.74, bottom=.2)
    fig.suptitle('ChunkSize 8K / 1K：Prefill 阶段处理速度' if prefill
                 else 'ChunkSize 8K / 1K：首 token 后平均输出速度',
                 fontsize=19, fontweight='bold', y=.97)
    formula = ('每条速度 = 13,184 个未命中输入 token / 服务端 Prefill 阶段时间'
               if prefill else '每条速度 = 299 个后续输出 token / (客户端 E2E − TTFT) = 1000 / TPOT(ms)')
    fig.text(.5, .909, formula, ha='center', fontsize=11.2, color='#536579')
    x = np.arange(5)
    width = .33
    ax.axvspan(3.52, 4.52, color='#EDF2F7', zorder=0)
    for j, c in enumerate(CHUNKS):
        values = [r[key] for r in records if r['chunk']==c] + [summaries[c]['means'][key]]
        bars = ax.bar(x + (-.18 if j == 0 else .18), values, width,
                      color=COLORS[c], label=f'Chunk {c//1024}K')
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value + (30 if prefill else .028),
                    f'{value:.1f}' if prefill else f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS[c])
    ax.set_xticks(x, ['请求 A', '请求 B', '请求 C', '请求 D', '四请求均值'])
    ax.set_xlim(-.6, 4.6)
    ax.set_ylim(0, 2750 if prefill else 2.52)
    ax.set_ylabel('输入 token/s；越高越好' if prefill else '输出 token/s；越高越好')
    style(ax, 'y')
    fig.legend(handles=[Patch(facecolor=COLORS[c], label=f'Chunk {c//1024}K') for c in CHUNKS],
               loc='upper center', bbox_to_anchor=(.37, .869), ncol=2, frameon=False)
    fig.text(.73, .84, f'均值：8K / 1K = {ratio:.3f}×（+{(ratio-1)*100:.2f}%）',
             ha='center', fontsize=11.4, fontweight='bold', color=COLORS[8192])
    note = ('速度使用实际未命中输入量；缓存命中的 19,584 token 不进入分子。\n'
            '服务端阶段区间可能包含调度等开销；不是纯 NPU 算子速度，也不是多请求服务总吞吐。'
            if prefill else
            '首 token 后时间包含其他请求 Prefill 造成的等待、生成、传输和结束收尾。\n'
            '本轮为错开到达的混合负载；此图不能解释为纯 Decode 内核提速。')
    fig.text(.085, .065, note, fontsize=10, color='#607286', linespacing=1.7)
    save(fig, '03_prefill_speed_bars' if prefill else '04_decode_speed_bars')


def make_report(records, summaries, effects, checks):
    asset = '../chunk_redo_20260928/analysis/prefill_comparison_20260928'
    source = '../chunk_redo_20260928/results/20260928_043035'
    doc = [
        '# ChunkSize 8K / 1K：Prefill、延迟与速度图表', '',
        '整理日期：**2026-09-28（北京时间）**。本页使用 **2026-09-28 的恢复权重后复测**，从八条原始 SSE 重新计算；'
        '两档分别为 `--chunked-prefill-size=8192 / 1024`。本页不混入 9 月 26 日的 54.6K 历史负载。', '',
        '**本轮结果：8K 的 Prefill 更快，平均首 token 等待更短，插入长 Prefill 时的输出停顿也更短。** '
        '以下均为这次四请求小样本的观察。先给 TTFT/TPOT 表，再给四张图，重点放大首 token 前的 Prefill 阶段。', '',
        '## 1. 数据和统计口径', '',
        '| 条件 | 本轮设置 |', '|---|---|',
        '| 输入 / 输出 | 每条 **32,768 总输入 / 300 输出**；A/B/C/D 四条输入，两档复用相同输入 |',
        '| 缓存 | 每条实际命中 **19,584 / 32,768 = 59.765625%**；实际新算 **13,184 token** |',
        '| 环境 | 节点 23，单机 TP16/EP16，GLM-5.2 W8A8，固定镜像 `fe694986`；MTP OFF |',
        '| 调度 | `enable_mixed_chunk=False`、`prefill_decode_interval=0`；Graph 实际关闭；服务端最大运行请求 4 |',
        '| 发请求规则 | A 输出至少 20 token 后发 B；B 至少 20 token 后发 C；A 完成且 C 至少 20 token、仍在 Decode 时发 D；实际活跃上限 3 |',
        '| 请求级样本数 | **4 条/Chunk**，先测 8K，再测 1K；不是随机顺序多轮重复 |',
        '| ITL 样本数 | 每条 300 次新增输出事件，得到 299 个间隔；每档 **1,196 个间隔，来自 4 条请求** |',
        '| 成功条件 | 八条均 HTTP 200、输出 300、`num_retractions=0`；本次已从原始流核对 |', '',
        f'原始实验说明：[复测报告]({source}/README.md)；实际参数：'
        f'[8K]({source}/node_23/chunk_8192/before_bench/server_info.json)、'
        f'[1K]({source}/node_23/chunk_1024/before_bench/server_info.json)。', '',
        '## 2. TTFT / TPOT：均值、P95、P99', '',
        '- **TTFT**：客户端开始请求到首次收到输出。',
        '- **TPOT**：`(客户端 E2E − TTFT) / (300 − 1)`；本轮含首 token 后的生成、等待、输出传输及结束收尾。',
        '- **P95/P99**：采用最近秩法，排序后取第 `ceil(n × p)` 个值；每档 `n=4`，所以 **P95=P99=该档最大值**。'
        '保留这些统计满足查看需求，但它们不能估计稳定的业务尾延迟。', '',
        '| 指标 | Chunk 8K n | 8K 均值 | 8K P95 | 8K P99 | Chunk 1K n | 1K 均值 | 1K P95 | 1K P99 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key, title, digits in [('ttft_s', 'TTFT（s）', 3), ('tpot_ms', 'TPOT（ms/token）', 2)]:
        a, b = summaries[8192]['latency'][key], summaries[1024]['latency'][key]
        doc.append(f'| {title} | 4 | {a["mean"]:.{digits}f} | {a["p95"]:.{digits}f} | {a["p99"]:.{digits}f} | 4 | {b["mean"]:.{digits}f} | {b["p95"]:.{digits}f} | {b["p99"]:.{digits}f} |')
    a, b = summaries[8192]['means'], summaries[1024]['means']
    doc += ['', f'8K 相对 1K：平均 TTFT **降低 {(1-a["ttft_s"]/b["ttft_s"])*100:.2f}%**；'
            f'平均 TPOT **降低 {(1-a["tpot_ms"]/b["tpot_ms"])*100:.2f}%**。'
            'TPOT 变化来自这次有 Prefill 注入的全过程，不能单独解释为 Decode 内核更快。', '',
        '## 3. E2E（s）及四段分解', '',
        '### 3.1 Queue 有记录，IPC 尚无独立测量', '',
        '**建议用“服务端 Queue → 服务端 Prefill → 首 token 前未归因 → 首 token 后至结束”四段。** '
        'Queue 并非完全猜测：最终 SSE 的 `meta_info.queue_time` 给出了该字段。'
        '但它只是服务端报告的排队口径，不代表请求在所有环节的排队都已被独立测量。'
        '**IPC 没有独立起止时间，不能把剩余时间直接命名为 IPC。**', '',
        '| 图中阶段 | 计算方式 | 可以说明什么 |', '|---|---|---|',
        '| Queue：服务端报告 | `meta_info.queue_time` | 框架记录的排队用时，不是推测值；不是整个请求链路的所有排队 |',
        '| Prefill：服务端阶段 | `prefill_finished_time − forward_entry_time` | 从进入 forward 至 Prefill 结束的阶段区间，未拆出纯算子时间 |',
        '| 首 token 前未归因 | `TTFT − Queue − Prefill` | 可能包含请求收发、分派、交付及计时边界差；**不等于实测 IPC** |',
        '| 首 token 后至结束 | `E2E − TTFT` | 主要是后续生成，也含被其他 Prefill 阻塞、传输、调度与收尾；**不等于纯 NPU Decode** |', '',
        '四段按定义严格相加为客户端 E2E。这是**账面分解**：Queue/Prefill 来自服务端时间，TTFT/E2E 来自客户端单调时钟；'
        '各段不是已证明互不重叠的硬件时间线。图中顺序用于阅读，不证明未归因开销都发生在 Prefill 之后。', '',
        '### 3.2 完整 E2E、TTFT 与小开销分别放大', '',
        f'![两种 Chunk 的 E2E 四段分解及 Prefill 放大]({asset}/01_e2e_breakdown.png)', '',
        '图 A 为完整 E2E；图 B 放大首 token 前区间。Queue 很短，在整体图中几乎看不见，'
        '因此图 C/D 分别显示 Queue 和未归因时间，单位改用 ms，且两图横轴量程不同；未人为拉长堆叠中的短阶段。', '',
        '| Chunk | Queue 均值（ms） | Prefill 均值（s） | 首 token 前未归因均值（ms） | TTFT 均值（s） | 首 token 后均值（s） | E2E 均值（s） |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for c in CHUNKS:
        m=summaries[c]['means']
        doc.append(f'| {c//1024}K | {m["queue_s"]*1000:.3f} | {m["prefill_s"]:.3f} | {m["residual_s"]*1000:.3f} | {m["ttft_s"]:.3f} | {m["after_first_s"]:.3f} | {m["e2e_s"]:.3f} |')
    doc += ['', f'主要差距在 Prefill：1K **{b["prefill_s"]:.3f} s**，8K **{a["prefill_s"]:.3f} s**，'
            f'8K 减少 **{b["prefill_s"]-a["prefill_s"]:.3f} s（{(1-a["prefill_s"]/b["prefill_s"])*100:.2f}%）**。'
            f'TTFT 的均值差为 **{b["ttft_s"]-a["ttft_s"]:.3f} s**，与 Prefill 差额接近。'
            'Queue 是毫秒级，无法解释秒级差距。', '',
        f'E2E 则是 **{b["e2e_s"]:.3f} → {a["e2e_s"]:.3f} s**，8K 降低 **{(1-a["e2e_s"]/b["e2e_s"])*100:.2f}%**。'
        '首 token 后仍占大部分时间，所以 Prefill 改善比例不能直接当整请求的改善比例。', '',
        '### 3.3 八条请求的阶段用时', '',
        '| Chunk | 请求 | Queue（ms） | Prefill（s） | 未归因（ms） | 首 token 后（s） | E2E（s） |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        doc.append(f'| {r["chunk"]//1024}K | {r["request"]} | {r["queue_s"]*1000:.3f} | {r["prefill_s"]:.3f} | {r["residual_s"]*1000:.3f} | {r["after_first_s"]:.3f} | {r["e2e_s"]:.3f} |')
    doc += ['', '## 4. ITL 条形图：常见间隔和稀有停顿都要看', '',
        f'![两种 Chunk 的 ITL 分位、逐请求最大间隔及注入窗口最大间隔]({asset}/02_itl_bars.png)', '',
        '图 A 将每档四条请求的所有 **1,196 个 ITL** 汇合后计算；每请求输出长度相同，各有 299 个间隔。'
        '图 B 保留每条请求的最长停顿；图 C 对应三次 Prefill 注入。'
        '**图 A 使用 ms，图 B/C 使用 s；每个面板内两档使用相同尺度。**', '',
        '| Chunk | 请求数 / ITL 数 | ITL 均值（ms） | P50（ms） | P95（ms） | P99（ms） | 全组最大 ITL（s） |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for c in CHUNKS:
        q=summaries[c]['pooled_itl']
        doc.append(f'| {c//1024}K | 4 / {q["n"]} | {q["mean_ms"]:.2f} | {q["p50_ms"]:.2f} | {q["p95_ms"]:.2f} | {q["p99_ms"]:.2f} | {q["max_ms"]/1000:.3f} |')
    doc += ['', 'ITL P95/P99 是**间隔级**分位数，不能视为 1,196 个独立请求的尾延迟。'
        '两档 P99 只有约 0.58–0.59 s，但 1K 单次最长停顿达到 **9.620 s**；'
        '说明稀有长停顿占比不到 1% 时，P99 仍可能看起来接近正常值。', '',
        '| 正在 Decode / 后来 Prefill | 8K 注入前 P50（ms） | 1K 注入前 P50（ms） | 8K 窗口最大 ITL（s） | 1K 窗口最大 ITL（s） |',
        '|---|---:|---:|---:|---:|',
    ]
    for old,new in [('A','B'),('B','C'),('C','D')]:
        x=next(e for e in effects if e['chunk']==8192 and e['decode_request']==old)
        y=next(e for e in effects if e['chunk']==1024 and e['decode_request']==old)
        doc.append(f'| {old} 遇到 {new} | {x["before_p50_ms"]:.2f} | {y["before_p50_ms"]:.2f} | {x["overlap_max_ms"]/1000:.3f} | {y["overlap_max_ms"]/1000:.3f} |')
    doc += ['', '注入窗口从后来请求的发起时刻到其首 token 到达；凡与窗口相交的 ITL 均计入。'
        '窗口包含请求传输、排队、Prefill 与首 token 交付，不是纯 Prefill 时段；三次注入也不是三个独立重复实验。', '',
        '**当前结果支持 8K：**注入前常见间隔约 0.46–0.51 s；注入时 8K 最大停顿约 3.1 s，1K 约 9.3–9.6 s。'
        '8K 仍有秒级停顿，不能理解成已经消除了 Prefill 干扰。', '',
        '## 5. Prefill 速度柱状图', '',
        '每请求速度采用 **未命中输入 token / 服务端 Prefill 阶段时间**：', '',
        '```text\nPrefill speed = (32768 − 19584) / (prefill_finished_time − forward_entry_time)\n              = 13184 / Prefill秒数\n```', '',
        f'![两种 Chunk 的逐请求和均值 Prefill 处理速度]({asset}/03_prefill_speed_bars.png)', '',
        '| 请求 | 8K Prefill 速度（token/s） | 1K Prefill 速度（token/s） | 8K / 1K |',
        '|---|---:|---:|---:|',
    ]
    for name in list(NAMES)+['均值']:
        x=a if name=='均值' else next(r for r in records if r['chunk']==8192 and r['request']==name)
        y=b if name=='均值' else next(r for r in records if r['chunk']==1024 and r['request']==name)
        v,w=x['prefill_speed_tok_s'],y['prefill_speed_tok_s']
        doc.append(f'| {name} | {v:.2f} | {w:.2f} | {v/w:.3f}× |')
    doc += ['', f'本图均值为 **8K {a["prefill_speed_tok_s"]:.2f} token/s，1K {b["prefill_speed_tok_s"]:.2f} token/s**；'
        f'8K 为 1K 的 **{a["prefill_speed_tok_s"]/b["prefill_speed_tok_s"]:.3f}×**。'
        '各请求分子都为 13,184，所以差别来自 Prefill 阶段用时。', '',
        '**平均方式：**先计算每条速度，再取四条的算术均值。它不是 `总未命中 token / Prefill 时间之和`。'
        f'后者本轮为 8K **{summaries[8192]["prefill_ratio_of_sums_tok_s"]:.2f}**、1K **{summaries[1024]["prefill_ratio_of_sums_tok_s"]:.2f} token/s**；'
        '此前综合报告使用过该口径，轻微数值差异来自平均方式，不是数据变化。'
        '所有这些值都是服务端阶段速率，不能称为纯算子速度或服务总吞吐。', '',
        '## 6. Decode 速度柱状图', '',
        '为与 TPOT 一致，使用**客户端首 token 后的平均输出速度**：', '',
        '```text\nDecode-observed speed = (300 − 1) / (E2E − TTFT)\n                      = 1000 / TPOT(ms)\n```', '',
        f'![两种 Chunk 的逐请求和均值客户端 Decode 输出速度]({asset}/04_decode_speed_bars.png)', '',
        '| 请求 | 8K 首 token 后速度（token/s） | 1K 首 token 后速度（token/s） | 8K / 1K |',
        '|---|---:|---:|---:|',
    ]
    for name in list(NAMES)+['均值']:
        x=a if name=='均值' else next(r for r in records if r['chunk']==8192 and r['request']==name)
        y=b if name=='均值' else next(r for r in records if r['chunk']==1024 and r['request']==name)
        v,w=x['decode_speed_tok_s'],y['decode_speed_tok_s']
        doc.append(f'| {name} | {v:.4f} | {w:.4f} | {v/w:.3f}× |')
    doc += ['', f'均值为 **8K {a["decode_speed_tok_s"]:.3f} token/s，1K {b["decode_speed_tok_s"]:.3f} token/s**，'
        f'8K 高约 **{(a["decode_speed_tok_s"]/b["decode_speed_tok_s"]-1)*100:.2f}%**。'
        '这也是先计算每请求速度再平均，不等于 `1000 / 全组平均TPOT`。', '',
        '这段时间含后来 Prefill 造成的阻塞，结果更适合描述“这次混合负载的输出体验”。'
        '本轮没有配套的同输入 Decode-only 对照，不能将差额都归为 Decode 内核性能。'
        'A/C/D 两档实际输出 ID 不一致，只有 B 完全一致；输入与长度相同不等于生成内容相同。', '',
        '## 7. 如何解释 Prefill 的差距', '',
        '两档实际新计算量相同，为 13,184 token。仅按分块上限作工作量示意，8K 约需 2 块，1K 约需 13 块；'
        '这不是实际 forward 次数的独立跟踪。当前 `enable_mixed_chunk=False`、`prefill_decode_interval=0`，'
        '不能假设切成小块后，Decode 就一定在块间获得执行机会。', '',
        '实测中，1K 的 Prefill 阶段更长，已有 Decode 遭遇的停顿也更长；这与该调度条件相符。'
        '但现有记录没有把 kernel、通信、调度和 IPC 分开计时，不能将差额精确归到其中一项。'
        '切换 mixed chunk、Graph、高并发或 PD 分离后应重新比较。', '',
        '本轮的 TTFT/TPOT 请求级 P95/P99 仅供查看样本最大值；不要用它们承诺生产尾延迟。'
        '速度与分解图都保留八条单请求数据或列出明细，方便识别均值背后的样本差异。', '',
        '## 8. 导出、原始数据与复算', '',
        f'四张图均有 PNG 与 SVG；另有四页 [合并 PDF]({asset}/ChunkSize_8K_vs_1K_四图.pdf)。'
        'SVG 的文字以路径保存，便于在没有相同中文字体的机器上查看。', '',
        '| 图 | PNG | SVG |', '|---|---|---|',
    ]
    for name,label in [('01_e2e_breakdown','E2E / Prefill 分解'),('02_itl_bars','ITL'),
                       ('03_prefill_speed_bars','Prefill 速度'),('04_decode_speed_bars','Decode 速度')]:
        doc.append(f'| {label} | [PNG]({asset}/{name}.png) | [SVG]({asset}/{name}.svg) |')
    doc += ['',
        f'派生数据：[逐请求阶段与速度 CSV]({asset}/per_request_metrics.csv)、'
        f'[TTFT/TPOT 分位表 CSV]({asset}/latency_summary.csv)、'
        f'[ITL 汇总 CSV]({asset}/itl_summary.csv)、'
        f'[全部 ITL CSV]({asset}/itl_intervals.csv)、'
        f'[注入窗口 CSV]({asset}/interference.csv)、'
        f'[统计与校验 JSON]({asset}/summary.json)。', '',
        f'生成代码：[plot_report.py]({asset}/plot_report.py)；'
        f'输入文件摘要：[input_manifest.json]({asset}/input_manifest.json)。'
        '脚本只读归档，重新生成本目录图表和本页，不调用在线服务。', '',
        '```bash\npython3 /home/chengyetai/workspace/current_exp/chunk_redo_20260928/analysis/prefill_comparison_20260928/plot_report.py\n```', '',
        f'原始流：[8K 请求目录]({source}/node_23/chunk_8192/bench/requests/)、'
        f'[1K 请求目录]({source}/node_23/chunk_1024/bench/requests/)。'
        '实际最终输出采用 `output_ids_final.json`，不使用早期错误拼接累计数组的 `output_ids.json`。', '',
        f'本次校验：八条请求各有 300 个逐 token 输出事件，{checks["itl_intervals_checked"]} 个相邻间隔；'
        '输入 SHA 两档逐对一致；最终 ID 与 SSE 末帧一致；四段之和等于 E2E；'
        '与存档请求指标和注入窗口最大 ITL 交叉核对通过。', '',
    ]
    DOC.write_text('\n'.join(doc), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(c,n) for c in CHUNKS for n in NAMES]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(load_request, tasks))
    records = [item[0] for item in results]
    ticks = {(r['chunk'],r['request']):times for r,times,gaps in results}
    all_gaps = {(r['chunk'],r['request']):gaps for r,times,gaps in results}
    for name in NAMES:
        pair = [r for r in records if r['request']==name]
        assert pair[0]['input_sha256'] == pair[1]['input_sha256']
    summaries = {}
    latency_rows = []
    interval_rows = []
    pooled_rows = []
    effects = []
    metric_keys = ['queue_s','prefill_s','residual_s','after_first_s','ttft_s','tpot_ms',
                   'e2e_s','prefill_speed_tok_s','decode_speed_tok_s']
    for c in CHUNKS:
        rows=[r for r in records if r['chunk']==c]
        gaps=[v for name in NAMES for v in all_gaps[(c,name)]]
        means={key:mean(r[key] for r in rows) for key in metric_keys}
        lat={}
        for key in ('ttft_s','tpot_ms'):
            values=[r[key] for r in rows]
            lat[key]={'n':len(values),'mean':mean(values),'p95':nearest(values,.95),'p99':nearest(values,.99)}
            assert lat[key]['p95'] == lat[key]['p99'] == max(values)
            latency_rows.append({'chunk':c,'metric':key,**lat[key],'method':'nearest_rank'})
        pooled={'n':len(gaps),'mean_ms':mean(gaps),'p50_ms':nearest(gaps,.5),
                'p95_ms':nearest(gaps,.95),'p99_ms':nearest(gaps,.99),'max_ms':max(gaps)}
        assert pooled['n'] == 1196
        pooled_rows.append({'chunk':c,'requests':4,**pooled})
        summaries[c]={'means':means,'latency':lat,'pooled_itl':pooled,
                      'prefill_ratio_of_sums_tok_s':sum(r['uncached_tokens'] for r in rows)/sum(r['prefill_s'] for r in rows)}
        for name in NAMES:
            for i,v in enumerate(all_gaps[(c,name)],1):
                interval_rows.append({'chunk':c,'request':name,'from_output_token':i,
                                      'to_output_token':i+1,'itl_ms':v,
                                      'start_perf_ns':ticks[(c,name)][i-1],
                                      'end_perf_ns':ticks[(c,name)][i]})
        dispatch={r['name']:r for r in load(RUN/f'node_23/chunk_{c}/bench/dispatches.json')}
        for old,new in [('A','B'),('B','C'),('C','D')]:
            start=dispatch[new]['dispatch_perf_ns']; end=ticks[(c,new)][0]
            ts=ticks[(c,old)]
            before=[(b-a)/1e6 for a,b in zip(ts,ts[1:]) if b<=start]
            overlap=[(b-a)/1e6 for a,b in zip(ts,ts[1:]) if a<end and b>start]
            effects.append({'chunk':c,'decode_request':old,'incoming_prefill':new,
                            'window_start_perf_ns':start,'window_end_perf_ns':end,
                            'before_count':len(before),'before_p50_ms':nearest(before,.5),
                            'overlap_count':len(overlap),'overlap_max_ms':max(overlap)})
    # Independent consistency check against already archived derived CSVs.
    old_csv=RUN/'analysis/requests.csv'; INPUT_FILES.add(old_csv)
    with old_csv.open() as f: old=list(csv.DictReader(f))
    comparisons={'ttft_s':'ttft_s','tpot_ms':'tpot_ms','e2e_s':'e2e_s',
                 'prefill_s':'server_prefill_s','after_first_s':'after_first_s','itl_max_ms':'itl_max_ms'}
    for r in records:
        old_r=next(x for x in old if int(x['chunk'])==r['chunk'] and x['name']==r['request'])
        for key,old_key in comparisons.items():assert abs(r[key]-float(old_r[old_key]))<1e-8
        assert abs(r['queue_s']*1000-float(old_r['queue_ms']))<1e-8
        assert abs(r['residual_s']*1000-float(old_r['first_residual_ms']))<1e-8
    old_csv=RUN/'analysis/interference.csv'; INPUT_FILES.add(old_csv)
    with old_csv.open() as f: old=list(csv.DictReader(f))
    for e in effects:
        old_e=next(x for x in old if int(x['chunk'])==e['chunk'] and x['decode_request']==e['decode_request'])
        for key in ['before_count','before_p50_ms','overlap_count','overlap_max_ms']:
            assert abs(e[key]-float(old_e[key]))<1e-8
    checks={'status':'PASS','raw_requests_checked':8,'itl_intervals_checked':len(interval_rows),
            'output_events_per_request':300,'http200_and_complete_all':True,
            'same_input_sha256_all_pairs':True,'all_stage_sums_match_e2e':True,
            'matches_archived_request_metrics':True,'matches_archived_interference':True,
            'output_ids_equal_pairs':[name for name in NAMES
                if len({r['output_ids_sha256'] for r in records if r['request']==name})==1]}
    write_csv('per_request_metrics.csv',records)
    write_csv('latency_summary.csv',latency_rows)
    write_csv('itl_summary.csv',pooled_rows)
    write_csv('itl_intervals.csv',interval_rows)
    write_csv('interference.csv',effects)
    (OUT/'summary.json').write_text(json.dumps({'source_run':str(RUN),'checks':checks,
        'quantile_method':'nearest_rank','speed_mean_method':'arithmetic_mean_of_per_request_rates',
        'groups':summaries},ensure_ascii=False,indent=2)+'\n')
    font_file=configure_fonts()
    figure_e2e(summaries)
    figure_itl(summaries,records,effects)
    figure_speed(records,summaries,True)
    figure_speed(records,summaries,False)
    with PdfPages(OUT/'ChunkSize_8K_vs_1K_四图.pdf') as pdf:
        for stem,fig in FIGURES:pdf.savefig(fig,bbox_inches='tight')
    for stem,fig in FIGURES:plt.close(fig)
    make_report(records,summaries,effects,checks)
    manifest={'run':str(RUN),'font_file':font_file,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(INPUT_FILES)]}
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(DOC),'figures':[stem for stem,_ in FIGURES],
                      'checks':checks,'font':font_file},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
