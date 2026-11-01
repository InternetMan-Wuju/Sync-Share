#!/usr/bin/env python3
"""Summarize the saved request-level benchmark data; never infer a missing measurement."""
import argparse
import json
import math
import statistics
from pathlib import Path

BASE = Path('/home/chengyetai/workspace/current_exp/chunk_single23')

def rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result

def fmt(v, n=2):
    return f'{v:.{n}f}' if isinstance(v, (int, float)) else '—'

def q(values, p):
    clean = sorted(x for x in values if isinstance(x, (float, int)))
    return (statistics.median(clean) if p == .5 else clean[math.ceil(len(clean) * p) - 1]) if clean else None

def stat(rows_, field):
    return (fmt(q([r.get(field) for r in rows_], .5)), fmt(q([r.get(field) for r in rows_], .95)))

def cache(rows_):
    usable = [r for r in rows_ if isinstance(r.get('prompt_tokens'), int)
              and isinstance(r.get('cached_tokens'), int)]
    denominator = sum(r['prompt_tokens'] for r in usable)
    return fmt(100 * sum(r['cached_tokens'] for r in usable) / denominator, 1) + '%' if denominator else '—'

def group(rows_, role, prefix=None):
    return [r for r in rows_ if r.get('role') == role and
            (prefix is None or r.get('stage', '').startswith(prefix))]

def event_progress(req, start, end):
    """Client-visible completion count within a wall-clock window; SSE batching limits precision."""
    if not isinstance(req.get('started_at'), (float, int)):
        return None
    events = [(req['started_at'] + e.get('t', 0), e.get('completion_tokens'))
              for e in req.get('events', []) if isinstance(e.get('completion_tokens'), int)]
    if not events:
        return None
    earlier = max((count for t, count in events if t <= start), default=0)
    later = max((count for t, count in events if t <= end), default=earlier)
    return max(0, later - earlier)

def append_table(out, name, records):
    out.extend((f'### {name}', '', '| 阶段 | 有效/总请求 | TTFT P50/P95 (s) | TPOT P50/P95 (ms) | 实测缓存 token 比例 |',
                '|---|---:|---:|---:|---:|'))
    for stage, entries in records:
        valid = [r for r in entries if not r.get('error') and isinstance(r.get('ttft_s'), (float, int))]
        ttft = stat(valid, 'ttft_s')
        tpot = stat(valid, 'tpot_ms')
        out.append(f'| {stage} | {len(valid)}/{len(entries)} | {ttft[0]} / {ttft[1]} | {tpot[0]} / {tpot[1]} | {cache(valid)} |')
    out.append('')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_id')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    root = BASE / 'results' / a.run_id
    out = ['# 23 节点 Chunk Size 单机实验结果', '', '> **口径更正**：原需求的 32K 指总输入长 32768、输出 300、约 60% token 缓存命中。本轮将其误解为公共前缀长 32768 并把总输入 54614 当主场景；真正对应目标的只有 32K 暖态长度对照，实测缓存命中 19584/32768（约 59.77%），两种 Chunk 各 1 条；缺少 32K + Decode 干扰对照。以下主场景等称呼沿用历史设计。', '', f'运行目录：`{root}`。', '',
           '口径：公共前缀 32768 token、总输入 54614 token、目标缓存命中约 60%、输出 300；'
           '背景 Decode 为 1024 输入/900 输出。TTFT 从请求发出到首输出，'
           'TPOT = (完成时间−TTFT)/(完成 token−1)。P50 为样本中位数，P95 用 nearest-rank 算法；仅对实际完成请求计算，样本过少不能解释尾延迟。', '']
    out.extend(('## 实验条件与方法', '',
                '| 条目 | 固定设置 |', '|---|---|',
                '| 日期与硬件 | 2026-09-26；npu1-23 单机 16 张 Ascend 910C；TP16 / EP16 / DP1。 |',
                '| 权重与镜像 | `/mnt/raid/user/data/models/glm-qq`（GLM-5.2 W8A8）；`sglang-glm52-pd:fe694986`；镜像 ID、脚本哈希见运行目录的 `manifest.txt`、`code_sha256.txt`。 |',
                '| 服务参数 | Chunk=8192 或 1024，其他相同：context 131072、max-prefill 65536、max-running 4、mem-fraction 0.88、DeepEP normal、HCCL buffer 650 MB；关闭 CUDA graph 与 MTP，开启 cache report。 |',
                '| 容量 | 两种 Chunk 的启动日志都显示 `max_total_num_tokens=63744`，权重约 47.97 GB / rank；单机主场景及背景 Decode 已实际运行。 |',
                '| 服务端调度 | `enable_mixed_chunk=False`（默认值）；在 pinned 镜像源码中有新 Prefill batch 时优先处理它。 |',
                '| 请求 | 随机 token ID，固定种子 1024、贪心解码 `temperature=0`、`ignore_eos=True`；每次先 flush，再预热公共前缀。主场景共享前缀本身 32768 token、总输入 54614 token、输出 300；背景 Decode 1024 输入/900 输出。 |',
                '| 采样 | 长请求 cold / warm 各一条，两轮配对的纯 Decode / 混合注入，16K/32K 总输入暖缓存各一条；服务端 `/metrics` 每约 3 秒采集一次，保留逐请求 SSE JSONL。 |', '',
                '主场景的目标 token 命中率为 `32768/54614≈60%`。SGLang 的 page 对齐可能让其他长度场景的实测命中略低；表格均用服务端 `cached_tokens` 计算。', ''))
    all_data = {}
    for chunk in (8192, 1024):
        folder = root / f'chunk_{chunk}'
        smoke = rows(folder / 'smoke.jsonl')
        full = rows(folder / 'full.jsonl')
        all_data[chunk] = (smoke, full)
        out.extend((f'## Chunk Size = {chunk}', ''))
        if not smoke:
            msg = (folder / 'failed.txt').read_text(encoding='utf-8').strip() if (folder / 'failed.txt').is_file() else '尚无冒烟结果'
            out.extend((f'当前状态：{msg}。日志：`{folder / "server.log"}`。', ''))
            continue
        append_table(out, '容量冒烟与缓存命中', [('前缀预热', group(smoke, 'warmup')),
                                           ('54.6K 输入/300 输出', group(smoke, 'long_warm'))])
        main_rows = group(smoke, 'long_warm')
        if main_rows:
            r = main_rows[-1]
            out.append(f'实际输出 {r.get("output_tokens", "—")} / 300 token；请求失败原因：{r.get("error", "无")}。')
            out.append('')
        if not full:
            out.extend(('完整矩阵尚无结果。', ''))
            continue
        stages = [
            ('54.6K 冷缓存', group(full, 'long_cold', 'cold')),
            ('54.6K 暖缓存（Mixed 注入）', group(full, 'long_warm', 'mixed_')),
            ('16K 输入/60% token 命中', group(full, 'long_warm', 'length_16384')),
            ('32K 输入/60% token 命中', group(full, 'long_warm', 'length_32768')),
        ]
        append_table(out, '长上下文', stages)
        timeline = rows(folder / 'metrics_timeline.jsonl')
        out.extend(('长上下文的近似输入处理速率（未命中 prompt token / TTFT，含排队与网络开销）：', '',
                    '| 场景 | 新算 token/TTFT (tok/s) | 输出 token/E2E (tok/s) | 期间 KV 空闲最少 (token) |',
                    '|---|---:|---:|---:|'))
        for label, entries in stages:
            valid = [r for r in entries if not r.get('error') and isinstance(r.get('ttft_s'), (int, float)) and r['ttft_s'] > 0]
            prefill_rates = [(r['prompt_tokens'] - r['cached_tokens']) / r['ttft_s'] for r in valid
                             if isinstance(r.get('prompt_tokens'), int) and isinstance(r.get('cached_tokens'), int)]
            output_rates = [r['output_tokens'] / r['e2e_s'] for r in valid if isinstance(r.get('output_tokens'), int) and isinstance(r.get('e2e_s'), (int, float)) and r['e2e_s'] > 0]
            mins = []
            for r in valid:
                if not isinstance(r.get('started_at'), (int, float)) or not isinstance(r.get('e2e_s'), (int, float)):
                    continue
                sample = [x['kv_available_tokens'] for x in timeline
                          if r['started_at'] <= x.get('at', 0) <= r['started_at'] + r['e2e_s']
                          and isinstance(x.get('kv_available_tokens'), (int, float))]
                if sample:
                    mins.append(min(sample))
            out.append(f'| {label} | {fmt(statistics.median(prefill_rates) if prefill_rates else None, 1)} | '
                       f'{fmt(statistics.median(output_rates) if output_rates else None)} | '
                       f'{fmt(min(mins) if mins else None, 0)} |')
        out.append('')
        standalone = group(full, 'decode', 'baseline_')
        mixed = group(full, 'decode', 'mixed_')
        append_table(out, 'Decode 干扰', [('纯 Decode（前缀已预热）', standalone),
                                         ('长 Prefill 注入期间的 Decode', mixed)])
        by_stage = {r['stage']: r for r in full if r.get('role') == 'decode'}
        for idx in range(30):
            normal = by_stage.get(f'baseline_{idx}')
            perturbed = by_stage.get(f'mixed_{idx}')
            if not (normal and perturbed):
                continue
            t1, t2 = normal.get('tpot_ms'), perturbed.get('tpot_ms')
            long = group(full, 'long_warm', f'mixed_{idx}')
            overlap = None
            if long and isinstance(perturbed.get('started_at'), (float, int)) and isinstance(perturbed.get('e2e_s'), (float, int)):
                at = long[-1].get('started_at')
                overlap = isinstance(at, (float, int)) and perturbed['started_at'] <= at < perturbed['started_at'] + perturbed['e2e_s']
            ratio = f'{(t2 / t1 - 1) * 100:+.1f}%' if isinstance(t1, (int, float)) and isinstance(t2, (int, float)) and t1 else '—'
            base_rate = normal.get('output_tokens', 0) / normal['e2e_s'] if isinstance(normal.get('e2e_s'), (float, int)) and normal['e2e_s'] else None
            mixed_rate = perturbed.get('output_tokens', 0) / perturbed['e2e_s'] if isinstance(perturbed.get('e2e_s'), (float, int)) and perturbed['e2e_s'] else None
            out.append(f'- 第 {idx+1} 轮：Decode TPOT {fmt(t1)} → {fmt(t2)} ms，变化 {ratio}；'
                       f'单请求输出 {fmt(base_rate)} → {fmt(mixed_rate)} tok/s；'
                       f'长请求发起时 Decode 是否仍在运行：{overlap}。')
            if long and isinstance(long[-1].get('ttft_s'), (int, float)):
                start = long[-1]['started_at']
                duration = long[-1]['ttft_s']
                bstart = normal.get('started_at', 0) + (normal.get('ttft_s') or 0)
                bcount = event_progress(normal, bstart, bstart + duration)
                mcount = event_progress(perturbed, start, start + duration)
                during = [r for r in timeline if isinstance(r.get('at'), (int, float)) and start <= r['at'] <= start + duration]
                kvmin = min((r['kv_available_tokens'] for r in during if isinstance(r.get('kv_available_tokens'), (int, float))), default=None)
                queued = max((r['num_queue_reqs'] for r in during if isinstance(r.get('num_queue_reqs'), (int, float))), default=None)
                out.append(f'  长请求 TTFT {fmt(duration)} s 窗口内，客户端观察到的 Decode 输出：'
                           f'基线 {bcount if bcount is not None else "—"} / 混合 {mcount if mcount is not None else "—"} token；'
                           f'混合期间 rank0 KV 可用最小 {fmt(kvmin, 0)} token，排队峰值 {fmt(queued, 0)}。')
        out.append('')
    s8, f8 = all_data[8192]
    s1, f1 = all_data[1024]
    if f8 and f1:
        out.extend(('## 两种 Chunk 的比较', ''))
        for label, role, prefix in [('长暖缓存 TTFT', 'long_warm', 'mixed_'),
                                     ('混合态 Decode TPOT', 'decode', 'mixed_'),
                                     ('纯 Decode TPOT', 'decode', 'baseline_')]:
            field = 'ttft_s' if 'TTFT' in label else 'tpot_ms'
            a8 = q([r.get(field) for r in group(f8, role, prefix)], .5)
            a1 = q([r.get(field) for r in group(f1, role, prefix)], .5)
            out.append(f'- {label} P50：8K={fmt(a8)}，1K={fmt(a1)} '
                       f'（1K 相对 8K 变化 {fmt((a1 / a8 - 1) * 100, 1) + "%" if a1 is not None and a8 else "—"}）。')
        out.append('')
        out.extend(('### 长上下文横向对照', '',
                    '| 场景 | 8K TTFT(s) | 1K TTFT(s) | 1K 相对 8K | 8K TPOT(ms) | 1K TPOT(ms) |',
                    '|---|---:|---:|---:|---:|---:|'))
        for label, role, prefix in [
                ('54.6K 冷缓存', 'long_cold', 'cold'),
                ('54.6K 暖缓存（有背景 Decode）', 'long_warm', 'mixed_'),
                ('16K 输入/60% 命中', 'long_warm', 'length_16384'),
                ('32K 输入/60% 命中', 'long_warm', 'length_32768')]:
            stage8 = group(f8, role, prefix)
            stage1 = group(f1, role, prefix)
            tt8 = q([r.get('ttft_s') for r in stage8], .5)
            tt1 = q([r.get('ttft_s') for r in stage1], .5)
            pt8 = q([r.get('tpot_ms') for r in stage8], .5)
            pt1 = q([r.get('tpot_ms') for r in stage1], .5)
            change = (tt1 / tt8 - 1) * 100 if tt8 and tt1 is not None else None
            out.append(f'| {label} | {fmt(tt8)} | {fmt(tt1)} | '
                       f'{fmt(change, 1) + "%" if change is not None else "—"} | {fmt(pt8)} | {fmt(pt1)} |')
        out.append('')
        for chunk, full in [(8192, f8), (1024, f1)]:
            by_id = {r['stage']: r for r in full if r.get('role') == 'decode'}
            penalties = []
            for idx in range(30):
                base_tpot = by_id.get(f'baseline_{idx}', {}).get('tpot_ms')
                mixed_tpot = by_id.get(f'mixed_{idx}', {}).get('tpot_ms')
                if isinstance(base_tpot, (float, int)) and isinstance(mixed_tpot, (float, int)) and base_tpot > 0:
                    penalties.append((mixed_tpot / base_tpot - 1) * 100)
            out.append(f'- Chunk={chunk}：混合态相对自身纯 Decode 的成对 TPOT 增幅'
                       f'（{len(penalties)} 轮）为 {[round(v, 2) for v in penalties]}%，中位数 {fmt(statistics.median(penalties) if penalties else None, 1)}%。')
        out.append('')
    if f8 and f1:
        cold8 = q([r.get('ttft_s') for r in group(f8, 'long_cold', 'cold')], .5)
        cold1 = q([r.get('ttft_s') for r in group(f1, 'long_cold', 'cold')], .5)
        warm8 = q([r.get('ttft_s') for r in group(f8, 'long_warm', 'mixed_')], .5)
        warm1 = q([r.get('ttft_s') for r in group(f1, 'long_warm', 'mixed_')], .5)
        out.extend(('## 主要发现', '',
                    f'- 在这个单机非混合调度条件下，Chunk 从 8192 缩到 1024，54.6K 冷缓存请求的 TTFT 从 {fmt(cold8)} s 升到 {fmt(cold1)} s；有背景 Decode 的 60% 命中请求从 {fmt(warm8)} s 升到 {fmt(warm1)} s。16K、32K 对照的变化见上表。',
                    '- 背景 Decode 在两种 Chunk 下都有短时停顿，2 轮成对 TPOT 增幅见上方列表；这组小样本没有显示 1K 能改善干扰。缩小 Chunk 的成本主要体现在 Prefill/首 token 时间，单独 Decode 的 TPOT 接近。',
                    '- 该结果对应 `enable_mixed_chunk=False`、禁用 graph、最大同时运行 4 请求与本机 16 张 NPU；不能直接外推到启用混合调度、更高并发或 PD 分离服务。', ''))
    end_state = root / 'end_state.txt'
    if end_state.is_file():
        out.extend(('## 实验结束时服务状态', '', '```text', end_state.read_text(encoding='utf-8').strip(), '```',
                    '完整 NPU 状态快照保存在运行目录的 `npu_smi_after.txt`。', ''))
    out.extend(('## 调度器源码与归因', '',
                '- 本轮两种配置的启动日志均显示 `enable_mixed_chunk=False`。固定镜像的 `sglang/srt/managers/scheduler.py` 中，约 1225 行由该开关决定 `is_mixed_chunk`；约 3305 行显示一旦有新 Prefill batch，先运行 Prefill，只有没有新 Prefill batch 时才选运行中的 Decode；约 3669 行的 Prefill/Decode 合并逻辑只在 `is_mixed_chunk=True` 时执行。',
                '- 因此当前矩阵测的是**默认不混合调度**下两种 Prefill 分块上限的性能。分块变小增加了 Prefill 轮数，但不会自动保证 Decode 在两个块之间执行；短时 Decode 输出停顿须结合这项调度规则解释。',
                '- `--enable-mixed-chunk` 会改变调度方式，若后续想测试交错执行带来的收益，应在固定其余参数的前提下单独做开关对照，并先验证 Ascend 后端兼容性。', ''))
    out.extend(('## 启动过程与容量修正', '',
                '- `results/20260926_143359` 首次启动在部分 rank 读取权重未完成时触发镜像内 480 秒的模型加载 barrier；已将固定镜像相应常量的单行补丁存于 `patches/`。',
                '- `results/20260926_144841` 把 barrier 延至 3600 秒后，16 个 rank 均完成权重加载，但 `mem-fraction-static=0.93` 时只剩约 2.65 GB，服务预热的 DeepEP/HCCL 显存申请失败。正式两组统一改为 0.88、normal 模式 HCCL 650 MB，KV 池 63744 token、算子可用约 6.25 GB。', '',
                '## 解释边界', '',
                '- 冷态输入 token 未命中率与暖态实际 `cached_tokens` 分开报告；若服务没有给出 `cached_tokens`，不能声称达到了 60%。',
                '- SSE 可能合批回传，事件计数窗口和间隔只能辅助确认时序，逐 token 尾部 ITL 需要服务端细粒度时间戳；长请求的 TTFT 窗口还包括排队、网络时间。',
                '- 模型加载失败、单机 KV 不足、输出 token 数不足时，相应配置没有可比较的性能结果。', ''))
    target = a.out or root / 'report.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(target)

if __name__ == '__main__':
    main()
