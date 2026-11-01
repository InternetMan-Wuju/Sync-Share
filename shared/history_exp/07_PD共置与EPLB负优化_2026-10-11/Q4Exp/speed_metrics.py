#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prefill / decode 速度（严格按计算时间，不含排队等待）。

两个口径：
  A. 步级（系统）：把窗口里 prefill/decode 的逐步时间戳按时间合并 → 相邻差 = 该步的计算时长；
     token 数 = 该步计数 / 600（1 token 过一遍模型 = 75 MoE 层 x top-8 = 600 次计数）
     速度 = token / 步时长
  B. 请求级（单请求）：
     prefill 速度 = prompt_tokens / (prefill_finished_time - forward_entry_time)   ← 真正的 prefill 计算时间
     decode  速度 = 1 / TPOT，或 output_tokens / (request_finished_ts - response_sent_to_client_ts)

用法: python3 speed_metrics.py --run results/20261012_smokeA
"""
import argparse, glob, json, pathlib
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOE_LAYERS, TOPK = 75, 8
PREFILL_COUNTS_PER_TOKEN = MOE_LAYERS * TOPK   # 600  —— 已用真实 prompt token 数验证 ✓
# ⚠️ decode 阶段的计数单位与 prefill 不同：同一份数据里
#    prefill 计数 / 真实 prompt token = 581~600   (≈600 ✓)
#    decode  计数 / 真实 output token = 4800      (是 600 的 8 倍 —— LL dispatch 路径的计数口径差异)
# 所以 decode 必须用 4800 换算，否则 token 数会被高估 8 倍。
DECODE_COUNTS_PER_TOKEN = 4800


def step_table(win_path):
    o = torch.load(win_path, map_location="cpu", weights_only=False)
    ts = o["step_timestamps_by_phase"]
    rows = []
    for key, ckey in (("prefill", "logical_count_prefill"), ("decode", "logical_count_decode")):
        t = o.get(ckey); times = list(ts.get(key) or [])
        if t is None or not times:
            continue
        # 计数行数可能比时间戳多 1~2 行（计数是全局 all_reduce 的并集，时间戳只有 rank0 的），
        # 对齐时保留末尾 len(times) 行。
        if t.shape[0] > len(times):
            t = t[-len(times):]
        elif t.shape[0] < len(times):
            times = times[-t.shape[0]:]
        for i in range(t.shape[0]):
            unit = PREFILL_COUNTS_PER_TOKEN if key == "prefill" else DECODE_COUNTS_PER_TOKEN
            rows.append(dict(phase=key, t=float(times[i]), counts=int(t[i].sum()),
                             tokens=int(t[i].sum()) / unit))
    rows.sort(key=lambda r: r["t"])
    # 时间戳记录的是【该步结束时刻】→ 步时长 = 本步时间戳 - 上一步时间戳（后向差分）
    for i, r in enumerate(rows):
        r["dur"] = (r["t"] - rows[i - 1]["t"]) if i > 0 else np.nan   # 每个窗口第一步无上一步
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    wins = sorted(glob.glob(str(run / "windows" / "*.pt")))
    all_rows = []
    for w in wins:
        rs = step_table(w)
        for r in rs:
            r["win"] = pathlib.Path(w).stem[:6]
        all_rows += rs
    df = [r for r in all_rows if not np.isnan(r.get("dur", np.nan))]
    print(f"窗口 {len(wins)} 个；可用步 {len(df)}（prefill {sum(1 for r in df if r['phase']=='prefill')} / "
          f"decode {sum(1 for r in df if r['phase']=='decode')}）")

    # ---------- 步级 ----------
    print("\n=== A. 步级（系统口径）===")
    print(f"{'阶段':>8} {'步数':>5} {'token合计':>12} {'计算时间合计':>12} {'速度 tokens/s':>14} {'单步token中位':>13} {'单步时长中位':>13}")
    summary = {}
    for ph in ("prefill", "decode"):
        rs = [r for r in df if r["phase"] == ph]
        if not rs:
            continue
        tok = sum(r["tokens"] for r in rs); dur = sum(r["dur"] for r in rs)
        sp = tok / dur
        summary[ph] = (len(rs), tok, dur, sp)
        print(f"{ph:>8} {len(rs):>5} {tok:>12,.0f} {dur:>12.1f} {sp:>14,.0f} "
              f"{np.median([r['tokens'] for r in rs]):>13,.0f} {np.median([r['dur'] for r in rs]):>13.3f}")

    # ---------- 请求级 ----------
    print("\n=== B. 请求级（单请求口径，服务端时间戳）===")
    reqs = [json.loads(l) for l in open(run / "bench2" / "requests.jsonl")]
    reqs = [r for r in reqs if isinstance(r.get("meta_info"), dict) and r.get("status") == "ok"]
    pre_sp, dec_sp, pre_t, dec_t, ptok, otok = [], [], [], [], [], []
    for r in reqs:
        m = r["meta_info"]
        p_t = m["prefill_finished_time"] - m["forward_entry_time"]          # 真正的 prefill 计算时间
        d_t = m["request_finished_ts"] - m["response_sent_to_client_ts"]    # decode 段
        pt, ot = r["prompt_tokens"], r["output_tokens"]
        if p_t > 0: pre_sp.append(pt / p_t); pre_t.append(p_t); ptok.append(pt)
        if d_t > 0 and ot: dec_sp.append(ot / d_t); dec_t.append(d_t); otok.append(ot)
    f = lambda v: (np.mean(v), np.median(v), np.percentile(v, 95)) if len(v) else (np.nan,) * 3
    print(f"  prefill: 单请求计算时间 mean={np.mean(pre_t):.2f}s p50={np.median(pre_t):.2f}s "
          f"| 速度 mean={np.mean(pre_sp):,.0f} p50={np.median(pre_sp):,.0f} p95low={np.percentile(pre_sp,5):,.0f} tokens/s")
    print(f"  decode : 单请求时长   mean={np.mean(dec_t):.2f}s p50={np.median(dec_t):.2f}s "
          f"| 速度 mean={np.mean(dec_sp):.1f} p50={np.median(dec_sp):.1f} p5={np.percentile(dec_sp,5):.1f} tokens/s")
    print(f"  （总 prompt tokens {sum(ptok):,}；总输出 tokens {sum(otok):,}）")

    # ---------- 归一化：整轮平均 ----------
    tot_dur = sum(r["dur"] for r in df)
    print("\n=== C. 整轮平均（分母 = 所有步的计算时间之和）===")
    print(f"  prefill: {summary['prefill'][1]:,.0f} tokens / {summary['prefill'][2]:.1f}s = "
          f"{summary['prefill'][1]/summary['prefill'][2]:,.0f} tokens/s")
    print(f"  decode : {summary['decode'][1]:,.0f} tokens / {summary['decode'][2]:.1f}s = "
          f"{summary['decode'][1]/summary['decode'][2]:,.0f} tokens/s")
    print(f"  （prefill 计算时间占总计算时间 {summary['prefill'][2]/tot_dur:.1%}，"
          f"decode 占 {summary['decode'][2]/tot_dur:.1%}）")

    # ---------- 输出 ----------
    (run / "tables").mkdir(exist_ok=True); (run / "figs").mkdir(exist_ok=True)
    with open(run / "tables" / "speed_steps.csv", "w") as fh:
        fh.write("win,phase,tokens,dur_s,tokens_per_s\n")
        for r in df:
            fh.write(f"{r['win']},{r['phase']},{r['tokens']:.0f},{r['dur']:.4f},{r['tokens']/r['dur']:.1f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    for ax, ph, c in ((axes[0], "prefill", "tab:blue"), (axes[1], "decode", "tab:red")):
        rs = [r for r in df if r["phase"] == ph]
        ax.scatter([r["dur"] for r in rs], [r["tokens"] for r in rs], s=22, color=c)
        ax.set_xlabel("step compute time (s)"); ax.set_ylabel("tokens in step")
        ax.set_title(f"{ph}: {summary[ph][3]:,.0f} tokens/s  (n={len(rs)} steps)")
        ax.grid(alpha=.3)
    fig.suptitle("prefill / decode step speed (system wide, 64 cards)")
    fig.tight_layout(); fig.savefig(run / "figs" / "fig6_speed.png", dpi=110); plt.close(fig)
    print("\n✓ tables/speed_steps.csv, figs/fig6_speed.png")


if __name__ == "__main__":
    main()
