#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4Exp 图（**只用 agentX 负载**）：A / B / C 三组对比。输出到 Q4Exp/figs/。

约定
  * layer 轴统一 origin="lower"（0 在下、77 在上）
  * 图内文字一律英文（本机无中文字体）；中文说明放 figs/表格.md
  * 逻辑热度 = logical_count_* 对"步"求和 [78,256]（固定逻辑编号，不排序）
  * 物理热度 = 按副本数均分（q4exp_batch.phys_heat；冗余时不做重复计数）
  * 合计 = prefill + decode/8（decode 计数单位是 prefill 的 8 倍）
  * 跨组比较"每槽"物理量时共用色标；跨时间看累积也共用色标
"""
import argparse, csv, json, pathlib, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q4exp_batch import (load_windows, rank_load, phys_heat, pd_overlap, RES, COL,
                         PRE_UNIT, DEC_UNIT, DEC_FIX)

OUT = HERE / "figs"
RUNS = {"A": "A_agentx_20261012_050236", "B": "B_agentx_20261012_052314",
        "C": "C_agentx_20261012_054412"}
G3 = ("A", "B", "C")
DEC = DEC_FIX


def sum_heat(ws, ph="total"):
    if ph == "prefill":
        return sum(w["pre"] for w in ws)
    if ph == "decode":
        return sum(w["dec"] for w in ws)
    return sum(w["pre"] + w["dec"] / DEC for w in ws)


def nrm(H):
    s = H.sum(1, keepdims=True)
    return H / np.where(s == 0, 1, s)


def ylab(ax):
    ax.set_ylabel("Layer")


def heat_title(group, phase, kind, window=None):
    title = f"{group} | {kind} | {phase.title()}"
    if window is not None:
        title += f" | win{window}"
    return title


def rebal_durations(run):
    f = RES / run / "rebalance_events.txt"
    if not f.exists():
        return []
    v = sorted(set(float(x) for x in re.findall(r"rebalance end time=([0-9.]+)s", f.read_text())))
    if not v:
        return []
    grp, cur = [], [v[0]]
    for t in v[1:]:
        if t - cur[-1] <= 1.0:
            cur.append(t)
        else:
            grp.append(cur); cur = [t]
    grp.append(cur)
    return [float(np.median(g)) for g in grp]


def main(*, plots_only=False, extra_figures=False):
    W = {g: load_windows(r) for g, r in RUNS.items()}
    print("windows per group:", {g: len(w) for g, w in W.items()})

    # ---------------- 图1 speed ----------------
    sp = {}
    for g in G3:
        tp = td = dp = dd = 0.0
        for w in W[g]:
            st = w["steps"]
            for i, (t, ph, c) in enumerate(st):
                if i == 0:
                    continue
                dur = t - st[i - 1][0]
                if dur <= 0 or dur > 300:
                    continue
                if ph == "prefill":
                    tp += c / PRE_UNIT; dp += dur
                else:
                    td += c / DEC_UNIT; dd += dur
        sp[g] = (tp / dp, td / dd)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, i, ttl in ((axes[0], 0, "Prefill"),
                       (axes[1], 1, "Decode")):
        v = [sp[g][i] for g in G3]
        ax.bar(range(3), v, color=[COL[g] for g in G3])
        ax.set_xticks(range(3)); ax.set_xticklabels(G3)
        ax.set_ylabel("tokens/s"); ax.set_title(ttl); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Fig1 | Prefill / Decode speed")
    fig.tight_layout(); fig.savefig(OUT / "fig1_speed_bar.png", dpi=110); plt.close(fig)

    # ---------------- 图2 E2E breakdown ----------------
    seg = {}
    for g in G3:
        rows = [json.loads(l) for l in open(RES / RUNS[g] / "bench" / "requests.jsonl")]
        rows = [r for r in rows if isinstance(r.get("meta_info"), dict) and r.get("status") == "ok"]
        acc = {k: [] for k in ("wait", "queue", "prefill", "first_send", "decode")}
        for r in rows:
            m = r["meta_info"]
            try:
                rf = m["forward_entry_time"] - m["request_received_ts"]; q = m.get("queue_time", 0.0)
                pf = m["prefill_finished_time"] - m["forward_entry_time"]
                fs = m["response_sent_to_client_ts"] - m["prefill_finished_time"]
                dc = m["request_finished_ts"] - m["response_sent_to_client_ts"]
            except KeyError:
                continue
            acc["wait"].append(max(rf - q, 0)); acc["queue"].append(q); acc["prefill"].append(pf)
            acc["first_send"].append(fs); acc["decode"].append(dc)
        seg[g] = {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    order = [("wait", "Wait", "#f0ad4e"),
             ("queue", "Queue", "#d9534f"),
             ("prefill", "Prefill", "#5bc0de"),
             ("first_send", "First-token send", "#cccccc"),
             ("decode", "Decode", "#5cb85c")]
    left = np.zeros(3)
    for k, nm, c in order:
        v = np.array([seg[g][k] for g in G3]); ax.barh(range(3), v, left=left, color=c, label=nm); left += v
    ax.set_yticks(range(3)); ax.set_yticklabels(G3)
    ax.set_xlabel("seconds (mean per request)"); ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3, axis="x"); ax.set_title("Fig2 | E2E breakdown")
    fig.tight_layout(); fig.savefig(OUT / "fig2_e2e_breakdown_barh.png", dpi=110); plt.close(fig)

    # ---------------- 图3+4：逻辑热度（TOTAL / PREFILL / DECODE × A/B/C）----------------
    Hs = {g: sum_heat(W[g]) for g in G3}
    grid = {}
    for g in G3:
        grid[("total", g)] = Hs[g]
        grid[("prefill", g)] = sum_heat(W[g], "prefill")
        grid[("decode", g)] = sum_heat(W[g], "decode")
    av = np.concatenate([nrm(v).ravel() for v in grid.values()])
    vmax = float(np.percentile(av[av > 0], 99.5))
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    for i, ph in enumerate(("total", "prefill", "decode")):
        for j, g in enumerate(G3):
            ax = axes[i][j]
            im = ax.imshow(nrm(grid[(ph, g)]), aspect="auto", cmap="turbo",
                           interpolation="nearest", vmin=0, vmax=vmax, origin="lower")
            ax.set_title(heat_title(g, ph, "Logical"), fontsize=12)
            ax.set_xlabel("Logical expert ID"); ylab(ax)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig3+4 | Logical heat", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fig3_4_logical_heat.png", dpi=110); plt.close(fig)

    # ---------------- 图3b：归一化（log 色标，所有格子可见）----------------
    ratio = {g: nrm(Hs[g]) * 256 for g in G3}
    avv = np.concatenate([v[v > 0].ravel() for v in ratio.values()])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, g in zip(axes, G3):
        im = ax.imshow(ratio[g], aspect="auto", cmap="turbo", interpolation="nearest",
                       norm=LogNorm(vmin=0.1, vmax=np.percentile(avv, 99.9)), origin="lower")
        ax.set_title(heat_title(g, "total", "Logical"), fontsize=10)
        ax.set_xlabel("Logical expert ID"); ylab(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, label="Routing share / layer mean")
    fig.suptitle("Fig3b | Normalized logical heat")
    fig.tight_layout()
    if extra_figures:
        fig.savefig(OUT / "fig3b_logical_normalized.png", dpi=110)
    plt.close(fig)

    # ---------------- 图3c：人眼版 ----------------
    S = {g: nrm(Hs[g]) for g in G3}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[0]
    x, y = S["B"].ravel(), S["C"].ravel(); m = (x > 0) & (y > 0)
    ax.scatter(x[m], y[m], s=3, alpha=.25, edgecolors="none", color="tab:purple")
    lim = [1e-5, 0.3]; ax.plot(lim, lim, "r--", lw=1.2, label="y = x (identical)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("share  B (EPLB on)"); ax.set_ylabel("share  C (EPLB + red64)")
    ax.set_title("Logical | Total | B / C", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    for g in G3:
        ax.plot(range(78), [S[g][l].max() for l in range(78)], "o-", ms=3, lw=1.2, color=COL[g], label=g)
    ax.set_xlabel("layer"); ax.set_ylabel("top-1 expert share")
    ax.set_title("Logical | Total | Top expert share", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    for g in G3:
        cums = [np.cumsum(np.sort(S[g][l])[::-1]) for l in range(78)]
        ax.plot(np.mean(cums, axis=0)[:80], "o-", ms=3, lw=1.3, color=COL[g], label=g)
    ax.axhline(0.5, color="gray", ls=":"); ax.set_xlabel("#experts (sorted desc)")
    ax.set_ylabel("cumulative share"); ax.set_title("Logical | Total | Cumulative share", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Fig3c | Logical heat comparison", fontsize=12)
    fig.tight_layout()
    if extra_figures:
        fig.savefig(OUT / "fig3c_logical_readable.png", dpi=110)
    plt.close(fig)

    # ---------------- 图5：累计快照（按时间对齐 A/B/C）----------------
    BINS = [180, 360, 540, 720, 900]
    t00 = {g: min(w["t0"] for w in W[g] if w["t0"]) for g in G3}
    pick = {g: [[w for w in W[g] if w["t1"] and (w["t1"] - t00[g]) <= T] for T in BINS] for g in G3}
    cum = {(g, T, ph): (sum_heat(pick[g][bi], ph) if pick[g][bi] else None)
           for g in G3 for bi, T in enumerate(BINS) for ph in ("total", "prefill", "decode")}
    v5 = max(float(np.percentile(H, 99.5)) for H in cum.values() if H is not None and H.sum() > 0)
    # 只保留【最终】的累计热度：3 行 = TOTAL / PREFILL / DECODE，3 列 = A/B/C（原始计数，每行共用色标）
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    for i_, ph in enumerate(("total", "prefill", "decode")):
        vm = max(float(np.percentile(grid[(ph, g)], 99.5)) for g in G3)
        for j_, g in enumerate(G3):
            H = grid[(ph, g)]
            ax = axes[i_][j_]
            im = ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest",
                           vmin=0, vmax=vm, origin="lower")
            ax.set_title(heat_title(g, ph, "Logical"), fontsize=10)
            ax.set_xlabel("Logical expert ID")
            plt.colorbar(im, ax=ax, fraction=0.046)
        axes[i_][0].set_ylabel(f"{ph.title()}\nLayer", fontsize=9)
    fig.suptitle("Fig5a | Final cumulative logical heat", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig5a_logical_snapshots_total_by_time.png", dpi=110); plt.close(fig)

    # 图5b：拆成三张（每张一个组），避免 6 行太挤；色标仍与 5a 共用（v5）
    for g in G3:
        fig, axes = plt.subplots(2, len(BINS), figsize=(2.9 * len(BINS), 2.3 * 2), squeeze=False)
        for i, ph in enumerate(("prefill", "decode")):
            for k, T in enumerate(BINS):
                ax = axes[i][k]; ax.set_xticks([]); ax.set_yticks([])
                H = cum[(g, T, ph)]
                if H is None:
                    ax.text(.5, .5, "no window", ha="center", fontsize=8); continue
                ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=v5,
                          origin="lower")
                ax.set_title(heat_title(g, ph, "Logical", k + 1), fontsize=10)
                if k == 0:
                    ax.set_ylabel(ph.title(), fontsize=12)
        fig.suptitle(f"Fig5b-{g} | Cumulative logical heat", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"fig5b_logical_snapshots_pd_{g}.png", dpi=110); plt.close(fig)

    # ---------------- 图6：物理槽热度 ----------------
    Ph = {g: phys_heat(Hs[g], W[g][-1]["pm"]) for g in G3}
    # fig6a: 与 fig5a 同布局（3 行 TOTAL/PREFILL/DECODE × 3 列 A/B/C），物理槽、原始计数、每行共用色标
    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    for i_, ph in enumerate(("total", "prefill", "decode")):
        Hp = {g: phys_heat(grid[(ph, g)], W[g][-1]["pm"]) for g in G3}
        vm = max(float(np.percentile(Hp[g], 99.5)) for g in G3)
        for j_, g in enumerate(G3):
            H = Hp[g]
            ax = axes[i_][j_]
            im = ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest",
                           vmin=0, vmax=vm, origin="lower")
            ax.set_title(heat_title(g, ph, "Physical"), fontsize=10)
            ax.set_xlabel("Physical slot")
            plt.colorbar(im, ax=ax, fraction=0.046)
        axes[i_][0].set_ylabel(f"{ph.title()}\nLayer", fontsize=9)
    fig.suptitle("Fig6a | Final cumulative physical heat", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig6a_physical_heat_total.png", dpi=110); plt.close(fig)

    v6b = max(float(np.percentile(phys_heat(sum_heat(W[g], ph), W[g][-1]["pm"]), 99.5))
              for g in G3 for ph in ("prefill", "decode"))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for j, g in enumerate(G3):
        for i, ph in enumerate(("prefill", "decode")):
            H = phys_heat(sum_heat(W[g], ph), W[g][-1]["pm"])
            im = axes[i][j].imshow(H, aspect="auto", cmap="turbo", interpolation="nearest",
                                   vmin=0, vmax=v6b, origin="lower")
            axes[i][j].set_title(heat_title(g, ph, "Physical"), fontsize=10)
            axes[i][j].set_xlabel("Physical slot"); ylab(axes[i][j])
            plt.colorbar(im, ax=axes[i][j], fraction=0.046)
    fig.suptitle("Fig6b | Physical heat", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig6b_physical_heat_pd.png", dpi=110); plt.close(fig)

    # 图6c：物理槽热度，PD 分离 + 按时间桶累计，拆成 A/B/C 三张（与 fig5b 同布局；线性 turbo + 共用色标）
    v6c = 0.0
    ph_cum = {}
    for g in G3:
        for k in range(len(BINS)):
            for ph in ("prefill", "decode"):
                H = phys_heat(sum_heat(pick[g][k], ph), W[g][-1]["pm"])
                ph_cum[(g, k, ph)] = H
                v6c = max(v6c, float(np.percentile(H, 99.5)))
    for g in G3:
        fig, axes = plt.subplots(2, len(BINS), figsize=(2.9 * len(BINS), 2.3 * 2), squeeze=False)
        for i, ph in enumerate(("prefill", "decode")):
            for k in range(len(BINS)):
                H = ph_cum[(g, k, ph)]
                ax = axes[i][k]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest",
                          vmin=0, vmax=v6c, origin="lower")
                ax.set_title(heat_title(g, ph, "Physical", k + 1), fontsize=10)
                if k == 0:
                    ax.set_ylabel(ph.title(), fontsize=12)
        fig.suptitle(f"Fig6c-{g} | Cumulative physical heat", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"fig6c_physical_snapshots_{g}.png", dpi=110); plt.close(fig)

    # ---------------- 图7 逐窗口 rank 不均衡 ----------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for g in G3:
        rr = []
        for w in W[g]:
            pm, ep, P = w["pm"], w["ep"], w["P"]
            lp = rank_load(w["pre"], pm, ep, P)
            lt = rank_load(w["pre"] + w["dec"] / DEC, pm, ep, P)
            rr.append((lp.max() / lp.mean(), lt.max() / lt.mean()))
        ax.plot(range(len(rr)), [a for a, _ in rr], "o--", ms=5, color=COL[g], label=f"{g} | Physical | Prefill")
        ax.plot(range(len(rr)), [b for _, b in rr], "s-", ms=5, color=COL[g], label=f"{g} | Physical | Total")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("Window index")
    ax.set_ylabel("rank load max/mean"); ax.set_title("Fig7 | Physical rank imbalance")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout()
    if extra_figures:
        fig.savefig(OUT / "fig7_rank_imbalance_time.png", dpi=110)
    plt.close(fig)

    # ---------------- 图8 结论图 ----------------
    hot, imb = {}, {}
    for g in G3:
        H = Hs[g]; pm, ep, P = W[g][-1]["pm"], W[g][-1]["ep"], W[g][-1]["P"]
        k = P // ep
        ranks = []
        for l in range(78):
            if H[l].sum() == 0:
                continue
            e = int(H[l].argmax()); ranks.append(int(np.where(pm[l] == e)[0][0]) // k)
        hot[g] = float(np.mean([x == 0 for x in ranks]))
        lt = rank_load(H, pm, ep, P); imb[g] = lt.max() / lt.mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.bar(range(3), [hot[g] for g in G3], color=[COL[g] for g in G3])
    ax.set_xticks(range(3)); ax.set_xticklabels(G3)
    ax.set_ylabel("layers whose hottest expert is in rank0"); ax.set_title("Physical | Total | Peak alignment")
    ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    ax.bar(range(3), [imb[g] for g in G3], color=[COL[g] for g in G3])
    ax.axhline(1.0, color="gray", ls=":"); ax.set_xticks(range(3))
    ax.set_xticklabels(G3)
    ax.set_ylabel("rank load max/mean"); ax.set_title("Physical | Total | Rank imbalance")
    ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    for i, g in enumerate(G3):
        for d in rebal_durations(RUNS[g]):
            ax.scatter(i, d, s=70, color=COL[g])
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(G3)
    ax.set_ylabel("rebalance duration (s, log)"); ax.set_title("EPLB rebalance")
    ax.grid(alpha=.3, axis="y")
    fig.suptitle("Fig8 | A / B / C comparison")
    fig.tight_layout()
    if extra_figures:
        fig.savefig(OUT / "fig8_story.png", dpi=110)
    plt.close(fig)

    if plots_only:
        print("agentX plots regenerated:", OUT)
        return

    # ---------------- 表格 ----------------
    L = ["### agentX（输入 16384 / 输出 512 token；只与同负载的其他组比）\n",
         "| **指标** | **EPLB off (A)** | **EPLB on (B)** | **EPLB+冗余 (C)** |",
         "| :---: | :---: | :---: | :---: |"]
    cols = {}
    for g in G3:
        run = RUNS[g]
        d = json.load(open(RES / run / "bench" / "summary.json"))
        rows = [json.loads(l) for l in open(RES / run / "bench" / "requests.jsonl")]
        rows = [r for r in rows if r.get("status") == "ok"]
        tt = np.array([r["ttft_s"] for r in rows]); tp = np.array([r["tpot_s"] for r in rows]) * 1000
        ee = np.array([r["e2e_s"] for r in rows])
        cols[g] = dict(done=f"{d['completed']} / {d['errors']}", wall=f"{d['actual_wall_s']:.1f}s",
                       tput=f"{d['output_tokens_per_s']:.1f} tok/s",
                       pf=f"{sp[g][0]:,.1f} tok/s", dc=f"{sp[g][1]:,.1f} tok/s",
                       tt=(tt.mean(), np.percentile(tt, 50), np.percentile(tt, 90), np.percentile(tt, 99)),
                       tp=(tp.mean(), np.percentile(tp, 50), np.percentile(tp, 90)),
                       ee=(ee.mean(), np.percentile(ee, 50), np.percentile(ee, 90)))
    for name, key, fmt in (("完成/错误", "done", "{}"), ("墙钟", "wall", "{}"), ("吞吐(输出)", "tput", "{}"),
                           ("Prefill速度", "pf", "{}"), ("Decode速度", "dc", "{}"),
                           ("TTFT mean", "tt0", "{:.2f}s"), ("TTFT p50", "tt1", "{:.2f}s"),
                           ("TTFT p90", "tt2", "{:.2f}s"), ("TTFT p99", "tt3", "{:.2f}s"),
                           ("TPOT mean", "tp0", "{:.1f}ms"), ("TPOT p50", "tp1", "{:.1f}ms"),
                           ("TPOT p90", "tp2", "{:.1f}ms"),
                           ("E2E mean", "ee0", "{:.1f}s"), ("E2E p50", "ee1", "{:.1f}s"),
                           ("E2E p90", "ee2", "{:.1f}s")):
        cells = []
        for g in G3:
            c = cols[g]
            if key in ("done", "wall", "tput", "pf", "dc"):
                cells.append(c[key])
            else:
                i = int(key[-1]); cells.append(fmt.format(c[key[:-1]][i]))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["\n### 补充：rank 不均衡 与 尖峰对齐（来自 windows/*.pt）\n",
          "| **指标** | **EPLB off (A)** | **EPLB on (B)** | **EPLB+冗余 (C)** |",
          "| :---: | :---: | :---: | :---: |"]
    pre_imb = {g: (lambda v: v.max() / v.mean())(rank_load(sum_heat(W[g], "prefill"),
                W[g][-1]["pm"], W[g][-1]["ep"], W[g][-1]["P"])) for g in G3}
    for nm, d, fmt in (("不均衡 prefill", pre_imb, "{:.2f}×"), ("不均衡 合计(corrected)", imb, "{:.2f}×"),
                       ("最热专家落 rank0 的层占比", hot, "{:.0%}")):
        L.append(f"| {nm} | " + " | ".join(fmt.format(d[g]) for g in G3) + " |")
    L += ["""
### 负载是怎么打的（输入详细描述）

**编排**：`run_q4exp.sh` 起服务 → 预热 → 跑 `bench_agentx.py` 负载 900s → 每 60s 抓 `/metrics` 与 `npu-smi` → 收尾 dump。
三组（A/B/C）用**同一份数据集、同一并发、同一样本参数**，只有 EPLB 配置不同 → 组间可比。

**并发模型（closed loop，8 个 worker）**
- 8 个 worker 共享一个**顺序游标**：每个 worker 取"数据集的下一行"→ 发送 → 流式收完 → **等这个请求完全结束**，才取下一行。
- 所以**每个 worker 是串行的**（上一个跑完才发下一个），**8 个 worker 之间并行** → 稳态在飞 ≈ 8 个请求。
- **不是**固定 QPS、**不是**按到达时间重放、**不是** round-robin 轮询队列；数据集行用完才会从头循环（本轮 900s 没跑完 4096 行）。

**请求格式**
- payload = `{"input_ids": [...], "sampling_params": {...}, "stream": true}` → **直接送 token id，服务端不做 tokenize**（所以 TTFT 里没有 tokenize 开销）。
- `sampling = {temperature: 0, ignore_eos: True, max_new_tokens: N}` → 每个请求**固定输出 N 个 token**（不会被 EOS 提前截断），因此"输入/输出长度"完全确定。

**两种数据集**
| 负载 | 数据集 | 行数 | 每请求 | 采样参数 |
|---|---|---|---|---|
| agentX | `history_exp/05_.../data/agentx_16k_512.json` | 4096 行（取自 391 条 trace；源 `cc-traces-.../traces.jsonl`，token 由 SHA256 确定重建） | **16384 in / 512 out** | temperature 0, ignore_eos |
| random | `pd_colloc_20261011/data/random_8k_64.json` | 256 行 | **8192 in / 64 out** | temperature 0, ignore_eos |

⚠️ **agentX 实测有个特点**：本轮只用到前 160 行，而这 160 行来自 **20 条 trace、内容只有 20 种**（每 8 行内容一致）。
所以**同一"波次"里 8 个在飞请求的 prompt 内容完全一样**（只是行号/trace 不同）；下一波（≈42s 后）才换内容。
random 则是 256 行内容各不相同，8 个 worker 顺序走。

**实测规模（本轮）**
| | 请求数 | 墙钟 | 输出 token | 吞吐 | TTFT p50 | TPOT p50 | E2E p50 |
|---|---|---|---|---|---|---|---|
| agentX | 160 | 914–947s | 81,920 | 86–90 tok/s | 20.3–20.4s | 50.0–50.8ms | 46.0–46.5s |
| random | 584–624 | ~903s | 37,376–39,936 | 41–44 tok/s | 5.7–7.8s | 49.6–51.2ms | 8.9–10.9s |

**对分析的影响（重要）**
1. `cached_tokens = 0`（同波次 8 个请求同时发出，前缀缓存来不及命中）→ prefill 是全量计算，热度统计干净。
2. 同波次 prompt 内容相同 → 窗口之间的热度天然稳定；但 random（内容各异）也稳定 → 说明稳定性**不是**内容造成的（这正是结论之一）。
3. 三组 dump 策略不同：A 每 60s（13 个窗口）；B/C 只在重排时（每窗口 = 一个重排周期，5 个）→ 图 5/6 的"按时间桶"对齐就是为此。
"""]

    L.append("\n> 口径：合计 = prefill + decode/8；物理热度/负载 = 冗余副本按均分。"
             "A 组窗口 60s（13 个），B/C 每窗口 = 一个重排周期（5 个）→ 图5 按时间对齐。")
    (OUT / "表格.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("\n✓ agentX 图与表已生成:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true", help="Regenerate plots without rewriting the Markdown tables.")
    parser.add_argument("--extra-figures", action="store_true", help="Also save the additional diagnostic figures.")
    args = parser.parse_args()
    main(plots_only=args.plots_only, extra_figures=args.extra_figures)
