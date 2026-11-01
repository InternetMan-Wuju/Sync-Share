#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按用户清单出图（Q4Exp/figs/）+ 报告用表格（Q4Exp/figs/表格.md）。

清单：
  图1 prefill/decode 速度，A/B/C 对比，柱形图
  图2 E2E breakdown，A/B/C 对比，横向堆叠柱
  图3 逻辑专家热度（总和），A/B/C 对比（每层归一化，形状应基本相同）
  图4 逻辑专家热度（PD 分离），A/B/C 对比
  图5 累计 snapshot 热度（总和 + PD 分离），按时间排序（会看到冷->热）
  图6 物理专家热度（总和）A/B/C 对比；以及 PD 分离版、snapshot 版
  表格：figs/表格.md（格式按用户给的样例）

口径：
  * layer 轴统一 0 在顶部（从上往下 0..77）—— 按要求"倒过来"
  * 逻辑热度 = logical_count_* 对步求和 [78,256]
  * 物理热度 phys[l][p] = H[l][ pmap[l][p] ]，维度 256(A/B) / 320(C)
  * decode 计数单位是 prefill 的 8 倍 → 相加时 total = prefill + decode/8
  * 对比图一律按"每层归一化"（除以该层总量），这样窗口数/时长不同的组也能比形状
"""
import csv, glob, json, pathlib, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q4exp_batch import load_windows, rank_load, phys_heat, pd_overlap, RES, COL, PRE_UNIT, DEC_UNIT, DEC_FIX

OUT = HERE / "figs"; OUT.mkdir(parents=True, exist_ok=True)
LOAD = "agentX"
RUNS = {  # 组 -> run 目录
    "A": {"agentX": "A_agentx_20261012_050236", "random": "A_random_20261012_062417"},
    "B": {"agentX": "B_agentx_20261012_052314", "random": "B_random_20261012_064400"},
    "C": {"agentX": "C_agentx_20261012_054412", "random": "C_random_20261012_071801"},
}


def norm_rows(H):
    s = H.sum(1, keepdims=True)
    return H / np.where(s == 0, 1, s)


def phys(H, pm):
    """物理槽热度：冗余时按副本数均分（见 q4exp_batch.phys_heat 的说明）"""
    return phys_heat(H, pm)


def heat_of(ws, phase="total", upto=None):
    ws = ws if upto is None else ws[:upto]
    if phase == "prefill":
        return sum(w["pre"] for w in ws)
    if phase == "decode":
        return sum(w["dec"] for w in ws)
    return sum(w["pre"] + w["dec"] / DEC_FIX for w in ws)


def style_layer_axis(ax, L=78):
    ax.set_ylabel("layer  (0 bottom -> 77 top)")


def panel(ax, H, ttl, norm=False, vmin=0, vmax=None):
    A = norm_rows(H) if norm else H
    if vmax is None:
        vmax = np.percentile(A[A > 0], 99.5) if (A > 0).any() else 1
    im = ax.imshow(A, aspect="auto", cmap="turbo", interpolation="nearest", vmin=vmin, vmax=vmax,
                   origin="lower")
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("logical expert / physical slot")
    style_layer_axis(ax)
    return im


def main():
    W = {g: {ld: load_windows(r) for ld, r in RUNS[g].items()} for g in RUNS}
    ws = {g: W[g][LOAD] for g in ("A", "B", "C")}

    # ================= 图1：prefill / decode 速度 =================
    speeds = {}
    for g in ("A", "B", "C"):
        for ld in ("agentX", "random"):
            tp = td = dp = dd = 0.0
            for w in W[g][ld]:
                st = w["steps"]
                for i, (t, ph, c) in enumerate(st):
                    if i == 0:
                        continue
                    dur = t - st[i - 1][0]
                    if dur <= 0 or dur > 300:
                        continue
                    if ph == "prefill": tp += c / PRE_UNIT; dp += dur
                    else:              td += c / DEC_UNIT; dd += dur
            speeds[(g, ld)] = (tp / dp if dp else np.nan, td / dd if dd else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3); labels = list("ABC")
    for ax, idx, ttl in ((axes[0], 0, "prefill speed (system, 64 cards)"), (axes[1], 1, "decode speed (system, 64 cards)")):
        v1 = [speeds[(g, "agentX")][idx] for g in labels]
        v2 = [speeds[(g, "random")][idx] for g in labels]
        ax.bar(x - 0.2, v1, 0.4, color=[COL[g] for g in labels], label="agentX")
        ax.bar(x + 0.2, v2, 0.4, color=[COL[g] for g in labels], hatch="//", label="random")
        for i, v in enumerate(v1): ax.text(i - 0.2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        for i, v in enumerate(v2): ax.text(i + 0.2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["A: EPLB off", "B: EPLB on", "C: EPLB+red64"])
        ax.set_ylabel("tokens/s"); ax.set_title(ttl); ax.grid(alpha=.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("图1  prefill / decode speed, A/B/C\n"
                 "agentX = in 16384 / out 512 tok   |   random = in 8192 / out 64 tok\n"
                 "⚠️ 只能做【同负载内】的 A/B/C 比较（跨负载的输入输出长度不同，不可比）", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig1_speed_bar.png", dpi=110); plt.close(fig)

    # ================= 图2：E2E breakdown（横向堆叠柱）=================
    seg = {}
    for g in ("A", "B", "C"):
        for ld in ("agentX", "random"):
            rows = [json.loads(l) for l in open(RES / RUNS[g][ld] / "bench" / "requests.jsonl")]
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
            seg[(g, ld)] = {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    order = ["wait", "queue", "prefill", "first_send", "decode"]
    names = ["wait (recv->scheduler, no metric)", "queue_time", "prefill forward", "first-token send", "decode"]
    colors = ["#f0ad4e", "#d9534f", "#5bc0de", "#cccccc", "#5cb85c"]
    for ax, ld in zip(axes, ("agentX", "random")):
        y = np.arange(3); left = np.zeros(3)
        for k, nm, c in zip(order, names, colors):
            v = np.array([seg[(g, ld)][k] for g in ("A", "B", "C")])
            ax.barh(y, v, left=left, color=c, label=nm); left += v
        for i, tot in enumerate(left):
            ax.text(tot, y[i], f"  {tot:.1f}s", va="center", fontsize=9)
        ax.set_yticks(y); ax.set_yticklabels(["A: EPLB off", "B: EPLB on", "C: EPLB+red64"])
        ax.set_xlabel("seconds (mean per request)"); ax.set_title(f"E2E breakdown — {ld}")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3, axis="x")
    fig.suptitle("图2  E2E breakdown, A/B/C (horizontal stacked)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_e2e_breakdown_barh.png", dpi=110); plt.close(fig)

    # ================= 图3：逻辑专家热度（总和）ABC =================
    for ld in ("agentX", "random"):
        Hn = {g: heat_of(W[g][ld]) for g in ("A", "B", "C")}
        vmax = np.percentile(np.concatenate([norm_rows(Hn[g]).ravel() for g in "ABC"]), 99.5)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
        for ax, g in zip(axes, "ABC"):
            im = panel(ax, Hn[g], f"{g}: logical heat (total, per-layer normalized)", norm=True, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"图3  logical expert heat TOTAL, A/B/C — {ld} (should be nearly identical)"); fig.tight_layout()
        fig.savefig(OUT / f"fig3_logical_heat_total_{ld}.png", dpi=110); plt.close(fig)

        # ============ 图4：逻辑专家热度（PD 分离）============
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        vmax = max(np.percentile(norm_rows(heat_of(W[g][ld], ph)), 99.5) for g in "ABC" for ph in ("prefill", "decode"))
        for j, g in enumerate("ABC"):
            for i, ph in enumerate(("prefill", "decode")):
                H = heat_of(W[g][ld], ph)
                im = panel(axes[i][j], H, f"{g} — {ph}", norm=True, vmax=vmax)
                plt.colorbar(im, ax=axes[i][j], fraction=0.046)
        fig.suptitle(f"图4  logical expert heat, PD separated, A/B/C — {ld}"); fig.tight_layout()
        fig.savefig(OUT / f"fig4_logical_heat_pd_{ld}.png", dpi=110); plt.close(fig)

    # ================= 图5：累计 snapshot（总和 / PD 分离）=================
    for g, tag in (("A", "A_agentX"), ("B", "B_agentX")):
        ws_ = W[g]["agentX"]
        n = len(ws_)
        fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 10), squeeze=False)
        vmax = np.percentile(heat_of(ws_), 99.5)
        for k in range(n):
            for i, ph in enumerate(("total", "prefill", "decode")):
                H = heat_of(ws_, ph, upto=k + 1)
                ax = axes[i][k]
                ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax,
                          origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"+win{k}\n({ws_[k]['kind'][:9]})", fontsize=8)
                if k == 0:
                    ax.set_ylabel({"total": "TOTAL", "prefill": "PREFILL", "decode": "DECODE"}[ph], fontsize=9)
        fig.suptitle(f"图5  cumulative snapshots (cold -> hot), {tag}, pooled windows 1..k")
        fig.tight_layout(); fig.savefig(OUT / f"fig5_snapshots_{tag}.png", dpi=110); plt.close(fig)

    # ================= 图6：物理专家热度 =================
    for ld in ("agentX", "random"):
        # 总和
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
        vmax = 0
        Hs = {g: phys(heat_of(W[g][ld]), W[g][ld][-1]["pm"]) for g in "ABC"}
        for g in "ABC":
            vmax = max(vmax, np.percentile(Hs[g], 99.5))
        for ax, g in zip(axes, "ABC"):
            H = Hs[g]
            im = ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax,
                           origin="lower")
            ax.set_title(f"{g}: physical heat (total), {H.shape[1]} slots/rank={H.shape[1]//64}", fontsize=10)
            ax.set_xlabel("physical slot"); style_layer_axis(ax)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"图6  physical expert heat TOTAL, A/B/C — {ld} (layout-dependent!)"); fig.tight_layout()
        fig.savefig(OUT / f"fig6_physical_heat_total_{ld}.png", dpi=110); plt.close(fig)

        # PD 分离
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        vmax = max(np.percentile(phys(heat_of(W[g][ld], ph), W[g][ld][-1]["pm"]), 99.5)
                   for g in "ABC" for ph in ("prefill", "decode"))
        for j, g in enumerate("ABC"):
            for i, ph in enumerate(("prefill", "decode")):
                H = phys(heat_of(W[g][ld], ph), W[g][ld][-1]["pm"])
                ax = axes[i][j]
                im = ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax,
                               origin="lower")
                ax.set_title(f"{g} — physical {ph}", fontsize=10); ax.set_xlabel("physical slot")
                style_layer_axis(ax); plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"图6b  physical expert heat, PD separated, A/B/C — {ld}"); fig.tight_layout()
        fig.savefig(OUT / f"fig6b_physical_heat_pd_{ld}.png", dpi=110); plt.close(fig)

    # 物理 snapshot（B 组，累计）
    ws_ = W["B"]["agentX"]; n = len(ws_)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 5.4), squeeze=False)
    vmax = max(np.percentile(phys(heat_of(ws_, upto=k + 1), ws_[-1]["pm"]), 99.5) for k in range(n))
    for k in range(n):
        H = phys(heat_of(ws_, upto=k + 1), ws_[-1]["pm"])
        ax = axes[0][k]
        ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax, origin="lower")
        ax.set_title(f"+win{k}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("图6c  physical heat cumulative snapshots (B = EPLB on, agentX)")
    fig.tight_layout(); fig.savefig(OUT / "fig6c_physical_snapshots_B_agentX.png", dpi=110); plt.close(fig)

    # ================= 表格（figs/表格.md）=================
    def stats(run, key):
        d = json.load(open(RES / run / "bench" / "summary.json"))
        rows = [json.loads(l) for l in open(RES / run / "bench" / "requests.jsonl")]
        rows = [r for r in rows if r.get("status") == "ok"]
        v = np.array([r[key] for r in rows], dtype=float)
        return d, v

    L = []
    for ld in ("agentX", "random"):
        shape = ("in 16384 / out 512" if ld == "agentX" else "in 8192 / out 64")
        L.append(f"### {ld}   （输入/输出 = {shape} token；**只能与同负载的其他组比，跨负载不可比**）\n")
        L.append("| **指标** | **EPLB off (A)** | **EPLB on (B)** | **EPLB+冗余 (C)** |")
        L.append("| :---: | :---: | :---: | :---: |")
        cols = {}
        for g in "ABC":
            run = RUNS[g][ld]
            d = json.load(open(RES / run / "bench" / "summary.json"))
            rows = [json.loads(l) for l in open(RES / run / "bench" / "requests.jsonl")]
            rows = [r for r in rows if r.get("status") == "ok"]
            tt = np.array([r["ttft_s"] for r in rows]); tp = np.array([r["tpot_s"] for r in rows]) * 1000
            ee = np.array([r["e2e_s"] for r in rows])
            pf, dc = speeds[(g, ld)]
            cols[g] = dict(done=f"{d['completed']} / {d['errors']}", wall=f"{d['actual_wall_s']:.1f}s",
                           pf=f"{pf:,.1f} tok/s", dc=f"{dc:,.1f} tok/s",
                           tt=(tt.mean(), np.percentile(tt, 50), np.percentile(tt, 90), np.percentile(tt, 99)),
                           tp=(tp.mean(), np.percentile(tp, 50), np.percentile(tp, 90)),
                           ee=(ee.mean(), np.percentile(ee, 50), np.percentile(ee, 90)))
        for name, key, fmt in (("完成/错误", "done", "{}"), ("墙钟", "wall", "{}"),
                               ("Prefill速度", "pf", "{}"), ("Decode速度", "dc", "{}"),
                               ("TTFT mean", "tt0", "{:.2f}s"), ("TTFT p50", "tt1", "{:.2f}s"),
                               ("TTFT p90", "tt2", "{:.2f}s"), ("TTFT p99", "tt3", "{:.2f}s"),
                               ("TPOT mean", "tp0", "{:.1f}ms"), ("TPOT p50", "tp1", "{:.1f}ms"),
                               ("TPOT p90", "tp2", "{:.1f}ms"),
                               ("E2E mean", "ee0", "{:.1f}s"), ("E2E p50", "ee1", "{:.1f}s"),
                               ("E2E p90", "ee2", "{:.1f}s")):
            cells = []
            for g in "ABC":
                c = cols[g]
                if key in ("done", "wall", "pf", "dc"):
                    cells.append(c[key])
                else:
                    i = int(key[-1]); tup = c[key[:-1]]
                    cells.append(fmt.format(tup[i]))
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    # 补充表：热力/不均衡
    L.append("### 补充：rank 不均衡 与 尖峰对齐（来自 windows/*.pt）\n")
    L.append("| **指标** | **EPLB off (A)** | **EPLB on (B)** | **EPLB+冗余 (C)** |")
    L.append("| :---: | :---: | :---: | :---: |")
    for ld in ("agentX", "random"):
        row = {}
        for g in "ABC":
            ws_ = W[g][ld]
            H = heat_of(ws_); pm = ws_[-1]["pm"]; ep = ws_[-1]["ep"]; P = ws_[-1]["P"]
            lt = rank_load(H, pm, ep, P)
            Hp = heat_of(ws_, "prefill"); Hd = heat_of(ws_, "decode")
            lp = rank_load(Hp, pm, ep, P)
            ranks = []
            for l in range(78):
                if H[l].sum() == 0: continue
                e = int(H[l].argmax()); ranks.append(int(np.where(pm[l] == e)[0][0]) // (P // ep))
            row[g] = (lp.max() / lp.mean(), lt.max() / lt.mean(), float(np.mean([r == 0 for r in ranks])))
        L.append(f"| 不均衡 prefill ({ld}) | {row['A'][0]:.2f}× | {row['B'][0]:.2f}× | {row['C'][0]:.2f}× |")
        L.append(f"| 不均衡 合计 ({ld}) | {row['A'][1]:.2f}× | {row['B'][1]:.2f}× | {row['C'][1]:.2f}× |")
        L.append(f"| 最热专家落 rank0 的层占比 ({ld}) | {row['A'][2]:.0%} | {row['B'][2]:.0%} | {row['C'][2]:.0%} |")
    (OUT / "表格.md").write_text("\n".join(L), encoding="utf-8")
    # ============ 图3-normalized：每格份额 ÷ (1/256)，log 色标（所有格子可见）============
    from matplotlib.colors import LogNorm
    ratio = {}
    for ld in ("agentX", "random"):
        for g in "ABC":
            H = heat_of(W[g][ld])
            share = H / np.maximum(H.sum(1, keepdims=True), 1)
            ratio[(ld, g)] = share * H.shape[1]              # 1.0 = 该层平均
    allv = np.concatenate([r[r > 0].ravel() for r in ratio.values()])
    vmin, vmax = 0.1, float(np.percentile(allv, 99.9))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for i, ld in enumerate(("agentX", "random")):
        for j, g in enumerate("ABC"):
            R = ratio[(ld, g)]
            ax = axes[i][j]
            im = ax.imshow(R, aspect="auto", cmap="turbo", interpolation="nearest",
                           norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower")
            ttl = f"{g} - {ld}   峰值={R.max():.0f}x 该层均值"
            if i == 0:
                Ra, Rr = ratio[("agentX", g)], ratio[("random", g)]
                cs = [np.corrcoef(Ra[l], Rr[l])[0, 1] for l in range(78) if Ra[l].sum() > 0]
                ovs = [len(set(np.argsort(-Ra[l])[:8]) & set(np.argsort(-Rr[l])[:8])) / 8
                       for l in range(78) if Ra[l].sum() > 0]
                ttl += f"\nvs random: r={np.mean(cs):.4f}, top-8={np.mean(ovs):.0%}"
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel("logical expert"); ax.set_ylabel("layer  (0 bottom -> 77 top)")
            if j == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, label="该格份额 / (1/256)")
    fig.suptitle("图3-normalized   逻辑专家热度 = 每层内份额 / (1/256)   (1.0 = 该层平均)，log 色标 -> 所有格子都可见\n"
                 "A/B/C 三列几乎相同(与摆放无关) ; 上下两行(两种负载)也几乎相同", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig3_normalized.png", dpi=110); plt.close(fig)

    # ============ 图3-readable：人眼看的版本（散点 + 逐层 top-1 + 累积曲线）============
    from matplotlib.colors import LogNorm
    def share_of(run):
        H = heat_of(load_windows(run))
        return H / np.maximum(H.sum(1, keepdims=True), 1)
    sa = share_of(RUNS["A"]["agentX"]); sr = share_of(RUNS["A"]["random"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    # (1) 散点：每个 (层,专家) 一格，x=agentX 份额, y=random 份额
    ax = axes[0]
    x, y = sa.ravel(), sr.ravel()
    m = (x > 0) & (y > 0)
    ax.scatter(x[m], y[m], s=3, alpha=.25, color="tab:blue", edgecolors="none")
    lim = [1e-5, 0.3]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x（完全一样）")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("share  agentX（每层内份额）"); ax.set_ylabel("share  random")
    r = np.corrcoef(np.log10(x[m]), np.log10(y[m]))[0, 1]
    within = (np.abs(np.log10(y[m] / x[m])) < np.log10(1.1)).mean()
    ax.set_title(f"(1) 每格对比：{m.sum():,} 个格子\nlog 相关 r={r:.3f}；{within:.0%} 的点在 ±10% 以内", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    # (2) 逐层 top-1 份额曲线
    ax = axes[1]
    for lab, S, c in (("agentX", sa, "tab:blue"), ("random", sr, "tab:orange")):
        ax.plot(range(78), [S[l].max() for l in range(78)], "o-", ms=3, lw=1.2, color=c, label=lab)
    ax.set_xlabel("layer"); ax.set_ylabel("top-1 expert share (of that layer)")
    ax.set_title("(2) 每层最热专家的份额：两条线几乎重合", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # (3) 累积曲线（专家按热度排序）
    ax = axes[2]
    for lab, S, c in (("agentX", sa, "tab:blue"), ("random", sr, "tab:orange")):
        cums = []
        for l in range(78):
            v = np.sort(S[l])[::-1]; cums.append(np.cumsum(v))
        ax.plot(np.mean(cums, axis=0)[:80], "o-", ms=3, lw=1.4, color=c,
                label=f"{lab}  (median 50%杯={int(np.median([np.searchsorted(c,0.5)+1 for c in cums]))} 个专家)")
    ax.set_xlabel("#experts (sorted desc, avg over layers)"); ax.set_ylabel("cumulative share")
    ax.axhline(0.5, color="gray", ls=":"); ax.set_title("(3) 累积曲线：两条线几乎重合", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("图3-readable   A 组（EPLB off）：agentX 16K vs random 8K —— 同样的热点专家", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig3_readable.png", dpi=110); plt.close(fig)

    print("\n".join(L))
    print("\n✓ 产出在", OUT)


if __name__ == "__main__":
    main()
