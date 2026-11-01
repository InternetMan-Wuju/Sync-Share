#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4Exp 分析器：出表格 + 热力图（纯 CPU）。

用法:
  python3 analyze_q4exp.py --windows <results/xxx/windows> [--out <dir>] [--tag xxx]

产出:
  <out>/tables/window_summary.csv      每个窗口的概况
  <out>/tables/rank_load.csv           每 rank 负载（prefill/decode/合计）
  <out>/tables/layer_stats.csv         每层的热点/集中度/PD 重合
  <out>/tables/top_experts.csv         最热的专家（按层聚合）
  <out>/figs/fig1_layer_expert.png     Layer x Expert 热力图（prefill/decode/合计）
  <out>/figs/fig2_layer_rank.png       Layer x Rank  热力图（prefill/decode/合计）
  <out>/figs/fig3_rank_time.png        rank 负载曲线 + rank x 窗口热力图
  <out>/figs/fig4_pd.png               PD 对比（top-8 重合分布 + 累积集中度）
"""
import argparse, pathlib, csv
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

TOPK = 8


def load_windows(d):
    out = []
    for p in sorted(pathlib.Path(d).glob("*.pt")):
        if p.name.startswith("expert_distribution_recorder_"):
            continue
        o = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(o, dict) or "physical_to_logical_map" not in o:
            continue
        def H(key):
            t = o.get(key)
            return None if t is None else t.float().sum(0).numpy()
        out.append(dict(name=p.stem, idx=int(o.get("dump_index", len(out))),
                        kind=str(o.get("window_kind", "?")),
                        t0=o.get("window_start_timestamp"), t1=o.get("window_end_timestamp"),
                        L=int(o["num_layers"]), E=int(o["num_logical_experts"]),
                        P=int(o["num_physical_experts"]), ep=int(o["ep_size"]),
                        pm=np.asarray(o["physical_to_logical_map"], dtype=np.int64),
                        pre=H("logical_count_prefill"), dec=H("logical_count_decode"),
                        oth=H("logical_count_other"),
                        steps=dict(o.get("n_steps_by_phase") or {})))
    out.sort(key=lambda w: w["idx"])
    return out


def rank_load(H, pm, ep, P):
    """逻辑热度 -> 每 rank 负载（冗余时副本按均分；统一调用 q4exp_batch.phys_heat，避免口径漂移）"""
    from q4exp_batch import phys_heat
    Ph = phys_heat(H, pm)
    k = P // ep
    return np.stack([Ph[l].reshape(ep, k).sum(1) for l in range(Ph.shape[0])]).sum(0)


def pd_overlap(Hp, Hd, k=TOPK, full=False):
    """full=True 时返回长度 = 层数 的数组（空层为 nan），否则只返回非空层"""
    out = np.full(Hp.shape[0], np.nan) if full else []
    for l in range(Hp.shape[0]):
        if Hp[l].sum() == 0 or Hd[l].sum() == 0:
            continue
        v = len(set(np.argsort(-Hp[l])[:k]) & set(np.argsort(-Hd[l])[:k])) / k
        if full: out[l] = v
        else: out.append(v)
    return out if full else np.array(out)


def n_to_half(v):
    if v.sum() == 0:
        return np.nan
    c = np.cumsum(np.sort(v)[::-1]) / v.sum()
    return int(np.searchsorted(c, 0.5) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--decode-fix", type=float, default=8.0,
                    help="decode 计数相对 prefill 的放大倍数（本 run 实测 8.0）；合计按真实 token 权重算")
    a = ap.parse_args()
    W_all = load_windows(a.windows)
    # 有些窗口某一相没有步（prefill 或 decode 为 None）→ 逐窗口分析时跳过，避免 None 崩
    W = [w for w in W_all if w["pre"] is not None and w["dec"] is not None]
    if len(W) != len(W_all):
        print(f"  跳过 {len(W_all)-len(W)} 个缺少某一相的窗口")
    if not W:
        raise SystemExit("没有可用窗口")
    out = pathlib.Path(a.out or pathlib.Path(a.windows).parent)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    L, E, P, ep = W[0]["L"], W[0]["E"], W[0]["P"], W[0]["ep"]
    print(f"载入 {len(W)} 个窗口  L={L} E={E} P={P} ep={ep}")

    # 合计（跨窗口求和，用于主热力图）
    TOT_pre = sum(w["pre"] for w in W if w["pre"] is not None)
    TOT_dec = sum(w["dec"] for w in W if w["dec"] is not None)
    # ⚠️ decode 的计数单位是 prefill 的 DEC_FIX 倍（实测 8.0）→ 合计必须先把 decode 缩回真实权重
    DEC_FIX = a.decode_fix
    TOT = TOT_pre + TOT_dec / DEC_FIX
    pm = W[-1]["pm"]

    # ---------- 表 1：窗口概况 ----------
    rows = []
    for w in W:
        rp = rank_load(w["pre"], w["pm"], ep, P) if w["pre"] is not None else None
        rd = rank_load(w["dec"], w["pm"], ep, P) if w["dec"] is not None else None
        rt = (rp + rd) if (rp is not None and rd is not None) else None
        ov = pd_overlap(w["pre"], w["dec"]) if (w["pre"] is not None and w["dec"] is not None) else np.array([])
        rt_fix = (rp + rd / a.decode_fix) if (rp is not None and rd is not None) else None
        rows.append(dict(idx=w["idx"], kind=w["kind"],
                         span_s=round((w["t1"] - w["t0"]), 1) if w["t0"] and w["t1"] else "",
                         steps_pre=w["steps"].get("prefill", ""), steps_dec=w["steps"].get("decode", ""),
                         tokens_pre=int(w["pre"].sum()) if w["pre"] is not None else 0,
                         tokens_dec=int(w["dec"].sum()) if w["dec"] is not None else 0,
                         nonzero_layers=int((w["pre"] + w["dec"]).sum(1).astype(bool).sum()) if w["pre"] is not None else "",
                         rank_maxmean_pre=round(float(rp.max() / rp.mean()), 3) if rp is not None else "",
                         rank_maxmean_dec=round(float(rd.max() / rd.mean()), 3) if rd is not None else "",
                         rank_maxmean_tot_asrec=round(float(rt.max() / rt.mean()), 3) if rt is not None else "",
                         rank_maxmean_tot_corrected=round(float(rt_fix.max() / rt_fix.mean()), 3) if rt_fix is not None else "",
                         pd_top8_overlap=round(float(ov.mean()), 3) if ov.size else ""))
    with open(out / "tables" / "window_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("  ✓ tables/window_summary.csv")

    # ---------- 表 2：每 rank 负载 ----------
    with open(out / "tables" / "rank_load.csv", "w", newline="") as f:
        wr = csv.writer(f)
        head = ["rank", "rank_node"]
        for w in W:
            head += [f"win{w['idx']}_prefill", f"win{w['idx']}_decode", f"win{w['idx']}_total"]
        wr.writerow(head)
        for r in range(ep):
            node = ["19", "20", "23", "25"][r // 16]
            row = [r, node]
            for w in W:
                rp = rank_load(w["pre"], w["pm"], ep, P) if w["pre"] is not None else np.zeros(ep)
                rd = rank_load(w["dec"], w["pm"], ep, P) if w["dec"] is not None else np.zeros(ep)
                row += [int(rp[r]), int(rd[r]), int(rp[r] + rd[r])]
            wr.writerow(row)
    print("  ✓ tables/rank_load.csv")

    # ---------- 表 3：每层统计 ----------
    OV = pd_overlap(TOT_pre, TOT_dec, full=True)
    rows = []
    for l in range(L):
        tot = TOT[l]
        if tot.sum() == 0:
            rows.append(dict(layer=l, tokens=0, top1_expert="", top1_share="", top1_rank="",
                             pre_share="", n_exp_50pct_total="", n_exp_50pct_prefill="",
                             n_exp_50pct_decode="", pd_top8_overlap=""))
            continue
        top = int(np.argsort(-tot)[0])
        rank_of_top = int(np.where(pm[l] == top)[0][0]) // (P // ep)
        rows.append(dict(layer=l, tokens=int(tot.sum()),
                         top1_expert=top, top1_share=round(float(tot[top] / tot.sum()), 4),
                         top1_rank=rank_of_top,
                         pre_share=round(float(TOT_pre[l].sum() / tot.sum()), 3),
                         n_exp_50pct_total=n_to_half(tot),
                         n_exp_50pct_prefill=n_to_half(TOT_pre[l]),
                         n_exp_50pct_decode=n_to_half(TOT_dec[l]),
                         pd_top8_overlap="" if np.isnan(OV[l]) else round(float(OV[l]), 3)))
    with open(out / "tables" / "layer_stats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[-1].keys()) if rows else []); wr.writeheader(); wr.writerows(rows)
    print("  ✓ tables/layer_stats.csv")

    # ---------- 表 4：最热专家 ----------
    tot_per_exp = TOT.sum(0)
    order = np.argsort(-tot_per_exp)[:40]
    with open(out / "tables" / "top_experts.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["expert", "tokens", "share", "n_layers_where_top8"])
        for e in order:
            n_in_top8 = sum(1 for l in range(L) if e in np.argsort(-TOT[l])[:TOPK])
            wr.writerow([int(e), int(tot_per_exp[e]), round(float(tot_per_exp[e] / tot_per_exp.sum()), 5), n_in_top8])
    print("  ✓ tables/top_experts.csv")

    # ---------- 图 1：Layer x Expert ----------
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, Hh, ttl in zip(axes, (TOT_pre, TOT_dec, TOT),
                           ("PREFILL", "DECODE (as recorded)",
                            f"TOTAL = prefill + decode/{DEC_FIX:g} (true token share)")):
        vmax = np.percentile(Hh[Hh > 0], 99.5) if (Hh > 0).any() else 1
        im = ax.imshow(Hh, aspect="auto", cmap="turbo", interpolation="nearest",
                       vmin=0, vmax=vmax, origin="lower")
        ax.set_title(f"{ttl}\nLayer x Expert (logical), {len(W)} windows pooled", fontsize=11)
        ax.set_xlabel("logical expert id (0..255)"); ax.set_ylabel("layer (0 bottom -> 77 top)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axhline(2.5, color="w", ls="--", lw=1)
    fig.suptitle(f"Layer x Expert heat  {a.tag}", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "figs" / "fig1_layer_expert.png", dpi=110); plt.close(fig)

    # ---------- 图 2：Layer x Rank ----------
    k = P // ep
    def to_rank_heat(H):
        out_ = np.zeros((L, ep))
        for l in range(L):
            out_[l] = H[l][pm[l]].reshape(ep, k).sum(1)
        return out_
    R_pre, R_dec, R_tot = to_rank_heat(TOT_pre), to_rank_heat(TOT_dec), to_rank_heat(TOT)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, Hh, ttl in zip(axes, (R_pre, R_dec, R_tot),
                           ("PREFILL", "DECODE (as recorded)",
                            f"TOTAL = prefill + decode/{DEC_FIX:g}")):
        vmax = np.percentile(Hh[Hh > 0], 99) if (Hh > 0).any() else 1
        im = ax.imshow(Hh, aspect="auto", cmap="turbo", interpolation="nearest",
                       vmin=0, vmax=vmax, origin="lower")
        ax.set_title(f"{ttl}\nLayer x EP-rank", fontsize=11)
        ax.set_xlabel("EP rank (0..63)"); ax.set_ylabel("layer (0 bottom -> 77 top)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Layer x Rank heat  {a.tag}", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "figs" / "fig2_layer_rank.png", dpi=110); plt.close(fig)

    # ---------- 图 3：rank 负载曲线 + rank x 窗口 ----------
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [2, 1.2]})
    ax = axes[0]
    for Hh, nm, c in ((TOT_pre, "prefill", "tab:blue"), (TOT_dec, "decode(as recorded)", "tab:red"),
                      (TOT, f"total (=prefill+decode/{DEC_FIX:g})", "k")):
        v = rank_load(Hh, pm, ep, P); ax.plot(range(ep), v / v.mean(), "o-", ms=4, lw=1.8, color=c,
                                              label=f"{nm}  max/mean={v.max()/v.mean():.2f}x")
    ax.axhline(1.0, color="gray", ls="--", lw=1); ax.set_xlabel("EP rank"); ax.set_ylabel("load / mean")
    ax.set_title("Per-rank load (pooled over windows)"); ax.legend(); ax.grid(alpha=.3)
    ax = axes[1]
    M = np.zeros((ep, len(W)))
    for j, w in enumerate(W):
        rp = rank_load(w["pre"], w["pm"], ep, P); rd = rank_load(w["dec"], w["pm"], ep, P)
        v = rp + rd; M[:, j] = v / v.mean()
    im = ax.imshow(M, aspect="auto", cmap="coolwarm", origin="upper", vmin=0.5, vmax=1.5)
    ax.set_xticks(range(len(W))); ax.set_xticklabels([f"win{w['idx']}" for w in W])
    ax.set_xlabel("window"); ax.set_ylabel("EP rank")
    ax.set_title("rank x window (load/mean)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Rank load  {a.tag}", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "figs" / "fig3_rank_time.png", dpi=110); plt.close(fig)

    # ---------- 图 4：PD 对比 ----------
    ov = pd_overlap(TOT_pre, TOT_dec)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[0]
    ax.hist(ov, bins=np.arange(0, 1.001, 0.125), color="tab:purple", edgecolor="w")
    ax.axvline(0.25, color="r", ls="--", label="random baseline 8*8/256 = 0.25")
    ax.set_xlabel("top-8 overlap (prefill vs decode) per layer"); ax.set_ylabel("#layers")
    ax.set_title(f"PD top-8 overlap: mean={ov.mean():.1%}"); ax.legend()
    ax = axes[1]
    for Hh, nm in ((TOT_pre, "prefill"), (TOT_dec, "decode")):
        c = np.sort(Hh[Hh > 0])[::-1]; c = np.cumsum(c) / c.sum()
        ax.plot(np.arange(1, len(c) + 1), c, lw=2, label=nm)
    ax.axhline(0.5, color="gray", ls=":"); ax.set_xlabel("#experts (sorted desc)"); ax.set_ylabel("cumulative share")
    ax.set_title("Concentration (all layers pooled)"); ax.legend(); ax.grid(alpha=.3)
    ax = axes[2]
    cm = TOT / np.maximum(TOT.sum(1, keepdims=True), 1)
    im = ax.imshow(cm * 1e3, aspect="auto", cmap="turbo", interpolation="nearest", origin="lower")
    ax.set_title("Per-layer shares (x1000) of TOTAL heat"); ax.set_xlabel("expert"); ax.set_ylabel("layer (0 bottom -> 77 top)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Prefill vs Decode  {a.tag}", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "figs" / "fig4_pd.png", dpi=110); plt.close(fig)

    # ---------- 图 7：合计口径对比（as-recorded vs 修正）----------
    TOT_bad = TOT_pre + TOT_dec
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.0), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, Hh, ttl in zip(axes[:2], (TOT_bad, TOT),
                           ("TOTAL as recorded (decode x8)", f"TOTAL corrected (decode /{DEC_FIX:g})")):
        vmax = np.percentile(Hh[Hh > 0], 99.5) if (Hh > 0).any() else 1
        im = ax.imshow(Hh, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax,
                       origin="lower")
        ax.set_title(f"{ttl}\nLayer x Expert"); ax.set_xlabel("logical expert"); ax.set_ylabel("layer (0 bottom -> 77 top)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    for Hh, nm, c in ((TOT_pre, "prefill", "tab:blue"), (TOT_dec, "decode(as recorded)", "tab:red"),
                      (TOT_bad, "total as recorded", "gray"), (TOT, "total corrected", "k")):
        v = rank_load(Hh, pm, ep, P); ax.plot(range(ep), v / v.mean(), "o-", ms=3, lw=1.6, color=c,
                                              label=f"{nm}  max/mean={v.max()/v.mean():.2f}x")
    ax.axhline(1.0, color="gray", ls="--", lw=1); ax.set_xlabel("EP rank"); ax.set_ylabel("load / mean")
    ax.set_title("Per-rank load: as-recorded vs corrected"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"decode count unit matters for the TOTAL  {a.tag}", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "figs" / "fig7_total_weight.png", dpi=110); plt.close(fig)

    print(f"\n[合计口径] as-recorded total max/mean = {rank_load(TOT_bad, pm, ep, P).max()/rank_load(TOT_bad, pm, ep, P).mean():.2f}x"
          f"   corrected = {rank_load(TOT, pm, ep, P).max()/rank_load(TOT, pm, ep, P).mean():.2f}x"
          f"   (prefill only = {rank_load(TOT_pre, pm, ep, P).max()/rank_load(TOT_pre, pm, ep, P).mean():.2f}x)")
    print(f"\n产出目录: {out}")
    print(f"  tables: window_summary / rank_load / layer_stats / top_experts")
    print(f"  figs  : fig1_layer_expert / fig2_layer_rank / fig3_rank_time / fig4_pd")


if __name__ == "__main__":
    main()
