#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4Exp 八轮总分析：图 + 表（纯 CPU）。

输入: pd_colloc_20261011/results/<RUN>/
输出: Q4Exp/tables/*.csv, Q4Exp/figs/*.png

口径要点:
  * prefill / decode 分开统计（明确要求），合计按真实 token 权重: total = prefill + decode/8
    （实测 decode 的计数单位是 prefill 的 8 倍，见 results/20261012_smokeA/README.md）
  * 周期 = 两次重排之间（B/C 的 eplb_rebalance 窗口就是一个完整周期）
  * rank = 物理槽号 // (物理槽数 / ep_size)
"""
import csv, glob, json, pathlib, re
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/home/chengyetai/workspace/current_exp/pd_colloc_20261011/results")
OUT = pathlib.Path("/home/chengyetai/workspace/current_exp/Q4Exp")
MOE_L = 75; TOPK = 8
PRE_UNIT = MOE_L * TOPK          # 600  counts per token (prefill)
DEC_FIX = 8.0                    # decode 计数放大倍数
DEC_UNIT = PRE_UNIT * DEC_FIX    # 4800

RUNS = [
    ("A0", "agentX", "A0_agentx_20261012_044134"),
    ("A0", "random", "A0_random_20261012_060410"),
    ("A",  "agentX", "A_agentx_20261012_050236"),
    ("A",  "random", "A_random_20261012_062417"),
    ("B",  "agentX", "B_agentx_20261012_052314"),
    ("B",  "random", "B_random_20261012_064400"),
    ("C",  "agentX", "C_agentx_20261012_054412"),
    ("C",  "random", "C_random_20261012_071801"),
]
COL = {"A0": "gray", "A": "tab:blue", "B": "tab:red", "C": "tab:green"}
LS  = {"agentX": "-", "random": "--"}


def expert_replica_count(pm):
    """[L, E] 每个逻辑专家在物理槽里出现几次（无冗余 = 全 1；冗余 64 → 热点专家 2~10）"""
    E = int(pm.max()) + 1
    cnt = np.zeros((pm.shape[0], E), dtype=np.float64)
    for l in range(pm.shape[0]):
        np.add.at(cnt[l], pm[l], 1.0)
    return cnt


def phys_heat(H, pm):
    """逻辑热度 -> 物理槽热度。⚠️ 冗余时按副本数【均分】，
    否则会把专家的全部热度复制到每一份副本上（会放大热专家 2~10 倍）。"""
    cnt = expert_replica_count(pm)
    P = pm.shape[1]
    out = np.zeros((H.shape[0], P))
    for l in range(H.shape[0]):
        out[l] = H[l][pm[l]] / cnt[l][pm[l]]
    return out


def rank_load(H, pm, ep, P):
    """每 rank 负载（冗余时副本均分；无冗余时退化为原始定义）"""
    k = P // ep
    Ph = phys_heat(H, pm)
    return Ph.reshape(Ph.shape[0], ep, k).sum(1).sum(0) if False else \
        np.stack([Ph[l].reshape(ep, k).sum(1) for l in range(Ph.shape[0])]).sum(0)


def pd_overlap(Hp, Hd, k=TOPK):
    o = []
    for l in range(Hp.shape[0]):
        if Hp[l].sum() == 0 or Hd[l].sum() == 0:
            continue
        o.append(len(set(np.argsort(-Hp[l])[:k]) & set(np.argsort(-Hd[l])[:k])) / k)
    return np.array(o)


def n_to_half(v):
    if v.sum() == 0:
        return np.nan
    c = np.cumsum(np.sort(v)[::-1]) / v.sum()
    return int(np.searchsorted(c, 0.5) + 1)


def load_windows(run):
    ws = []
    for p in sorted((RES / run / "windows").glob("*.pt")):
        o = torch.load(p, map_location="cpu", weights_only=False)
        pre = o.get("logical_count_prefill"); dec = o.get("logical_count_decode")
        if pre is None or dec is None:
            continue
        ts = o.get("step_timestamps_by_phase") or {}
        steps = []
        for key, t in (("prefill", pre), ("decode", dec)):
            times = list(ts.get(key) or [])
            if not times:
                continue
            if t.shape[0] > len(times):
                t = t[-len(times):]
            elif t.shape[0] < len(times):
                times = times[-t.shape[0]:]
            for i in range(t.shape[0]):
                steps.append((float(times[i]), key, int(t[i].sum())))
        steps.sort()
        ws.append(dict(
            name=p.stem, idx=int(o.get("dump_index", 0)), kind=str(o.get("window_kind", "?")),
            t0=o.get("window_start_timestamp"), t1=o.get("window_end_timestamp"),
            L=int(o["num_layers"]), E=int(o["num_logical_experts"]),
            P=int(o["num_physical_experts"]), ep=int(o["ep_size"]),
            pm=np.asarray(o["physical_to_logical_map"], dtype=np.int64),
            pre=pre.float().sum(0).numpy(), dec=dec.float().sum(0).numpy(),
            n_pre=int(pre.shape[0]), n_dec=int(dec.shape[0]), steps=steps))
        del o
    return ws


def perf_of(run):
    d = json.load(open(RES / run / "bench" / "summary.json"))
    return dict(req=int(d.get("requests_sent", 0)), done=int(d.get("completed", 0)),
                err=int(d.get("errors", 0)), wall=float(d.get("actual_wall_s", 0)),
                out_tok=int(d.get("output_tokens", 0)), tput=float(d.get("output_tokens_per_s", 0)),
                ttft_p50=d["ttft_s"]["p50"], ttft_p95=d["ttft_s"]["p95"],
                tpot_p50=d["tpot_s"]["p50"], tpot_p95=d["tpot_s"]["p95"],
                e2e_p50=d["e2e_s"]["p50"], e2e_p95=d["e2e_s"]["p95"])


def latency_breakdown(run):
    rows = [json.loads(l) for l in open(RES / run / "bench" / "requests.jsonl")]
    rows = [r for r in rows if isinstance(r.get("meta_info"), dict) and r.get("status") == "ok"]
    seg = {k: [] for k in ("recv_to_fwd", "queue", "prefill", "first_send", "decode", "pre_sp", "dec_sp")}
    for r in rows:
        m = r["meta_info"]
        try:
            recv = m["forward_entry_time"] - m["request_received_ts"]
            q = m.get("queue_time", np.nan)
            pf = m["prefill_finished_time"] - m["forward_entry_time"]
            fs = m["response_sent_to_client_ts"] - m["prefill_finished_time"]
            dc = m["request_finished_ts"] - m["response_sent_to_client_ts"]
        except KeyError:
            continue
        seg["recv_to_fwd"].append(recv); seg["queue"].append(q); seg["prefill"].append(pf)
        seg["first_send"].append(fs); seg["decode"].append(dc)
        if pf > 0: seg["pre_sp"].append(r["prompt_tokens"] / pf)
        if dc > 0 and r.get("output_tokens"): seg["dec_sp"].append(r["output_tokens"] / dc)
    return {k: (np.mean(v) if v else np.nan) for k, v in seg.items()}, len(rows)


def rebalance_times(run):
    f = RES / run / "rebalance_events.txt"
    if not f.exists():
        return []
    vals = [float(x) for x in re.findall(r"rebalance end time=([0-9.]+)s", f.read_text())]
    if not vals:
        return []
    # 每次重排 64 个 rank 各报一次 → 聚类
    vals = sorted(set(vals))
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= 1.0:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    return [round(float(np.median(g)), 2) for g in groups]


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figs" / "extra").mkdir(parents=True, exist_ok=True)
    data = {}
    for grp, load, run in RUNS:
        key = f"{grp}_{load}"
        print(f"--- {key} ({run})")
        d = dict(grp=grp, load=load, run=run)
        d["perf"] = perf_of(run)
        d["seg"], d["n_ok"] = latency_breakdown(run)
        d["rebal"] = rebalance_times(run)
        d["win"] = load_windows(run)
        # 每窗口指标
        wrows = []
        for w in d["win"]:
            ep, P = w["ep"], w["P"]
            lp = rank_load(w["pre"], w["pm"], ep, P)
            ld = rank_load(w["dec"], w["pm"], ep, P)
            lt = rank_load(w["pre"] + w["dec"] / DEC_FIX, w["pm"], ep, P)
            ov = pd_overlap(w["pre"], w["dec"])
            wrows.append(dict(run=key, idx=w["idx"], kind=w["kind"],
                              span_s=round((w["t1"] - w["t0"]), 1) if w["t0"] and w["t1"] else "",
                              n_pre=w["n_pre"], n_dec=w["n_dec"],
                              pre=lp.max() / lp.mean(), dec=ld.max() / ld.mean(),
                              tot=lt.max() / lt.mean(), pd_ov=float(ov.mean()) if ov.size else np.nan))
        d["wrows"] = wrows
        # 汇聚（全部窗口）
        if d["win"]:
            hp = sum(w["pre"] for w in d["win"]); hd = sum(w["dec"] for w in d["win"])
            pm = d["win"][-1]["pm"]; ep, P = d["win"][-1]["ep"], d["win"][-1]["P"]
            d["pool"] = dict(pre=hp, dec=hd, pm=pm, ep=ep, P=P)
            lp = rank_load(hp, pm, ep, P); ld = rank_load(hd, pm, ep, P)
            lt = rank_load(hp + hd / DEC_FIX, pm, ep, P)
            d["pool"]["loads"] = (lp, ld, lt)
            ranks = []
            for l in range(hp.shape[0]):
                T = hp[l] + hd[l] / DEC_FIX
                if T.sum() == 0:
                    continue
                e = int(T.argmax()); ranks.append(int(np.where(pm[l] == e)[0][0]) // (P // ep))
            d["pool"]["hot_rank"] = ranks
            d["pool"]["rank0_frac"] = float(np.mean([r == 0 for r in ranks]))
            ov = pd_overlap(hp, hd)
            d["pool"]["pd_ov"] = ov
            d["pool"]["n50_pre"] = np.median([n_to_half(hp[l]) for l in range(78) if hp[l].sum() > 0])
            d["pool"]["n50_dec"] = np.median([n_to_half(hd[l]) for l in range(78) if hd[l].sum() > 0])
            # 步级速度（只统计各窗口第一步之外）
            tp = td = dp = dd = 0.0
            for w in d["win"]:
                for i, (t, ph, c) in enumerate(w["steps"]):
                    if i == 0:
                        continue
                    dur = t - w["steps"][i - 1][0]
                    if dur <= 0 or dur > 300:
                        continue
                    if ph == "prefill":
                        tp += c / PRE_UNIT; dp += dur
                    else:
                        td += c / DEC_UNIT; dd += dur
            d["pool"]["pref_speed"] = tp / dp if dp else np.nan
            d["pool"]["dec_speed"] = td / dd if dd else np.nan
        data[key] = d

    # ---------------- 表 1：性能 ----------------
    with open(OUT / "tables" / "perf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "load", "requests", "errors", "throughput_tok_s", "ttft_p50_s", "ttft_p95_s",
                    "tpot_p50_ms", "tpot_p95_ms", "e2e_p50_s", "e2e_p95_s", "rebalances", "rebalance_durations_s"])
        for k, d in data.items():
            p = d["perf"]
            w.writerow([k, d["grp"], d["load"], p["req"], p["err"], round(p["tput"], 1),
                        round(p["ttft_p50"], 2), round(p["ttft_p95"], 2), round(p["tpot_p50"] * 1000, 1),
                        round(p["tpot_p95"] * 1000, 1), round(p["e2e_p50"], 2), round(p["e2e_p95"], 2),
                        len(d["rebal"]), ";".join(str(x) for x in d["rebal"])])
    # ---------------- 表 2：延迟分解 ----------------
    with open(OUT / "tables" / "latency_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "load", "requests_ok", "recv_to_fwd_s", "queue_s", "prefill_forward_s",
                    "first_send_s", "decode_s", "prefill_tok_s", "decode_tok_s"])
        for k, d in data.items():
            s = d["seg"]
            w.writerow([k, d["grp"], d["load"], d["n_ok"], round(s["recv_to_fwd"], 3), round(s["queue"], 4),
                        round(s["prefill"], 2), round(s["first_send"], 4), round(s["decode"], 2),
                        round(s["pre_sp"], 0), round(s["dec_sp"], 1)])
    # ---------------- 表 3：逐窗口 rank 不均衡 ----------------
    with open(OUT / "tables" / "rank_imbalance_by_window.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "load", "window", "kind", "span_s", "steps_pre", "steps_decode",
                    "rank_maxmean_prefill", "rank_maxmean_decode", "rank_maxmean_total_corrected", "pd_top8_overlap"])
        for k, d in data.items():
            for r in d["wrows"]:
                w.writerow([k, d["grp"], d["load"], r["idx"], r["kind"], r["span_s"], r["n_pre"], r["n_dec"],
                            round(r["pre"], 3), round(r["dec"], 3), round(r["tot"], 3), round(r["pd_ov"], 3)])
    # ---------------- 表 4：汇聚（PD 分开）+ 速度 + 热点 rank ----------------
    with open(OUT / "tables" / "pooled_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "load", "windows", "rank_maxmean_prefill", "rank_maxmean_decode",
                    "rank_maxmean_total_corr", "hot_in_rank0_frac", "pd_top8_overlap", "n_exp_for_50pct_prefill",
                    "n_exp_for_50pct_decode", "prefill_tok_s_system", "decode_tok_s_system",
                    "physical_slots", "slots_per_rank"])
        for k, d in data.items():
            if not d["win"]:
                w.writerow([k, d["grp"], d["load"], 0] + [""] * 11); continue
            p = d["pool"]; lp, ld, lt = p["loads"]
            w.writerow([k, d["grp"], d["load"], len(d["win"]),
                        round(lp.max() / lp.mean(), 3), round(ld.max() / ld.mean(), 3), round(lt.max() / lt.mean(), 3),
                        round(p["rank0_frac"], 3), round(float(p["pd_ov"].mean()), 3),
                        p["n50_pre"], p["n50_dec"], round(p["pref_speed"], 0), round(p["dec_speed"], 0),
                        p["P"], p["P"] // p["ep"]])
    # ---------------- 表 5：稳定性（相邻窗口） ----------------
    with open(OUT / "tables" / "window_stability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "pair", "layer_corr_mean", "top8_overlap", "pd_top8_overlap"])
        for k, d in data.items():
            ws = d["win"]
            for i in range(1, len(ws)):
                A = ws[i - 1]["pre"] + ws[i - 1]["dec"] / DEC_FIX
                B = ws[i]["pre"] + ws[i]["dec"] / DEC_FIX
                cs = [np.corrcoef(A[l], B[l])[0, 1] for l in range(78)
                      if A[l].sum() > 0 and B[l].sum() > 0 and A[l].std() > 0 and B[l].std() > 0]
                ovs = [len(set(np.argsort(-A[l])[:8]) & set(np.argsort(-B[l])[:8])) / 8 for l in range(78)
                       if A[l].sum() > 0 and B[l].sum() > 0]
                o_pd = pd_overlap(ws[i - 1]["pre"], ws[i - 1]["dec"])
                o_pd2 = pd_overlap(ws[i]["pre"], ws[i]["dec"])
                w.writerow([k, f"{ws[i-1]['idx']}->{ws[i]['idx']}", round(float(np.mean(cs)), 3),
                            round(float(np.mean(ovs)), 3), round(float(np.mean(o_pd2)), 3)])

    # ================= 图 =================
    keys = list(data.keys())
    # --- 图1：性能对比 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xs = np.arange(len(keys)); labels = [k.replace("_", "\n") for k in keys]
    for ax, fld, ttl, f in ((axes[0], "tput", "Throughput (tok/s)", 1), (axes[1], "ttft_p50", "TTFT p50 (s)", 1),
                            (axes[2], "tpot_p50", "TPOT p50 (ms)", 1000)):
        v = [data[k]["perf"][fld] * f for k in keys]
        ax.bar(xs, v, color=[COL[data[k]["grp"]] for k in keys],
               hatch=["" if data[k]["load"] == "agentX" else "//" for k in keys])
        for i, vv in enumerate(v):
            ax.text(i, vv, f"{vv:.0f}" if fld == "tput" else f"{vv:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.set_title(ttl); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Q4Exp: performance (plain = agentX, hatched = random)")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "extra" / "fig1_perf.png", dpi=110); plt.close(fig)

    # --- 图2：逐窗口 rank 不均衡（含重排）---
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), sharey=True)
    for ax, load in zip(axes, ("agentX", "random")):
        for grp in ("A0", "A", "B", "C"):
            k = f"{grp}_{load}"
            if k not in data or not data[k]["wrows"]:
                continue
            rr = data[k]["wrows"]
            ax.plot([r["idx"] for r in rr], [r["pre"] for r in rr], "o--", ms=4, lw=1.4,
                    color=COL[grp], alpha=.85, label=f"{grp} prefill")
            ax.plot([r["idx"] for r in rr], [r["tot"] for r in rr], "s-", ms=4, lw=1.8,
                    color=COL[grp], label=f"{grp} total(corr)")
        ax.axhline(1.0, color="gray", ls=":", lw=1); ax.set_xlabel("window index"); ax.set_title(load)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("rank load max/mean")
    fig.suptitle("Rank imbalance per window (squares = total corrected, circles = prefill only)")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "extra" / "fig2_rank_imbalance_time.png", dpi=110); plt.close(fig)

    # --- 图3：per-rank 负载曲线（agentX / random）---
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for ax, load in zip(axes, ("agentX", "random")):
        for grp in ("A", "B", "C"):
            k = f"{grp}_{load}"
            if k not in data or not data[k]["win"]:
                continue
            lt = data[k]["pool"]["loads"][2]; v = lt / lt.mean()
            ax.plot(range(len(v)), v, "-", lw=1.8, color=COL[grp],
                    label=f"{grp}  max/mean={v.max()/v.mean():.2f}x")
        ax.axhline(1.0, color="gray", ls=":"); ax.set_xlabel("EP rank"); ax.set_ylabel("load / mean")
        ax.set_title(f"{load}  (total, new window)"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.suptitle("Per-rank load: A (no EPLB) vs B (EPLB) vs C (EPLB+redundant 64)")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "extra" / "fig3_rank_load_curve.png", dpi=110); plt.close(fig)

    # --- 图4：Layer x Expert 热力图（A / B重排前 / B重排后 / C）---
    panels = []
    if "A_agentX" in data and data["A_agentX"]["win"]:
        H = data["A_agentX"]["pool"]["pre"] + data["A_agentX"]["pool"]["dec"] / DEC_FIX
        panels.append(("A: no EPLB (pooled 14 windows)", H))
    if "B_agentX" in data and len(data["B_agentX"]["win"]) > 1:
        ws = data["B_agentX"]["win"]
        pre0 = ws[0]
        H0 = pre0["pre"] + pre0["dec"] / DEC_FIX
        panels.append(("B: before 1st rebalance", H0))
        H1 = data["B_agentX"]["pool"]["pre"] + data["B_agentX"]["pool"]["dec"] / DEC_FIX
        panels.append(("B: after rebalances (pooled)", H1))
    if "C_agentX" in data and data["C_agentX"]["win"]:
        H = data["C_agentX"]["pool"]["pre"] + data["C_agentX"]["pool"]["dec"] / DEC_FIX
        panels.append(("C: EPLB + redundant 64 (pooled)", H))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.2))
        for ax, (ttl, H) in zip(np.atleast_1d(axes), panels):
            vmax = np.percentile(H[H > 0], 99.5) if (H > 0).any() else 1
            im = ax.imshow(H, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax,
                           origin="lower")
            ax.set_title(ttl, fontsize=10); ax.set_xlabel("logical expert"); ax.set_ylabel("layer (0 bottom -> 77 top)")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Layer x Expert (agentX, total = prefill + decode/8)"); fig.tight_layout()
        fig.savefig(OUT / "figs" / "extra" / "fig4_layer_expert.png", dpi=110); plt.close(fig)

    # --- 图5：Layer x Rank（A vs B-after vs C）---
    sel = [k for k in ("A_agentX", "B_agentX", "C_agentX") if k in data and data[k]["win"]]
    if sel:
        fig, axes = plt.subplots(1, len(sel), figsize=(6 * len(sel), 6.2))
        for ax, k in zip(np.atleast_1d(axes), sel):
            d = data[k]; p = d["pool"]; ep, P = p["ep"], p["P"]; k_ = P // ep
            H = p["pre"] + p["dec"] / DEC_FIX
            R = np.zeros((78, ep))
            for l in range(78):
                R[l] = H[l][p["pm"][l]].reshape(ep, k_).sum(1)
            vmax = np.percentile(R[R > 0], 99) if (R > 0).any() else 1
            im = ax.imshow(R, aspect="auto", cmap="turbo", interpolation="nearest", vmin=0, vmax=vmax,
                           origin="lower")
            lp = p["loads"][2]
            ax.set_title(f"{k.replace('_', ' ')}\nrank max/mean={lp.max()/lp.mean():.2f}x", fontsize=10)
            ax.set_xlabel("EP rank"); ax.set_ylabel("layer (0 bottom -> 77 top)"); plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Layer x Rank (agentX, total corrected)"); fig.tight_layout()
        fig.savefig(OUT / "figs" / "extra" / "fig5_layer_rank.png", dpi=110); plt.close(fig)

    # --- 图6：PD 分开（不均衡 / 重合 / 集中度）---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    lab = [k.replace("_", "\n") for k in keys]
    ax = axes[0]
    vp = [data[k]["pool"]["loads"][0].max() / data[k]["pool"]["loads"][0].mean() if data[k]["win"] else np.nan for k in keys]
    vd = [data[k]["pool"]["loads"][1].max() / data[k]["pool"]["loads"][1].mean() if data[k]["win"] else np.nan for k in keys]
    vt = [data[k]["pool"]["loads"][2].max() / data[k]["pool"]["loads"][2].mean() if data[k]["win"] else np.nan for k in keys]
    ax.bar(xs - 0.25, vp, 0.25, label="prefill", color="tab:blue")
    ax.bar(xs, vd, 0.25, label="decode (shape)", color="tab:red")
    ax.bar(xs + 0.25, vt, 0.25, label="total (corrected)", color="k")
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("rank load max/mean")
    ax.set_title("PD split: rank imbalance"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    ax.bar(xs, [float(data[k]["pool"]["pd_ov"].mean()) if data[k]["win"] else np.nan for k in keys],
           color=[COL[data[k]["grp"]] for k in keys])
    ax.axhline(0.25, color="r", ls="--", label="random baseline 0.25")
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("top-8 overlap (prefill vs decode)")
    ax.set_title("PD: are they the same experts?"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    ax.bar(xs - 0.2, [data[k]["pool"]["n50_pre"] if data[k]["win"] else np.nan for k in keys], 0.4,
           label="prefill", color="tab:blue")
    ax.bar(xs + 0.2, [data[k]["pool"]["n50_dec"] if data[k]["win"] else np.nan for k in keys], 0.4,
           label="decode", color="tab:red")
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("#experts for 50% of a layer")
    ax.set_title("Concentration (median over layers)"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Prefill vs Decode"); fig.tight_layout()
    fig.savefig(OUT / "figs" / "extra" / "fig6_pd_split.png", dpi=110); plt.close(fig)

    # --- 图7：延迟分解（每请求平均，堆叠）---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    for ax, load in zip(axes, ("agentX", "random")):
        ks = [k for k in keys if data[k]["load"] == load]
        bot = np.zeros(len(ks))
        for fld, nm, c in (("recv_to_fwd", "recv->forward (wait)", "#f0ad4e"), ("prefill", "prefill forward", "#5bc0de"),
                           ("first_send", "first-token send", "#cccccc"), ("decode", "decode", "#5cb85c")):
            v = np.array([data[k]["seg"][fld] for k in ks])
            ax.bar(range(len(ks)), v, bottom=bot, color=c, label=nm)
            bot += v
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([k.split("_")[0] for k in ks])
        ax.set_ylabel("seconds (mean per request)"); ax.set_title(load); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("E2E breakdown (server-side timestamps)")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "extra" / "fig7_latency_breakdown.png", dpi=110); plt.close(fig)

    # --- 图8：稳定性 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, fld, ttl in ((axes[0], "layer_corr_mean", "layer-wise corr of heat (consecutive windows)"),
                         (axes[1], "top8_overlap", "top-8 overlap (consecutive windows)")):
        for k in keys:
            rs = list(csv.DictReader(open(OUT / "tables" / "window_stability.csv")))
            rs = [r for r in rs if r["run"] == k]
            if not rs:
                continue
            ax.plot(range(len(rs)), [float(r[fld]) for r in rs], "o-", ms=3, color=COL[data[k]["grp"]],
                    ls=LS[data[k]["load"]], label=k)
        ax.set_xlabel("window pair"); ax.set_title(ttl); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "extra" / "fig8_stability.png", dpi=110); plt.close(fig)

    # --- 图9：结论图（热点落 rank0 比例 / 合计不均衡 / 重排耗时）---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    v = [data[k]["pool"]["rank0_frac"] if data[k]["win"] else np.nan for k in keys]
    ax.bar(xs, v, color=[COL[data[k]["grp"]] for k in keys],
           hatch=["" if data[k]["load"] == "agentX" else "//" for k in keys])
    for i, vv in enumerate(v):
        if vv == vv: ax.text(i, vv, f"{vv:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("fraction of layers whose hottest expert sits in rank 0")
    ax.set_title("Peak alignment"); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    v = [data[k]["pool"]["loads"][2].max() / data[k]["pool"]["loads"][2].mean() if data[k]["win"] else np.nan for k in keys]
    ax.bar(xs, v, color=[COL[data[k]["grp"]] for k in keys],
           hatch=["" if data[k]["load"] == "agentX" else "//" for k in keys])
    for i, vv in enumerate(v):
        if vv == vv: ax.text(i, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="gray", ls=":"); ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("rank load max/mean (total, corrected)"); ax.set_title("Imbalance"); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    for i, k in enumerate(keys):
        for j, d in enumerate(data[k]["rebal"]):
            ax.scatter(i, d, s=60, color=COL[data[k]["grp"]])
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("rebalance duration (s, log)"); ax.set_title("EPLB rebalance cost"); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Q4Exp conclusion: alignment -> imbalance; redundant experts fix it")
    fig.tight_layout(); fig.savefig(OUT / "figs" / "extra" / "fig9_story.png", dpi=110); plt.close(fig)

    print("\n产出：")
    for p in sorted((OUT / "tables").glob("*.csv")):
        print("  ", p.relative_to(OUT))
    for p in sorted((OUT / "figs").glob("fig1*.png")) + sorted((OUT / "figs").glob("fig[2-8]*.png")):
        print("  ", p.relative_to(OUT))


if __name__ == "__main__":
    main()
