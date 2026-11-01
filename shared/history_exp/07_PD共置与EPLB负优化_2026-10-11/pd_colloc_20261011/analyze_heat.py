#!/usr/bin/env python3
"""CPU-only analysis of the phase_window recorder dumps.

Reads <windows>/*.pt (our expert_distribution_phase_window patch output) and produces
per-window and time-evolution heatmaps of Layer x Expert and Layer x EP-Rank,
using the physical_to_logical_map that was in effect for EACH window.
"""
import argparse, json, os, pathlib, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_windows(d):
    items = []
    for p in sorted(pathlib.Path(d).glob("*.pt")):
        if p.name.startswith("expert_distribution_recorder_"):
            continue          # 那是 stock 的 rank0 全量文件，另算
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  跳过 {p.name}: {e}")
            continue
        if not isinstance(obj, dict) or "physical_to_logical_map" not in obj:
            continue
        items.append((obj.get("dump_index", len(items)), p.name, obj))
    items.sort(key=lambda x: x[0])
    return items


def heat_of(step_tensor):
    """[steps, layers, experts] -> [layers, experts] (sum over steps), or None"""
    if step_tensor is None:
        return None
    a = np.asarray(step_tensor)
    if a.size == 0:
        return None
    return a.sum(axis=0)


def rank_heat(layer_expert, pmap, ep_size):
    """layer_expert: [L, E_logical]; pmap: [L, E_phys] -> [L, ep_size]"""
    L = layer_expert.shape[0]
    Ephys = pmap.shape[1]
    if Ephys % ep_size:
        return None
    local = Ephys // ep_size
    out = np.zeros((L, ep_size), dtype=np.int64)
    for layer in range(L):
        phys = layer_expert[layer][pmap[layer]]
        out[layer] = phys.reshape(ep_size, local).sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep-size", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    wins = load_windows(a.windows)
    print(f"载入 {len(wins)} 个窗口")
    if not wins:
        sys.exit("没有可用窗口")

    idx, series = [], []
    for di, name, obj in wins:
        L = int(obj["num_layers"]); E = int(obj["num_logical_experts"])
        hp = heat_of(obj.get("logical_count_prefill"))
        hd = heat_of(obj.get("logical_count_decode"))
        pmap = np.asarray(obj["physical_to_logical_map"], dtype=np.int64) \
            if obj.get("physical_to_logical_map") is not None else None
        rec = dict(dump_index=di, file=name, window_kind=obj.get("window_kind"),
                   t_end=obj.get("window_end_timestamp"), t_start=obj.get("window_start_timestamp"),
                   n_steps=obj.get("n_steps_by_phase"),
                   layers=L, experts=E,
                   prefill_total=None if hp is None else int(hp.sum()),
                   decode_total=None if hd is None else int(hd.sum()),
                   rank_prefill=None, rank_decode=None)
        if pmap is not None:
            if hp is not None: rec["rank_prefill"] = rank_heat(hp, pmap, a.ep_size)
            if hd is not None: rec["rank_decode"] = rank_heat(hd, pmap, a.ep_size)
        rec["_hp"], rec["_hd"] = hp, hd
        rec["_pmap"] = pmap
        idx.append(di); series.append(rec)
        print(f"  窗口 {di}: kind={rec['window_kind']} steps={rec['n_steps']} "
              f"prefill={rec['prefill_total']} decode={rec['decode_total']}")

    # --- 每窗口的 PNG ---
    for r in series:
        for tag, h in (("prefill", r["_hp"]), ("decode", r["_hd"])):
            if h is None: continue
            fig, ax = plt.subplots(figsize=(11, 5))
            im = ax.imshow(np.log1p(h), aspect="auto", origin="lower", cmap="viridis")
            ax.set_xlabel("logical expert"); ax.set_ylabel("layer")
            ax.set_title(f"window {r['dump_index']} [{tag}] {r['window_kind']}  steps={r['n_steps']}")
            fig.colorbar(im, ax=ax, label="log1p(count)")
            fig.tight_layout(); fig.savefig(out / f"win{r['dump_index']:03d}_{tag}_layer_expert.png", dpi=110)
            plt.close(fig)
        for tag, h in (("prefill", r["rank_prefill"]), ("decode", r["rank_decode"])):
            if h is None: continue
            fig, ax = plt.subplots(figsize=(11, 5))
            im = ax.imshow(np.log1p(h), aspect="auto", origin="lower", cmap="magma")
            ax.set_xlabel(f"EP rank (0..{a.ep_size-1})"); ax.set_ylabel("layer")
            ax.set_title(f"window {r['dump_index']} [{tag}] Layer x EP-Rank")
            fig.colorbar(im, ax=ax, label="log1p(count)")
            fig.tight_layout(); fig.savefig(out / f"win{r['dump_index']:03d}_{tag}_layer_rank.png", dpi=110)
            plt.close(fig)

    # --- 时间演化（把所有 layer 加起来）---
    t = [r["t_end"] for r in series]
    t0 = min([x for x in t if x] or [0])
    xs = [ (x - t0)/60.0 if x else i for i, x in enumerate(t) ]
    def stack(key):
        arr = [r[key] for r in series]
        return None if any(x is None for x in arr) else np.stack(arr)
    for tag in ("prefill", "decode"):
        H = stack("_h" + ("p" if tag == "prefill" else "d"))
        if H is not None:
            tot = H.sum(axis=1)     # [windows, experts]
            fig, ax = plt.subplots(figsize=(13, 5))
            im = ax.imshow(np.log1p(tot.T), aspect="auto", origin="lower", cmap="viridis",
                           extent=[xs[0], xs[-1], 0, tot.shape[1]])
            ax.set_xlabel("time (min)"); ax.set_ylabel("logical expert")
            ax.set_title(f"[{tag}] expert heat over time")
            for i, r in enumerate(series):
                if r["window_kind"] == "eplb_rebalance":
                    ax.axvline(xs[i], color="w", ls=":", lw=1)
            fig.colorbar(im, ax=ax, label="log1p(count)")
            fig.tight_layout(); fig.savefig(out / f"evolution_{tag}_expert.png", dpi=110)
            plt.close(fig)
        R = stack("rank_" + tag)
        if R is not None:
            tot = R.sum(axis=1)     # [windows, ranks]
            fig, ax = plt.subplots(figsize=(13, 5))
            im = ax.imshow(np.log1p(tot.T), aspect="auto", origin="lower", cmap="magma",
                           extent=[xs[0], xs[-1], 0, tot.shape[1]])
            ax.set_xlabel("time (min)"); ax.set_ylabel("EP rank")
            ax.set_title(f"[{tag}] EP-Rank heat over time  (竖虚线=重排窗口边界)")
            for i, r in enumerate(series):
                if r["window_kind"] == "eplb_rebalance":
                    ax.axvline(xs[i], color="w", ls=":", lw=1)
            fig.colorbar(im, ax=ax, label="log1p(count)")
            fig.tight_layout(); fig.savefig(out / f"evolution_{tag}_rank.png", dpi=110)
            plt.close(fig)

    # --- 汇总 json/csv ---
    summary = []
    for r in series:
        # keep only JSON-serializable scalars/dicts/lists (drop ndarray payloads)
        s = {k: v for k, v in r.items()
             if not k.startswith("_") and k not in ("rank_prefill", "rank_decode")
             and not isinstance(v, np.ndarray)}
        if r["rank_prefill"] is not None:
            v = r["rank_prefill"].sum(axis=0).astype(float)
            s["rank_prefill_min"] = float(v.min()); s["rank_prefill_max"] = float(v.max())
        if r["rank_decode"] is not None:
            v = r["rank_decode"].sum(axis=0).astype(float)
            s["rank_decode_min"] = float(v.min()); s["rank_decode_max"] = float(v.max())
        summary.append(s)
    (out / "windows_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (out / "windows_summary.csv").open("w") as f:
        cols = ["dump_index", "window_kind", "t_start", "t_end", "prefill_total", "decode_total",
                "rank_prefill_min", "rank_prefill_max", "rank_decode_min", "rank_decode_max"]
        f.write(",".join(cols) + "\n")
        for s in summary:
            f.write(",".join(str(s.get(c, "")) for c in cols) + "\n")
    print(f"\n输出目录: {out}")
    for p in sorted(out.glob("*")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
