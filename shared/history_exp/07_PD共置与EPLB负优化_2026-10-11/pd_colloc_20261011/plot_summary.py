#!/usr/bin/env python3
"""Summary figures for the 4-node EP64 collocated run (CPU only)."""
import pathlib, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

WIN = pathlib.Path("results/20261012_000935/windows")
OUT = pathlib.Path("results/20261012_000935/figs"); OUT.mkdir(parents=True, exist_ok=True)
EP = 64

def load():
    ws = []
    for p in sorted(WIN.glob("0*.pt")):
        o = torch.load(p, map_location="cpu", weights_only=False)
        pm = np.asarray(o["physical_to_logical_map"])
        def per_rank(key):
            t = o.get(key)
            if t is None: return None
            h = np.asarray(t).sum(axis=0)                       # [L,256] logical
            phys = np.stack([h[l][pm[l]] for l in range(h.shape[0])])
            return phys.reshape(phys.shape[0], EP, 256 // EP).sum(axis=(0, 2))   # [64]
        ws.append(dict(idx=o["dump_index"], kind=o["window_kind"],
                       t=o.get("window_end_timestamp"),
                       pre=per_rank("logical_count_prefill"),
                       dec=per_rank("logical_count_decode")))
    return ws

ws = load()
t0 = min(w["t"] for w in ws if w["t"])
xs = [ (w["t"]-t0)/60 if w["t"] else float(w["idx"]) for w in ws ]
rebal = [i for i, w in enumerate(ws) if w["kind"] == "eplb_rebalance"]

# ---- Fig 1: per-rank load, before vs after the rebalance ----
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
before = [i for i, w in enumerate(ws) if w["kind"] != "eplb_rebalance" and i <= 2 and w["pre"] is not None]
after  = [i for i, w in enumerate(ws) if w["kind"] != "eplb_rebalance" and i >= 4 and w["pre"] is not None]
for ax, key, name in ((axes[0], "pre", "PREFILL"), (axes[1], "dec", "DECODE")):
    for i in before[:1]:
        v = ws[i][key]; ax.plot(range(EP), v/v.mean(), "o-", lw=2, ms=5,
                                label=f"before rebalance (win{ws[i]['idx']})  max/mean={v.max()/v.mean():.2f}x", color="tab:blue")
    for i in after[:1]:
        v = ws[i][key]; ax.plot(range(EP), v/v.mean(), "s-", lw=2, ms=5,
                                label=f"after rebalance (win{ws[i]['idx']})  max/mean={v.max()/v.mean():.2f}x", color="tab:red")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_ylabel("rank load / mean")
    ax.set_title(f"{name}: per-EP-rank load (normalised)  — EPLB rebalance makes it WORSE")
    ax.legend(); ax.grid(alpha=.3)
axes[1].set_xlabel("EP rank (0..63)")
fig.tight_layout(); fig.savefig(OUT/"fig1_rank_load_before_vs_after.png", dpi=120); plt.close(fig)

# ---- Fig 2: max/mean over time ----
fig, ax = plt.subplots(figsize=(12, 5))
for key, name, c in (("pre","prefill","tab:blue"), ("dec","decode","tab:red")):
    yy = [ (w[key].max()/w[key].mean()) if w[key] is not None else np.nan for w in ws ]
    ax.plot(xs, yy, "o-", lw=2, color=c, label=f"{name}  max/mean")
for i in rebal:
    ax.axvline(xs[i], color="green", ls=":", lw=2)
    ax.text(xs[i], ax.get_ylim()[1]*0.95, " EPLB rebalance", color="green", fontsize=9, va="top")
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.set_xlabel("time (min)"); ax.set_ylabel("hottest EP rank / mean")
ax.set_title("EP64 collocated, agentX workload: rank imbalance over time  (green = EPLB rebalance)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/"fig2_imbalance_over_time.png", dpi=120); plt.close(fig)

# ---- Fig 3: rank load heatmap over time ----
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for ax, key, name in ((axes[0],"pre","PREFILL"), (axes[1],"dec","DECODE")):
    M = np.stack([ (w[key]/w[key].mean()) if w[key] is not None else np.full(EP, np.nan) for w in ws ])
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="hot", norm=LogNorm(vmin=0.2, vmax=max(8, np.nanmax(M))),
                   extent=[0, EP, 0, len(ws)])
    ax.set_ylabel("window # (time \u2192)")
    ax.set_title(f"{name}: rank load / mean per window  (red = overloaded)")
    plt.colorbar(im, ax=ax, label="load / mean")
    for i in rebal: ax.axhline(i+0.5, color="cyan", ls=":", lw=1.5)
axes[1].set_xlabel("EP rank")
fig.tight_layout(); fig.savefig(OUT/"fig3_rank_heat_over_time.png", dpi=120); plt.close(fig)

print("figs written to", OUT)
for p in sorted(OUT.glob("*.png")): print("  ", p.name)
