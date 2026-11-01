#!/usr/bin/env python3
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "cache" / "matplotlib")
os.environ["TMPDIR"] = str(HERE / "cache")
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load(HERE / "example_rank_loads.npz")
rows = list(csv.DictReader((HERE / "tables" / "aggregate_metrics.csv").open()))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, method, label in zip(axes[0], ("stock", "shifted"), ("Stock EPLB", "Whole-rank shift by layer")):
    demand = data[method][3:]
    normalized = demand / demand.mean(axis=1, keepdims=True)
    im = ax.imshow(normalized, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=8)
    ax.set_title(label + ": per-layer demand")
    ax.set_xlabel("EP rank")
    ax.set_ylabel("MoE layer index")
fig.colorbar(im, ax=axes[0].tolist(), label="Demand / layer mean", fraction=.025, pad=.02)

for method, label, color in (("stock", "Stock", "#c2410c"), ("shifted", "Rank shift", "#0369a1"),
                             ("global_permutation", "Global rank permutation", "#047857")):
    total = data[method].sum(axis=0)
    axes[1, 0].plot(total / total.mean(), label=label, color=color, linewidth=2)
axes[1, 0].axhline(1, color="grey", linestyle="--", linewidth=1)
axes[1, 0].set_title("Cumulative demand becomes balanced")
axes[1, 0].set_xlabel("EP rank")
axes[1, 0].set_ylabel("Cumulative demand / rank mean")
axes[1, 0].legend(fontsize=9)

methods = ["identity", "stock_fit_same_window", "stock_rank_shift", "stock_global_rank_permutation", "replicas64_fit_equal_split"]
selected = {r["method"]: r for r in rows if r["run"] == "B_agentx_20261012_052314" and r["phase"] == "prefill"}
values = [float(selected[m]["sequential_work_ratio"]) for m in methods]
bars = axes[1, 1].bar(range(len(methods)), values, color=["#64748b", "#c2410c", "#0369a1", "#047857", "#7e22ce"])
axes[1, 1].set_xticks(range(len(methods)), ["Identity", "Stock", "Rank\nshift", "Global rank\npermutation", "+64 replicas\n(equal split)"], fontsize=9)
axes[1, 1].set_ylim(0, 8.5)
axes[1, 1].set_ylabel("Sum of step/layer maxima / sum of means")
axes[1, 1].set_title("Sequential critical work stays the same")
for bar, value in zip(bars, values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, value + .12, f"{value:.3f}", ha="center", fontsize=10)
fig.suptitle("Layer shifts redistribute work but preserve each layer's bottleneck", fontsize=15)
fig.supxlabel("Archived agentX routing replay; demand proxies, not measured latency. Heatmaps: one prefill window; bars: all prefill samples.", fontsize=9)
fig.savefig(HERE / "figures" / "layer_shift_vs_critical_work.png", dpi=180)
fig.savefig(HERE / "figures" / "layer_shift_vs_critical_work.pdf")
plt.close(fig)
