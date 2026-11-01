#!/usr/bin/env python3
"""Combine P0/P1/D heat JSON files and plot P/D-separated heatmaps."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(values):
    vals = np.asarray(values, dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std())
    return {
        "n": int(vals.size),
        "total": float(vals.sum()),
        "mean": mean,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "std": std,
        "cv": std / mean if mean else None,
        "peak_to_mean": float(vals.max()) / mean if mean else None,
        "max_to_min": float(vals.max()) / float(vals.min())
        if float(vals.min()) > 0
        else None,
    }


def write_matrix_csv(path, matrix, prefix):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"{prefix}_{i}" for i in range(matrix.shape[1])])
        for layer, row in enumerate(matrix):
            writer.writerow([layer] + [int(x) for x in row])


def heat(ax, matrix, title, xlabel):
    image = ax.imshow(np.log1p(matrix), aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Layer")
    plt.colorbar(image, ax=ax, label="log1p(expert-token assignments)")


def save_heat_pair(prefill, decode, path, title, xlabels):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    heat(axes[0], prefill, f"Prefill - {title}", xlabels[0])
    heat(axes[1], decode, f"Decode - {title}", xlabels[1])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill0", required=True)
    ap.add_argument("--prefill1", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="PD heat")
    args = ap.parse_args()

    p0 = load(args.prefill0)
    p1 = load(args.prefill1)
    dec = load(args.decode)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p0_expert = np.asarray(p0["layer_expert_tokens"], dtype=np.int64)
    p1_expert = np.asarray(p1["layer_expert_tokens"], dtype=np.int64)
    d_expert = np.asarray(dec["layer_expert_tokens"], dtype=np.int64)
    if p0_expert.shape != p1_expert.shape or p0_expert.shape != d_expert.shape:
        raise SystemExit(
            f"Layer x Expert shapes differ: {p0_expert.shape}, "
            f"{p1_expert.shape}, {d_expert.shape}"
        )

    # Prefill consists of two independent EP16 workers. Token heat is summed;
    # physical rank heat is concatenated as P0-rank0..15 + P1-rank0..15.
    p_expert = p0_expert + p1_expert
    p_rank = np.concatenate(
        [
            np.asarray(p0["layer_rank_tokens"], dtype=np.int64),
            np.asarray(p1["layer_rank_tokens"], dtype=np.int64),
        ],
        axis=1,
    )
    d_rank = np.asarray(dec["layer_rank_tokens"], dtype=np.int64)

    write_matrix_csv(out_dir / "prefill_layer_expert_heat.csv", p_expert, "expert")
    write_matrix_csv(out_dir / "decode_layer_expert_heat.csv", d_expert, "expert")
    write_matrix_csv(out_dir / "prefill_layer_ep_rank_heat.csv", p_rank, "prefill_rank")
    write_matrix_csv(out_dir / "decode_layer_ep_rank_heat.csv", d_rank, "decode_rank")

    save_heat_pair(
        p_expert,
        d_expert,
        out_dir / "pd_separated_layer_expert_heat.png",
        "Layer x Logical Expert",
        ("Logical Expert", "Logical Expert"),
    )
    save_heat_pair(
        p_rank,
        d_rank,
        out_dir / "pd_separated_layer_ep_rank_heat.png",
        "Layer x Physical EP Rank",
        ("P0 ranks 0-15, then P1 ranks 0-15", "Decode EP Rank"),
    )

    p_rank_totals = p_rank.sum(axis=0)
    d_rank_totals = d_rank.sum(axis=0)
    p_expert_totals = p_expert.sum(axis=0)
    d_expert_totals = d_expert.sum(axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].bar(np.arange(len(p_rank_totals)), p_rank_totals, color="#2a6f97")
    axes[0].set_title("Prefill total heat by physical rank")
    axes[0].set_xlabel("P0 ranks 0-15, then P1 ranks 0-15")
    axes[0].set_ylabel("Expert-token assignments")
    axes[1].bar(np.arange(len(d_rank_totals)), d_rank_totals, color="#bc4749")
    axes[1].set_title("Decode total heat by physical EP rank")
    axes[1].set_xlabel("Decode EP Rank")
    axes[1].set_ylabel("Expert-token assignments")
    fig.suptitle(args.title)
    fig.savefig(out_dir / "pd_ep_rank_total_heat.png", dpi=180)
    plt.close(fig)

    def top_ranks(vals, labels):
        order = np.argsort(vals)[::-1][:5]
        return [
            {"rank": labels[int(i)], "tokens": int(vals[int(i)])} for i in order
        ]

    def top_experts(vals):
        order = np.argsort(vals)[::-1][:5]
        return [
            {"logical_expert": int(i), "tokens": int(vals[int(i)])}
            for i in order
        ]

    p_labels = [f"P0-r{i}" for i in range(p0["ep_size"])] + [
        f"P1-r{i}" for i in range(p1["ep_size"])
    ]
    d_labels = [f"D-r{i}" for i in range(dec["ep_size"])]
    summary = {
        "title": args.title,
        "count_semantics": "expert-token assignments from logical_count",
        "visual_scale": "log1p for heatmap colors; CSV/JSON retain raw counts",
        "prefill": {
            "workers": [p0["role"], p1["role"]],
            "coverage_validated": bool(
                p0.get("coverage_validated") and p1.get("coverage_validated")
            ),
            "coverage_by_worker": {
                p0["role"]: p0.get("coverage_validated", False),
                p1["role"]: p1.get("coverage_validated", False),
            },
            "window_count_by_worker": {
                p0["role"]: p0.get("window_count"),
                p1["role"]: p1.get("window_count"),
            },
            "layer_expert_shape": list(p_expert.shape),
            "layer_rank_shape": list(p_rank.shape),
            "recorder_window_steps": [
                p0["recorder_window_steps"],
                p1["recorder_window_steps"],
            ],
            "total_expert_token_assignments": int(p_expert.sum()),
            "expert_summary": summarize(p_expert_totals),
            "top5_hot_logical_experts": top_experts(p_expert_totals),
            "rank_summary": summarize(p_rank_totals),
            "top5_hot_ranks": top_ranks(p_rank_totals, p_labels),
        },
        "decode": {
            "worker": dec["role"],
            "coverage_validated": bool(dec.get("coverage_validated")),
            "window_count": dec.get("window_count"),
            "layer_expert_shape": list(d_expert.shape),
            "layer_rank_shape": list(d_rank.shape),
            "recorder_window_steps": dec["recorder_window_steps"],
            "total_expert_token_assignments": int(d_expert.sum()),
            "expert_summary": summarize(d_expert_totals),
            "top5_hot_logical_experts": top_experts(d_expert_totals),
            "rank_summary": summarize(d_rank_totals),
            "top5_hot_ranks": top_ranks(d_rank_totals, d_labels),
        },
    }
    (out_dir / "pd_heat_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
