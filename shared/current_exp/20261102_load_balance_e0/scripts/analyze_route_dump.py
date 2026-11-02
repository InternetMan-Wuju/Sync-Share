#!/usr/bin/env python3
"""Analyze the patched phase-window expert distribution dump."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def analyze_phase(name: str, counts: torch.Tensor, pmap: torch.Tensor, ep_size: int) -> dict:
    c = counts.sum(dim=0).to(torch.int64).numpy()
    m = pmap.numpy()
    n_layers, n_experts = c.shape
    slots_per_rank = n_experts // ep_size
    rank_load = np.zeros((n_layers, ep_size), dtype=np.int64)
    for layer in range(n_layers):
        for rank in range(ep_size):
            slots = m[layer, rank * slots_per_rank : (rank + 1) * slots_per_rank]
            rank_load[layer, rank] = c[layer, slots].sum()
    totals = c.sum(axis=1)
    maxima = rank_load.max(axis=1)
    means = rank_load.mean(axis=1)
    imbalance = np.divide(maxima, means, out=np.zeros_like(maxima, dtype=float), where=means > 0)
    cv = np.divide(rank_load.std(axis=1), means, out=np.zeros_like(means, dtype=float), where=means > 0)
    moe = totals > 0
    aggregate = rank_load[moe].sum(axis=0)
    aggregate_ratio = float(aggregate.max() / aggregate.mean()) if aggregate.mean() else None
    order_load = np.argsort(-maxima)
    order_imb = np.argsort(-imbalance)
    top_load = [
        {
            "layer": int(i),
            "total_tasks": int(totals[i]),
            "rank_max": int(maxima[i]),
            "max_over_mean": float(imbalance[i]),
            "rank_cv": float(cv[i]),
            "rank_load": rank_load[i].tolist(),
            "top_experts": [
                {"expert": int(j), "count": int(c[i, j])}
                for j in np.argsort(-c[i])[:10]
            ],
        }
        for i in order_load[:15]
    ]
    top_imbalance = [
        {
            "layer": int(i),
            "total_tasks": int(totals[i]),
            "rank_max": int(maxima[i]),
            "max_over_mean": float(imbalance[i]),
            "rank_cv": float(cv[i]),
        }
        for i in order_imb[:15]
    ]
    return {
        "phase": name,
        "steps": int(counts.shape[0]),
        "observed_task_count": int(totals.sum()),
        "moe_layer_count": int(moe.sum()),
        "aggregate_rank_load": aggregate.tolist(),
        "aggregate_max_over_mean": aggregate_ratio,
        "top_layers_by_absolute_rank_max": top_load,
        "top_layers_by_imbalance_ratio": top_imbalance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-prefill-tasks", type=int, default=4096 * 75 * 8)
    args = parser.parse_args()
    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    ep_size = int(d["ep_size"])
    pmap = d["physical_to_logical_map"]
    result = {
        "dump": str(Path(args.dump)),
        "window_kind": d.get("window_kind"),
        "num_layers": d.get("num_layers"),
        "num_logical_experts": d.get("num_logical_experts"),
        "num_physical_experts": d.get("num_physical_experts"),
        "ep_size": ep_size,
        "n_steps_by_phase": d.get("n_steps_by_phase"),
        "phases": {},
        "warnings": [],
    }
    for name, key in (("prefill", "logical_count_prefill"), ("decode", "logical_count_decode")):
        value = d.get(key)
        if value is None:
            continue
        result["phases"][name] = analyze_phase(name, value, pmap, ep_size)
    prefill = result["phases"].get("prefill", {})
    observed = prefill.get("observed_task_count")
    result["expected_prefill_tasks"] = args.expected_prefill_tasks
    result["prefill_conservation_ok"] = observed == args.expected_prefill_tasks
    if not result["prefill_conservation_ok"]:
        result["warnings"].append("prefill task count does not match fixed-input expectation")
    decode = result["phases"].get("decode", {})
    if decode:
        observed_decode = decode["observed_task_count"]
        result["decode_observed_task_count"] = observed_decode
        result["warnings"].append(
            "decode count is reported as-is and is not treated as a conserved token count; "
            "the historical recorder path can duplicate decode tasks across parallelism."
        )
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({
        "out": args.out,
        "prefill_tasks": observed,
        "expected_prefill": args.expected_prefill_tasks,
        "prefill_conserved": result["prefill_conservation_ok"],
        "decode_tasks_observed": decode.get("observed_task_count"),
        "prefill_aggregate_max_over_mean": prefill.get("aggregate_max_over_mean"),
        "decode_aggregate_max_over_mean": decode.get("aggregate_max_over_mean"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
