#!/usr/bin/env python3
"""从全局 logical_count 直接算 rank 负载（无 EPLB 时 logical==physical）。"""

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--ep-size", type=int, default=64)
    ap.add_argument("--out", default="/tmp/rank_load_global.json")
    a = ap.parse_args()

    obj = torch.load(a.pt, map_location="cpu", weights_only=False)
    logical_count = obj["logical_count"]
    # [time, layer, num_logical_experts]
    num_logical = logical_count.shape[-1]
    num_local = num_logical // a.ep_size

    # 无 EPLB：logical i 由 rank (i // num_local) 持有
    per_rank = logical_count.reshape(
        logical_count.shape[0], logical_count.shape[1], a.ep_size, num_local
    )
    rank_load = per_rank.sum(dim=(0, 1, 3)).float()

    mean = rank_load.mean().item()
    max_v = rank_load.max().item()
    min_v = rank_load.min().item()
    std = rank_load.std().item()
    cv = std / mean if mean else None
    peak_to_mean = max_v / mean if mean else None

    out = {
        "num_logical_experts": num_logical,
        "num_local_per_rank": num_local,
        "ep_size": a.ep_size,
        "logical_count_shape": list(logical_count.shape),
        "rank_token_count": [round(float(x), 2) for x in rank_load.tolist()],
        "rank_load": {
            "min": round(min_v, 4),
            "max": round(max_v, 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(cv, 6) if cv is not None else None,
            "peak_to_mean": round(peak_to_mean, 6) if peak_to_mean is not None else None,
        },
    }

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[rank_load_global] wrote -> {a.out}")
    print(json.dumps(out["rank_load"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
