#!/usr/bin/env python3
"""用 physical_to_logical_map + logical_count 算真实物理 rank 负载。

需要 torch，请在容器内运行。
"""

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)          # expert_distribution.pt
    ap.add_argument("--map", required=True)         # parse_layout_from_log.py 的输出
    ap.add_argument("--ep-size", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    obj = torch.load(a.pt, map_location="cpu", weights_only=False)
    logical_count = obj["logical_count"].float()  # [T, L, 256 logical]

    m = json.load(open(a.map))
    lm = m.get("last_map", m)
    p2l = lm["physical_to_logical_map"]  # [L][num_physical]
    num_layers = lm["num_layers"]
    num_physical = lm["num_physical_experts"]
    ep_size = a.ep_size
    num_local = num_physical // ep_size

    p2l_t = torch.tensor(p2l, dtype=torch.long)  # [L, num_physical]

    rank_load = torch.zeros(ep_size)
    for layer in range(num_layers):
        gathered = logical_count[:, layer, p2l_t[layer]]  # [T, num_physical]
        per_slot = gathered.sum(dim=0)                     # [num_physical]
        per_rank = per_slot.reshape(ep_size, num_local).sum(dim=1)
        rank_load += per_rank

    mean = rank_load.mean().item()
    max_v = rank_load.max().item()
    min_v = rank_load.min().item()
    std = rank_load.std().item()
    cv = std / mean if mean else None
    peak_to_mean = max_v / mean if mean else None
    max_to_min = max_v / min_v if min_v else None

    out = {
        "num_layers": num_layers,
        "num_physical_experts": num_physical,
        "num_local_per_rank": num_local,
        "ep_size": ep_size,
        "rank_token_count": [round(float(x), 2) for x in rank_load.tolist()],
        "rank_load": {
            "min": round(min_v, 4),
            "max": round(max_v, 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(cv, 6) if cv is not None else None,
            "peak_to_mean": round(peak_to_mean, 6) if peak_to_mean is not None else None,
            "max_to_min": round(max_to_min, 6) if max_to_min is not None else None,
        },
    }

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[rank_load_with_map] wrote {a.out}")
    print(json.dumps(out["rank_load"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
