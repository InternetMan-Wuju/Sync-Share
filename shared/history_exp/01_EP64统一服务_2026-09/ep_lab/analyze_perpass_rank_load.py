#!/usr/bin/env python3
"""合并 per_pass dump，计算每个 EP rank 的物理专家负载（token 计数）。"""

import argparse
import glob
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ep-size", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "rank_*.pt")))
    if not files:
        files = sorted(
            glob.glob(os.path.join(a.dir, "expert_distribution_recorder_*.pt"))
        )

    if not files:
        raise SystemExit(f"没有找到 .pt 文件: {a.dir}")

    total = None
    num_layers = None
    num_physical = None
    num_records = 0
    ranks_seen = set()
    physical_to_logical = None

    for f in files:
        obj = torch.load(f, map_location="cpu", weights_only=False)
        if physical_to_logical is None and isinstance(obj, dict):
            physical_to_logical = obj.get("last_physical_to_logical_map")
        records = obj.get("records", []) if isinstance(obj, dict) else []
        for rec in records:
            gpc = rec.get("global_physical_count")
            if gpc is None:
                continue
            gpc = gpc.cpu()
            if total is None:
                num_layers, num_physical = gpc.shape
                total = torch.zeros((num_layers, num_physical), dtype=torch.int64)
            total += gpc.long()
            num_records += 1
            if isinstance(rec.get("rank"), int):
                ranks_seen.add(rec["rank"])

    ep_size = a.ep_size
    num_local = num_physical // ep_size

    # rank r 拥有物理专家段 [num_local*r, num_local*(r+1)]
    rank_load = total.reshape(num_layers, ep_size, num_local).sum(dim=(0, 2))
    rank_load = rank_load.float()

    mean = rank_load.mean().item()
    max_v = rank_load.max().item()
    min_v = rank_load.min().item()
    std = rank_load.std().item()
    cv = std / mean if mean else None
    peak_to_mean = max_v / mean if mean else None

    # 每层 rank 负载的 CV（层间不均衡的另一视角）
    per_layer = total.reshape(num_layers, ep_size, num_local).sum(dim=2).float()
    per_layer_max = per_layer.max(dim=1).values
    per_layer_mean = per_layer.mean(dim=1)
    per_layer_peak_to_mean = (per_layer_max / per_layer_mean).tolist()

    out = {
        "num_files": len(files),
        "ranks_seen": sorted(ranks_seen),
        "num_records": num_records,
        "num_layers": num_layers,
        "num_physical_experts": num_physical,
        "num_local_physical_experts_per_rank": num_local,
        "ep_size": ep_size,
        "rank_token_count": [round(float(x), 2) for x in rank_load.tolist()],
        "rank_load": {
            "min": round(min_v, 4),
            "max": round(max_v, 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(cv, 6) if cv is not None else None,
            "peak_to_mean": round(peak_to_mean, 6) if peak_to_mean is not None else None,
        },
        "per_layer_peak_to_mean": [
            round(float(x), 4) for x in per_layer_peak_to_mean
        ],
    }

    if physical_to_logical is not None:
        p2l = physical_to_logical
        if torch.is_tensor(p2l):
            p2l = p2l.cpu()
            out["physical_to_logical_map"] = {
                "type": "tensor",
                "shape": list(p2l.shape),
                "dtype": str(p2l.dtype),
            }
        else:
            out["physical_to_logical_map"] = {"type": type(p2l).__name__}

    out_path = a.out or os.path.join(a.dir, "perpass_rank_load.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[analyze] wrote -> {out_path}")
    print(
        json.dumps(
            {
                "num_files": out["num_files"],
                "ranks_seen_count": len(ranks_seen),
                "num_records": num_records,
                "num_physical_experts": num_physical,
                "rank_load": out["rank_load"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
