#!/usr/bin/env python3
"""读取 SGLang expert_distribution_recorder_*.pt，输出可读 JSON。

先做通用 introspection，避免对 .pt 内部结构做过度假设。
"""

import argparse
import json
import math

import torch


def tensor_stats(t):
    t = t.detach().cpu().reshape(-1)
    if t.numel() == 0:
        return {
            "numel": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    if t.is_floating_point():
        vals = t.float()
    else:
        vals = t.float()
    return {
        "numel": int(t.numel()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std().item()),
    }


def walk(obj, prefix=""):
    out = []
    if isinstance(obj, torch.Tensor):
        info = {
            "key": prefix or "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
        info.update(tensor_stats(obj))
        out.append(info)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.extend(walk(v, child_prefix))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]"
            out.extend(walk(v, child_prefix))
    return out


def gini(values):
    values = [float(v) for v in values if v >= 0]
    if not values or sum(values) == 0:
        return None
    values = sorted(values)
    n = len(values)
    total = sum(values)
    cum = 0.0
    s = 0.0
    for i, v in enumerate(values, 1):
        cum += v
        s += (i / n) - (cum / total)
    return round(2 * s / n, 6)


def infer_metrics(tensor_infos):
    metrics = []
    interesting_keys = (
        "logical_count",
        "physical_count",
        "expert_count",
        "rank",
        "token",
    )
    for info in tensor_infos:
        key = info["key"].lower()
        if not any(k in key for k in interesting_keys):
            continue
        if not info.get("numel"):
            continue
        # 大多数情况下最后一维是 expert，倒数第二维是 layer。
        shape = info["shape"]
        if len(shape) >= 2:
            layer_axis = len(shape) - 2
            expert_axis = len(shape) - 1
            # 这里不做完整张量载入，只基于已有全局统计给出提示。
            metrics.append(
                {
                    "key": info["key"],
                    "shape": shape,
                    "note": "last axis assumed expert; second-last axis assumed layer",
                    "global_peak_to_mean": (
                        round(info["max"] / info["mean"], 4)
                        if info.get("mean")
                        else None
                    ),
                    "global_cv": (
                        round(info["std"] / info["mean"], 4)
                        if info.get("mean")
                        else None
                    ),
                }
            )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", default="/tmp/expert_distribution.json")
    a = ap.parse_args()

    obj = torch.load(a.pt, map_location="cpu", weights_only=False)
    tensor_infos = walk(obj)

    out = {
        "source": a.pt,
        "top_level_type": type(obj).__name__,
        "tensor_count": len(tensor_infos),
        "tensors": tensor_infos,
        "inferred_metrics": infer_metrics(tensor_infos),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[analyze] wrote {len(tensor_infos)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
