#!/usr/bin/env python3
"""从 expert_distribution_recorder_*.pt 的 logical_count 算专家热度指标。

logical_count 形状 [time, layer, expert]（如 [2077, 78, 256]），
int32，含义是每个 (时间步, 层, 逻辑专家) 被路由到的 token 数。

本脚本在**容器内**运行（宿主机无 torch），产出 ep_exp2.md §3.2 需要的指标：
  1) 全局专家热度总和（256 个专家）
  2) 逐层专家热度（78 层）
  3) 某一时刻专家热度快照（首/中/尾）
  4) rank 负载（identity 映射，把专家热度汇总到 32/64 个 rank）
  5) 不均衡三件套：CV、peak/mean、max/min
"""

import argparse
import json
import math

import torch


def pct(vals, q):
    if not vals:
        return None
    s = sorted(float(v) for v in vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return round(s[idx], 4)


def summarize(vals):
    """对一组非负数值做统计，返回 total/mean/min/max/std/CV/peak_to_mean。"""
    vals = [float(v) for v in vals]
    if not vals:
        return None
    total = sum(vals)
    mean = total / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    mx = max(vals)
    return {
        "n": len(vals),
        "total": round(total, 2),
        "mean": round(mean, 4),
        "min": round(min(vals), 4),
        "max": round(mx, 4),
        "std": round(std, 4),
        "cv": round(std / mean, 4) if mean > 0 else None,
        "peak_to_mean": round(mx / mean, 4) if mean > 0 else None,
    }


def topk_concentration(vals, ks=(1, 5, 10, 20)):
    total = sum(vals)
    if total <= 0:
        return {}
    s = sorted(vals, reverse=True)
    out = {}
    for k in ks:
        share = sum(s[: min(k, len(s))]) / total
        out[f"top{k}_share"] = round(share, 4)
    return out


def rank_load_from_experts(expert_totals, ep_size):
    """identity 映射：逻辑专家 e 落在 rank e // (num_experts // ep_size)。

    256 专家 / EP 个 rank = 每 rank K 个专家，rank r 负责 [r*K, (r+1)*K)。
    """
    num_experts = len(expert_totals)
    if num_experts % ep_size != 0:
        raise ValueError(f"experts={num_experts} not divisible by EP={ep_size}")
    k = num_experts // ep_size
    ranks = []
    for r in range(ep_size):
        ranks.append(sum(expert_totals[r * k : (r + 1) * k]))
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", default="/tmp/expert_heat.json")
    ap.add_argument("--ep-size", type=int, default=64)
    ap.add_argument("--snapshots", type=int, default=3, help="首/中/尾取几个时刻快照")
    a = ap.parse_args()

    obj = torch.load(a.pt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "logical_count" in obj:
        lc = obj["logical_count"]
    else:
        raise SystemExit("no 'logical_count' tensor found in .pt")

    lc = lc.detach().cpu().to(torch.float32)
    if lc.dim() != 3:
        raise SystemExit(f"expected logical_count shape [T,L,E], got {list(lc.shape)}")
    T, L, E = lc.shape

    # 1) 全局专家热度总和：对 time、layer 求和 -> [E]
    expert_totals = lc.sum(dim=0).sum(dim=0).tolist()  # [E]
    global_heat = {
        "shape": [T, L, E],
        "per_expert": summarize(expert_totals),
        "zero_heat_experts": int(sum(1 for v in expert_totals if v == 0)),
        "concentration": topk_concentration(expert_totals),
    }

    # 2) 逐层专家热度：对 time 求和 -> [L, E]
    layer_totals = lc.sum(dim=0)  # [L, E]
    layer_cv = []
    layer_peak_mean = []
    layer_total = []
    for i in range(L):
        row = layer_totals[i].tolist()
        s = summarize(row)
        layer_total.append(s["total"])
        layer_cv.append(s["cv"])
        layer_peak_mean.append(s["peak_to_mean"])
    layer_cv_clean = [v for v in layer_cv if v is not None]
    layer_peak_clean = [v for v in layer_peak_mean if v is not None]
    per_layer = {
        "layer_count": L,
        "layer_total_tokens": layer_total,
        "layer_cv": {
            "min": round(min(layer_cv_clean), 4) if layer_cv_clean else None,
            "mean": round(sum(layer_cv_clean) / len(layer_cv_clean), 4)
            if layer_cv_clean
            else None,
            "max": round(max(layer_cv_clean), 4) if layer_cv_clean else None,
        },
        "layer_peak_to_mean": {
            "min": round(min(layer_peak_clean), 4) if layer_peak_clean else None,
            "mean": round(sum(layer_peak_clean) / len(layer_peak_clean), 4)
            if layer_peak_clean
            else None,
            "max": round(max(layer_peak_clean), 4) if layer_peak_clean else None,
        },
    }

    # 3) 某一时刻专家热度快照（首/中/尾）
    snap_idx = sorted(set([0, T // 2, T - 1]))
    snapshots = []
    for t in snap_idx:
        row = lc[t].sum(dim=0).tolist()  # [E]
        snapshots.append(
            {
                "t": t,
                "per_expert": summarize(row),
                "zero_heat_experts": int(sum(1 for v in row if v == 0)),
                "concentration": topk_concentration(row),
            }
        )

    # 4) rank 负载（identity 映射）
    ranks = rank_load_from_experts(expert_totals, a.ep_size)
    rank_summary = summarize(ranks)
    rank_load = {
        "ep_size": a.ep_size,
        "mapping": "identity (expert e -> rank e // (E/EP))",
        "ranks": [round(v, 2) for v in ranks],
        "summary": rank_summary,
        "max_over_min": round(rank_summary["max"] / rank_summary["min"], 4)
        if rank_summary["min"] > 0
        else None,
    }

    out = {
        "source": a.pt,
        "global_heat": global_heat,
        "per_layer": per_layer,
        "snapshots": snapshots,
        "rank_load": rank_load,
        # 不均衡三件套（rank 负载口径）
        "imbalance": {
            "cv": rank_summary["cv"],
            "peak_to_mean": rank_summary["peak_to_mean"],
            "max_to_min": rank_load["max_over_min"],
        },
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[expert_heat] wrote -> {a.out}")


if __name__ == "__main__":
    main()
