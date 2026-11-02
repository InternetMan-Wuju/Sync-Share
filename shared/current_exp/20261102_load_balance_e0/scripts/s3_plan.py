#!/usr/bin/env python3
"""S3 CPU planner: legal copy allocations and measured-cost quota selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

EPS = 1e-12


def load_counts(path: Path) -> dict[str, dict[int, int]]:
    rows = list(csv.DictReader(open(path)))
    return {
        "calibration": {int(row["expert"]): int(row["calibration_count"]) for row in rows},
        "holdout": {int(row["expert"]): int(row["holdout_count"]) for row in rows},
    }


def make_mapping(kind: str) -> list[list[int]]:
    if kind == "corrected":
        result = []
        for layer in range(78):
            slots = []
            for rank in range(16):
                base = rank * 16
                slots.extend(range(base, base + 16))
                extra = base
                if layer == 49 and rank == 0:
                    extra = 43
                elif layer == 49 and rank == 1:
                    extra = 40
                slots.append(extra)
            result.append(slots)
        return result[49]
    if kind == "old":
        slots = list(range(256)) + [43, 40] + [rank * 16 for rank in range(2, 16)]
        return slots
    raise ValueError(kind)


def rank_slots(mapping: list[int]) -> list[list[int]]:
    if len(mapping) == 16 * 17:
        return [mapping[rank * 17:(rank + 1) * 17] for rank in range(16)]
    raise ValueError(len(mapping))


def target_slots(mapping: list[int], kind: str) -> dict:
    slots = rank_slots(mapping)
    def find(rank: int, expert: int) -> int:
        positions = [index for index, value in enumerate(slots[rank]) if value == expert]
        if not positions:
            raise RuntimeError((rank, expert))
        return positions[-1] if len(positions) > 1 else positions[0]
    if kind == "corrected":
        return {
            "43": {"owner": (2, find(2, 43)), "replica": (0, find(0, 43))},
            "40": {"owner": (2, find(2, 40)), "replica": (1, find(1, 40))},
        }
    if kind == "old":
        return {
            "43": {"owner": (2, find(2, 43)), "replica": (15, find(15, 43))},
            "40": {"owner": (2, find(2, 40)), "replica": (15, find(15, 40))},
        }
    raise ValueError(kind)


def slots_for_expert(slots: list[list[int]], expert: int) -> list[tuple[int, int]]:
    return [(rank, index) for rank, vector in enumerate(slots) for index, value in enumerate(vector) if value == expert]


def build_vectors(mapping: list[int], counts: dict[int, int], replica_quotas: dict[int, int] | None, kind: str) -> dict:
    slots = rank_slots(mapping)
    targets = target_slots(mapping, kind)
    quotas = replica_quotas or {}
    vectors = [[0] * len(vector) for vector in slots]
    special = set()
    for expert_text, placement in targets.items():
        expert = int(expert_text)
        owner_rank, owner_index = placement["owner"]
        replica_rank, replica_index = placement["replica"]
        removed = min(max(int(quotas.get(expert, 0)), 0), counts[expert])
        vectors[owner_rank][owner_index] = counts[expert] - removed
        vectors[replica_rank][replica_index] = removed
        special.add((owner_rank, owner_index))
        special.add((replica_rank, replica_index))
    for expert in range(256):
        duplicate_slots = slots_for_expert(slots, expert)
        if any(position in special for position in duplicate_slots):
            continue
        n = len(duplicate_slots)
        for order, (rank, index) in enumerate(duplicate_slots):
            vectors[rank][index] = counts[expert] // n + (counts[expert] % n if order == 0 else 0)
    row_loads = [sum(vector) for vector in vectors]
    by_expert = {expert: 0 for expert in range(256)}
    for rank, vector in enumerate(vectors):
        for index, expert in enumerate(slots[rank]):
            by_expert[expert] += vector[index]
    return {
        "vectors": vectors,
        "row_loads": row_loads,
        "by_expert": by_expert,
        "conserved": by_expert == counts and sum(row_loads) == sum(counts.values()),
        "slots": slots,
    }


def feature_vector(counts, active, largest, sqrt_sum, small):
    return np.asarray([1.0, float(sum(counts)), float(active), float(largest), float(sqrt_sum), float(small)])


def features(counts: list[int]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    nonzero = values[values > 0]
    return feature_vector(
        values,
        len(nonzero),
        nonzero.max() if len(nonzero) else 0,
        np.sqrt(values).sum(),
        int((nonzero < 128).sum()),
    )


def fit_cost_model(s2: dict, s3: dict) -> dict:
    train = []
    holdout = []
    for case in s2["shape_curve"]:
        if "timing" in case:
            train.append((case["counts"], case["timing"]["device_ms_median"], "s2_shape"))
    for case in s2["distribution_cases"]:
        train.append((case["counts"], case["timing"]["device_ms_median"], "s2_distribution"))
    train.append((s2["real_observed"]["counts"], s2["real_observed"]["timing"]["device_ms_median"], "s2_real"))
    for case in s3["cases"]:
        target = holdout if case["set"] == "holdout" else train
        target.append((case["counts"], case["timing"]["device_ms_median"], f"s3_{case['set']}_{case['kind']}"))
    X = np.stack([features(item[0]) for item in train])
    y = np.asarray([item[1] for item in train])
    Xh = np.stack([features(item[0]) for item in holdout])
    yh = np.asarray([item[1] for item in holdout])
    ridge = 0.1
    regularizer = np.eye(X.shape[1]) * ridge
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + regularizer, X.T @ y)
    train_pred = X @ beta
    hold_pred = Xh @ beta
    train_nrmse = float(np.linalg.norm(train_pred - y) / np.linalg.norm(y))
    hold_nrmse = float(np.linalg.norm(hold_pred - yh) / np.linalg.norm(yh))
    relative = np.abs(hold_pred - yh) / np.maximum(yh, EPS)
    return {
        "features": ["intercept", "total_rows", "active_experts", "largest_expert_rows", "sum_sqrt_rows", "active_experts_below_128_rows"],
        "ridge": ridge,
        "coefficients": beta.tolist(),
        "train_points": len(train),
        "holdout_points": len(holdout),
        "train_nrmse": train_nrmse,
        "holdout_nrmse": hold_nrmse,
        "holdout_mae_ms": float(np.mean(np.abs(hold_pred - yh))),
        "holdout_max_abs_relative_error": float(relative.max()),
        "holdout_p95_abs_relative_error": float(np.quantile(relative, 0.95)),
        "validation_threshold_nrmse": 0.05,
        "validation_pass": hold_nrmse <= 0.05,
        "holdout_examples": [
            {"label": holdout[i][2], "measured_ms": float(yh[i]), "predicted_ms": float(hold_pred[i]), "relative_error": float(relative[i])}
            for i in range(len(holdout))
        ],
    }


def predict(model: dict, counts: list[int]) -> float:
    return float(features(counts) @ np.asarray(model["coefficients"]))


def row_quota_search(vectors: dict, counts: dict[int, int]) -> dict:
    base = vectors["row_loads"]
    n43, n40 = counts[43], counts[40]
    b0, b1, b2 = base[0], base[1], base[2]
    lo = max(0, min(b0, b1, b2))
    hi = max(b0 + n43, b1 + n40, b2)
    while lo < hi:
        mid = (lo + hi) // 2
        max43 = min(n43, mid - b0)
        max40 = min(n40, mid - b1)
        required = b2 + n43 + n40 - mid
        feasible = max43 >= 0 and max40 >= 0 and max43 + max40 >= required
        if feasible:
            hi = mid
        else:
            lo = mid + 1
    target = lo
    q43 = min(n43, target - b0)
    q40 = min(n40, target - b1)
    required = b2 + n43 + n40 - target
    # Keep the feasible owner allocation while avoiding needless overload.
    while q43 > 0 and q43 + q40 - 1 >= required:
        q43 -= 1
    while q40 > 0 and q43 + q40 - 1 >= required:
        q40 -= 1
    return {"replica_quotas": {43: q43, 40: q40}, "minimax_row_load": target}


def time_quota_search(model: dict, vectors: dict, counts: dict[int, int]) -> dict:
    base = vectors["vectors"]
    n43, n40 = counts[43], counts[40]
    target = target_slots(make_mapping("corrected"), "corrected")
    owner43 = target["43"]["owner"]
    owner40 = target["40"]["owner"]
    replica43 = target["43"]["replica"]
    replica40 = target["40"]["replica"]
    unaffected = [predict(model, vector) for rank, vector in enumerate(base) if rank not in (0, 1, 2)]
    floor = max(unaffected)
    best = {"objective": float("inf"), "q43": 0, "q40": 0}
    for q43 in range(n43 + 1):
        vec0 = list(base[0]); vec0[replica43[1]] = q43
        t0 = predict(model, vec0)
        if t0 >= best["objective"]:
            continue
        vec2 = list(base[2]); vec2[owner43[1]] = n43 - q43
        def t1(q40):
            vec1 = list(base[1]); vec1[replica40[1]] = q40
            return predict(model, vec1)
        def t2(q40):
            values = list(vec2); values[owner40[1]] = n40 - q40
            return predict(model, values)
        lo, hi = 0, n40
        while lo < hi:
            mid = (lo + hi) // 2
            if t1(mid) < t2(mid):
                lo = mid + 1
            else:
                hi = mid
        candidates = {0, n40, lo, max(0, lo - 1), min(n40, lo + 1)}
        for q40 in candidates:
            objective = max(floor, t0, t1(q40), t2(q40))
            if objective < best["objective"]:
                best = {"objective": objective, "q43": q43, "q40": q40}
    return {
        "replica_quotas": {43: best["q43"], 40: best["q40"]},
        "predicted_minimax_rank_time_ms": best["objective"],
        "predicted_unaffected_floor_ms": floor,
    }


def evaluate_policy(model: dict, plan: dict, counts: dict[int, int], name: str) -> dict:
    vectors = build_vectors(make_mapping(plan["mapping"]), counts, plan["replica_quotas"], plan["mapping"])
    predicted = [predict(model, vector) for vector in vectors["vectors"]]
    return {
        "policy": name,
        "mapping": plan["mapping"],
        "replica_quotas": {str(k): v for k, v in plan["replica_quotas"].items()},
        "rank_vectors": vectors["vectors"],
        "rank_row_loads": vectors["row_loads"],
        "conserved": vectors["conserved"],
        "predicted_rank_times_ms": predicted,
        "predicted_max_rank_time_ms": max(predicted),
        "predicted_mean_rank_time_ms": float(np.mean(predicted)),
        "predicted_max_over_mean": max(predicted) / float(np.mean(predicted)),
        "max_row_load": max(vectors["row_loads"]),
        "max_over_mean_rows": max(vectors["row_loads"]) / float(np.mean(vectors["row_loads"])),
        "active_weight_instances": 2 if sum(plan["replica_quotas"].values()) > 0 else 0,
    }


def add_relative(policy: dict, baseline: dict) -> None:
    policy["predicted_max_time_reduction_vs_owner"] = 1.0 - policy["predicted_max_rank_time_ms"] / baseline["predicted_max_rank_time_ms"]
    policy["max_row_load_reduction_vs_owner"] = 1.0 - policy["max_row_load"] / baseline["max_row_load"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--results-root", default="/exp/results")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.results_root).resolve()
    s2 = json.loads((root / "S2_LOCAL_COST.json").read_text())
    s3 = json.loads((root / "S3_COST_CALIBRATION.json").read_text())
    counts = load_counts(root / "s0_layer49_expert_loads.csv")
    model = fit_cost_model(s2, s3)

    summary = {"stage": "S3", "kind": "copy_budget_allocation_plan", "model": model, "splits": {}}
    for split in ("calibration", "holdout"):
        split_counts = counts[split]
        owner = build_vectors(make_mapping("corrected"), split_counts, {}, "corrected")
        corrected_zero = build_vectors(make_mapping("corrected"), split_counts, {}, "corrected")
        row_choice = row_quota_search(corrected_zero, split_counts)
        time_choice = time_quota_search(model, corrected_zero, split_counts)
        plans = {
            "owner_only": {"mapping": "corrected", "replica_quotas": {}},
            "equal_corrected": {"mapping": "corrected", "replica_quotas": {43: split_counts[43] // 2, 40: split_counts[40] // 2}},
            "row_quota_corrected": {"mapping": "corrected", "replica_quotas": row_choice["replica_quotas"]},
            "time_quota_corrected": {"mapping": "corrected", "replica_quotas": time_choice["replica_quotas"]},
            "equal_old_rank15": {"mapping": "old", "replica_quotas": {43: split_counts[43] // 2, 40: split_counts[40] // 2}},
        }
        evaluated = {}
        baseline = None
        for name, plan in plans.items():
            policy = evaluate_policy(model, plan, split_counts, name)
            if baseline is None:
                baseline = policy
            add_relative(policy, baseline)
            evaluated[name] = policy
        summary["splits"][split] = {
            "counts": {str(k): v for k, v in split_counts.items()},
            "row_quota_search": row_choice,
            "time_quota_search": time_choice,
            "policies": evaluated,
            "conservation_pass": all(policy["conserved"] for policy in evaluated.values()),
        }
    summary["status"] = "COMPLETE" if model["validation_pass"] and all(
        summary["splits"][split]["conservation_pass"] for split in ("calibration", "holdout")
    ) else "EVIDENCE_INSUFFICIENT"
    summary["scope_limits"] = [
        "The model predicts local expert-chain time only; it has no HCCL, dispatch, combine, router, attention or SLO terms.",
        "Quota choice uses a measured ridge cost model validated on held-out rank vectors.",
        "Replica transport/rearrangement cost and placement beyond the calibrated copy set are not included.",
        "The old-rank15 policy is an allocation comparison, not a re-run of the old service experiment.",
    ]
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "status": summary["status"],
        "model_holdout_nrmse": model["holdout_nrmse"],
        "model_holdout_p95_relative": model["holdout_p95_abs_relative_error"],
        "calibration": {
            name: {
                "quotas": value["replica_quotas"],
                "predicted_max_ms": value["predicted_max_rank_time_ms"],
                "time_reduction": value["predicted_max_time_reduction_vs_owner"],
                "row_reduction": value["max_row_load_reduction_vs_owner"],
            }
            for name, value in summary["splits"]["calibration"]["policies"].items()
        },
        "holdout": {
            name: {
                "quotas": value["replica_quotas"],
                "predicted_max_ms": value["predicted_max_rank_time_ms"],
                "time_reduction": value["predicted_max_time_reduction_vs_owner"],
                "row_reduction": value["max_row_load_reduction_vs_owner"],
            }
            for name, value in summary["splits"]["holdout"]["policies"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
