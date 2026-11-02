#!/usr/bin/env python3
"""S3 calibration: measured local-chain cost for up to 17 active expert groups."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_local_cost import ExpertBank, benchmark_call, load_experts, make_case, sync  # noqa: E402

SEED = 20261103


def interleaved_mapping() -> list[list[int]]:
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
    return result


def rank_vectors(counts: dict[int, int]) -> list[list[int]]:
    mapping = interleaved_mapping()[49]
    result = []
    for rank in range(16):
        slots = mapping[rank * 17:(rank + 1) * 17]
        vector = []
        for slot in slots:
            expert = int(slot)
            positions = [i for i, value in enumerate(mapping) if int(value) == expert]
            share = counts[expert] // len(positions)
            if slot == positions[0]:
                share += counts[expert] % len(positions)
            vector.append(share)
        result.append(vector)
    return result


def distribute(total: int, weights: list[float]) -> list[int]:
    exact = [total * value / sum(weights) for value in weights]
    result = [math.floor(value) for value in exact]
    remainder = total - sum(result)
    for index in sorted(range(len(result)), key=lambda i: exact[i] - result[i], reverse=True)[:remainder]:
        result[index] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/glm-q")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError(f"expected one NPU, got {torch.npu.device_count()}")
    sync()
    free_before, total_hbm = torch.npu.mem_get_info()
    torch.npu.reset_peak_memory_stats()

    ids = [43, 40, 192, 67, 82, 172, 112, 79, 36, 41, 106, 69, 132, 63, 191, 109]
    raw = load_experts(Path(args.model), ids)
    bank = ExpertBank("calibration17", ids + [43], raw)

    audit = json.loads(Path("/exp/results/S0_MAPPING_AUDIT.json").read_text())
    cal_csv = list(csv.DictReader(open("/exp/results/s0_layer49_expert_loads.csv")))
    cal_counts = {int(row["expert"]): int(row["calibration_count"]) for row in cal_csv}
    hold_counts = {int(row["expert"]): int(row["holdout_count"]) for row in cal_csv}
    cal_vectors = rank_vectors(cal_counts)
    hold_vectors = rank_vectors(hold_counts)

    cases = []
    for total in (1024, 4096):
        for active in (1, 2, 4, 8, 12, 16, 17):
            counts = distribute(total, [1.0] * active) + [0] * (17 - active)
            cases.append({"set": "calibration", "kind": "uniform_active", "total": total,
                          "active": active, "counts": counts, "label": f"uniform_{active}_m{total}"})
        skew = distribute(total, [8.0] + [1.0] * 16)
        cases.append({"set": "calibration", "kind": "dominant_plus_16", "total": total,
                      "active": 17, "counts": skew, "label": f"dominant17_m{total}"})
        geometric = distribute(total, [1.0 / (index + 1) ** 0.7 for index in range(17)])
        cases.append({"set": "calibration", "kind": "geometric_17", "total": total,
                      "active": 17, "counts": geometric, "label": f"geometric17_m{total}"})
    for split, vectors in (("calibration", cal_vectors), ("holdout", hold_vectors)):
        for rank, vector in enumerate(vectors):
            cases.append({"set": split, "kind": "real_rank_vector", "total": sum(vector),
                          "active": sum(value > 0 for value in vector), "counts": vector,
                          "label": f"{split}_rank{rank}"})

    measured = []
    for item in cases:
        case = make_case(bank, item["counts"], item["label"], item["kind"])
        timing = benchmark_call(lambda c=case: bank.chain(c))
        measured.append({**item, "timing": timing})

    sync()
    free_after, _ = torch.npu.mem_get_info()
    result = {
        "stage": "S3",
        "kind": "local_cost_model_calibration",
        "status": "COMPLETE",
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": int(torch.npu.device_count()),
            "device_name": torch.npu.get_device_name(0),
        },
        "bank": {
            "unique_expert_ids": ids,
            "group_count": 17,
            "duplicate_group_expert": 43,
            "per_expert_bytes": int(bank.per_expert_bytes),
            "bank_bytes": int(bank.per_expert_bytes * 17),
        },
        "calibration_rank_vectors": cal_vectors,
        "holdout_rank_vectors": hold_vectors,
        "s0_rank_load_cal": audit["corrected_mapping"]["layer49_rank_load_cal"],
        "s0_rank_load_hold": audit["corrected_mapping"]["layer49_rank_load_hold"],
        "cases": measured,
        "memory": {
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "total_hbm_bytes": int(total_hbm),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        },
        "limitations": [
            "Synthetic active-group shapes use 17 resident representative experts; they are not a full-256-expert rank.",
            "Real rank row vectors are replayed with representative weights, not every rank's exact expert weights.",
            "No HCCL, dispatch, combine, router, attention or service metrics are included.",
        ],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "cases": len(measured),
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
        "bank_bytes": result["bank"]["bank_bytes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
