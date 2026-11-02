#!/usr/bin/env python3
"""S3 replay: measure affected-rank vectors with their exact resident experts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_local_cost import ExpertBank, benchmark_call, load_experts, make_case, sync  # noqa: E402
import s3_plan  # noqa: E402

SEED = 20261103
CORRECTED_AFFECTED = (0, 1, 2)
OLD_AFFECTED = (2, 15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/glm-q")
    parser.add_argument("--plans", default="/exp/results/S3_PLANS.json")
    parser.add_argument("--calibration", default="/exp/results/S3_COST_CALIBRATION.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plans = json.loads(Path(args.plans).read_text())
    calibration = json.loads(Path(args.calibration).read_text())

    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError(f"expected one NPU, got {torch.npu.device_count()}")
    sync()
    free_before, total_hbm = torch.npu.mem_get_info()
    torch.npu.reset_peak_memory_stats()

    corrected_slots = s3_plan.rank_slots(s3_plan.make_mapping("corrected"))
    old_slots = s3_plan.rank_slots(s3_plan.make_mapping("old"))
    rank_experts = {
        ("corrected", 0): corrected_slots[0],
        ("corrected", 1): corrected_slots[1],
        ("corrected", 2): corrected_slots[2],
        ("old", 2): old_slots[2],
        ("old", 15): old_slots[15],
    }
    all_ids = sorted({expert for slots in rank_experts.values() for expert in slots})
    raw = load_experts(Path(args.model), all_ids)
    banks = {
        key: ExpertBank(f"{key[0]}_rank{key[1]}", slots, raw)
        for key, slots in rank_experts.items()
    }

    # Generic-bank measurements for unaffected ranks, keyed by route split.
    baseline_measurements = {}
    for case in calibration["cases"]:
        if case["kind"] != "real_rank_vector":
            continue
        split = case["set"]
        rank = int(case["label"].rsplit("rank", 1)[1])
        baseline_measurements.setdefault(split, {})[rank] = case["timing"]["device_ms_median"]

    results = {"stage": "S3", "kind": "exact_affected_rank_replay", "splits": {}}
    for split in ("calibration", "holdout"):
        split_result = {"policies": {}}
        for policy_name, policy in plans["splits"][split]["policies"].items():
            mapping_kind = policy["mapping"]
            affected = OLD_AFFECTED if mapping_kind == "old" else CORRECTED_AFFECTED
            exact = {}
            for rank in affected:
                counts = policy["rank_vectors"][rank]
                slot_key = (mapping_kind, rank)
                bank = banks[slot_key]
                case = make_case(bank, counts, f"{split}_{policy_name}_rank{rank}", policy_name)
                # Keep the input distribution fixed across policies for this rank.
                case["x"] = torch.randn(
                    (case["rows"], 6144),
                    generator=torch.Generator().manual_seed(SEED),
                    dtype=torch.float32,
                ).to(torch.bfloat16).npu()
                timing = benchmark_call(lambda c=case: bank.chain(c))
                exact[str(rank)] = {
                    "rows": case["rows"],
                    "active_experts": sum(value > 0 for value in counts),
                    "counts": counts,
                    "timing": timing,
                }
            unaffected = {
                str(rank): timing
                for rank, timing in baseline_measurements[split].items()
                if rank not in affected
            }
            all_measured = {**unaffected, **{rank: item["timing"]["device_ms_median"] for rank, item in exact.items()}}
            exact_max = max(item["timing"]["device_ms_median"] for item in exact.values())
            split_result["policies"][policy_name] = {
                "mapping": mapping_kind,
                "replica_quotas": policy["replica_quotas"],
                "affected_ranks_exact": exact,
                "unaffected_rank_representative_ms": unaffected,
                "all_rank_measured_or_representative_ms": all_measured,
                "affected_exact_max_ms": exact_max,
                "affected_exact_sum_ms": sum(item["timing"]["device_ms_median"] for item in exact.values()),
                "all_rank_max_ms": max(all_measured.values()),
                "all_rank_mean_ms": sum(all_measured.values()) / len(all_measured),
            }
        baseline = split_result["policies"]["owner_only"]
        for policy in split_result["policies"].values():
            policy["measured_max_reduction_vs_owner"] = 1.0 - policy["all_rank_max_ms"] / baseline["all_rank_max_ms"]
            policy["affected_max_reduction_vs_owner"] = 1.0 - policy["affected_exact_max_ms"] / baseline["affected_exact_max_ms"]
        results["splits"][split] = split_result

    sync()
    free_after, _ = torch.npu.mem_get_info()
    results.update({
        "status": "COMPLETE",
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": int(torch.npu.device_count()),
            "device_name": torch.npu.get_device_name(0),
        },
        "exact_rank_experts": {
            f"{kind}_rank{rank}": slots for (kind, rank), slots in rank_experts.items()
        },
        "unique_expert_count_loaded": len(all_ids),
        "working_set_bytes": int(sum(bank.per_expert_bytes * 17 for bank in banks.values())),
        "memory": {
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "total_hbm_bytes": int(total_hbm),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        },
        "scope_limits": [
            "Affected ranks 0/1/2 (and rank15 for the old layout) use their exact layer-49 experts.",
            "Unaffected ranks use prior representative 17-group replays, not their exact expert weights.",
            "Values are sequentially measured local-chain times, not a simultaneous EP critical path.",
            "No HCCL or service-level cost is included.",
        ],
    })
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    summary = {}
    for split, split_result in results["splits"].items():
        summary[split] = {
            name: {
                "quotas": value["replica_quotas"],
                "max_ms": value["all_rank_max_ms"],
                "reduction": value["measured_max_reduction_vs_owner"],
                "affected": {rank: round(item["timing"]["device_ms_median"], 4) for rank, item in value["affected_ranks_exact"].items()},
            }
            for name, value in split_result["policies"].items()
        }
    print(json.dumps({
        "out": str(out),
        "unique_experts": len(all_ids),
        "working_set_bytes": results["working_set_bytes"],
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
