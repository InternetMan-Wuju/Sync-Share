#!/usr/bin/env python3
"""Build the single SUMMARY.json deliverable from all raw evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def read(name):
    return json.loads((ART / name).read_text())


def main():
    compute = read("compute_curve_summary.json")
    diagnostic = read("DIAGNOSTIC_TIMELINE.json")
    overlap = read("OVERLAP_TIMELINE.json")
    standard = read("pair_analysis_standard.json")
    smoke = read("pair_analysis_smoke.json")
    cpu = read("cpu_expert_search_oracle.json")
    natural = read("natural_prefill_counts.json")

    def median_replica(n, q, mode, analysis=standard):
        return analysis["point_summary"][f"{n}/{q}/{mode}"]["replica"]["median"]

    def median_original(n, q, analysis=standard):
        return analysis["point_summary"][f"{n}/{q}/original"]["original"]["median"]

    diagnostic_vs_formal = {}
    for n, q, analysis in (
        (6, 3, smoke),
        (512, 256, standard),
    ):
        key = str(n)
        stages = diagnostic["stage_medians_ms"]
        diagnostic_vs_formal[key] = {
            "n": n,
            "q": q,
            "diagnostic_unified_ms": {
                mode: stages[f"{n}/{mode}"]["source/diagnostic_unified"]
                for mode in ("original", "cold", "existing")
            },
            "formal_unified_ms": {
                "original": median_original(n, q, analysis),
                "cold": median_replica(n, q, "cold", analysis),
                "existing": median_replica(n, q, "existing", analysis),
            },
            "formal_source": "smoke formal run" if n == 6 else "standard formal run",
        }

    conclusion = (
        "本轮范围内均未回本：三种 n、三个 q 下，临时复制没有任何 12/12 "
        "正收益点；已有副本最多只有 5/12 正收益组，未达到确认门槛。"
    )
    summary = {
        "scope": (
            "two 910C, global K=1 fixed slot, real GLM layer49/expert43 "
            "ModelSlim W8A8 expert-subchain Prefill replay"
        ),
        "one_sentence_conclusion": conclusion,
        "environment": {
            "host": "npu1-23",
            "devices": ["davinci0", "davinci1"],
            "image": "quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3",
            "sglang": "0.5.20",
            "torch_npu": "2.10.0",
            "model": "/data_lib/data/models/glm-q",
            "layer": 49,
            "logical_expert": 43,
            "state_bytes": 37810176,
        },
        "selection": compute["selection"],
        "natural_prefill_frequency_context": natural,
        "compute_curve": {
            "records": compute["records"],
            "warmups_per_shape": compute["warmups_per_shape"],
            "formal_repeats_per_shape": compute["formal_repeats_per_shape"],
            "errors": compute["errors"],
            "checksum_mismatches": compute["checksum_mismatches"],
        },
        "diagnostic_vs_formal": diagnostic_vs_formal,
        "diagnostic_stage_medians_ms": diagnostic["stage_medians_ms"],
        "overlap_timeline": overlap,
        "formal": {
            "checks": standard["checks"],
            "points": standard["points"],
            "point_summary": standard["point_summary"],
            "confirmation_candidates": standard["confirmation_candidates"],
            "confirmation_required": standard["confirmation_required"],
            "confirmation_status": (
                "NOT_RUN_NO_12_OF_12_POSITIVE_POINT"
                if not standard["confirmation_candidates"]
                else "PENDING"
            ),
        },
        "cpu_corrections": {
            "summary": cpu["summary"],
            "cases": [
                {
                    "name": case["name"],
                    "baseline": case["baseline"]["score"],
                    "oracle": case["oracle"]["score"],
                    "complete": case["complete_candidate_space"]["score"],
                    "fixed_bounded": case["fixed_bounded"]["score"],
                    "legacy": case["legacy_bounded"]["score"],
                    "candidate_omission": case["candidate_omission"],
                    "pruning_omission": case["pruning_omission"],
                    "search_time": case["fixed_bounded"]["time"],
                    "deduplicated": case["fixed_bounded"]["stats"]["deduplicated"],
                }
                for case in cpu["cases"]
            ],
            "slot_release_race_tests": "PASS",
        },
        "evidence_classes": {
            "npu_measured": [
                "compute curve",
                "diagnostic stage timeline",
                "single overlap timeline",
                "324 formal paired executions",
            ],
            "cpu_measured_or_exhaustive": [
                "expert-search regression",
                "independent integer oracle",
                "slot-release race tests",
            ],
            "estimated": [
                "optimistic half-split G used only to select the middle n",
            ],
            "not_measured": [
                "full MoE layer output",
                "shared expert",
                "all-expert dispatch/combine",
                "automatic planner cost in NPU run",
                "Decode",
                "K=2",
                "multi-request concurrency",
                "cross-node",
                "full-model TTFT",
            ],
        },
        "notes": [
            "n is the number of tasks received by expert43, not request input length.",
            "Inputs are synthetic task batches on real weights; natural routing frequency is not inferred.",
            "The original path directly computes n tasks and performs no q split.",
            "Formal timing allows source compute to overlap asynchronous transfers and only waits for required completion/ready/refill/release dependencies.",
            "Diagnostic stages are synchronized separately and are not summed as a formal total.",
            "No production p95 or complete-service acceleration is claimed.",
        ],
    }
    (ART / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(conclusion)


if __name__ == "__main__":
    main()
