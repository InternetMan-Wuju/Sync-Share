#!/usr/bin/env python3
"""Combine source/destination paired records and calculate differences/noise."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, int(q * (len(values) - 1)))
    return values[index]


def dist_summary(values):
    values = [float(v) for v in values]
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    return {
        "n": len(values),
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "mad": mad,
        "min": min(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "raw": values,
    }


def main():
    source = json.loads((ART / "two_rank_paired_rank0.json").read_text())
    dest = json.loads((ART / "two_rank_paired_rank1.json").read_text())
    source_by_id = {s["sample_id"]: s for s in source["samples"]}
    dest_by_id = {s["sample_id"]: s for s in dest["samples"]}
    if set(source_by_id) != set(dest_by_id):
        raise SystemExit("source/destination sample IDs differ")

    by_mode = defaultdict(list)
    by_group = defaultdict(dict)
    records = []
    for sample_id in sorted(source_by_id):
        src = source_by_id[sample_id]
        dst = dest_by_id[sample_id]
        if src["mode"] != dst["mode"] or src["group"] != dst["group"]:
            raise SystemExit(f"metadata mismatch for {sample_id}")
        by_mode[(src["load_case"], src["mode"])].append(src["unified_ms"])
        by_group[(src["load_case"], src["group"])][src["mode"]] = src["unified_ms"]
        records.append(
            {
                "sample_id": sample_id,
                "load_case": src["load_case"],
                "group": src["group"],
                "mode": src["mode"],
                "order": src["order"],
                "tokens": src["tokens"],
                "quota_main": src["quota_main"],
                "quota_external": src["quota_external"],
                "input_fingerprint": src["input_fingerprint"],
                "unified_ms": src["unified_ms"],
                "correctness_max_abs": src["correctness_max_abs"],
                "task_count_conserved": src["task_count_conserved"],
                "source_segments_ms": src["segments_ms"],
                "dest_segments_ms": dst["segments_ms"],
                "source_memory": src["memory"],
                "dest_memory": dst["memory"],
                "source_preinstall": src["preinstall"],
                "dest_preinstall": dst["preinstall"],
                "state_payload_bytes": src["state_payload_bytes"],
                "activation_payload_bytes": src["activation_payload_bytes"],
                "result_payload_bytes": src["result_payload_bytes"],
            }
        )

    mode_summary = {
        f"{load_case}/{mode}": dist_summary(values)
        for (load_case, mode), values in by_mode.items()
    }
    deltas = defaultdict(list)
    for (load_case, group), times in by_group.items():
        if set(times) != {"original", "cold", "existing"}:
            raise SystemExit(f"incomplete paired group {load_case}/{group}: {times}")
        deltas[f"{load_case}/original_minus_cold"].append(
            times["original"] - times["cold"]
        )
        deltas[f"{load_case}/original_minus_existing"].append(
            times["original"] - times["existing"]
        )
        deltas[f"{load_case}/cold_minus_existing"].append(
            times["cold"] - times["existing"]
        )
    delta_summary = {name: dist_summary(values) for name, values in deltas.items()}

    segment_summary = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = f"{record['load_case']}/{record['mode']}"
        for name, value in record["source_segments_ms"].items():
            if value is not None:
                segment_summary[key][f"source/{name}"].append(value)
        for name, value in record["dest_segments_ms"].items():
            if value is not None:
                segment_summary[key][f"dest/{name}"].append(value)
    segment_summary = {
        key: {name: dist_summary(values) for name, values in stages.items()}
        for key, stages in segment_summary.items()
    }

    memory = {}
    for rank_name, first in (("source", source), ("destination", dest)):
        samples = first["samples"]
        memory[rank_name] = {
            "initial": first["memory_initial"],
            "final": first["memory_final"],
            "sample_max": {
                field: max(int(s["memory"].get(field, 0)) for s in samples)
                for field in (
                    "allocated",
                    "reserved",
                    "max_allocated",
                    "max_reserved",
                )
            },
            "persistent_slot_committed_bytes": (
                first.get("final_pool_stats", {}).get("committed_bytes")
            ),
        }

    summary = {
        "scope": "two-rank expert-child replay; no full layer or TTFT claim",
        "groups_per_load_case": source["groups"],
        "load_cases": source["load_cases"],
        "samples": len(records),
        "correctness": {
            "all_max_abs_zero": all(
                r["correctness_max_abs"] == 0.0 for r in records
            ),
            "all_task_counts_conserved": all(
                r["task_count_conserved"] for r in records
            ),
            "max_abs": max(r["correctness_max_abs"] for r in records),
        },
        "mode_summary_ms": mode_summary,
        "paired_delta_summary_ms": delta_summary,
        "segment_summary_ms": segment_summary,
        "memory_bytes": memory,
        "source_preparation_ms": source.get("source_preparation_ms"),
        "actual_state": source.get("actual_state"),
        "records": records,
        "note": (
            "Unified time starts before the go notification and ends after return, "
            "refill, release, synchronization, and acknowledgement. Segment values "
            "are separately synchronized and are not required to sum to unified."
        ),
    }
    (ART / "two_rank_paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        "scope: two-rank expert-child replay; no full layer/TTFT claim",
        f"samples={len(records)} all_max_abs_zero={summary['correctness']['all_max_abs_zero']}",
        "mode\tmedian_ms\tp95_ms\tmin_ms\tmax_ms",
    ]
    for name, item in sorted(mode_summary.items()):
        lines.append(
            f"{name}\t{item['median']:.6f}\t{item['p95']:.6f}\t"
            f"{item['min']:.6f}\t{item['max']:.6f}"
        )
    lines.append("paired_delta\tmedian_ms\tp95_ms\tmin_ms\tmax_ms")
    for name, item in sorted(delta_summary.items()):
        lines.append(
            f"{name}\t{item['median']:.6f}\t{item['p95']:.6f}\t"
            f"{item['min']:.6f}\t{item['max']:.6f}"
        )
    (ART / "two_rank_paired_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not summary["correctness"]["all_max_abs_zero"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
