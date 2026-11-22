#!/usr/bin/env python3
"""Join source/target samples, write paired CSV, and apply the fixed gate rule."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def dist_summary(values):
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "raw": values,
    }


def read_pair(mode, tag):
    source = json.loads(
        (ART / f"prefill_pair_{mode}_{tag}_rank0.json").read_text()
    )
    target = json.loads(
        (ART / f"prefill_pair_{mode}_{tag}_rank1.json").read_text()
    )
    source_by_id = {sample["sample_id"]: sample for sample in source["samples"]}
    target_by_id = {sample["sample_id"]: sample for sample in target["samples"]}
    if set(source_by_id) != set(target_by_id):
        raise SystemExit("source/target sample IDs differ")
    return source, target, source_by_id, target_by_id


def analyze_diagnostic():
    source, target, src_by, dst_by = read_pair("diagnostic", "diagnostic")
    stages = defaultdict(lambda: defaultdict(list))
    joined = []
    for sample_id in sorted(src_by):
        src = src_by[sample_id]
        dst = dst_by[sample_id]
        for name, value in src.get("segments_ms", {}).items():
            if value is not None:
                stages[(src["n"], src["mode"])][f"source/{name}"].append(value)
        for name, value in dst.get("segments_ms", {}).items():
            if value is not None:
                stages[(src["n"], src["mode"])][f"target/{name}"].append(value)
        joined.append({"source": src, "target": dst})
    summary = {
        "scope": "diagnostic timeline with stage-level synchronization",
        "samples": len(joined),
        "stage_medians_ms": {
            f"{n}/{mode}": {
                name: statistics.median(values)
                for name, values in sorted(stage_values.items())
            }
            for (n, mode), stage_values in sorted(stages.items())
        },
    }
    (ART / "DIAGNOSTIC_TIMELINE.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def analyze_formal(tag):
    source, target, src_by, dst_by = read_pair("formal", tag)
    rows = []
    by_group = defaultdict(dict)
    for sample_id in sorted(src_by):
        src = src_by[sample_id]
        dst = dst_by[sample_id]
        rows.append(
            {
                "tag": tag,
                "n": src["n"],
                "q": src["q"],
                "group": src["group"],
                "seed_id": src["seed_id"],
                "input_seed": src["input_seed"],
                "input_fingerprint": src["input_fingerprint"],
                "order": ">".join(src["order"]),
                "mode": src["mode"],
                "unified_ms": src["unified_ms"],
                "correctness_max_abs": src["correctness_max_abs"],
                "relative_error": src["relative_error"],
                "nonfinite_count": src["nonfinite_count"],
                "task_count_conserved": src["task_count_conserved"],
                "source_preinstall": json.dumps(src.get("preinstall")),
                "target_preinstall": json.dumps(dst.get("preinstall")),
                "slot_generation": dst.get("slot_generation"),
                "slot_owner": (dst.get("slot_state") or {}).get("owner"),
                "slot_local": (dst.get("slot_state") or {}).get("local_slot"),
                "slot_layer": (dst.get("slot_state") or {}).get("layer_id"),
                "slot_expert": (dst.get("slot_state") or {}).get("expert_id"),
                "slot_request_id": (dst.get("slot_state") or {}).get("request_id"),
                "state_payload_bytes": src["state_payload_bytes"],
                "activation_payload_bytes": src["activation_payload_bytes"],
                "result_payload_bytes": src["result_payload_bytes"],
                "source_allocated": src["memory"].get("allocated"),
                "source_reserved": src["memory"].get("reserved"),
                "source_peak_allocated": src["memory"].get("max_allocated"),
                "source_peak_reserved": src["memory"].get("max_reserved"),
                "dest_allocated": dst["memory"].get("allocated"),
                "dest_reserved": dst["memory"].get("reserved"),
                "dest_peak_allocated": dst["memory"].get("max_allocated"),
                "dest_peak_reserved": dst["memory"].get("max_reserved"),
                "dest_pool_free": dst.get("pool_stats_after", {})
                .get("local", {})
                .get("1", {})
                .get("free"),
            }
        )
        by_group[(src["n"], src["q"], src["group"])][src["mode"]] = src["unified_ms"]

    csv_path = ART / ("PAIRED_TIMINGS.csv" if tag == "standard" else f"PAIRED_TIMINGS_{tag}.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    point_modes = defaultdict(list)
    for row in rows:
        point_modes[(row["n"], row["q"], row["mode"])].append(row["unified_ms"])
    point_summary = {}
    for (n, q, mode), values in sorted(point_modes.items()):
        orig_values = [
            by_group[(n, q, group)]["original"]
            for group in range(12)
        ]
        replica_values = [
            by_group[(n, q, group)][mode]
            for group in range(12)
        ]
        deltas = [o - r for o, r in zip(orig_values, replica_values)]
        point_summary[f"{n}/{q}/{mode}"] = {
            "n": n,
            "q": q,
            "mode": mode,
            "original": dist_summary(orig_values),
            "replica": dist_summary(replica_values),
            "paired_delta": dist_summary(deltas),
            "slowed_groups": sum(delta <= 0 for delta in deltas),
            "positive_groups": sum(delta > 0 for delta in deltas),
            "all_12_positive": all(delta > 0 for delta in deltas),
        }

    candidates = {}
    for mode in ("cold", "existing"):
        eligible = [
            item
            for item in point_summary.values()
            if item["mode"] == mode and item["all_12_positive"]
        ]
        if eligible:
            min_n = min(item["n"] for item in eligible)
            same_n = [item for item in eligible if item["n"] == min_n]
            best = max(
                same_n,
                key=lambda item: (item["paired_delta"]["median"], -item["q"]),
            )
            candidates[mode] = {
                "n": best["n"],
                "q": best["q"],
                "median_paired_delta_ms": best["paired_delta"]["median"],
            }

    checks = {
        "expected_samples": sum(len(source["points"]) * 12 * 3 for _ in [0]),
        "actual_samples": len(rows),
        "all_max_abs_zero": all(row["correctness_max_abs"] == 0.0 for row in rows),
        "all_finite": all(row["nonfinite_count"] == 0 for row in rows),
        "all_task_counts_conserved": all(row["task_count_conserved"] for row in rows),
        "all_slots_free_after": all(
            row["dest_pool_free"] == 1 for row in rows
        ),
    }
    summary = {
        "tag": tag,
        "scope": "two-rank manual Prefill expert-subchain replay; not service TTFT",
        "points": source["points"],
        "groups_per_point": source["groups_per_point"],
        "input_seeds": source["input_seeds"],
        "checks": checks,
        "point_summary": point_summary,
        "confirmation_candidates": candidates,
        "confirmation_required": bool(candidates),
        "note": (
            "Positive paired deltas are in the same group. Medians are not "
            "subtracted across columns. Twelve groups do not define production p95."
        ),
    }
    output = ART / ("pair_analysis_standard.json" if tag == "standard" else f"pair_analysis_{tag}.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(
        (
            checks["all_max_abs_zero"],
            checks["all_finite"],
            checks["all_task_counts_conserved"],
            checks["all_slots_free_after"],
        )
    ):
        raise SystemExit(1)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="standard")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.diagnostic:
        analyze_diagnostic()
    else:
        analyze_formal(args.tag)


if __name__ == "__main__":
    main()
