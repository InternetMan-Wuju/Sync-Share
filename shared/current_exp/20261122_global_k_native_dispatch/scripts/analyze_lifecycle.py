#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def stats(values):
    values = [float(x) for x in values]
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "raw": values,
    }


def load_condition(tag):
    s = json.loads((ART / f"pair_lifecycle_{tag}_rank0.json").read_text())
    t = json.loads((ART / f"pair_lifecycle_{tag}_rank1.json").read_text())
    sb = {x["sample_id"]: x for x in s["pairs"]}
    tb = {x["sample_id"]: x for x in t["pairs"]}
    if set(sb) != set(tb):
        raise SystemExit(f"sample mismatch {tag}")
    return s, t, sb, tb


def main():
    conditions = ["old_round1", "resident_round1", "resident_round2"]
    loaded = {tag: load_condition(tag) for tag in conditions}
    rows = []
    summaries = {}
    for tag, (source, target, sb, tb) in loaded.items():
        by_pair = defaultdict(dict)
        for sample_id, src in sb.items():
            dst = tb[sample_id]
            row = {
                "condition": tag,
                "n": src["n"],
                "q": src["q"],
                "pair": src["pair"],
                "seed_id": src["seed_id"],
                "input_seed": src["input_seed"],
                "input_fingerprint": src["input_fingerprint"],
                "position": src["position"],
                "previous_mode": src["previous_mode"],
                "mode": src["mode"],
                "install_location": src["install_location"],
                "unified_ms": src["unified_ms"],
                "correctness_max_abs": src["correctness_max_abs"],
                "relative_error": src["relative_error"],
                "nonfinite_count": src["nonfinite_count"],
                "task_count_conserved": src["task_count_conserved"],
                "source_preinstall": json.dumps(src.get("preinstall")),
                "target_preinstall": json.dumps(dst.get("preinstall")),
                "target_release_ms": dst.get("release_ms"),
                "slot_generation": dst.get("slot_generation"),
                "slot_state": json.dumps(dst.get("slot_state")),
                "source_memory": json.dumps(src.get("memory")),
                "target_memory": json.dumps(dst.get("memory")),
                "target_pool_stats": json.dumps(dst.get("pool_stats_after")),
            }
            rows.append(row)
            by_pair[(src["n"], src["q"], src["pair"])][src["mode"]] = src

        point_results = {}
        grouped = defaultdict(list)
        for (n, q, pair), modes in sorted(by_pair.items()):
            if set(modes) != {"original", "split"}:
                raise SystemExit(f"incomplete pair {tag}/{n}/{q}/{pair}")
            original = modes["original"]
            split = modes["split"]
            original_first = original["position"] == 0
            delta = original["unified_ms"] - split["unified_ms"]
            item = {
                "n": n,
                "q": q,
                "pair": pair,
                "original_first": original_first,
                "previous_mode_before_split": split["previous_mode"],
                "delta_ms": delta,
            }
            grouped[(n, q)].append(item)

        for (n, q), items in sorted(grouped.items()):
            all_values = [x["delta_ms"] for x in items]
            first_values = [x["delta_ms"] for x in items if x["original_first"]]
            split_values = [x["delta_ms"] for x in items if not x["original_first"]]
            all_stats = stats(all_values)
            first_stats = stats(first_values)
            split_stats = stats(split_values)
            point_results[f"{n}/{q}"] = {
                "all": all_stats,
                "original_first": first_stats,
                "split_first": split_stats,
                "positive_pairs": sum(x > 0 for x in all_values),
                "slowed_pairs": sum(x <= 0 for x in all_values),
                "all_12_positive": all(x > 0 for x in all_values),
                "both_order_medians_positive": (
                    first_stats["median"] > 0 and split_stats["median"] > 0
                ),
            }
        summaries[tag] = {
            "lifecycle": source["lifecycle"],
            "round": source["round"],
            "seeds": source["seeds"],
            "point_summary": point_results,
            "install_events": source.get("block_install_events", []),
            "checks": {
                "samples": len(source["pairs"]),
                "all_max_abs_zero": all(x["correctness_max_abs"] == 0.0 for x in sb.values()),
                "all_finite": all(x["nonfinite_count"] == 0 for x in sb.values()),
                "all_task_counts_conserved": all(x["task_count_conserved"] for x in sb.values()),
            },
        }

    stable = {}
    for nq in ("8192/2048", "4096/1024"):
        r1 = summaries["resident_round1"]["point_summary"].get(nq)
        r2 = summaries["resident_round2"]["point_summary"].get(nq)
        stable[nq] = bool(
            r1
            and r2
            and r1["all_12_positive"]
            and r2["all_12_positive"]
            and r1["both_order_medians_positive"]
            and r2["both_order_medians_positive"]
        )

    fields = list(rows[0])
    with (ART / "PAIRED_TIMINGS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": "manual expert-subchain Prefill; not complete MoE or TTFT",
        "conditions": summaries,
        "resident_stable_break_even": stable,
        "resident_stable_any": any(stable.values()),
        "note": "Old lifecycle installs per call; resident lifecycle installs once per point block and evicts at block end.",
    }
    (ART / "lifecycle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(x["checks"]["all_max_abs_zero"] and x["checks"]["all_finite"] and x["checks"]["all_task_counts_conserved"] for x in summaries.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
