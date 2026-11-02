#!/usr/bin/env python3
"""Combine S3 planner and replay into compact tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    plans = json.loads(Path(args.plans).read_text())
    replay = json.loads(Path(args.replay).read_text())
    rows = []
    for split in ("calibration", "holdout"):
        owner_replay = replay["splits"][split]["policies"]["owner_only"]
        for name, measured in replay["splits"][split]["policies"].items():
            predicted = plans["splits"][split]["policies"][name]
            rows.append({
                "split": split,
                "policy": name,
                "mapping": measured["mapping"],
                "replica_quotas": json.dumps(measured["replica_quotas"], sort_keys=True),
                "predicted_max_rank_time_ms": predicted["predicted_max_rank_time_ms"],
                "measured_all_rank_max_ms": measured["all_rank_max_ms"],
                "measured_reduction_vs_owner": measured["measured_max_reduction_vs_owner"],
                "predicted_reduction_vs_owner": predicted["predicted_max_time_reduction_vs_owner"],
                "max_row_load": predicted["max_row_load"],
                "row_reduction_vs_owner": predicted["max_row_load_reduction_vs_owner"],
                "affected_exact_max_ms": measured["affected_exact_max_ms"],
                "rank0_ms": measured["affected_ranks_exact"].get("0", {}).get("timing", {}).get("device_ms_median"),
                "rank1_ms": measured["affected_ranks_exact"].get("1", {}).get("timing", {}).get("device_ms_median"),
                "rank2_ms": measured["affected_ranks_exact"].get("2", {}).get("timing", {}).get("device_ms_median"),
                "rank15_ms": measured["affected_ranks_exact"].get("15", {}).get("timing", {}).get("device_ms_median"),
            })
    result = {
        "stage": "S3",
        "status": plans["status"],
        "model": plans["model"],
        "rows": rows,
        "best_policy_by_split": {
            split: min(
                (row for row in rows if row["split"] == split),
                key=lambda row: row["measured_all_rank_max_ms"],
            )["policy"]
            for split in ("calibration", "holdout")
        },
        "scope_limits": replay["scope_limits"] + plans["scope_limits"],
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with open(args.csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "out": args.out,
        "csv": args.csv,
        "best": result["best_policy_by_split"],
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
