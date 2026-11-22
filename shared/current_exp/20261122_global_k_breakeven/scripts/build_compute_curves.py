#!/usr/bin/env python3
"""Combine two-rank compute-curve JSON into COMPUTE_CURVES.csv and summary."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def main():
    rank0 = json.loads((ART / "compute_curve_rank0.json").read_text())
    rank1 = json.loads((ART / "compute_curve_rank1.json").read_text())
    records = rank0["records"] + rank1["records"] + rank0.get("subscale_records", []) + rank1.get("subscale_records", [])
    records.sort(key=lambda r: (r["card"], r["size"], r["repeat"]))
    fields = [
        "card",
        "size",
        "repeat",
        "input_seed",
        "input_fingerprint",
        "time_ms",
        "allocated",
        "reserved",
        "max_allocated",
        "max_reserved",
        "device_free",
        "device_total",
    ]
    with (ART / "COMPUTE_CURVES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields[:6]}
            row.update({key: record.get("memory", {}).get(key) for key in fields[6:]})
            writer.writerow(row)

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["card"], record["size"])].append(record["time_ms"])
    selection = json.loads((ART / "compute_selection.json").read_text())
    checks = {
        f"{card}/{size}": {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
        for (card, size), values in sorted(grouped.items())
    }
    expected_capacity = set(selection["capacity_sizes"])
    errors = []
    for card in ("source", "target"):
        for size in expected_capacity:
            count = len(grouped[(card, size)])
            if count != 12:
                errors.append(f"{card}/{size}: expected 12 samples, got {count}")
    source_checks = rank0.get("capacity_checksums", {})
    target_checks = rank1.get("capacity_checksums", {})
    checksum_mismatches = []
    for key, source_value in source_checks.items():
        size = key.split("/", 1)[1]
        target_value = target_checks.get(f"target/{size}")
        if target_value is None or abs(source_value - target_value) > 1e-9:
            checksum_mismatches.append(
                {"size": int(size), "source": source_value, "target": target_value}
            )
    summary = {
        "scope": "real-weight synthetic-task expert-chain capacity scan",
        "records": len(records),
        "warmups_per_shape": rank0["warmups"],
        "formal_repeats_per_shape": rank0["repeats"],
        "group_summary": checks,
        "selection": selection,
        "errors": errors,
        "checksum_mismatches": checksum_mismatches,
        "note": "Times are local full-chain measurements; no communication is included.",
    }
    (ART / "compute_curve_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors or checksum_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
