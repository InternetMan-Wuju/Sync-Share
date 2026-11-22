#!/usr/bin/env python3
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
r0 = json.loads((ART / "overlap_timeline_rank0.json").read_text())
r1 = json.loads((ART / "overlap_timeline_rank1.json").read_text())
by_size0 = {x["n"]: x for x in r0["records"]}
by_size1 = {x["n"]: x for x in r1["records"]}
out = {"scope": "same-host diagnostic timeline, existing replica, one sample per n", "records": []}
for n in sorted(by_size0):
    s, t = by_size0[n], by_size1[n]
    source_start, source_end = s["compute_start_ns"], s["compute_end_ns"]
    target_start, target_end = t["compute_start_ns"], t["compute_end_ns"]
    overlap_ns = max(0, min(source_end, target_end) - max(source_start, target_start))
    out["records"].append({
        "n": n,
        "q": s["q"],
        "source_compute_ms": (source_end - source_start) / 1e6,
        "target_compute_ms": (target_end - target_start) / 1e6,
        "overlap_ms": overlap_ns / 1e6,
        "source_relative_start_ms": (source_start - s["whole_start_ns"]) / 1e6,
        "source_relative_end_ms": (source_end - s["whole_start_ns"]) / 1e6,
        "target_compute_start_ns": target_start,
        "target_compute_end_ns": target_end,
        "whole_existing_ms": (s["whole_end_ns"] - s["whole_start_ns"]) / 1e6,
        "source_memory": s["memory"],
        "target_memory": t["memory"],
    })
(ART / "OVERLAP_TIMELINE.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
