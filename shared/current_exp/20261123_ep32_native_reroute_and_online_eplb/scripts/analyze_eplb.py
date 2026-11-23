#!/usr/bin/env python3
"""Summarize patched EP32 online-EPLB logs and observer task counts."""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path

LAYER = 49
RE_LAYER = re.compile(r"^layer=(\d+):\s+(.*)$")
RE_EP = re.compile(r"ep(\d+)=\[([0-9, ]+)\]")

def parse_layouts(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    starts = sum(1 for x in lines if "[EPLBManager] rebalance start" in x)
    ends = sum(1 for x in lines if "[EPLBManager] rebalance end" in x)
    layouts = []
    for i, line in enumerate(lines):
        if "rebalance layout after:" not in line:
            continue
        block = lines[i + 1:i + 100]
        for row in block:
            m = RE_LAYER.match(row)
            if not m or int(m.group(1)) != LAYER:
                continue
            values = []
            for ep, raw in RE_EP.findall(m.group(2)):
                values.extend(int(x) for x in raw.split(",") if x.strip())
            if len(values) == 288:
                layouts.append(values)
            break
    return starts, ends, layouts

def load_routes(dirs):
    by_seq = defaultdict(lambda: defaultdict(lambda: {"physical_counts": [0] * 288, "logical_counts": [0] * 256, "local_rows": 0, "ranks": 0}))
    for directory in dirs:
        for p in sorted(Path(directory).glob("routes_rank*.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                seq = int(row["seq"])
                dst = by_seq[seq][int(row["rank"])]
                dst["local_rows"] += int(row.get("local_rows", 0))
                dst["ranks"] += 1
                pc = row.get("physical_counts", [])
                lc = row.get("logical_counts", [])
                for i, v in enumerate(pc[:288]):
                    dst["physical_counts"][i] += int(v)
                for i, v in enumerate(lc[:256]):
                    dst["logical_counts"][i] += int(v)
    out = []
    for seq in sorted(by_seq):
        rank_rows = by_seq[seq]
        physical = [0] * 288
        logical = [0] * 256
        local_rows = 0
        for row in rank_rows.values():
            local_rows += row["local_rows"]
            physical = [a + b for a, b in zip(physical, row["physical_counts"])]
            logical = [a + b for a, b in zip(logical, row["logical_counts"])]
        out.append({"seq": seq, "active_ranks": len(rank_rows), "local_rows": local_rows,
                    "physical_counts": physical, "logical_counts": logical})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node23-server", type=Path, required=True)
    ap.add_argument("--node25-server", type=Path, required=True)
    ap.add_argument("--node23-observer", type=Path, required=True)
    ap.add_argument("--node25-observer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    starts23, ends23, layouts23 = parse_layouts(a.node23_server)
    starts25, ends25, layouts25 = parse_layouts(a.node25_server)
    layouts = layouts23 or layouts25
    if not layouts:
        raise SystemExit("no layer49 layout snapshots found")
    final = layouts[-1]
    first = layouts[0]
    changes = [i for i, (x,y) in enumerate(zip(first, final)) if x != y]
    counts = [0] * 256
    for e in final: counts[e] += 1
    replica_logical = [i for i,c in enumerate(counts) if c > 1]
    replica_physical = [i for i,e in enumerate(final) if counts[e] > 1]
    routes = load_routes([a.node23_observer, a.node25_observer])
    chunk_labels = [
        'calibration_12288_chunk0', 'calibration_12288_chunk1',
        'formal_8192_chunk0', 'formal_16384_chunk0', 'formal_16384_chunk1',
        'formal_32768_chunk0', 'formal_32768_chunk1', 'formal_32768_chunk2', 'formal_32768_chunk3',
    ]
    for i, row in enumerate(routes):
        if i < len(chunk_labels): row['label'] = chunk_labels[i]
        pc = row["physical_counts"]
        row["physical_task_count"] = sum(pc)
        row["active_physical_positions"] = sum(1 for x in pc if x)
        row["replica_tasks"] = sum(pc[i] for i in replica_physical)
        row["replica_positions_active"] = sum(1 for i in replica_physical if pc[i])
        row["logical_task_count"] = sum(row["logical_counts"])
    out = {
        "layer": LAYER,
        "ep_size": 32,
        "num_logical_experts": 256,
        "num_physical_experts": len(final),
        "local_positions_per_rank": len(final) // 32,
        "node23_rebalance_starts_rank_events": starts23,
        "node23_rebalance_ends_rank_events": ends23,
        "node23_rebalance_events_per_rank": starts23 / 16,
        "node25_rebalance_starts_rank_events": starts25,
        "node25_rebalance_ends_rank_events": ends25,
        "node25_rebalance_events_per_rank": starts25 / 16,
        "layout_snapshots": len(layouts),
        "layout_changed": first != final,
        "changed_physical_positions": changes,
        "final_layer49_physical_to_logical": final,
        "final_layer49_sha256": hashlib.sha256(json.dumps(final,separators=(",", ":")).encode()).hexdigest(),
        "replica_logical_experts": replica_logical,
        "replica_physical_positions": replica_physical,
        "route_chunks": routes,
        "route_total_tasks": sum(x["physical_task_count"] for x in routes),
        "route_total_replica_tasks": sum(x["replica_tasks"] for x in routes),
        "calibration_route_chunks": routes[:2],
        "formal_route_chunks": routes[2:],
        "formal_route_total_tasks": sum(x["physical_task_count"] for x in routes[2:]),
        "formal_route_total_replica_tasks": sum(x["replica_tasks"] for x in routes[2:]),
        "provenance": "last layer49 'rebalance layout after' plus observer physical/logical counts"
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k:v for k,v in out.items() if k not in ("final_layer49_physical_to_logical","route_chunks")}, indent=2))

if __name__ == "__main__":
    main()
