#!/usr/bin/env python3
"""Build full-run Layer x Expert and Layer x physical EP-rank heat.

Pass --pt repeatedly for every EPLB interval plus the final manual dump. Each
full-window dump embeds the physical_to_logical_map that was active during that
interval. The script maps and sums each interval independently, so rebalance
boundaries do not lose data or apply the wrong layout.

For old single-window dumps without an embedded map, --map remains supported.
This script needs torch and is intended to run in the SGLang container.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


def summarize(values):
    vals = [float(x) for x in values]
    total = sum(vals)
    mean = total / len(vals) if vals else 0.0
    var = sum((x - mean) ** 2 for x in vals) / len(vals) if vals else 0.0
    std = math.sqrt(var)
    return {
        "n": len(vals),
        "total": round(total, 4),
        "mean": round(mean, 4),
        "min": round(min(vals), 4) if vals else None,
        "max": round(max(vals), 4) if vals else None,
        "std": round(std, 4),
        "cv": round(std / mean, 6) if mean else None,
        "peak_to_mean": round(max(vals) / mean, 6) if mean else None,
        "max_to_min": round(max(vals) / min(vals), 6)
        if vals and min(vals) > 0
        else None,
    }


def write_csv(path, matrix, column_prefix):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"{column_prefix}_{i}" for i in range(len(matrix[0]))])
        for layer, row in enumerate(matrix):
            writer.writerow([layer] + [int(x) for x in row])


def validate_mapping(rows, num_layers, num_experts, ep_size, source, map_ep_size=None):
    if torch.is_tensor(rows):
        mapping = rows.detach().cpu().to(torch.long)
    else:
        mapping = torch.tensor(rows, dtype=torch.long)
    if list(mapping.shape) != [num_layers, num_experts]:
        raise SystemExit(
            f"map shape {list(mapping.shape)} != recorder [L,E] "
            f"{[num_layers, num_experts]} ({source})"
        )
    if map_ep_size is not None and int(map_ep_size) != ep_size:
        raise SystemExit(
            f"map ep_size={map_ep_size} != requested ep_size={ep_size} ({source})"
        )

    expected = list(range(num_experts))
    for layer, row in enumerate(mapping.tolist()):
        if sorted(row) != expected:
            raise SystemExit(
                f"layer {layer} map is not a permutation ({source}). "
                "logical_count alone cannot split traffic across duplicated "
                "physical experts"
            )
    return mapping


def load_external_mapping(path, num_layers, num_experts, ep_size):
    if path is None:
        return (
            torch.arange(num_experts, dtype=torch.long)
            .repeat(num_layers, 1)
            .contiguous(),
            "identity",
        )

    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if "last_map" in obj:
        obj = obj["last_map"]
    rows = obj.get("physical_to_logical_map")
    if rows is None:
        raise SystemExit(f"no physical_to_logical_map in {path}")
    mapping = validate_mapping(
        rows,
        num_layers,
        num_experts,
        ep_size,
        str(path),
        obj.get("ep_size"),
    )
    return mapping, "final_physical_to_logical_map"


def load_window_mapping(obj, external_map, num_layers, num_experts, ep_size, source):
    for key in ("physical_to_logical_map", "last_physical_to_logical_map"):
        if key in obj:
            return (
                validate_mapping(
                    obj[key], num_layers, num_experts, ep_size, source
                ),
                f"embedded_{key}",
            )
    return load_external_mapping(external_map, num_layers, num_experts, ep_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", action="append", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--ep-size", type=int, required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.pt) > 1 and args.map:
        raise SystemExit("--map is only valid for one legacy --pt file")

    layer_expert = None
    layer_rank = None
    num_layers = None
    num_experts = None
    num_local = None
    total_steps = 0
    windows = []

    for pt_path in args.pt:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "logical_count" not in obj:
            raise SystemExit(f"recorder dump has no logical_count tensor: {pt_path}")
        logical_count = obj["logical_count"].detach().cpu().to(torch.int64)
        if logical_count.dim() != 3:
            raise SystemExit(
                f"expected [T,L,E], got {list(logical_count.shape)}: {pt_path}"
            )

        steps, window_layers, window_experts = logical_count.shape
        if num_layers is None:
            num_layers, num_experts = window_layers, window_experts
            if num_experts % args.ep_size != 0:
                raise SystemExit(
                    f"num_experts={num_experts} is not divisible by "
                    f"ep_size={args.ep_size}"
                )
            num_local = num_experts // args.ep_size
            layer_expert = torch.zeros(
                (num_layers, num_experts), dtype=torch.int64
            )
            layer_rank = torch.zeros(
                (num_layers, args.ep_size), dtype=torch.int64
            )
        elif (window_layers, window_experts) != (num_layers, num_experts):
            raise SystemExit(
                f"shape changed across windows: {(window_layers, window_experts)} "
                f"!= {(num_layers, num_experts)}"
            )

        window_layer_expert = logical_count.sum(dim=0)
        mapping, mapping_mode = load_window_mapping(
            obj,
            args.map,
            num_layers,
            num_experts,
            args.ep_size,
            pt_path,
        )
        physical_slot = torch.gather(window_layer_expert, 1, mapping)
        window_layer_rank = physical_slot.reshape(
            num_layers, args.ep_size, num_local
        ).sum(dim=2)
        layer_expert += window_layer_expert
        layer_rank += window_layer_rank
        total_steps += steps
        map_hash = hashlib.sha256(mapping.numpy().tobytes()).hexdigest()[:16]
        windows.append(
            {
                "source": str(Path(pt_path).resolve()),
                "steps": steps,
                "expert_token_assignments": int(window_layer_expert.sum().item()),
                "mapping_mode": mapping_mode,
                "mapping_sha256_16": map_hash,
                "window_kind": obj.get("window_kind", "legacy_single_window"),
                "full_window_dump_index": obj.get("full_window_dump_index"),
                "dump_timestamp": obj.get("dump_timestamp"),
                "recorder_rank": obj.get("rank"),
            }
        )

    expert_totals = layer_expert.sum(dim=0)
    rank_totals = layer_rank.sum(dim=0)
    active_layers = [
        i for i, total in enumerate(layer_expert.sum(dim=1).tolist()) if total > 0
    ]
    per_layer_rank_cv = []
    for row in layer_rank.tolist():
        per_layer_rank_cv.append(summarize(row)["cv"])

    layer_expert_list = layer_expert.tolist()
    layer_rank_list = layer_rank.tolist()
    windows.sort(
        key=lambda x: (
            x["dump_timestamp"] is None,
            x["dump_timestamp"] if x["dump_timestamp"] is not None else 0,
        )
    )
    indices = [w["full_window_dump_index"] for w in windows]
    kinds = [w["window_kind"] for w in windows]
    timestamps_present = all(w["dump_timestamp"] is not None for w in windows)
    embedded_maps = all(
        w["mapping_mode"].startswith("embedded_") for w in windows
    )
    indexed = all(i is not None for i in indices)
    contiguous = False
    ordered = False
    if indexed:
        integer_indices = [int(i) for i in indices]
        unique_indices = sorted(set(integer_indices))
        contiguous = (
            len(unique_indices) == len(indices)
            and unique_indices == list(range(len(indices)))
        )
        ordered = integer_indices == list(range(len(indices)))
    final_kind_valid = bool(
        kinds
        and kinds[-1] == "manual_final"
        and all(k == "eplb_rebalance" for k in kinds[:-1])
    )
    full_coverage = (
        embedded_maps
        and indexed
        and contiguous
        and ordered
        and timestamps_present
        and final_kind_valid
    )
    result = {
        "sources": [str(Path(p).resolve()) for p in args.pt],
        "legacy_map_source": str(Path(args.map).resolve()) if args.map else None,
        "role": args.role,
        "coverage_mode": "all_rebalance_windows" if full_coverage else "legacy_or_partial",
        "coverage_validated": full_coverage,
        "coverage_checks": {
            "all_windows_embed_map": embedded_maps,
            "all_windows_have_index": indexed,
            "indices_are_exactly_0_to_n": contiguous,
            "indices_follow_timestamp_order": ordered,
            "all_windows_have_timestamp": timestamps_present,
            "last_is_manual_final_and_earlier_are_eplb": final_kind_valid,
        },
        "window_index_range": [min(indices), max(indices)] if indexed else None,
        "window_count": len(windows),
        "windows": windows,
        "logical_count_shapes": [
            [w["steps"], num_layers, num_experts] for w in windows
        ],
        "recorder_window_steps": total_steps,
        "num_layers": num_layers,
        "num_logical_experts": num_experts,
        "ep_size": args.ep_size,
        "num_experts_per_rank": num_local,
        "mapping_mode": "per_window" if len(windows) > 1 else windows[0]["mapping_mode"],
        "active_layers": active_layers,
        "total_expert_token_assignments": int(layer_expert.sum().item()),
        "expert_totals": [int(x) for x in expert_totals.tolist()],
        "rank_totals": [int(x) for x in rank_totals.tolist()],
        "rank_summary": summarize(rank_totals.tolist()),
        "per_layer_rank_cv": per_layer_rank_cv,
        "layer_expert_tokens": layer_expert_list,
        "layer_rank_tokens": layer_rank_list,
    }

    json_path = out_dir / f"{args.role}_heat.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        out_dir / f"{args.role}_layer_expert_heat.csv",
        layer_expert_list,
        "expert",
    )
    write_csv(
        out_dir / f"{args.role}_layer_ep_rank_heat.csv",
        layer_rank_list,
        "ep_rank",
    )
    print(json.dumps({k: result[k] for k in (
        "role",
        "coverage_mode",
        "window_count",
        "logical_count_shapes",
        "mapping_mode",
        "total_expert_token_assignments",
        "rank_summary",
    )}, ensure_ascii=False, indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
