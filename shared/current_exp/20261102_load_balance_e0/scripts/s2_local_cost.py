#!/usr/bin/env python3
"""S2: measure the local W8A8 layer-49 expert chain on one Ascend NPU.

Production-compatible primitives:
  dynamic per-token int8 quant -> fused gate/up grouped matmul -> SwiGLU ->
  dynamic activation quant -> down grouped matmul -> row-order restore.

This is a local expert-chain experiment.  It does not include cross-rank
dispatch/combine, HCCL waiting, router work, or full-service metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

from sglang.srt.hardware_backend.npu.utils import npu_format_cast

SEED = 20261103
WARMUP_CALLS = 3
MEASURE_BLOCKS = 5
TARGET_BLOCK_MS = 100.0
SEGMENT_REPEATS = 5


def sync() -> None:
    torch.npu.synchronize()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def event_block_ms(fn, repeats: int) -> tuple[float, float]:
    sync()
    event_start = torch.npu.Event(enable_timing=True)
    event_end = torch.npu.Event(enable_timing=True)
    host_start = time.perf_counter()
    event_start.record()
    for _ in range(repeats):
        fn()
    event_end.record()
    event_end.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1000.0 / repeats
    return event_start.elapsed_time(event_end) / repeats, host_ms


def median_segment_ms(fn) -> dict:
    for _ in range(2):
        fn()
    sync()
    device, host = [], []
    for _ in range(SEGMENT_REPEATS):
        event_start = torch.npu.Event(enable_timing=True)
        event_end = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        event_start.record()
        fn()
        event_end.record()
        event_end.synchronize()
        device.append(event_start.elapsed_time(event_end))
        host.append((time.perf_counter() - started) * 1000.0)
    return {
        "device_ms_median": statistics.median(device),
        "host_ms_median": statistics.median(host),
        "device_ms_samples": device,
        "host_ms_samples": host,
    }


def load_experts(model: Path, expert_ids: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for expert_id in expert_ids:
        keys = sorted(k for k in index if k.startswith(f"model.layers.49.mlp.experts.{expert_id}."))
        if len(keys) != 9:
            raise RuntimeError(f"expert {expert_id} has {len(keys)} tensors")
        for key in keys:
            by_shard.setdefault(index[key], []).append(key)
    result = {expert_id: {} for expert_id in expert_ids}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(model / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                expert_id = int(key.split(".experts.", 1)[1].split(".", 1)[0])
                result[expert_id][key.rsplit(".", 2)[-2] + "." + key.rsplit(".", 1)[-1]] = handle.get_tensor(key)
    return result


class ExpertBank:
    def __init__(self, bank_name: str, expert_ids: list[int], raw: dict[int, dict[str, torch.Tensor]]):
        self.bank_name = bank_name
        self.expert_ids = expert_ids
        self.raw = raw
        w13_raw, down_raw, w13_scale, down_scale = [], [], [], []
        for expert_id in expert_ids:
            tensors = raw[expert_id]
            gate = tensors["gate_proj.weight"].npu()
            up = tensors["up_proj.weight"].npu()
            down = tensors["down_proj.weight"].npu()
            w13_raw.append(torch.cat([gate, up], dim=0).transpose(0, 1).contiguous())
            down_raw.append(down.transpose(0, 1).contiguous())
            w13_scale.append(
                torch.cat(
                    [
                        tensors["gate_proj.weight_scale"].flatten(),
                        tensors["up_proj.weight_scale"].flatten(),
                    ]
                ).to(torch.bfloat16).npu()
            )
            down_scale.append(tensors["down_proj.weight_scale"].flatten().to(torch.bfloat16).npu())
        self.w13 = npu_format_cast(torch.stack(w13_raw, dim=0).contiguous())
        self.down = npu_format_cast(torch.stack(down_raw, dim=0).contiguous())
        self.w13_scale = torch.stack(w13_scale, dim=0).contiguous()
        self.down_scale = torch.stack(down_scale, dim=0).contiguous()
        sync()
        self.per_expert_bytes = (
            self.w13.numel() * self.w13.element_size()
            + self.down.numel() * self.down.element_size()
            + self.w13_scale.numel() * self.w13_scale.element_size()
            + self.down_scale.numel() * self.down_scale.element_size()
        ) // len(expert_ids)

    def make_input(self, rows: int, permutation: torch.Tensor, group_list: torch.Tensor) -> dict:
        x = torch.randn(
            (rows, 6144),
            generator=torch.Generator().manual_seed(SEED + rows),
            dtype=torch.float32,
        ).to(torch.bfloat16).npu()
        return {"x": x, "permutation": permutation, "inverse": torch.argsort(permutation), "group_list": group_list}

    def chain(self, case: dict, include_restore: bool = True) -> torch.Tensor:
        x = case["x"]
        permutation = case["permutation"]
        group_list = case["group_list"]
        qx, x_scale = torch_npu.npu_dynamic_quant(x)
        qx = qx[permutation]
        x_scale = x_scale[permutation]
        gate_up = torch.ops.npu.npu_grouped_matmul(
            x=[qx],
            weight=[self.w13],
            scale=[self.w13_scale],
            per_token_scale=[x_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
        hidden = self.w13.shape[-1] // 2
        activation = torch.nn.functional.silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
        q_activation, activation_scale = torch_npu.npu_dynamic_quant(activation)
        output = torch.ops.npu.npu_grouped_matmul(
            x=[q_activation],
            weight=[self.down],
            scale=[self.down_scale],
            per_token_scale=[activation_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
        if include_restore:
            output = output[case["inverse"]]
        return output

    def segments(self, case: dict) -> dict:
        cache = {}
        permutation = case["permutation"]
        group_list = case["group_list"]

        def quantize():
            cache["qx"], cache["x_scale"] = torch_npu.npu_dynamic_quant(case["x"])
            cache["qx_packed"] = cache["qx"][permutation]
            cache["scale_packed"] = cache["x_scale"][permutation]

        def gmm1():
            cache["gate_up"] = torch.ops.npu.npu_grouped_matmul(
                x=[cache["qx_packed"]],
                weight=[self.w13],
                scale=[self.w13_scale],
                per_token_scale=[cache["scale_packed"]],
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.bfloat16,
            )[0]

        def activation_requant():
            hidden = self.w13.shape[-1] // 2
            cache["activation"] = torch.nn.functional.silu(cache["gate_up"][:, :hidden]) * cache["gate_up"][:, hidden:]
            cache["q_activation"], cache["activation_scale"] = torch_npu.npu_dynamic_quant(cache["activation"])

        def gmm2():
            cache["output"] = torch.ops.npu.npu_grouped_matmul(
                x=[cache["q_activation"]],
                weight=[self.down],
                scale=[self.down_scale],
                per_token_scale=[cache["activation_scale"]],
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.bfloat16,
            )[0]

        def restore():
            cache["restored"] = cache["output"][case["inverse"]]

        quantize(); gmm1(); activation_requant(); gmm2(); restore()
        sync()
        return {
            "dynamic_quant": median_segment_ms(quantize),
            "gmm1_gate_up": median_segment_ms(gmm1),
            "activation_requant": median_segment_ms(activation_requant),
            "gmm2_down": median_segment_ms(gmm2),
            "restore_order": median_segment_ms(restore),
        }


def benchmark_call(fn, warmups: int = WARMUP_CALLS) -> dict:
    for _ in range(warmups):
        fn()
    sync()
    calibration_event, calibration_host = event_block_ms(fn, 1)
    calibration = max(calibration_event, calibration_host)
    repeats = max(1, min(2048, math.ceil(TARGET_BLOCK_MS / max(calibration, 1e-6))))
    blocks = [event_block_ms(fn, repeats) for _ in range(MEASURE_BLOCKS)]
    device = [item[0] for item in blocks]
    host = [item[1] for item in blocks]
    return {
        "repeats_per_block": repeats,
        "device_ms_samples": device,
        "host_ms_samples": host,
        "device_ms_median": statistics.median(device),
        "host_ms_median": statistics.median(host),
        "device_ms_p05": percentile(device, 0.05),
        "device_ms_p95": percentile(device, 0.95),
        "host_ms_p05": percentile(host, 0.05),
        "host_ms_p95": percentile(host, 0.95),
    }


def largest_remainder(values: list[int], total: int) -> list[int]:
    exact = [value * total / sum(values) for value in values]
    result = [math.floor(value) for value in exact]
    remainder = total - sum(result)
    for index in sorted(range(len(result)), key=lambda i: exact[i] - result[i], reverse=True)[:remainder]:
        result[index] += 1
    return result


def make_case(bank: ExpertBank, counts: list[int], label: str, distribution: str) -> dict:
    rows = sum(counts)
    if rows == 0:
        x = torch.empty((0, 6144), dtype=torch.bfloat16, device="npu:0")
        permutation = torch.empty((0,), dtype=torch.int64, device="npu:0")
        inverse = torch.empty((0,), dtype=torch.int64, device="npu:0")
        case = {
            "x": x,
            "permutation": permutation,
            "inverse": inverse,
            "group_list": torch.tensor(counts, dtype=torch.int64, device="npu:0"),
        }
    else:
        permutation = torch.randperm(rows, generator=torch.Generator().manual_seed(SEED + rows), device="cpu").npu()
        group_list = torch.tensor(counts, dtype=torch.int64, device="npu:0")
        case = bank.make_input(rows, permutation, group_list)
    case.update({"label": label, "distribution": distribution, "counts": counts, "rows": rows})
    return case


def timestamp_profile(bank: ExpertBank, case: dict, timings: dict) -> dict:
    segments = bank.segments(case)
    segments["full_chain_measured_separately"] = timings
    segment_sum = sum(segments[name]["device_ms_median"] for name in (
        "dynamic_quant", "gmm1_gate_up", "activation_requant", "gmm2_down", "restore_order"
    ))
    segments["segment_sum_ms"] = segment_sum
    segments["segment_sum_over_full_chain"] = segment_sum / max(timings["device_ms_median"], 1e-12)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/glm-q")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    model = Path(args.model)
    out = Path(args.out).resolve()
    csv_path = Path(args.csv).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError(f"expected exactly one NPU, got {torch.npu.device_count()}")
    sync()
    free_before, total_hbm = torch.npu.mem_get_info()
    torch.npu.reset_peak_memory_stats()

    bank1_ids = [43, 40, 192, 67, 82, 172, 112, 79]
    bank2_ids = [36, 41, 106, 69, 132, 63, 191, 109]
    raw1 = load_experts(model, bank1_ids)
    bank1 = ExpertBank("top8_a", bank1_ids, raw1)
    raw2 = load_experts(model, bank2_ids)
    bank2 = ExpertBank("top8_b", bank2_ids, raw2)
    full_counts = [4062, 3978, 1321, 1014, 841, 818, 594, 561]
    actual = json.loads((Path("/exp/results/S0_MAPPING_AUDIT.json")).read_text())

    rows = []
    shape_cases = []
    for rows_count in (0, 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        case = make_case(bank1, [rows_count] + [0] * 7, f"single_e43_m{rows_count}", "single_expert")
        if rows_count == 0:
            boundary = {"rows": 0, "counts": [0] * 8}
            try:
                output = bank1.chain(case)
                sync()
                boundary.update(status="supported", output_shape=list(output.shape))
            except Exception as exc:
                boundary.update(status="unsupported", error=f"{type(exc).__name__}: {exc}")
            shape_cases.append(boundary)
            continue
        timing = benchmark_call(lambda c=case: bank1.chain(c))
        profile = timestamp_profile(bank1, case, timing)
        shape_cases.append({"rows": rows_count, "counts": case["counts"], "timing": timing, "segments": profile})
        rows.append({
            "case_group": "shape_curve",
            "label": case["label"],
            "distribution": "single_expert",
            "rows": rows_count,
            "active_experts": 1,
            "counts": json.dumps(case["counts"]),
            "device_ms_median": timing["device_ms_median"],
            "device_ms_p05": timing["device_ms_p05"],
            "device_ms_p95": timing["device_ms_p95"],
            "host_ms_median": timing["host_ms_median"],
            "quant_ms": profile["dynamic_quant"]["device_ms_median"],
            "gmm1_ms": profile["gmm1_gate_up"]["device_ms_median"],
            "activation_requant_ms": profile["activation_requant"]["device_ms_median"],
            "gmm2_ms": profile["gmm2_down"]["device_ms_median"],
            "restore_ms": profile["restore_order"]["device_ms_median"],
        })

    distribution_cases = []
    for total in (1024, 4096):
        plans = {
            "one_hot_1": [total] + [0] * 7,
            "two_hot_2": [total // 2, total // 2] + [0] * 6,
            "uniform_8": [total // 8] * 8,
            "hot_plus_small_7": [total * 5 // 8, total // 4, total // 16] + [total // 64] * 4 + [0],
            "real_top8_normalized": largest_remainder(full_counts, total),
        }
        for label, counts in plans.items():
            if sum(counts) != total:
                raise RuntimeError((label, counts, sum(counts), total))
            case = make_case(bank1, counts, f"{label}_m{total}", label)
            timing = benchmark_call(lambda c=case: bank1.chain(c))
            profile = timestamp_profile(bank1, case, timing)
            distribution_cases.append({
                "rows": total,
                "label": label,
                "counts": counts,
                "active_experts": sum(value > 0 for value in counts),
                "timing": timing,
                "segments": profile,
            })
            rows.append({
                "case_group": "distribution",
                "label": case["label"],
                "distribution": label,
                "rows": total,
                "active_experts": sum(value > 0 for value in counts),
                "counts": json.dumps(counts),
                "device_ms_median": timing["device_ms_median"],
                "device_ms_p05": timing["device_ms_p05"],
                "device_ms_p95": timing["device_ms_p95"],
                "host_ms_median": timing["host_ms_median"],
                "quant_ms": profile["dynamic_quant"]["device_ms_median"],
                "gmm1_ms": profile["gmm1_gate_up"]["device_ms_median"],
                "activation_requant_ms": profile["activation_requant"]["device_ms_median"],
                "gmm2_ms": profile["gmm2_down"]["device_ms_median"],
                "restore_ms": profile["restore_order"]["device_ms_median"],
            })

    real_case = make_case(bank1, full_counts, "real_top8_observed_m13189", "real_top8_observed")
    real_timing = benchmark_call(lambda: bank1.chain(real_case))
    real_profile = timestamp_profile(bank1, real_case, real_timing)
    rows.append({
        "case_group": "real_observed",
        "label": real_case["label"],
        "distribution": "real_top8_observed",
        "rows": real_case["rows"],
        "active_experts": 8,
        "counts": json.dumps(full_counts),
        "device_ms_median": real_timing["device_ms_median"],
        "device_ms_p05": real_timing["device_ms_p05"],
        "device_ms_p95": real_timing["device_ms_p95"],
        "host_ms_median": real_timing["host_ms_median"],
        "quant_ms": real_profile["dynamic_quant"]["device_ms_median"],
        "gmm1_ms": real_profile["gmm1_gate_up"]["device_ms_median"],
        "activation_requant_ms": real_profile["activation_requant"]["device_ms_median"],
        "gmm2_ms": real_profile["gmm2_down"]["device_ms_median"],
        "restore_ms": real_profile["restore_order"]["device_ms_median"],
    })

    # Cache-context diagnostic: both 8-expert banks are pre-resident.  This is
    # repeated-vs-rotating weight access, not a cold-cache measurement.
    rotation_rows = 1024
    rotation_counts = [128] * 8
    rotation_case_1 = make_case(bank1, rotation_counts, "rotation_bank_a", "uniform_8")
    rotation_case_2 = make_case(bank2, rotation_counts, "rotation_bank_b", "uniform_8")
    bank_a_timing = benchmark_call(lambda: bank1.chain(rotation_case_1))
    bank_b_timing = benchmark_call(lambda: bank2.chain(rotation_case_2))
    rotation_timing = benchmark_call(
        lambda: (bank1.chain(rotation_case_1), bank2.chain(rotation_case_2))
    )
    rotation_per_call = {
        key: (value / 2 if isinstance(value, float) else value // 2 if isinstance(value, int) else value)
        for key, value in rotation_timing.items()
    }
    repeated_mean = (bank_a_timing["device_ms_median"] + bank_b_timing["device_ms_median"]) / 2
    cache_context = {
        "context": "both 8-expert banks pre-resident; no artificial cache clear",
        "rotation_rows_per_bank": rotation_rows,
        "counts_per_bank": rotation_counts,
        "bank_a_repeated": bank_a_timing,
        "bank_b_repeated": bank_b_timing,
        "alternating_pair": rotation_timing,
        "alternating_per_call_estimate": rotation_per_call,
        "repeated_bank_mean_device_ms": repeated_mean,
        "alternating_mean_device_ms": rotation_timing["device_ms_median"] / 2,
        "rotation_over_repeated": (rotation_timing["device_ms_median"] / 2) / repeated_mean,
        "working_set_experts": len(bank1_ids) + len(bank2_ids),
        "working_set_bytes": int(bank1.per_expert_bytes * (len(bank1_ids) + len(bank2_ids))),
        "interpretation": "same-shape repeated weights versus round-robin between two resident banks",
    }

    # Explicit per-vector activation work: active experts and rows per expert.
    for entry in distribution_cases:
        entry["active_weight_bytes"] = int(sum(value > 0 for value in entry["counts"]) * bank1.per_expert_bytes)
    real_record = {
        "rows": real_case["rows"],
        "counts": full_counts,
        "active_experts": 8,
        "active_weight_bytes": int(8 * bank1.per_expert_bytes),
        "timing": real_timing,
        "segments": real_profile,
    }

    sync()
    free_after, _ = torch.npu.mem_get_info()
    result = {
        "stage": "S2",
        "status": "COMPLETE",
        "kind": "single_npu_local_expert_chain_cost_curve",
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": int(torch.npu.device_count()),
            "device_name": torch.npu.get_device_name(0),
            "seed": SEED,
        },
        "method": {
            "precision": "W8A8_DYNAMIC, int8 weights, bf16 weight scales for kernel, bf16 hidden states",
            "chain": "dynamic per-token quant -> fused gate/up grouped matmul -> SwiGLU -> dynamic activation quant -> down grouped matmul -> restore row order",
            "weights": "fused [gate; up] followed by down, processed to transposed production layout with npu_format_cast",
            "target_block_ms": TARGET_BLOCK_MS,
            "measure_blocks": MEASURE_BLOCKS,
            "warmup_calls": WARMUP_CALLS,
            "segment_profile_repeats": SEGMENT_REPEATS,
            "formal_timing": "only final synchronization inside a block; segment profile is separate",
        },
        "scope_limits": [
            "No cross-rank dispatch/combine, HCCL transfer or waiting.",
            "No router, attention or full-service TTFT/TPOT measurement.",
            "Both expert banks are pre-resident; the rotation test is not a cold-weight test.",
            "The real-top8 vector is an observed expert workload, not a full per-rank vector composed of all 16 experts.",
        ],
        "memory": {
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "total_hbm_bytes": int(total_hbm),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        },
        "expert_banks": {
            "top8_a": {
                "expert_ids": bank1_ids,
                "per_expert_bytes": int(bank1.per_expert_bytes),
                "bank_bytes": int(bank1.per_expert_bytes * len(bank1_ids)),
                "w13_shape": list(bank1.w13.shape),
                "down_shape": list(bank1.down.shape),
            },
            "top8_b": {
                "expert_ids": bank2_ids,
                "per_expert_bytes": int(bank2.per_expert_bytes),
                "bank_bytes": int(bank2.per_expert_bytes * len(bank2_ids)),
                "w13_shape": list(bank2.w13.shape),
                "down_shape": list(bank2.down.shape),
            },
        },
        "shape_curve": shape_cases,
        "distribution_cases": distribution_cases,
        "real_observed": real_record,
        "cache_context": cache_context,
        "s0_corrected_rank_load_cal": actual["corrected_mapping"]["layer49_rank_load_cal"],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "out": str(out),
        "csv": str(csv_path),
        "status": result["status"],
        "shape_points": len(shape_cases),
        "distribution_points": len(distribution_cases) + 1,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
        "rotation_over_repeated": cache_context["rotation_over_repeated"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
