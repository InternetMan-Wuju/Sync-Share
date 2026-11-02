#!/usr/bin/env python3
"""P2 single-NPU checks: shared-expert cost/overlap and one-slot lifecycle."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

from s2_local_cost import ExpertBank, benchmark_call, make_case, sync  # noqa: E402

SEED = 20261103


def load_prefix(model: Path, shard: str, prefix: str) -> dict[str, torch.Tensor]:
    with safe_open(model / shard, framework="pt", device="cpu") as handle:
        result = {key[len(prefix):]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    expected = {f"{proj}.{suffix}" for proj in ("gate_proj", "up_proj", "down_proj") for suffix in ("weight", "weight_scale", "weight_offset")}
    if set(result) != expected:
        raise RuntimeError(f"incomplete state for {prefix}: {sorted(set(result) ^ expected)}")
    return result


def chain(state, case: dict) -> torch.Tensor:
    qx, x_scale = torch_npu.npu_dynamic_quant(case["x"])
    qx = qx[case["permutation"]]
    x_scale = x_scale[case["permutation"]]
    group_list = case["group_list"]
    gate_up = torch.ops.npu.npu_grouped_matmul(
        x=[qx], weight=[state.w13], scale=[state.w13_scale], per_token_scale=[x_scale],
        split_item=2, group_list_type=1, group_type=0, group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    hidden = state.w13.shape[-1] // 2
    activation = torch.nn.functional.silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
    q_activation, activation_scale = torch_npu.npu_dynamic_quant(activation)
    output = torch.ops.npu.npu_grouped_matmul(
        x=[q_activation], weight=[state.down], scale=[state.down_scale],
        per_token_scale=[activation_scale], split_item=2, group_list_type=1,
        group_type=0, group_list=group_list, output_dtype=torch.bfloat16,
    )[0]
    return output[case["inverse"]]


def compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    sync()
    diff = candidate.float() - reference.float()
    return {
        "bitwise_equal": bool(torch.equal(candidate, reference)),
        "max_abs_error": float(diff.abs().max().item()),
        "nrmse": float(torch.linalg.vector_norm(diff) / max(torch.linalg.vector_norm(reference.float()), 1e-12)),
        "nonfinite_count": int((~torch.isfinite(candidate)).sum().item()),
    }


def concurrent_measure(shared_state, routed_state, shared_case, routed_case) -> dict:
    shared_stream = torch.npu.Stream()
    routed_stream = torch.npu.Stream()
    current = torch.npu.current_stream()

    def run():
        shared_stream.wait_stream(current)
        routed_stream.wait_stream(current)
        with torch.npu.stream(shared_stream):
            shared_out = chain(shared_state, shared_case)
        with torch.npu.stream(routed_stream):
            routed_out = chain(routed_state, routed_case)
        current.wait_stream(shared_stream)
        current.wait_stream(routed_stream)
        return shared_out, routed_out

    for _ in range(3):
        run()
    sync()
    samples = []
    host_samples = []
    for _ in range(5):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
        host_samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "device_ms_samples": samples,
        "host_ms_samples": host_samples,
        "device_ms_median": statistics.median(samples),
        "host_ms_median": statistics.median(host_samples),
    }


class Slot:
    def __init__(self, state):
        self.w13 = torch.empty_like(state.w13)
        self.down = torch.empty_like(state.down)
        self.w13_scale = torch.empty_like(state.w13_scale)
        self.down_scale = torch.empty_like(state.down_scale)
        self.version = 0
        self.completed_event = None

    def copy_from_npu(self, state):
        if self.completed_event is not None:
            torch.npu.current_stream().wait_event(self.completed_event)
        self.w13.copy_(state.w13)
        self.down.copy_(state.down)
        self.w13_scale.copy_(state.w13_scale)
        self.down_scale.copy_(state.down_scale)
        self.version += 1
        return self.version

    def mark_readers_done(self):
        event = torch.npu.Event()
        event.record()
        self.completed_event = event
        return event

    def pointer_switch(self, state):
        self.w13 = state.w13
        self.down = state.down
        self.w13_scale = state.w13_scale
        self.down_scale = state.down_scale
        self.version += 1
        return self.version


def timed_copy(copy_fn) -> dict:
    copy_fn()
    sync()
    device, host = [], []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        copy_fn()
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end))
        host.append((time.perf_counter() - started) * 1000.0)
    return {"device_ms_samples": device, "host_ms_samples": host, "device_ms_median": statistics.median(device), "host_ms_median": statistics.median(host)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/glm-q")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError(f"expected one NPU, got {torch.npu.device_count()}")
    sync()
    free_before, total_hbm = torch.npu.mem_get_info()
    torch.npu.reset_peak_memory_stats()
    model = Path(args.model)

    shared_raw = load_prefix(model, "quant_model_weights-00109-of-00181.safetensors", "model.layers.49.mlp.shared_experts.")
    routed_raw = load_prefix(model, "quant_model_weights-00107-of-00181.safetensors", "model.layers.49.mlp.experts.43.")
    layer48_raw = load_prefix(model, "quant_model_weights-00105-of-00181.safetensors", "model.layers.48.mlp.experts.0.")
    shared = ExpertBank("shared_layer49", [0], {0: shared_raw})
    routed = ExpertBank("routed_layer49_e43", [0], {0: routed_raw})
    layer48 = ExpertBank("routed_layer48_e0", [0], {0: layer48_raw})

    shared_vs_routed = []
    overlap = []
    for rows in (1, 128, 1024, 4096):
        shared_case = make_case(shared, [rows], f"shared_m{rows}", "shared_expert")
        routed_case = make_case(routed, [rows], f"routed_m{rows}", "routed_expert")
        shared_timing = benchmark_call(lambda s=shared_case: chain(shared, s))
        routed_timing = benchmark_call(lambda r=routed_case: chain(routed, r))
        shared_vs_routed.append({
            "rows": rows,
            "shared_timing": shared_timing,
            "routed_timing": routed_timing,
            "shared_to_routed_time_ratio": shared_timing["device_ms_median"] / routed_timing["device_ms_median"],
        })
        if rows in (1024, 4096):
            concurrent = concurrent_measure(shared, routed, shared_case, routed_case)
            sequential_sum = shared_timing["device_ms_median"] + routed_timing["device_ms_median"]
            ideal_overlap = max(shared_timing["device_ms_median"], routed_timing["device_ms_median"])
            overlap.append({
                "rows": rows,
                "shared_ms": shared_timing["device_ms_median"],
                "routed_ms": routed_timing["device_ms_median"],
                "sequential_sum_ms": sequential_sum,
                "concurrent_dual_stream": concurrent,
                "measured_overlap_saving_vs_sequential": 1.0 - concurrent["device_ms_median"] / sequential_sum,
                "ideal_overlap_max_ms": ideal_overlap,
                "ideal_overlap_saving": 1.0 - ideal_overlap / sequential_sum,
            })

    # Slot lifecycle: layer48 expert 0 and layer49 shared expert alternately use
    # one preallocated set of buffers.  Each overwrite waits for prior readers.
    rows = 257
    case = make_case(shared, [rows], "slot_lifecycle_m257", "slot_lifecycle")
    ref_a = chain(shared, case)
    ref_b = chain(layer48, case)
    slot = Slot(shared)
    lifecycle = {}
    try:
        version = slot.copy_from_npu(shared)
        sync()
        output_a = chain(slot, case)
        slot.mark_readers_done().synchronize()
        version_b = slot.copy_from_npu(layer48)
        sync()
        output_b = chain(slot, case)
        slot.mark_readers_done().synchronize()
        lifecycle = {
            "status": "PASS",
            "versions": [version, version_b],
            "layer49_shared_after_copy": compare(output_a, ref_a),
            "layer48_expert0_after_reuse": compare(output_b, ref_b),
            "version_monotonic": version < version_b,
        }
    except Exception as exc:
        lifecycle = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    npu_copy = timed_copy(lambda: slot.copy_from_npu(layer48))
    pointer_slot = Slot(layer48)
    pointer_samples = []
    for _ in range(1000):
        started = time.perf_counter_ns()
        pointer_slot.pointer_switch(shared)
        pointer_slot.pointer_switch(layer48)
        pointer_samples.append((time.perf_counter_ns() - started) / 1e6)
    h2d = {}
    try:
        cpu_state = type("CPUState", (), {
            "w13": layer48.w13.cpu(), "down": layer48.down.cpu(),
            "w13_scale": layer48.w13_scale.cpu(), "down_scale": layer48.down_scale.cpu(),
        })()
        h2d_timing = timed_copy(lambda: slot.copy_from_npu(cpu_state))
        h2d = {"status": "measured", "timing": h2d_timing}
    except Exception as exc:
        h2d = {"status": "unsupported", "error": f"{type(exc).__name__}: {exc}"}

    sync()
    free_after, _ = torch.npu.mem_get_info()
    result = {
        "stage": "P2",
        "status": "COMPLETE" if lifecycle.get("status") == "PASS" else "PARTIAL",
        "kind": "shared_expert_overlap_and_slot_lifecycle",
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": int(torch.npu.device_count()),
            "device_name": torch.npu.get_device_name(0),
        },
        "shared_expert": {
            "exists": True,
            "checkpoint_layer49_tensor_count": 9,
            "complete_weight_bytes": int(shared.per_expert_bytes),
            "runtime_code_note": "With DeepEP the GLM4 MoE Lite source sets shared experts to tp_rank=0,tp_size=1; full runtime residency was not launched in this check.",
            "shared_vs_routed": shared_vs_routed,
            "dual_stream_overlap": overlap,
            "prefill_note": "NPU prefill can run shared experts on a separate stream; decode does not enable this dual-stream branch.",
        },
        "slot_lifecycle": lifecycle,
        "slot_copy_cost": {
            "bytes_per_state": int(layer48.per_expert_bytes),
            "npu_to_npu_copy": npu_copy,
            "h2d_copy": h2d,
            "pointer_switch_host_ms_median": statistics.median(pointer_samples),
            "pointer_switch_is_not_weight_transport": True,
        },
        "memory": {
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "total_hbm_bytes": int(total_hbm),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        },
        "scope_limits": [
            "Single-NPU local chains only; no full-model graph, service trace or HCCL.",
            "Dual-stream overlap uses two local streams and representative resident experts, not the full layer.",
            "The slot copy is local; it does not measure cross-rank expert transfer.",
            "No claim is made that shared-expert work can be removed from a hot rank without additional memory or synchronization.",
        ],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "status": result["status"],
        "shared_vs_routed": [
            {"rows": item["rows"], "shared_ms": item["shared_timing"]["device_ms_median"], "routed_ms": item["routed_timing"]["device_ms_median"], "ratio": item["shared_to_routed_time_ratio"]}
            for item in shared_vs_routed
        ],
        "overlap": overlap,
        "lifecycle": lifecycle,
        "copy": result["slot_copy_cost"],
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
