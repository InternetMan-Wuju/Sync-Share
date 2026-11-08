#!/usr/bin/env python3
"""Single-NPU positive local case for copy-aware prefill routing.

The case replays an imbalanced logical routing ``4096/64`` through the same
controller/Planner used by the external hook, copies a real W8A8 expert state
to a second local slot, and measures the complete local expert chains before
and after a ``2080/2080`` split.

This is deliberately a local proxy:

* both virtual ranks execute sequentially on one Ascend NPU;
* only local expert-chain calls are timed;
* state transfer uses independent local buffers, not HCCL/P2P;
* dispatch/combine, global synchronization, format staging and contention are
  not included.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch_npu

TARGET_BLOCK_MS = 250.0
BLOCKS = 5
WARMUP = 20
RUN_DIR = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("M0_POSITIVE_OUT", RUN_DIR / "artifacts" / "positive_case.json"))


class LocalStateTransport:
    """Copy the complete runtime expert state into independent local buffers."""

    def __init__(self, source, destination, source_offsets):
        self.source = source
        self.destination = destination
        self.source_offsets = source_offsets
        self.calls = 0
        self.destination_offsets = {
            name: torch.empty_like(value) for name, value in source_offsets.items()
        }

    def is_cached(self, *_args):
        return False

    def prepare_copy(self, *_args):
        # Sentinel contents in the destination slot make a missed copy visible.
        self.destination.w13[1].copy_(self.source.w13[0])
        self.destination.down[1].copy_(self.source.down[0])
        self.destination.w13_scale[1].copy_(self.source.w13_scale[0])
        self.destination.down_scale[1].copy_(self.source.down_scale[0])
        for name, value in self.source_offsets.items():
            self.destination_offsets[name].copy_(value)
        self.calls += 1


def make_case(n_tokens: int, counts: list[int], seed: int, device: torch.device) -> dict:
    x = torch.randn(
        (n_tokens, 6144),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float32,
    ).to(torch.bfloat16).npu()
    permutation = torch.arange(n_tokens, device=device)
    return {
        "x": x,
        "permutation": permutation,
        "inverse": permutation,
        "group_list": torch.tensor(counts, dtype=torch.int64, device=device),
    }


def event_block(fn, calls: int) -> tuple[float, float, float]:
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms, total_ms / calls, (time.perf_counter() - host_start) * 1000.0 / calls


def bench(fn) -> dict:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    _, calibration_ms, _ = event_block(fn, 1)
    calls = max(20, math.ceil(TARGET_BLOCK_MS / max(calibration_ms, 1e-6)))
    samples: list[float] = []
    blocks = []
    for _ in range(BLOCKS):
        total_ms, per_call_ms, host_ms = event_block(fn, calls)
        samples.append(per_call_ms)
        blocks.append(
            {
                "calls": calls,
                "total_ms": total_ms,
                "ms_per_call": per_call_ms,
                "host_ms_per_call": host_ms,
            }
        )
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "blocks": blocks,
        "all_ge_100ms": all(block["total_ms"] >= 100.0 for block in blocks),
    }


def bench_cpu_planner(planner, kwargs: dict) -> dict:
    """Measure pure host planning from already-collected routing counts."""
    for _ in range(WARMUP):
        planner.plan(**kwargs)
    samples = []
    blocks = []
    for _ in range(BLOCKS):
        start = time.perf_counter()
        planner.plan(**kwargs)
        calibration_ms = (time.perf_counter() - start) * 1000.0
        calls = max(20, math.ceil(TARGET_BLOCK_MS / max(calibration_ms, 1e-6)))
        start = time.perf_counter()
        for _ in range(calls):
            planner.plan(**kwargs)
        total_ms = (time.perf_counter() - start) * 1000.0
        per_call_ms = total_ms / calls
        samples.append(per_call_ms)
        blocks.append(
            {
                "calls": calls,
                "total_ms": total_ms,
                "ms_per_call": per_call_ms,
            }
        )
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "blocks": blocks,
        "all_ge_100ms": all(block["total_ms"] >= 100.0 for block in blocks),
    }


def main() -> None:
    from s2_local_cost import ExpertBank, load_experts

    import m0_external.controller as controller

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    model = Path("/models/glm-q")
    raw = load_experts(model, [43, 40])

    source_43 = ExpertBank("src43", [43], {43: raw[43]})
    source_40 = ExpertBank("src40", [40], {40: raw[40]})
    rank1 = ExpertBank("rank1", [40, 43], {40: raw[40], 43: raw[43]})

    # A wrong sentinel is installed in the future replica slot. The state check
    # below would fail if Planner/transport forgot to copy the real e43 state.
    rank1.w13[1].copy_(source_40.w13[0])
    rank1.down[1].copy_(source_40.down[0])
    rank1.w13_scale[1].copy_(source_40.w13_scale[0])
    rank1.down_scale[1].copy_(source_40.down_scale[0])

    offsets = {
        name: raw[43][f"{name}.weight_offset"].npu()
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    transport = LocalStateTransport(source_43, rank1, offsets)

    counts = (4096, 64) + (0,) * 14
    stats = controller.RoutingStats(counts, (counts,), 4160, 0, 0)

    # Measure a cold local state copy with the same >=100 ms block policy as the
    # expert chains. The destination uses independent buffers.
    copy = bench(transport.prepare_copy)
    copy_ms = copy["median_ms"]

    planner = controller.Planner()
    planner_kwargs = {
        "stats": stats,
        "step_id": 1,
        "layer_id": 49,
        "epoch": 1,
        "rank_expert_counts": [[4096], [64]],
        "primary_rank_by_expert": [0, 1],
        "primary_local_slot_by_expert": [0, 0],
        "primary_physical_by_expert": [0, 17],
        "destination_rank": 1,
        "destination_local_slot": 16,
        "destination_physical_id": 33,
        "cost_model": controller.LinearCostModel(per_task_us=1),
        "planner_cost_us": 0.0,
        "prepare_cost_us": copy_ms * 1000.0,
        "dispatch_extra_us": 0.0,
        "min_gain_us": 0.0,
        "epsilon_us": 0.0,
        "slot_manager": None,
        "transport": None,
        "weight_version": 3,
        "layout_version": "positive-local",
        "execute_copy": False,
    }
    planner_measurement = bench_cpu_planner(planner, planner_kwargs)
    planner_ms = planner_measurement["median_ms"]

    plan = planner.plan(
        **{
            **planner_kwargs,
            "planner_cost_us": planner_ms * 1000.0,
            "prepare_cost_us": copy_ms * 1000.0,
            "slot_manager": controller.SlotManager(),
            "transport": transport,
            "execute_copy": True,
        }
    )
    if plan.action != "route" or plan.q_global != 2016:
        raise RuntimeError(f"unexpected plan: {plan.to_dict()}")

    old_rank0 = make_case(4096, [4096], 101, device)
    old_rank1 = make_case(64, [64], 102, device)
    old = {
        "r0": bench(lambda: source_43.chain(old_rank0)),
        "r1": bench(lambda: source_40.chain(old_rank1)),
    }

    new_rank0 = make_case(2080, [2080], 103, device)
    new_rank1 = make_case(2080, [64, 2016], 104, device)
    new = {
        "r0": bench(lambda: source_43.chain(new_rank0)),
        "r1": bench(lambda: rank1.chain(new_rank1)),
    }

    old_max_ms = max(old["r0"]["median_ms"], old["r1"]["median_ms"])
    new_max_ms = max(new["r0"]["median_ms"], new["r1"]["median_ms"])
    compute_saving_ms = old_max_ms - new_max_ms
    net_ms = compute_saving_ms - planner_ms - copy_ms

    copied_state = {
        "copied_w13": bool(torch.equal(source_43.w13[0], rank1.w13[1])),
        "copied_down": bool(torch.equal(source_43.down[0], rank1.down[1])),
        "copied_w13_scale": bool(torch.equal(source_43.w13_scale[0], rank1.w13_scale[1])),
        "copied_down_scale": bool(torch.equal(source_43.down_scale[0], rank1.down_scale[1])),
        "offsets_zero": all(int(value.abs().sum()) == 0 for value in transport.destination_offsets.values()),
    }
    all_blocks_ge_100ms = all(
        measurement["all_ge_100ms"]
        for measurement in (copy, planner_measurement, *old.values(), *new.values())
    )
    passed = (
        compute_saving_ms > planner_ms + copy_ms
        and all_blocks_ge_100ms
        and all(copied_state.values())
    )
    copy_bytes = int(
        source_43.w13[0].numel() * source_43.w13[0].element_size()
        + source_43.down[0].numel() * source_43.down[0].element_size()
        + source_43.w13_scale[0].numel() * source_43.w13_scale[0].element_size()
        + source_43.down_scale[0].numel() * source_43.down_scale[0].element_size()
        + sum(value.numel() * value.element_size() for value in transport.source_offsets.values())
    )

    result = {
        "status": "LOCAL_POSITIVE_TEST_ONLY" if passed else "LOCAL_NOT_POSITIVE",
        "device_name": torch.npu.get_device_name(0),
        "case": "4096/64 -> 2080/2080",
        "plan": plan.to_dict(),
        "old": old,
        "new": new,
        "old_max_ms": old_max_ms,
        "new_max_ms": new_max_ms,
        "compute_saving_ms": compute_saving_ms,
        "planner_ms": planner_ms,
        "copy_ms": copy_ms,
        "copy": copy,
        "planner": planner_measurement,
        "net_after_planner_and_local_copy_ms": net_ms,
        "plan_state_copy_calls": 1 if plan.copy_required else 0,
        "transport_calls_total": transport.calls,
        "copy_bytes": copy_bytes,
        "state": copied_state,
        "all_blocks_ge_100ms": all_blocks_ge_100ms,
        "measurement_range": "expert-chain call + local full-state copy; excludes routing/dispatch/combine/HCCL",
        "service_started": False,
        "full_model_loaded": False,
        "scope": "single NPU local compute + local full-state copy; no HCCL",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
