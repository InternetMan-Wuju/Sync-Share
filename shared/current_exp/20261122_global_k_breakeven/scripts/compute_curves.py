#!/usr/bin/env python3
"""Two-rank source/target expert-chain capacity curves for one expert.

This measures only the complete local W8A8 expert chain on each card. It does
not include dispatch, weight copy, return, or refill. n is the number of tasks
received by expert43, not request input length.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist

from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import (
    EXPERT,
    LAYER_ATTRS,
    LAYERS,
    PARTS,
    ExpertLayer,
    RuntimeChain,
    alloc,
    external_state_from_bank,
    load_raw_state,
    memory_snapshot,
    slot_state_views,
    state_layout,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "/runexp/artifacts"))
LAYER_ID = LAYERS[0]
FULL_SIZES = (512, 1024, 2048, 4096, 8192)
HALF_SIZES = tuple(n // 2 for n in FULL_SIZES)
CAPACITY_SIZES = tuple(sorted(set(FULL_SIZES + HALF_SIZES)))
MIDDLE_CANDIDATES = (1024, 2048, 4096)
WARMUPS = 3
REPEATS = 12


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def layout_json(layout):
    return [
        {
            "name": name,
            "offset": offset,
            "bytes": nbytes,
            "dtype": str(dtype),
            "shape": list(shape),
        }
        for name, offset, nbytes, dtype, shape in layout
    ]


def layout_from_json(items):
    dtype_map = {
        "torch.int8": torch.int8,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    return [
        (
            item["name"],
            item["offset"],
            item["bytes"],
            dtype_map[item["dtype"]],
            tuple(item["shape"]),
        )
        for item in items
    ]


def pack(state, flat, layout):
    for name, offset, nbytes, _, _ in layout:
        flat[offset : offset + nbytes].copy_(
            state[name].view(torch.uint8).reshape(-1)
        )
    torch.npu.synchronize()


def input_hash(hidden):
    return hashlib.sha256(
        hidden.detach().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def measure_chain(chain, hidden, size, seed, card):
    fingerprint = input_hash(hidden)
    for _ in range(WARMUPS):
        chain(hidden)
    torch.npu.synchronize()
    records = []
    last_output = None
    for repeat in range(REPEATS):
        torch.npu.synchronize()
        started = time.perf_counter_ns()
        last_output = chain(hidden)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        records.append(
            {
                "card": card,
                "size": size,
                "repeat": repeat,
                "input_seed": seed,
                "input_fingerprint": fingerprint,
                "time_ms": elapsed_ms,
                "memory": memory_snapshot(),
            }
        )
    checksum = float(last_output.float().sum().item())
    return records, checksum


def medians_from_records(records):
    values = {}
    for record in records:
        values.setdefault((record["card"], record["size"]), []).append(
            record["time_ms"]
        )
    return {f"{card}/{size}": statistics.median(vals) for (card, size), vals in values.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("requires exactly two ranks")
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", init_method="env://", rank=rank, world_size=world
    )

    import sglang.srt.layers.quantization  # break package import cycle
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW8A8Int8MoEMethod,
    )

    method = NPUW8A8Int8MoEMethod()
    matmul = GroupedMatmul()
    meta_path = OUT_DIR / "compute_meta.json"
    result = {
        "rank": rank,
        "role": "source" if rank == 0 else "target",
        "layer_id": LAYER_ID,
        "expert_id": EXPERT,
        "scope": "synthetic-task capacity scan on real W8A8 weights",
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "records": [],
    }

    if rank == 0:
        bank_state, cpu_slot_state, offsets, _, slot_hashes = load_raw_state(LAYER_ID)
        layer = ExpertLayer(bank_state)
        for name in PARTS:
            attr = LAYER_ATTRS[name]
            getattr(layer, attr).data = getattr(layer, attr).data.npu()
        method.process_weights_after_loading(layer, "w13")
        method.process_weights_after_loading(layer, "w2")
        bank_chain = RuntimeChain(method, matmul, layer, LAYER_ID)
        external_state = external_state_from_bank(layer)
        torch.npu.synchronize()
        layout, state_bytes = state_layout(external_state)
        flat = torch.empty(state_bytes, dtype=torch.uint8, device="npu")
        pack(external_state, flat, layout)
        meta = {
            "state_bytes": state_bytes,
            "layout": layout_json(layout),
            "offsets": offsets,
            "slot_state_sha256": slot_hashes,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        result["actually_loaded_offsets"] = offsets
        result["actual_state"] = meta
        result["memory_initial"] = memory_snapshot()
    else:
        bank_chain = None
        layout = None
        state_bytes = None
        flat = None

    dist.barrier()
    if rank == 1:
        meta = json.loads(meta_path.read_text())
        layout = layout_from_json(meta["layout"])
        state_bytes = meta["state_bytes"]
        pool = GlobalSlotPool(
            [1],
            state_bytes,
            alloc,
            name="curve-target-slot",
            synchronize=torch.npu.synchronize,
        )
        lease = pool.acquire(
            request_id="curve-target", layer_id=LAYER_ID, expert_id=EXPERT
        )
        result["memory_initial"] = memory_snapshot()
    if rank == 0:
        dist.send(flat, dst=1)
    else:
        slot = pool.slot_buffer(lease, allow_reserved=True)
        dist.recv(slot, src=0)
        torch.npu.synchronize()
        pool.mark_ready(lease)
        use_cm = pool.use(lease)
        slot = use_cm.__enter__()
        ext_layer = type("ExternalLayer", (), {})()
        views = slot_state_views(slot, layout)
        for name in PARTS:
            setattr(ext_layer, LAYER_ATTRS[name], views[name])
        bank_chain = RuntimeChain(method, matmul, ext_layer, LAYER_ID)
    dist.barrier()

    measured_sizes = []
    checksums = {}
    for size in CAPACITY_SIZES:
        seed = 600000 + size
        if rank == 0:
            torch.manual_seed(seed)
            hidden = torch.randn((size, 6144), dtype=torch.bfloat16, device="npu")
        else:
            hidden = torch.empty((size, 6144), dtype=torch.bfloat16, device="npu")
        dist.broadcast(hidden, src=0)
        dist.barrier()
        if rank == 0:
            records, checksum = measure_chain(
                bank_chain, hidden, size, seed, "source"
            )
            result["records"].extend(records)
            checksums[f"source/{size}"] = checksum
        dist.barrier()
        if rank == 1:
            records, checksum = measure_chain(
                bank_chain, hidden, size, seed, "target"
            )
            result["records"].extend(records)
            checksums[f"target/{size}"] = checksum
        dist.barrier()
        measured_sizes.append(size)

    result["capacity_checksums"] = checksums
    rank_path = OUT_DIR / f"compute_curve_rank{rank}.json"
    rank_path.write_text(json.dumps(result, indent=2, default=json_default))
    dist.barrier()

    if rank == 0:
        rank0 = json.loads((OUT_DIR / "compute_curve_rank0.json").read_text())
        rank1 = json.loads((OUT_DIR / "compute_curve_rank1.json").read_text())
        combined = rank0["records"] + rank1["records"]
        medians = medians_from_records(combined)
        half_scores = []
        for n in MIDDLE_CANDIDATES:
            source_full = medians[f"source/{n}"]
            source_half = medians[f"source/{n // 2}"]
            target_half = medians[f"target/{n // 2}"]
            gain = source_full - max(source_half, target_half)
            half_scores.append(
                {
                    "n": n,
                    "source_full_ms": source_full,
                    "source_half_ms": source_half,
                    "target_half_ms": target_half,
                    "optimistic_half_saving_ms": gain,
                }
            )
        selected_middle = max(
            half_scores,
            key=lambda item: (item["optimistic_half_saving_ms"], -item["n"]),
        )["n"]
        selected_n = sorted({512, selected_middle, 8192})
        selected_quotas = {
            str(n): [n // 4, n // 2, 3 * n // 4] for n in selected_n
        }
        selection = {
            "capacity_sizes": list(CAPACITY_SIZES),
            "middle_candidates": half_scores,
            "selected_middle_n": selected_middle,
            "selection_rule": (
                "n=512 and n=8192 always; middle maximizes "
                "source_full(n)-max(source(n/2),target(n/2)); ties choose smaller n"
            ),
            "selected_n": selected_n,
            "selected_quotas": selected_quotas,
            "curve_medians_ms": medians,
        }
        (OUT_DIR / "compute_selection.json").write_text(
            json.dumps(selection, indent=2)
        )
    dist.barrier()
    selection = json.loads((OUT_DIR / "compute_selection.json").read_text())
    quota_sizes = set()
    for n, quotas in selection["selected_quotas"].items():
        for q in quotas:
            quota_sizes.add(q)
            quota_sizes.add(int(n) - q)
    missing_sizes = sorted(quota_sizes - set(measured_sizes))
    subscale_records = []
    for size in missing_sizes:
        seed = 700000 + size
        if rank == 0:
            torch.manual_seed(seed)
            hidden = torch.randn(
                (size, 6144), dtype=torch.bfloat16, device="npu"
            )
        else:
            hidden = torch.empty(
                (size, 6144), dtype=torch.bfloat16, device="npu"
            )
        dist.broadcast(hidden, src=0)
        dist.barrier()
        if rank == 0:
            records, _ = measure_chain(bank_chain, hidden, size, seed, "source")
            subscale_records.extend(records)
        dist.barrier()
        if rank == 1:
            records, _ = measure_chain(bank_chain, hidden, size, seed, "target")
            subscale_records.extend(records)
        dist.barrier()
    result["subscale_records"] = subscale_records
    result["selection"] = selection
    result["memory_final_before_release"] = memory_snapshot()
    rank_path.write_text(json.dumps(result, indent=2, default=json_default))

    dist.barrier()
    if rank == 1:
        use_cm.__exit__(None, None, None)
        pool.release(lease)
        result["final_pool_stats"] = pool.stats()
        result["memory_final"] = memory_snapshot()
        rank_path.write_text(json.dumps(result, indent=2, default=json_default))
        del pool
    dist.destroy_process_group()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
