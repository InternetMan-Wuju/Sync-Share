#!/usr/bin/env python3
"""Short same-host timeline for existing-replica source/target compute overlap."""
from __future__ import annotations

import json
import os
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
SIZES = (512, 8192)


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def layout_from_json(items):
    dtype_map = {"torch.int8": torch.int8, "torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}
    return [
        (x["name"], x["offset"], x["bytes"], dtype_map[x["dtype"]], tuple(x["shape"]))
        for x in items
    ]


def pack(state, flat, layout):
    for name, offset, nbytes, _, _ in layout:
        flat[offset : offset + nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
    torch.npu.synchronize()


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://", rank=rank, world_size=2)
    import sglang.srt.layers.quantization
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod

    method = NPUW8A8Int8MoEMethod()
    matmul = GroupedMatmul()
    meta_path = OUT_DIR / "overlap_meta.json"
    result = {"rank": rank, "sizes": SIZES, "records": []}
    if rank == 0:
        bank_state, cpu_slot, offsets, _, _ = load_raw_state(LAYER_ID)
        layer = ExpertLayer(bank_state)
        for name in PARTS:
            attr = LAYER_ATTRS[name]
            getattr(layer, attr).data = getattr(layer, attr).data.npu()
        method.process_weights_after_loading(layer, "w13")
        method.process_weights_after_loading(layer, "w2")
        bank_chain = RuntimeChain(method, matmul, layer, LAYER_ID)
        state = external_state_from_bank(layer)
        torch.npu.synchronize()
        layout, state_bytes = state_layout(state)
        flat = torch.empty(state_bytes, dtype=torch.uint8, device="npu")
        pack(state, flat, layout)
        meta_path.write_text(json.dumps({"layout": [
            {"name": n, "offset": o, "bytes": b, "dtype": str(d), "shape": list(s)}
            for n, o, b, d, s in layout
        ], "state_bytes": state_bytes}, indent=2))
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
        pool = GlobalSlotPool([1], state_bytes, alloc, name="overlap-slot", synchronize=torch.npu.synchronize)
        lease = pool.acquire(request_id="overlap", layer_id=LAYER_ID, expert_id=EXPERT)
        result["memory_initial"] = memory_snapshot()
    if rank == 0:
        dist.send(flat, dst=1)
    else:
        slot = pool.slot_buffer(lease, allow_reserved=True)
        dist.recv(slot, src=0)
        pool.mark_ready(lease)
        views = slot_state_views(slot, layout)
        ext_layer = type("ExternalLayer", (), {})()
        for name in PARTS:
            setattr(ext_layer, LAYER_ATTRS[name], views[name])
        bank_chain = RuntimeChain(method, matmul, ext_layer, LAYER_ID)
    dist.barrier()

    go = torch.ones(1, dtype=torch.uint8, device="npu")
    ack = torch.ones(1, dtype=torch.uint8, device="npu")
    for n in SIZES:
        q = n // 2
        if rank == 0:
            torch.manual_seed(990000 + n)
            hidden = torch.randn((n, 6144), dtype=torch.bfloat16, device="npu")
            remote_result = torch.empty((q, 6144), dtype=torch.bfloat16, device="npu")
        else:
            hidden = torch.empty((n, 6144), dtype=torch.bfloat16, device="npu")
            remote_input = torch.empty((q, 6144), dtype=torch.bfloat16, device="npu")
        dist.broadcast(hidden, src=0)
        if rank == 0:
            bank_chain(hidden[: n - q])
        else:
            bank_chain(remote_input)
            torch.npu.synchronize()
        dist.barrier()
        if rank == 0:
            whole_start = time.perf_counter_ns()
            dist.send(go, dst=1)
            send_work = dist.isend(hidden[n - q :], dst=1)
            compute_start = time.perf_counter_ns()
            main_out = bank_chain(hidden[: n - q])
            torch.npu.synchronize()
            compute_end = time.perf_counter_ns()
            dist.recv(remote_result, src=1)
            send_work.wait()
            dist.recv(ack, src=1)
            torch.npu.synchronize()
            whole_end = time.perf_counter_ns()
            result["records"].append({
                "n": n,
                "q": q,
                "compute_start_ns": compute_start,
                "compute_end_ns": compute_end,
                "whole_start_ns": whole_start,
                "whole_end_ns": whole_end,
                "memory": memory_snapshot(),
            })
        else:
            dist.recv(go, src=0)
            dist.recv(remote_input, src=0)
            compute_start = time.perf_counter_ns()
            ext_out = bank_chain(remote_input)
            torch.npu.synchronize()
            compute_end = time.perf_counter_ns()
            dist.send(ext_out, dst=0)
            dist.send(ack, dst=0)
            result["records"].append({
                "n": n,
                "q": q,
                "compute_start_ns": compute_start,
                "compute_end_ns": compute_end,
                "memory": memory_snapshot(),
            })
        dist.barrier()
    result["memory_final"] = memory_snapshot()
    path = OUT_DIR / f"overlap_timeline_rank{rank}.json"
    path.write_text(json.dumps(result, indent=2, default=json_default))
    dist.barrier()
    if rank == 1:
        pool.release(lease)
        del pool
    dist.destroy_process_group()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
