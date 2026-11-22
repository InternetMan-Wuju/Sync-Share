#!/usr/bin/env python3
"""Two-rank device profiler for original and resident-replica main point."""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist
from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import (
    EXPERT, LAYER_ATTRS, LAYERS, PARTS, ExpertLayer, RuntimeChain, alloc,
    external_state_from_bank, load_raw_state, slot_state_views, state_layout,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "/runexp/artifacts"))
RAW = OUT_DIR / "profiler_raw"
LAYER_ID = LAYERS[0]
N, Q = 8192, 2048


def pack(state, flat, layout):
    for name, offset, nbytes, _, _ in layout:
        flat[offset:offset+nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
    torch.npu.synchronize()


def summarize_trace(base):
    dirs = sorted(glob.glob(str(Path(base) / "*_ascend_pt")))
    if not dirs:
        return {"error": "no profiler output directory", "base": base}
    root = Path(dirs[-1])
    out = {"trace_dir": str(root)}
    for name in ("kernel_details.csv", "operator_details.csv", "trace_view.json"):
        path = root / "ASCEND_PROFILER_OUTPUT" / name
        if path.exists():
            out.setdefault("files", {})[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    kernel_path = root / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if kernel_path.exists():
        kernels = []
        with kernel_path.open() as f:
            for row in csv.DictReader(f):
                kernels.append({
                    "name": row["Name"],
                    "start_us": float(row["Start Time(us)"].strip()),
                    "duration_us": float(row["Duration(us)"]),
                    "wait_us": float(row["Wait Time(us)"]),
                })
        out["kernels"] = kernels
        out["kernel_total_us"] = sum(x["duration_us"] for x in kernels)
        out["kernel_wait_us"] = sum(x["wait_us"] for x in kernels)
        out["top_kernels_by_duration"] = sorted(kernels, key=lambda x: x["duration_us"], reverse=True)[:20]
        out["top_kernels_by_wait"] = sorted(kernels, key=lambda x: x["wait_us"], reverse=True)[:20]
    op_path = root / "ASCEND_PROFILER_OUTPUT" / "operator_details.csv"
    if op_path.exists():
        ops = []
        with op_path.open() as f:
            for row in csv.DictReader(f):
                ops.append({
                    "name": row["Name"],
                    "host_total_us": float(row["Host Total Duration(us)"]),
                    "device_total_us": float(row["Device Total Duration(us)"]),
                })
        out["operators"] = ops
        out["top_operators_by_host_total"] = sorted(ops, key=lambda x: x["host_total_us"], reverse=True)[:20]
    return out


def profile_call(name, fn):
    base = RAW / name
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        on_trace_ready=tensorboard_trace_handler(str(base)),
    ):
        fn()
        torch.npu.synchronize()
    return summarize_trace(base)


def main():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl", init_method="env://", rank=rank, world_size=2)
    import sglang.srt.layers.quantization
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod

    method = NPUW8A8Int8MoEMethod(); matmul = GroupedMatmul(); meta_path = OUT_DIR / "profile_meta.json"
    result = {"rank": rank, "n": N, "q": Q, "runs": [], "clock_alignment": "not calibrated across devices"}
    if rank == 0:
        bank_state, _, offsets, _, _ = load_raw_state(LAYER_ID)
        layer = ExpertLayer(bank_state)
        for part in PARTS:
            attr = LAYER_ATTRS[part]
            getattr(layer, attr).data = getattr(layer, attr).data.npu()
        method.process_weights_after_loading(layer, "w13"); method.process_weights_after_loading(layer, "w2")
        bank_chain = RuntimeChain(method, matmul, layer, LAYER_ID)
        state = external_state_from_bank(layer); torch.npu.synchronize(); layout, state_bytes = state_layout(state)
        flat = torch.empty(state_bytes, dtype=torch.uint8, device="npu"); pack(state, flat, layout)
        meta_path.write_text(json.dumps({"state_bytes": state_bytes, "layout": [
            {"name": n, "offset": o, "bytes": b, "dtype": str(d), "shape": list(s)} for n,o,b,d,s in layout
        ]}, indent=2))
        torch.manual_seed(950000); hidden = torch.randn((N,6144), dtype=torch.bfloat16, device="npu")
        remote_result = torch.empty((Q,6144), dtype=torch.bfloat16, device="npu")
    else:
        bank_chain = None; layer = None; hidden = None; flat = None; remote_result = None
        pool = None; lease = None
    dist.barrier()
    if rank == 1:
        meta = json.loads(meta_path.read_text()); state_bytes = meta["state_bytes"]
        dtype_map = {"torch.int8": torch.int8, "torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}
        layout = [(x["name"],x["offset"],x["bytes"],dtype_map[x["dtype"]],tuple(x["shape"])) for x in meta["layout"]]
        pool = GlobalSlotPool([1], state_bytes, alloc, name="profile-slot", synchronize=torch.npu.synchronize)
        lease = pool.acquire(request_id="profile", layer_id=LAYER_ID, expert_id=EXPERT)
        slot = pool.slot_buffer(lease, allow_reserved=True); dist.recv(slot, src=0); pool.mark_ready(lease)
        views = slot_state_views(slot, layout); ext_layer = type("ExternalLayer", (), {})()
        for part in PARTS: setattr(ext_layer, LAYER_ATTRS[part], views[part])
        bank_chain = RuntimeChain(method, matmul, ext_layer, LAYER_ID)
        remote_input = torch.empty((Q,6144), dtype=torch.bfloat16, device="npu")
    else:
        dist.send(flat, dst=1)
    dist.barrier()
    go = torch.ones(1,dtype=torch.uint8,device="npu"); ack = torch.ones(1,dtype=torch.uint8,device="npu")

    # Shared prewarm of formal paths.
    for mode in ("original","existing"):
        if rank == 0:
            dist.send(go,dst=1)
            if mode == "original": bank_chain(hidden)
            else:
                work=dist.isend(hidden[N-Q:],dst=1); mo=bank_chain(hidden[:N-Q]); dist.recv(remote_result,src=1); torch.cat([mo,remote_result],0); work.wait()
            dist.recv(ack,src=1); torch.npu.synchronize()
        else:
            dist.recv(go,src=0)
            if mode == "existing":
                dist.recv(remote_input, src=0)
                with pool.use(lease):
                    out = bank_chain(remote_input)
                    dist.send(out, dst=0)
            dist.send(ack,dst=0); torch.npu.synchronize()
        dist.barrier()

    for mode in ("original","existing"):
        for sample in range(2):
            name=f"{mode}_{sample}_rank{rank}"
            def call():
                if rank == 0:
                    dist.send(go,dst=1)
                    if mode == "original": bank_chain(hidden)
                    else:
                        work=dist.isend(hidden[N-Q:],dst=1); mo=bank_chain(hidden[:N-Q]); dist.recv(remote_result,src=1); torch.cat([mo,remote_result],0); work.wait()
                    dist.recv(ack,src=1)
                else:
                    dist.recv(go, src=0)
                    if mode == "existing":
                        dist.recv(remote_input, src=0)
                        with pool.use(lease):
                            out = bank_chain(remote_input)
                            dist.send(out, dst=0)
                    dist.send(ack, dst=0)
            traced = profile_call(name, call)
            result["runs"].append({"mode":mode,"sample":sample,"trace":traced})
            dist.barrier()
    out=OUT_DIR/f"device_timeline_rank{rank}.json"; out.write_text(json.dumps(result,indent=2))
    dist.barrier()
    if rank==1: pool.release(lease); del pool
    dist.destroy_process_group(); torch.npu.synchronize()


if __name__ == "__main__":
    main()
