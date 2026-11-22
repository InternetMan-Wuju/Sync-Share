#!/usr/bin/env python3
"""Old lifecycle and continuously-resident replica timing for fixed points."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
REQUIRED_PAIRS = 12


def now_us():
    return time.perf_counter_ns() / 1000.0


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def layout_from_json(items):
    dtype_map = {"torch.int8": torch.int8, "torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}
    return [(x["name"], x["offset"], x["bytes"], dtype_map[x["dtype"]], tuple(x["shape"])) for x in items]


def pack_external_state(layer, flat, layout):
    state = external_state_from_bank(layer)
    for name, offset, nbytes, _, _ in layout:
        flat[offset : offset + nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
    torch.npu.synchronize()


def input_hash(hidden):
    return hashlib.sha256(hidden.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_schedule(point_index, round_id):
    rng = random.Random(7000 + point_index * 100 + round_id)
    schedule = []
    for seed_id in (0, 1):
        orders = [("original", "split")] * 3 + [("split", "original")] * 3
        rng.shuffle(orders)
        for order in orders:
            schedule.append({"seed_id": seed_id, "order": order})
    assert len(schedule) == REQUIRED_PAIRS
    return schedule


def build_ext_chain(method, matmul, slot, layout):
    layer = type("ExternalLayer", (), {})()
    views = slot_state_views(slot, layout)
    for name in PARTS:
        setattr(layer, LAYER_ATTRS[name], views[name])
    return RuntimeChain(method, matmul, layer, LAYER_ID)


def parse_points(value):
    result = []
    for item in value.split(","):
        n, q = item.split(":")
        result.append((int(n), int(q)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle", choices=("old", "resident"), required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--points", required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="hccl", init_method="env://", rank=rank, world_size=2)
    torch.npu.set_device(local_rank)
    import sglang.srt.layers.quantization
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod

    method = NPUW8A8Int8MoEMethod()
    matmul = GroupedMatmul()
    points = parse_points(args.points)
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != 2:
        raise ValueError("exactly two seeds required")
    tag = f"{args.lifecycle}_round{args.round}"
    meta_path = OUT_DIR / f"{tag}_meta.json"
    result = {
        "rank": rank,
        "tag": tag,
        "lifecycle": args.lifecycle,
        "round": args.round,
        "seeds": seeds,
        "points": points,
        "scope": "manual contiguous expert-subchain Prefill replay; not complete MoE or TTFT",
        "pairs": [],
        "block_install_events": [],
    }

    if rank == 0:
        bank_state, cpu_slot, offsets, bank_hashes, slot_hashes = load_raw_state(LAYER_ID)
        layer = ExpertLayer(bank_state)
        for name in PARTS:
            attr = LAYER_ATTRS[name]
            getattr(layer, attr).data = getattr(layer, attr).data.npu()
        method.process_weights_after_loading(layer, "w13")
        method.process_weights_after_loading(layer, "w2")
        bank_chain = RuntimeChain(method, matmul, layer, LAYER_ID)
        converted = external_state_from_bank(layer)
        torch.npu.synchronize()
        layout, state_bytes = state_layout(converted)
        flat = torch.empty(state_bytes, dtype=torch.uint8, device="npu")
        meta_path.write_text(json.dumps({
            "layout": [
                {"name": n, "offset": o, "bytes": b, "dtype": str(d), "shape": list(s)}
                for n, o, b, d, s in layout
            ],
            "state_bytes": state_bytes,
            "offsets": offsets,
            "bank_state_sha256": bank_hashes,
            "slot_state_sha256": slot_hashes,
        }, indent=2))
        del converted
        result["actual_state"] = {"state_bytes": state_bytes, "offsets": offsets, "bank_state_sha256": bank_hashes, "slot_state_sha256": slot_hashes}
        result["memory_initial"] = memory_snapshot()
    else:
        bank_chain = None
        layer = None
        layout = None
        state_bytes = None
        flat = None

    dist.barrier()
    if rank == 1:
        meta = json.loads(meta_path.read_text())
        layout = layout_from_json(meta["layout"])
        state_bytes = meta["state_bytes"]
        pool = GlobalSlotPool([1], state_bytes, alloc, name=f"{tag}-slot", synchronize=torch.npu.synchronize)
        result["actual_state"] = meta
        result["memory_initial"] = memory_snapshot()

    go = torch.ones(1, dtype=torch.uint8, device="npu")
    ack = torch.ones(1, dtype=torch.uint8, device="npu")

    def install(request_id):
        event = {"request_id": request_id}
        lease = None
        if rank == 0:
            started = now_us()
            pack_external_state(layer, flat, layout)
            dist.send(flat, dst=1)
            event["source_install_ms"] = (now_us() - started) / 1000.0
        else:
            started = now_us()
            lease = pool.acquire(request_id=request_id, layer_id=LAYER_ID, expert_id=EXPERT)
            slot = pool.slot_buffer(lease, allow_reserved=True)
            dist.recv(slot, src=0)
            pool.mark_ready(lease)
            event.update({
                "target_install_ms": (now_us() - started) / 1000.0,
                "slot_generation": lease.generation,
                "slot_owner": lease.owner_rank,
                "slot_local": lease.local_slot,
                "slot_layer": lease.layer_id,
                "slot_expert": lease.expert_id,
            })
        dist.barrier()
        return lease, event

    def evict(lease):
        started = now_us()
        if rank == 1:
            pool.release(lease)
        dist.barrier()
        return {"evict_ms": (now_us() - started) / 1000.0}

    for point_index, (n, q) in enumerate(points):
        main_n = n - q
        schedule = make_schedule(point_index, args.round)
        source_inputs = {}
        source_refs = {}
        fingerprints = {}
        if rank == 0:
            for seed_id in (0, 1):
                torch.manual_seed(seeds[seed_id] + n * 10 + seed_id)
                hidden = torch.randn((n, 6144), dtype=torch.bfloat16, device="npu")
                source_inputs[seed_id] = hidden
                source_refs[seed_id] = bank_chain(hidden)
                fingerprints[seed_id] = input_hash(hidden)
            torch.npu.synchronize()
        remote_input = torch.empty((q, 6144), dtype=torch.bfloat16, device="npu") if rank == 1 else None
        remote_result = torch.empty((q, 6144), dtype=torch.bfloat16, device="npu") if rank == 0 else None

        # Install once for resident blocks; for old semantics install a warm slot
        # only for shape warmup and then evict it.
        warm_lease, warm_event = install(f"warm-{tag}-{n}")
        if rank == 1:
            warm_chain = build_ext_chain(method, matmul, pool.slot_buffer(warm_lease), layout)
        if rank == 0:
            bank_chain(source_inputs[0])
            bank_chain(source_inputs[0][:main_n])
        else:
            for _ in range(3):
                warm_chain(remote_input)
            torch.npu.synchronize()
        dist.barrier()
        # Warm the complete formal path, including HCCL send/recv, cat/refill,
        # use-reference entry/exit and acknowledgement, without recording it.
        for warm_mode in ("original", "split"):
            if rank == 0:
                dist.send(go, dst=1)
                if warm_mode == "original":
                    bank_chain(source_inputs[0])
                else:
                    warm_work = dist.isend(source_inputs[0][main_n:], dst=1)
                    warm_main = bank_chain(source_inputs[0][:main_n])
                    dist.recv(remote_result, src=1)
                    torch.cat([warm_main, remote_result], dim=0)
                    warm_work.wait()
                dist.recv(ack, src=1)
                torch.npu.synchronize()
            else:
                dist.recv(go, src=0)
                if warm_mode == "split":
                    dist.recv(remote_input, src=0)
                    with pool.use(warm_lease):
                        warm_out = warm_chain(remote_input)
                        dist.send(warm_out, dst=0)
                dist.send(ack, dst=0)
                torch.npu.synchronize()
            dist.barrier()
        warm_evict = evict(warm_lease)

        if args.lifecycle == "resident":
            block_lease, block_install = install(f"block-{tag}-{n}")
            if rank == 1:
                resident_chain = build_ext_chain(method, matmul, pool.slot_buffer(block_lease), layout)
                for _ in range(3):
                    resident_chain(remote_input)
                torch.npu.synchronize()
            dist.barrier()
            result["block_install_events"].append({"n": n, "source": warm_event if rank == 0 else None, "target": warm_event if rank == 1 else None, "actual": block_install, "warm_evict": warm_evict})
        else:
            block_lease = None
            block_install = None

        for pair_index, entry in enumerate(schedule):
            seed_id = entry["seed_id"]
            order = list(entry["order"])
            if rank == 0:
                hidden = source_inputs[seed_id]
                reference = source_refs[seed_id]
                ext_hidden = hidden[main_n:]
                main_hidden = hidden[:main_n]
            else:
                hidden = None
                reference = None
                ext_hidden = None
                main_hidden = None

            for position, mode in enumerate(order):
                sample_id = f"{tag}-{n}-{q}-p{pair_index:02d}-{mode}"
                preinstall = None
                call_lease = None
                if mode == "split":
                    if args.lifecycle == "old":
                        call_lease, preinstall = install(f"install-{sample_id}")
                        if rank == 1:
                            ext_chain = build_ext_chain(method, matmul, pool.slot_buffer(call_lease), layout)
                    else:
                        call_lease = block_lease
                        if rank == 1:
                            ext_chain = resident_chain
                dist.barrier()

                if rank == 0:
                    started = now_us()
                    dist.send(go, dst=1)
                    if mode == "original":
                        output = bank_chain(hidden)
                    else:
                        activation_work = dist.isend(ext_hidden, dst=1)
                        main_output = bank_chain(main_hidden)
                        dist.recv(remote_result, src=1)
                        output = torch.cat([main_output, remote_result], dim=0)
                        activation_work.wait()
                    dist.recv(ack, src=1)
                    torch.npu.synchronize()
                    elapsed_ms = (now_us() - started) / 1000.0
                    diff = float((reference.float() - output.float()).abs().max().item())
                    denom = float(reference.float().abs().max().item())
                    sample = {
                        "sample_id": sample_id,
                        "n": n,
                        "q": q,
                        "pair": pair_index,
                        "seed_id": seed_id,
                        "input_seed": seeds[seed_id] + n * 10 + seed_id,
                        "input_fingerprint": fingerprints[seed_id],
                        "position": position,
                        "previous_mode": order[position - 1] if position else None,
                        "mode": mode,
                        "install_location": "none" if mode == "original" else ("before_call" if args.lifecycle == "old" else "block_start"),
                        "unified_ms": elapsed_ms,
                        "correctness_max_abs": diff,
                        "relative_error": diff / denom if denom else 0.0,
                        "nonfinite_count": int((~torch.isfinite(output)).sum().item()),
                        "task_count_conserved": bool(main_n + q == n),
                        "preinstall": preinstall,
                        "slot_generation": call_lease.generation if call_lease is not None else None,
                        "memory": memory_snapshot(),
                    }
                    result["pairs"].append(sample)
                else:
                    dist.recv(go, src=0)
                    release_ms = None
                    if mode == "split":
                        dist.recv(remote_input, src=0)
                        with pool.use(call_lease):
                            ext_out = ext_chain(remote_input)
                            dist.send(ext_out, dst=0)
                        if args.lifecycle == "old":
                            release_started = now_us()
                            pool.release(call_lease)
                            release_ms = (now_us() - release_started) / 1000.0
                    dist.send(ack, dst=0)
                    torch.npu.synchronize()
                    sample = {
                        "sample_id": sample_id,
                        "n": n,
                        "q": q,
                        "pair": pair_index,
                        "seed_id": seed_id,
                        "position": position,
                        "previous_mode": order[position - 1] if position else None,
                        "mode": mode,
                        "install_location": "none" if mode == "original" else ("before_call" if args.lifecycle == "old" else "block_start"),
                        "preinstall": preinstall,
                        "slot_generation": call_lease.generation if call_lease is not None else None,
                        "slot_state": (
                            {"owner": call_lease.owner_rank, "local_slot": call_lease.local_slot, "layer_id": call_lease.layer_id, "expert_id": call_lease.expert_id}
                            if call_lease is not None else None
                        ),
                        "release_ms": release_ms if mode == "split" else None,
                        "pool_stats_after": pool.stats(),
                        "memory": memory_snapshot(),
                    }
                    result["pairs"].append(sample)
                dist.barrier()

        if args.lifecycle == "resident":
            block_evict = evict(block_lease)
            result["block_install_events"].append({"n": n, "evict": block_evict, "rank": rank})
        if rank == 0:
            del source_inputs, source_refs, fingerprints

    result["memory_final"] = memory_snapshot()
    if rank == 1:
        result["final_pool_stats"] = pool.stats()
    out = OUT_DIR / f"pair_lifecycle_{tag}_rank{rank}.json"
    out.write_text(json.dumps(result, indent=2, default=json_default))
    print(f"PAIR_LIFECYCLE_{tag.upper()}_RANK{rank} " + json.dumps(result, default=json_default))
    dist.barrier()
    if rank == 1:
        del pool
    dist.destroy_process_group()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
