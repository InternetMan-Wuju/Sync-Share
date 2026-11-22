#!/usr/bin/env python3
"""Two-rank Prefill expert-subchain break-even timing.

Formal mode keeps only dependency/correctness waits and allows source compute to
overlap the asynchronous weight/activation sends. Diagnostic mode runs the same
operations with stage-level synchronization to expose where time goes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
    state_equal,
    state_layout,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "/runexp/artifacts"))
LAYER_ID = LAYERS[0]
ORDERS = tuple(itertools.permutations(("original", "cold", "existing")))
DIAGNOSTIC_ORDERS = ORDERS[:3]
FORMAL_GROUPS = 12
DIAGNOSTIC_GROUPS = 3


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def now_us():
    return time.perf_counter_ns() / 1000.0


def timed_sync(fn):
    torch.npu.synchronize()
    started = now_us()
    value = fn()
    torch.npu.synchronize()
    return value, (now_us() - started) / 1000.0


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


def pack_external_state(layer, flat, layout):
    # NZ->ND conversion plus flat packing; synchronize before temporary
    # conversion buffers leave scope.
    state = external_state_from_bank(layer)
    for name, offset, nbytes, _, _ in layout:
        flat[offset : offset + nbytes].copy_(
            state[name].view(torch.uint8).reshape(-1)
        )
    torch.npu.synchronize()


def input_fingerprint(hidden):
    return hashlib.sha256(
        hidden.detach().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def parse_points(value):
    points = []
    for item in value.split(","):
        n, q = item.split(":")
        points.append((int(n), int(q)))
    return points


def load_points(args):
    if args.points:
        return parse_points(args.points)
    selection = json.loads((OUT_DIR / "compute_selection.json").read_text())
    points = []
    for n in selection["selected_n"]:
        for q in selection["selected_quotas"][str(n)]:
            points.append((int(n), int(q)))
    return points


def make_inputs(n, seed, rank):
    if rank == 0:
        torch.manual_seed(seed)
        hidden = torch.randn((n, 6144), dtype=torch.bfloat16, device="npu")
    else:
        hidden = torch.empty((n, 6144), dtype=torch.bfloat16, device="npu")
    dist.broadcast(hidden, src=0)
    return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("diagnostic", "formal"), required=True)
    parser.add_argument("--points", default="")
    parser.add_argument("--seeds", default="810000,810001")
    parser.add_argument("--tag", default="standard")
    args = parser.parse_args()

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
    points = (
        [(6, 3), (512, 256)]
        if args.mode == "diagnostic"
        else load_points(args)
    )
    groups = DIAGNOSTIC_GROUPS if args.mode == "diagnostic" else FORMAL_GROUPS
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != 2:
        raise ValueError("exactly two input seeds are required")
    meta_path = OUT_DIR / "prefill_pair_meta.json"
    result = {
        "rank": rank,
        "role": "source" if rank == 0 else "target",
        "timing_mode": args.mode,
        "tag": args.tag,
        "points": points,
        "groups_per_point": groups,
        "input_seeds": seeds,
        "scope": (
            "two-rank manual expert-subchain Prefill replay; not full MoE layer "
            "or service TTFT"
        ),
        "samples": [],
    }

    if rank == 0:
        bank_state, cpu_slot_state, offsets, bank_hashes, slot_hashes = load_raw_state(
            LAYER_ID
        )
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
        meta = {
            "state_bytes": state_bytes,
            "layout": layout_json(layout),
            "offsets": offsets,
            "bank_state_sha256": bank_hashes,
            "slot_state_sha256": slot_hashes,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        del converted
        result["actual_state"] = meta
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
        pool = GlobalSlotPool(
            [1],
            state_bytes,
            alloc,
            name="prefill-breakeven-slot",
            synchronize=torch.npu.synchronize,
        )
        result["actual_state"] = meta
        result["memory_initial"] = memory_snapshot()

    go = torch.ones(1, dtype=torch.uint8, device="npu")
    ack = torch.ones(1, dtype=torch.uint8, device="npu")

    for n, q in points:
        main_n = n - q
        source_remote_result = (
            torch.empty((q, 6144), dtype=torch.bfloat16, device="npu")
            if rank == 0
            else None
        )
        target_remote_input = (
            torch.empty((q, 6144), dtype=torch.bfloat16, device="npu")
            if rank == 1
            else None
        )
        # Per-shape operator and communication warmup. This is outside every
        # recorded sample and leaves the slot free before the first cold trial.
        warm_hidden = make_inputs(n, seeds[0] + n * 10 + 99, rank)
        dist.barrier()
        if rank == 0:
            bank_chain(warm_hidden)
            main_output = bank_chain(warm_hidden[:main_n])
            pack_external_state(layer, flat, layout)
            weight_work = dist.isend(flat, dst=1)
            activation_work = dist.isend(warm_hidden[main_n:], dst=1)
            weight_work.wait()
            activation_work.wait()
            dist.recv(source_remote_result, src=1)
            torch.cat([main_output, source_remote_result], dim=0)
            dist.recv(ack, src=1)
            torch.npu.synchronize()
        else:
            warm_lease = pool.acquire(
                request_id=f"warm-{n}-{q}",
                layer_id=LAYER_ID,
                expert_id=EXPERT,
            )
            warm_slot = pool.slot_buffer(warm_lease, allow_reserved=True)
            dist.recv(warm_slot, src=0)
            pool.mark_ready(warm_lease)
            dist.recv(target_remote_input, src=0)
            warm_layer = type("WarmExternalLayer", (), {})()
            warm_views = slot_state_views(warm_slot, layout)
            for name in PARTS:
                setattr(warm_layer, LAYER_ATTRS[name], warm_views[name])
            warm_chain = RuntimeChain(method, matmul, warm_layer, LAYER_ID)
            for _ in range(3):
                warm_result = warm_chain(target_remote_input)
            torch.npu.synchronize()
            dist.send(warm_result, dst=0)
            pool.release(warm_lease)
            dist.send(ack, dst=0)
        dist.barrier()

        for group in range(groups):
            seed_id = group // 6 if groups == FORMAL_GROUPS else group % 2
            order = ORDERS[group % 6] if groups == FORMAL_GROUPS else DIAGNOSTIC_ORDERS[group]
            input_seed = seeds[seed_id] + n * 10 + seed_id
            hidden = make_inputs(n, input_seed, rank)
            fingerprint = input_fingerprint(hidden) if rank == 0 else None
            dist.barrier()

            reference = bank_chain(hidden) if rank == 0 else None
            torch.npu.synchronize()
            for mode in order:
                sample_id = f"{args.tag}-{n}-{q}-g{group:02d}-{mode}"
                preinstall = None
                existing_lease = None
                existing_slot = None

                if mode == "existing":
                    if rank == 0:
                        started = now_us()
                        pack_external_state(layer, flat, layout)
                        dist.send(flat, dst=1)
                        preinstall = {
                            "source_convert_pack_send_ms": (now_us() - started)
                            / 1000.0
                        }
                    else:
                        started = now_us()
                        existing_lease = pool.acquire(
                            request_id=f"{sample_id}-install",
                            layer_id=LAYER_ID,
                            expert_id=EXPERT,
                        )
                        existing_slot = pool.slot_buffer(
                            existing_lease, allow_reserved=True
                        )
                        dist.recv(existing_slot, src=0)
                        pool.mark_ready(existing_lease)
                        preinstall = {
                            "target_acquire_recv_ready_ms": (now_us() - started)
                            / 1000.0
                        }
                dist.barrier()

                source_segments = {}
                target_segments = {}
                if rank == 0:
                    diagnostic_started = now_us()
                    started = now_us()
                    if args.mode == "diagnostic":
                        _, source_segments["control_send"] = timed_sync(
                            lambda: dist.send(go, dst=1)
                        )
                    else:
                        dist.send(go, dst=1)

                    weight_work = None
                    activation_work = None
                    main_output = None
                    if mode == "cold":
                        if args.mode == "diagnostic":
                            _, source_segments["convert_pack"] = timed_sync(
                                lambda: pack_external_state(layer, flat, layout)
                            )
                            _, source_segments["weight_send"] = timed_sync(
                                lambda: dist.isend(flat, dst=1).wait()
                            )
                        else:
                            pack_external_state(layer, flat, layout)
                            weight_work = dist.isend(flat, dst=1)
                    if mode in ("cold", "existing"):
                        ext_hidden = hidden[main_n:]
                        main_hidden = hidden[:main_n]
                        if args.mode == "diagnostic":
                            _, source_segments["activation_send"] = timed_sync(
                                lambda: dist.isend(ext_hidden, dst=1).wait()
                            )
                            main_output, source_segments["main_compute"] = timed_sync(
                                lambda: bank_chain(main_hidden)
                            )
                            _, source_segments["result_recv"] = timed_sync(
                                lambda: dist.recv(source_remote_result, src=1)
                            )
                            output, source_segments["refill"] = timed_sync(
                                lambda: torch.cat(
                                    [main_output, source_remote_result], dim=0
                                )
                            )
                        else:
                            activation_work = dist.isend(ext_hidden, dst=1)
                            main_output = bank_chain(main_hidden)
                            dist.recv(source_remote_result, src=1)
                            output = torch.cat([main_output, source_remote_result], dim=0)
                    elif mode == "original":
                        if args.mode == "diagnostic":
                            output, source_segments["main_compute"] = timed_sync(
                                lambda: bank_chain(hidden)
                            )
                        else:
                            output = bank_chain(hidden)
                    else:
                        raise ValueError(mode)

                    if args.mode == "diagnostic":
                        _, source_segments["ack_recv"] = timed_sync(
                            lambda: dist.recv(ack, src=1)
                        )
                        source_segments["diagnostic_unified"] = (
                            now_us() - diagnostic_started
                        ) / 1000.0
                    else:
                        if weight_work is not None:
                            weight_work.wait()
                        if activation_work is not None:
                            activation_work.wait()
                        dist.recv(ack, src=1)
                        torch.npu.synchronize()
                        source_segments["unified"] = (now_us() - started) / 1000.0

                    diff = float(
                        (reference.float() - output.float()).abs().max().item()
                    )
                    denom = float(reference.float().abs().max().item())
                    relative = diff / denom if denom else 0.0
                    nonfinite = int((~torch.isfinite(output)).sum().item())
                    sample = {
                        "sample_id": sample_id,
                        "group": group,
                        "seed_id": seed_id,
                        "input_seed": input_seed,
                        "input_fingerprint": fingerprint,
                        "order": list(order),
                        "mode": mode,
                        "n": n,
                        "q": q,
                        "main_n": main_n,
                        "unified_ms": (
                            source_segments.get("unified")
                            if args.mode == "formal"
                            else None
                        ),
                        "segments_ms": source_segments,
                        "correctness_max_abs": diff,
                        "relative_error": relative,
                        "nonfinite_count": nonfinite,
                        "task_count_conserved": bool(main_n + q == n),
                        "preinstall": preinstall,
                        "state_payload_bytes": state_bytes,
                        "activation_payload_bytes": q * 6144 * 2,
                        "result_payload_bytes": q * 6144 * 2,
                        "slot_generation": None,
                        "slot_state": None,
                        "memory": memory_snapshot(),
                    }
                else:
                    if args.mode == "diagnostic":
                        _, target_segments["control_recv"] = timed_sync(
                            lambda: dist.recv(go, src=0)
                        )
                    else:
                        dist.recv(go, src=0)

                    lease = None
                    slot = None
                    if mode == "cold":
                        if args.mode == "diagnostic":
                            acquire_started = now_us()
                            lease = pool.acquire(
                                request_id=sample_id,
                                layer_id=LAYER_ID,
                                expert_id=EXPERT,
                            )
                            target_segments["slot_acquire"] = (
                                now_us() - acquire_started
                            ) / 1000.0
                            slot = pool.slot_buffer(lease, allow_reserved=True)
                            _, target_segments["weight_recv"] = timed_sync(
                                lambda: dist.recv(slot, src=0)
                            )
                            _, target_segments["mark_ready"] = timed_sync(
                                lambda: pool.mark_ready(lease)
                            )
                        else:
                            lease = pool.acquire(
                                request_id=sample_id,
                                layer_id=LAYER_ID,
                                expert_id=EXPERT,
                            )
                            slot = pool.slot_buffer(lease, allow_reserved=True)
                            dist.recv(slot, src=0)
                            pool.mark_ready(lease)
                    elif mode == "existing":
                        lease = existing_lease
                        slot = existing_slot

                    if mode in ("cold", "existing"):
                        if args.mode == "diagnostic":
                            _, target_segments["activation_recv"] = timed_sync(
                                lambda: dist.recv(target_remote_input, src=0)
                            )
                        else:
                            dist.recv(target_remote_input, src=0)
                        ext_layer = type("ExternalLayer", (), {})()
                        views = slot_state_views(slot, layout)
                        for name in PARTS:
                            setattr(ext_layer, LAYER_ATTRS[name], views[name])
                        ext_chain = RuntimeChain(
                            method, matmul, ext_layer, LAYER_ID
                        )
                        if args.mode == "diagnostic":
                            ext_result, target_segments["external_compute"] = timed_sync(
                                lambda: ext_chain(target_remote_input)
                            )
                            _, target_segments["result_send"] = timed_sync(
                                lambda: dist.send(ext_result, dst=0)
                            )
                        else:
                            ext_result = ext_chain(target_remote_input)
                            dist.send(ext_result, dst=0)

                    if lease is not None:
                        if args.mode == "diagnostic":
                            _, target_segments["release"] = timed_sync(
                                lambda: pool.release(lease)
                            )
                        else:
                            pool.release(lease)
                    if args.mode == "diagnostic":
                        _, target_segments["ack_send"] = timed_sync(
                            lambda: dist.send(ack, dst=0)
                        )
                    else:
                        dist.send(ack, dst=0)
                        torch.npu.synchronize()

                    sample = {
                        "sample_id": sample_id,
                        "group": group,
                        "seed_id": seed_id,
                        "input_seed": input_seed,
                        "order": list(order),
                        "mode": mode,
                        "n": n,
                        "q": q,
                        "main_n": main_n,
                        "segments_ms": target_segments,
                        "preinstall": preinstall,
                        "state_payload_bytes": state_bytes,
                        "activation_payload_bytes": q * 6144 * 2,
                        "result_payload_bytes": q * 6144 * 2,
                        "slot_generation": lease.generation if lease is not None else None,
                        "slot_state": (
                            {
                                "owner": lease.owner_rank,
                                "local_slot": lease.local_slot,
                                "layer_id": lease.layer_id,
                                "expert_id": lease.expert_id,
                                "request_id": lease.request_id,
                            }
                            if lease is not None
                            else None
                        ),
                        "pool_stats_after": pool.stats() if rank == 1 else None,
                        "memory": memory_snapshot(),
                    }
                result["samples"].append(sample)
                dist.barrier()

    result["memory_final"] = memory_snapshot()
    if rank == 1:
        result["final_pool_stats"] = pool.stats()
    out = OUT_DIR / f"prefill_pair_{args.mode}_{args.tag}_rank{rank}.json"
    out.write_text(json.dumps(result, indent=2, default=json_default))
    print(f"PREFILL_PAIR_{args.mode.upper()}_RANK{rank} " + json.dumps(result, default=json_default))
    dist.barrier()
    if rank == 1:
        del pool
    dist.destroy_process_group()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
