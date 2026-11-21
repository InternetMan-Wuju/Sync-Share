#!/usr/bin/env python3
"""Paired two-rank K=1 timing with one unified completion boundary.

The coordinator's clock starts before the go command and ends only after the
destination acknowledgement, return transfer, and refill are complete. Every
mode uses the same logical input and quota. This remains an expert-child replay,
not a complete MoE layer or service measurement.
"""
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
    LAYERS,
    LAYER_ATTRS,
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
GROUPS = 10
LOAD_CASES = (
    {"name": "interface_6", "tokens": 6, "main": 3, "external": 3},
    {"name": "prefill_subchain_512", "tokens": 512, "main": 256, "external": 256},
)
ORDERS = (
    ("original", "cold", "existing"),
    ("original", "existing", "cold"),
    ("cold", "original", "existing"),
    ("cold", "existing", "original"),
    ("existing", "original", "cold"),
    ("existing", "cold", "original"),
)


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def now_us():
    return time.perf_counter_ns() / 1000.0


def sync_us(fn):
    torch.npu.synchronize()
    started = now_us()
    value = fn()
    torch.npu.synchronize()
    return value, (now_us() - started) / 1000.0


def pack(state, flat, layout):
    """Pack runtime NZ->ND conversion plus all expert state into one flat buffer."""
    for name, offset, nbytes, _, _ in layout:
        flat[offset : offset + nbytes].copy_(
            state[name].view(torch.uint8).reshape(-1)
        )


def pack_from_bank(layer, flat, layout):
    # Conversion is intentionally inside the measured cold path. Synchronize
    # before temporary conversion buffers leave scope.
    pack(external_state_from_bank(layer), flat, layout)
    torch.npu.synchronize()


def split_indices(tokens, main_tokens, device="npu"):
    main_idx = torch.arange(main_tokens, dtype=torch.int64, device=device)
    ext_idx = torch.arange(main_tokens, tokens, dtype=torch.int64, device=device)
    return main_idx, ext_idx


def refill_output(reference_template, main_idx, ext_idx, main_out, ext_out):
    out = torch.empty_like(reference_template)
    if main_idx.numel():
        out.index_copy_(0, main_idx, main_out)
    if ext_idx.numel():
        out.index_copy_(0, ext_idx, ext_out)
    return out


def make_command(device="npu"):
    return torch.ones(1, dtype=torch.uint8, device=device)


def source_segments(mode, action):
    stage = {}
    if mode == "cold":
        _, stage["pack"] = sync_us(action["pack"])
        _, stage["weight_send"] = sync_us(action["weight_send"])
        stage["ready_ack"] = action["ready_ack"]
    if mode in ("cold", "existing"):
        _, stage["activation_prepare_send"] = sync_us(action["activation_send"])
        _, stage["main_compute"] = sync_us(action["main_compute"])
        _, stage["result_recv"] = sync_us(action["result_recv"])
        _, stage["refill"] = sync_us(action["refill"])
    else:
        _, stage["main_compute"] = sync_us(action["main_compute"])
    _, stage["done_recv"] = sync_us(action["done_recv"])
    return stage


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("this probe requires exactly two ranks")
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", init_method="env://", rank=rank, world_size=world_size
    )

    import sglang.srt.layers.quantization  # break import cycle intentionally
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW8A8Int8MoEMethod,
    )

    method = NPUW8A8Int8MoEMethod()
    matmul = GroupedMatmul()
    result = {
        "rank": rank,
        "role": "source" if rank == 0 else "destination",
        "layer_id": LAYER_ID,
        "expert_id": EXPERT,
        "groups": GROUPS,
        "warmup_groups_per_load_case": 1,
        "load_cases": LOAD_CASES,
        "scope": "two-rank expert-child replay; not full layer/service",
        "samples": [],
    }

    if rank == 0:
        preparation_started = now_us()
        bank_state, slot_state, offsets, bank_hashes, slot_hashes = load_raw_state(
            LAYER_ID
        )
        layer = ExpertLayer(bank_state)
        for name in PARTS:
            attr = LAYER_ATTRS[name]
            getattr(layer, attr).data = getattr(layer, attr).data.npu()
        method.process_weights_after_loading(layer, "w13")
        method.process_weights_after_loading(layer, "w2")
        bank_chain = RuntimeChain(method, matmul, layer, LAYER_ID)
        layout, state_bytes = state_layout(slot_state)
        flat = torch.empty((state_bytes,), dtype=torch.uint8, device="npu")
        torch.npu.synchronize()
        result["source_preparation_ms"] = (now_us() - preparation_started) / 1000.0
        result["actual_state"] = {
            "state_bytes": state_bytes,
            "slot_state_sha256": slot_hashes,
            "bank_state_sha256": bank_hashes,
            "offset_stats": offsets,
            "layout": [
                {
                    "name": name,
                    "offset": offset,
                    "bytes": nbytes,
                    "dtype": str(dtype),
                    "shape": list(shape),
                }
                for name, offset, nbytes, dtype, shape in layout
            ],
        }
        result["memory_initial"] = memory_snapshot()
    else:
        # Destination receives only the shape metadata; it never loads a full
        # permanent copy of the source expert.
        # Rank0 writes shape metadata before the following barrier.
        bank_chain = None
        layout = None
        state_bytes = None
        flat = None

    # Rank0 writes metadata and rank1 reads it through shared local filesystem.
    if rank == 0:
        result_meta = {
            "layout": result["actual_state"]["layout"],
            "state_bytes": result["actual_state"]["state_bytes"],
            "offset_stats": result["actual_state"]["offset_stats"],
            "slot_state_sha256": result["actual_state"]["slot_state_sha256"],
        }
        with open("/runexp/two_rank_meta.json", "w") as handle:
            json.dump(result_meta, handle)
    dist.barrier()
    if rank == 1:
        with open("/runexp/two_rank_meta.json") as handle:
            meta = json.load(handle)
        # Recover torch dtype from string stored by rank0.
        dtype_map = {
            "torch.int8": torch.int8,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }
        layout = [
            (
                item["name"],
                item["offset"],
                item["bytes"],
                dtype_map[item["dtype"]],
                tuple(item["shape"]),
            )
            for item in meta["layout"]
        ]
        state_bytes = meta["state_bytes"]
        pool = GlobalSlotPool(
            [1],
            state_bytes,
            alloc,
            name="rework-two-rank-dest",
            synchronize=torch.npu.synchronize,
        )
        result["actual_state"] = meta
        result["memory_initial"] = memory_snapshot()

    go_template = make_command()
    ack_template = make_command()

    for load_case in LOAD_CASES:
        tokens = load_case["tokens"]
        main_tokens = load_case["main"]
        ext_tokens = load_case["external"]
        main_idx, ext_idx = split_indices(tokens, main_tokens)
        if rank == 0:
            torch.manual_seed(9000 + tokens)
            hidden = torch.randn(
                (tokens, 6144), dtype=torch.bfloat16, device="npu"
            )
            reference = bank_chain(hidden)
            torch.npu.synchronize()
            remote_ext_output = torch.empty(
                (ext_tokens, 6144), dtype=torch.bfloat16, device="npu"
            )
            hidden_sha_input = hidden.detach().cpu()
            import hashlib

            input_fingerprint = hashlib.sha256(
                hidden_sha_input.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        else:
            hidden = None
            reference = None
            remote_ext_input = torch.empty(
                (ext_tokens, 6144), dtype=torch.bfloat16, device="npu"
            )
            input_fingerprint = None

        for group in range(-1, GROUPS):
            order = ORDERS[group % len(ORDERS)]
            for mode in order:
                sample_id = f"{load_case['name']}-g{group:02d}-{mode}"
                preinstall = None
                existing_lease = None
                existing_slot = None

                # Existing replica is preinstalled outside this sample's timer.
                if mode == "existing":
                    if rank == 0:
                        prep_start = now_us()
                        pack_from_bank(layer, flat, layout)
                        torch.npu.synchronize()
                        dist.send(flat, dst=1)
                        preinstall = {
                            "pack_and_send_ms": (now_us() - prep_start) / 1000.0
                        }
                    else:
                        prep_start = now_us()
                        existing_lease = pool.acquire(
                            request_id=f"{sample_id}-preinstall",
                            layer_id=LAYER_ID,
                            expert_id=EXPERT,
                        )
                        existing_slot = pool.slot_buffer(
                            existing_lease, allow_reserved=True
                        )
                        dist.recv(existing_slot, src=0)
                        torch.npu.synchronize()
                        pool.mark_ready(existing_lease)
                        preinstall = {
                            "acquire_recv_ready_ms": (
                                now_us() - prep_start
                            )
                            / 1000.0
                        }
                dist.barrier()

                if rank == 0:
                    started = now_us()
                    _, stage_control_send = sync_us(
                        lambda: dist.send(go_template, dst=1)
                    )
                    segments = {}
                    main_hidden = hidden.index_select(0, main_idx)
                    ext_hidden = hidden.index_select(0, ext_idx)
                    main_out = None
                    output = None

                    if mode == "cold":
                        _, segments["pack"] = sync_us(
                            lambda: pack_from_bank(layer, flat, layout)
                        )
                        _, segments["weight_send"] = sync_us(
                            lambda: dist.send(flat, dst=1)
                        )
                    if mode in ("cold", "existing"):
                        _, segments["activation_prepare_send"] = sync_us(
                            lambda: dist.send(ext_hidden, dst=1)
                        )
                        main_out, segments["main_compute"] = sync_us(
                            lambda: bank_chain(main_hidden)
                        )
                        _, segments["result_recv"] = sync_us(
                            lambda: dist.recv(remote_ext_output, src=1)
                        )

                        def do_refill():
                            return refill_output(
                                reference,
                                main_idx,
                                ext_idx,
                                main_out,
                                remote_ext_output,
                            )

                        output, segments["refill"] = sync_us(do_refill)
                    elif mode == "original":
                        output, segments["main_compute"] = sync_us(
                            lambda: bank_chain(hidden)
                        )
                    else:
                        raise ValueError(mode)

                    _, segments["done_recv"] = sync_us(
                        lambda: dist.recv(ack_template, src=1)
                    )
                    torch.npu.synchronize()
                    unified_ms = (now_us() - started) / 1000.0
                    diff = float(
                        (reference.float() - output.float()).abs().max().item()
                    )
                    sample = {
                        "sample_id": sample_id,
                        "group": group,
                        "mode": mode,
                        "order": list(order),
                        "load_case": load_case["name"],
                        "tokens": tokens,
                        "quota_main": main_tokens,
                        "quota_external": ext_tokens,
                        "input_fingerprint": input_fingerprint,
                        "unified_ms": unified_ms,
                        "segments_ms": segments,
                        "control_send_ms": stage_control_send,
                        "correctness_max_abs": diff,
                        "task_count": tokens,
                        "task_count_conserved": bool(
                            main_tokens + ext_tokens == tokens
                        ),
                        "preinstall": preinstall,
                        "state_payload_bytes": state_bytes,
                        "activation_payload_bytes": (
                            ext_tokens * 6144 * 2 if ext_tokens else 0
                        ),
                        "result_payload_bytes": (
                            ext_tokens * 6144 * 2 if ext_tokens else 0
                        ),
                        "memory": memory_snapshot(),
                    }
                else:
                    recv_start = now_us()
                    dist.recv(go_template, src=0)
                    torch.npu.synchronize()
                    segments = {"control_recv": (now_us() - recv_start) / 1000.0}
                    lease = None
                    slot = None

                    if mode == "cold":
                        def acquire_recv_ready():
                            nonlocal lease, slot
                            lease = pool.acquire(
                                request_id=sample_id,
                                layer_id=LAYER_ID,
                                expert_id=EXPERT,
                            )
                            slot = pool.slot_buffer(lease, allow_reserved=True)
                            dist.recv(slot, src=0)
                            torch.npu.synchronize()
                            pool.mark_ready(lease)

                        _, segments["acquire_weight_recv_ready"] = sync_us(
                            acquire_recv_ready
                        )
                    elif mode == "existing":
                        lease = existing_lease
                        slot = existing_slot

                    if mode in ("cold", "existing"):
                        _, segments["activation_recv"] = sync_us(
                            lambda: dist.recv(remote_ext_input, src=0)
                        )
                        ext_layer = type("ExternalLayer", (), {})()
                        views = slot_state_views(slot, layout)
                        for name in PARTS:
                            setattr(ext_layer, LAYER_ATTRS[name], views[name])
                        ext_chain = RuntimeChain(
                            method, matmul, ext_layer, LAYER_ID
                        )
                        ext_result, segments["external_compute"] = sync_us(
                            lambda: ext_chain(remote_ext_input)
                        )
                        _, segments["result_send"] = sync_us(
                            lambda: dist.send(ext_result, dst=0)
                        )

                    if lease is not None:
                        def release_then_ack():
                            pool.release(lease)
                            dist.send(ack_template, dst=0)

                        _, segments["release_and_done_send"] = sync_us(
                            release_then_ack
                        )
                    else:
                        _, segments["done_send"] = sync_us(
                            lambda: dist.send(ack_template, dst=0)
                        )
                    total_local_ms = (now_us() - recv_start) / 1000.0
                    sample = {
                        "sample_id": sample_id,
                        "group": group,
                        "mode": mode,
                        "order": list(order),
                        "load_case": load_case["name"],
                        "tokens": tokens,
                        "quota_main": main_tokens,
                        "quota_external": ext_tokens,
                        "local_ms": total_local_ms,
                        "segments_ms": segments,
                        "preinstall": preinstall,
                        "state_payload_bytes": state_bytes,
                        "activation_payload_bytes": (
                            ext_tokens * 6144 * 2 if ext_tokens else 0
                        ),
                        "result_payload_bytes": (
                            ext_tokens * 6144 * 2 if ext_tokens else 0
                        ),
                        "memory": memory_snapshot(),
                    }
                if group >= 0:
                    result["samples"].append(sample)
                dist.barrier()

    result["memory_final"] = memory_snapshot()
    if rank == 1:
        result["final_pool_stats"] = pool.stats()
    out_path = OUT_DIR / f"two_rank_paired_rank{rank}.json"
    out_path.write_text(json.dumps(result, indent=2, default=json_default))
    print(f"TWO_RANK_PAIRED_RANK{rank} " + json.dumps(result, default=json_default))
    dist.barrier()
    dist.destroy_process_group()
    if rank == 1:
        del pool
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
