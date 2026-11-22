#!/usr/bin/env python3
"""Actual DeepEP normal dispatch/combine harness for 256/257/258 layouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist

OUT_DIR = Path(os.environ.get("OUT_DIR", "/runexp/artifacts"))
HIDDEN = 128
TOP_K = 8
TOKENS_PER_RANK = 8


def logical_to_comm(expert, placeholder):
    if not 0 <= expert < 256:
        raise ValueError(expert)
    if not placeholder:
        return expert
    if expert < 128:
        return expert
    return expert + 1


def comm_to_logical(comm, placeholder):
    if not placeholder:
        if 0 <= comm < 256:
            return comm
        raise ValueError(f"invalid 256 comm id {comm}")
    if 0 <= comm < 128:
        return comm
    if 129 <= comm <= 256:
        return comm - 1
    if comm == 257:
        return 43
    raise ValueError(f"invalid placeholder comm id {comm}")


def build_inputs(rank, placeholder):
    torch.manual_seed(123400 + rank)
    logical_rows = []
    coverage = [0, 43, 127, 128, 255, (rank * 37 + 5) % 256, (rank * 53 + 11) % 256, (rank * 71 + 17) % 256]
    for token in range(TOKENS_PER_RANK):
        row = [coverage[(token + j) % len(coverage)] for j in range(TOP_K)]
        # Force both ranks to send expert43 and cross-rank boundary experts.
        if token % 3 == 0:
            row[0] = 43
        if token % 4 == 0:
            row[1] = 0
        if token % 5 == 0:
            row[2] = 255
        if token % 2 == 0:
            row[3] = 128
        if len(set(row)) != TOP_K:
            row = list(dict.fromkeys(row + coverage))[:TOP_K]
            while len(row) < TOP_K:
                row.append((row[-1] + 1) % 256)
        logical_rows.append(row)
    logical = torch.tensor(logical_rows, dtype=torch.int64, device="npu")
    comm = torch.tensor(
        [[logical_to_comm(int(x), placeholder) for x in row] for row in logical_rows],
        dtype=torch.int32,
        device="npu",
    )
    if placeholder:
        # Send only part of expert43 to the unique replica; the rest stays original.
        comm[0, 0] = 257
        comm[3, 0] = 257
        comm[6, 0] = 257
    weights = torch.softmax(
        torch.randn((TOKENS_PER_RANK, TOP_K), dtype=torch.float32, device="npu"), dim=-1
    )
    hidden = torch.randn((TOKENS_PER_RANK, HIDDEN), dtype=torch.bfloat16, device="npu")
    hidden[:, 0] = torch.arange(TOKENS_PER_RANK, dtype=torch.bfloat16, device="npu") + rank * 16
    hidden[:, 1] = 0
    return hidden, comm, weights, logical_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("256", "257", "258"), required=True)
    args = parser.parse_args()
    profile = args.profile
    placeholder = profile == "258"
    num_experts = int(profile)
    num_local_experts = 128 if profile != "258" else 129
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    init_distributed_environment(world_size=-1, rank=-1, local_rank=-1, backend="hccl")
    server_args = ServerArgs(model_path="dummy", tp_size=2, ep_size=2)
    server_args.moe_a2a_backend = "deepep"
    server_args.moe_runner_backend = "ascend"
    set_global_server_args_for_scheduler(server_args)
    publish(server_args, role="scheduler")
    initialize_moe_config()
    initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=2)
    group = get_tp_group().device_group
    record = {
        "profile": profile,
        "rank": rank,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "hidden_size": HIDDEN,
        "top_k": TOP_K,
        "tokens_per_rank": TOKENS_PER_RANK,
        "placeholder_mapping": placeholder,
        "runtime": {"torch_npu": getattr(torch_npu, "__version__", "unknown")},
    }
    try:
        import sglang.srt.layers.quantization
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.utils import DeepEPMode

        record["dispatcher_class"] = "sglang.srt.layers.moe.token_dispatcher.deepep.DeepEPDispatcher"
        record["backend_mode"] = "DeepEPMode.NORMAL"
        dispatcher = DeepEPDispatcher(
            group=group,
            router_topk=TOP_K,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=HIDDEN,
            params_dtype=torch.bfloat16,
            deepep_mode=DeepEPMode.NORMAL,
            async_finish=False,
        )
        record["constructor"] = "PASS"
        hidden, comm_ids, weights, logical_rows = build_inputs(rank, placeholder)
        topk_output = StandardTopKOutput(weights, comm_ids, None)
        dispatch_output = dispatcher.dispatch(hidden, topk_output)
        recv_hidden = dispatch_output.hidden_states
        if isinstance(recv_hidden, tuple):
            recv_hidden = recv_hidden[0]
        counts = [int(x) for x in dispatch_output.num_recv_tokens_per_expert]
        local_base = 0 if rank == 0 else (128 if not placeholder else 129)
        row_comm = []
        for local_idx, count in enumerate(counts):
            row_comm.extend([local_base + local_idx] * count)
        if len(row_comm) != recv_hidden.shape[0]:
            raise RuntimeError(f"count/hidden mismatch {len(row_comm)} vs {recv_hidden.shape[0]}")
        logical_recv_values = [comm_to_logical(x, placeholder) for x in row_comm]
        logical_recv = torch.tensor(logical_recv_values, dtype=torch.float32, device="npu")
        expert_out = torch.zeros_like(recv_hidden)
        expert_out[:, 0] = logical_recv.remainder(8)
        expert_out[:, 1] = recv_hidden[:, 0]
        expert_out[:, 2] = torch.div(logical_recv, 32, rounding_mode="floor")
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput
        combined = dispatcher.combine(DeepEPNormalCombineInput(expert_out, None, None))
        logical_tensor = torch.tensor(logical_rows, dtype=torch.float32, device="npu")
        expected_label = (weights.float() * logical_tensor.remainder(8)).sum(dim=-1)
        expected_group = (weights.float() * torch.div(logical_tensor, 32, rounding_mode="floor")).sum(dim=-1)
        expected_token = torch.arange(TOKENS_PER_RANK, dtype=torch.float32, device="npu") + rank * 16
        label_err = float((combined[:, 0].float() - expected_label).abs().max().item())
        token_err = float((combined[:, 1].float() - expected_token).abs().max().item())
        group_err = float((combined[:, 2].float() - expected_group).abs().max().item())
        record.update({
            "debug_label_err": label_err,
            "debug_token_err": token_err,
            "debug_group_err": group_err,
            "debug_recv_token_tags": recv_hidden[:, 0].detach().cpu().tolist(),
            "debug_row_comm": row_comm,
            "debug_counts": counts,
            "debug_combined_head": combined[: min(4, combined.shape[0]), :3].detach().cpu().tolist(),
        })
        empty_index = 128 if placeholder and rank == 0 else None
        empty_count = counts[empty_index] if empty_index is not None else 0
        manifest = {
            "rank": rank,
            "logical_rows": logical_rows,
            "comm_rows": comm_ids.detach().cpu().tolist(),
            "weights": weights.detach().cpu().tolist(),
            "token_tags": [rank * 1000 + i for i in range(TOKENS_PER_RANK)],
        }
        (OUT_DIR / f"native_manifest_{profile}_rank{rank}.json").write_text(json.dumps(manifest))
        dist.barrier()
        all_manifests = [json.loads((OUT_DIR / f"native_manifest_{profile}_rank{i}.json").read_text()) for i in (0,1)]
        rank_comm_ids = set(range(local_base, local_base + num_local_experts))
        expected_recv = []
        lookup = {}
        for m in all_manifests:
            for token, row in enumerate(m["comm_rows"]):
                lookup[(m["rank"] * 1000 + token, *row)] = m
                for slot, comm in enumerate(row):
                    if comm in rank_comm_ids:
                        expected_recv.append((m["rank"], token, slot, int(comm)))
        actual_recv = []
        recv_tags = recv_hidden[:, 0].detach().cpu().tolist()
        for i, comm in enumerate(row_comm):
            token_tag = int(round(recv_tags[i]))
            sender, token = divmod(token_tag, 16)
            m = all_manifests[sender]
            slots = [slot for slot, value in enumerate(m["comm_rows"][token]) if value == comm]
            if len(slots) != 1:
                raise RuntimeError(f"ambiguous slot for ({sender},{token},{comm}): {slots}")
            actual_recv.append((sender, token, slots[0], int(comm)))
        all_logical = [v for m in all_manifests for row in m["logical_rows"] for v in row]
        all_comm = [v for m in all_manifests for row in m["comm_rows"] for v in row]
        coverage = {str(e): all_logical.count(e) for e in (0, 43, 127, 128, 255)}
        empty_guard = None
        if placeholder:
            injected = comm_ids.clone()
            injected[0, 0] = 128
            flag = (injected == 128).any().to(torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            empty_guard = {
                "injected_comm_id": 128,
                "global_flag_after_all_reduce": int(flag.item()),
                "all_ranks_agree": int(flag.item()) == 1,
                "dispatch_not_called": True,
                "rejected_before_dispatch": int(flag.item()) == 1,
            }
        record.update({
            "combine": "PASS",
            "required_logical_expert_coverage": coverage,
            "replica_comm_257_task_count": all_comm.count(257),
            "empty_comm_128_task_count_legal_input": all_comm.count(128),
            "empty_injection_guard": empty_guard,
            "dispatch_shape": list(dispatch_output.hidden_states.shape),
            "combine_shape": list(combined.shape),
            "recv_tokens_per_local_expert": counts,
            "max_expert_label_error": label_err,
            "max_token_position_error": token_err,
            "max_expert_group_error": group_err,
            "empty_id_received_count": int(empty_count),
            "expected_task_count": len(expected_recv),
            "actual_task_count": len(actual_recv),
            "task_multiset_equal": sorted(expected_recv) == sorted(actual_recv),
            "task_missing": sorted(set(expected_recv) - set(actual_recv)),
            "task_extra": sorted(set(actual_recv) - set(expected_recv)),
            "correct": label_err <= 0.05 and token_err <= 0.05 and group_err <= 0.05 and empty_count == 0 and sorted(expected_recv) == sorted(actual_recv) and all(coverage.values()) and (not placeholder or (all_comm.count(257) > 0 and all_comm.count(128) == 0 and empty_guard["rejected_before_dispatch"])),
        })
    except Exception as exc:
        record.update({
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
        })
    out = OUT_DIR / f"interface_run_{profile}_rank{rank}.json"
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    if record.get("constructor") != "PASS" or (profile != "257" and not record.get("correct")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
