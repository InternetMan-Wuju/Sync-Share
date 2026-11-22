#!/usr/bin/env python3
"""258 native communication + finite real W8A8 experts + one K=1 slot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist

from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import (
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
LAYERS = (49, 50)
TOKENS = 8
TOP_K = 8
HIDDEN = 6144
ACTIVE = (0, 1, 43, 127, 128, 129, 130, 255)
MAIN_EXPERTS = {0: (0, 1, 43, 127), 1: (128, 129, 130, 255)}
BASE_COMM = {0: 0, 1: 129}
EMPTY_COMM = 128
REPLICA_COMM = 257
ATOL = 0.02
RTOL = 0.02


def logical_to_comm(e):
    return e if e < 128 else e + 1


def comm_to_logical(c):
    if 0 <= c < 128:
        return c
    if 129 <= c <= 256:
        return c - 1
    if c == REPLICA_COMM:
        return 43
    raise ValueError(f"invalid comm id {c}")


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_case(layer, rank, mode):
    seed = 21000 + layer * 100 + rank
    rng = random.Random(seed)
    logical_rows = []
    weight_rows = []
    for token in range(TOKENS):
        row = list(ACTIVE)
        rng.shuffle(row)
        logical_rows.append(row)
        w = [rng.random() + 1e-3 for _ in range(TOP_K)]
        s = sum(w)
        weight_rows.append([x / s for x in w])
    comm_rows = [[logical_to_comm(x) for x in row] for row in logical_rows]
    if mode in ("partial", "all"):
        for token, row in enumerate(comm_rows):
            for slot, logical in enumerate(logical_rows[token]):
                if logical == 43 and (mode == "all" or token % 2 == 0):
                    row[slot] = REPLICA_COMM
    torch.manual_seed(seed + 7)
    hidden = torch.randn((TOKENS, HIDDEN), dtype=torch.bfloat16, device="npu")
    hidden[:, 0] = torch.arange(TOKENS, dtype=torch.bfloat16, device="npu") + rank * 16
    hidden[:, 1] = 0
    return hidden, logical_rows, comm_rows, weight_rows


def storage_entries(modules):
    entries = []
    unique = {}
    for (layer, expert), module in modules.items():
        for attr in LAYER_ATTRS.values():
            t = getattr(module, attr, None)
            if t is None:
                continue
            st = t.untyped_storage()
            item = {
                "layer": layer, "expert": expert, "attr": attr,
                "shape": list(t.shape), "dtype": str(t.dtype),
                "data_ptr": int(t.data_ptr()), "storage_ptr": int(st.data_ptr()),
                "storage_nbytes": int(st.nbytes()), "offset": int(t.storage_offset()),
            }
            entries.append(item)
            unique[item["storage_ptr"]] = item["storage_nbytes"]
    return entries, unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="real258")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel, get_tp_group, initialize_model_parallel
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher, DeepEPNormalCombineInput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import DeepEPMode, initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    init_distributed_environment(world_size=-1, rank=-1, local_rank=-1, backend="hccl")
    server_args = ServerArgs(model_path="dummy", tp_size=2, ep_size=2)
    server_args.moe_a2a_backend = "deepep"
    server_args.moe_runner_backend = "ascend"
    set_global_server_args_for_scheduler(server_args); publish(server_args, role="scheduler")
    initialize_moe_config(); initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=2)
    group = get_tp_group().device_group
    dispatcher = DeepEPDispatcher(group=group, router_topk=TOP_K, num_experts=258, num_local_experts=129, hidden_size=HIDDEN, params_dtype=torch.bfloat16, deepep_mode=DeepEPMode.NORMAL, async_finish=False)

    import sglang.srt.layers.quantization
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    method = NPUW8A8Int8MoEMethod(); matmul = GroupedMatmul()
    result = {"rank": rank, "tag": args.tag, "layers": list(LAYERS), "active_logical_experts": list(ACTIVE), "rank_main_experts": list(MAIN_EXPERTS[rank]), "memory_before": memory_snapshot(), "native_runs": [], "install_events": [], "reuse_runs": []}

    bank = {}
    raw_hashes = {}
    for layer in LAYERS:
        for expert in MAIN_EXPERTS[rank]:
            bank_state, _, offsets, _, slot_hashes = load_raw_state(layer, expert)
            module = ExpertLayer(bank_state)
            for part in PARTS:
                attr = LAYER_ATTRS[part]
                getattr(module, attr).data = getattr(module, attr).data.npu()
            method.process_weights_after_loading(module, "w13")
            method.process_weights_after_loading(module, "w2")
            bank[(layer, expert)] = module
            if expert == 43:
                raw_hashes[layer] = slot_hashes
    torch.npu.synchronize()
    chains = {(layer, e): RuntimeChain(method, matmul, bank[(layer, e)], layer) for layer, e in bank}
    result["memory_after_bank"] = memory_snapshot()
    entries, unique = storage_entries(bank)
    result["main_storage"] = {"entries": entries, "unique_storage_count": len(unique), "unique_bytes": sum(unique.values()), "full_128_per_rank": False}

    meta_path = OUT_DIR / f"{args.tag}_meta.json"
    if rank == 0:
        sample = external_state_from_bank(bank[(49, 43)])
        layout, state_bytes = state_layout(sample)
        meta_path.write_text(json.dumps({"state_bytes": state_bytes, "layout": [
            {"name": n, "offset": o, "bytes": b, "dtype": str(d), "shape": list(s)} for n,o,b,d,s in layout
        ], "slot_hashes": raw_hashes}, indent=2))
        flat = torch.empty(state_bytes, dtype=torch.uint8, device="npu")
    dist.barrier()
    meta = json.loads(meta_path.read_text())
    state_bytes = meta["state_bytes"]
    if rank == 1:
        dtype_map = {"torch.int8": torch.int8, "torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}
        layout = [(x["name"],x["offset"],x["bytes"],dtype_map[x["dtype"]],tuple(x["shape"])) for x in meta["layout"]]
        pool = GlobalSlotPool([1], state_bytes, alloc, name="real-native-k1", synchronize=torch.npu.synchronize)
        lease = None
        replica_chain = None
    else:
        pool = None; lease = None; replica_chain = None

    def install(layer):
        nonlocal lease, replica_chain
        started = time.perf_counter_ns()
        info = {"rank": rank, "layer": layer, "install_id": f"{args.tag}-{layer}-{time.time_ns()}"}
        if rank == 0:
            state = external_state_from_bank(bank[(layer, 43)])
            for name, off, nbytes, _, _ in layout:
                flat[off:off+nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
            torch.npu.synchronize()
            dist.send(flat, dst=1)
            info["source_hash"] = sha_tensor(flat)
        else:
            if lease is not None:
                pool.release(lease)
            lease = pool.acquire(request_id=info["install_id"], layer_id=layer, expert_id=43)
            slot = pool.slot_buffer(lease, allow_reserved=True)
            dist.recv(slot, src=0)
            pool.mark_ready(lease)
            with pool.use(lease) as active:
                info.update({"generation": lease.generation, "slot_owner": lease.owner_rank, "slot_local": lease.local_slot, "slot_layer": lease.layer_id, "slot_expert": lease.expert_id, "slot_storage_ptr": int(active.untyped_storage().data_ptr()), "slot_storage_nbytes": int(active.untyped_storage().nbytes()), "target_hash": sha_tensor(active)})
                ext_layer = type("ReplicaLayer", (), {})()
                views = slot_state_views(active, layout)
                for part in PARTS:
                    setattr(ext_layer, LAYER_ATTRS[part], views[part])
                replica_chain = RuntimeChain(method, matmul, ext_layer, layer)
        dist.barrier()
        info["install_ms"] = (time.perf_counter_ns() - started) / 1e6
        result["install_events"].append(info)
        return info

    def reference_for(layer, hidden, logical_rows_by_rank, weights_by_rank):
        global_hidden = torch.empty((2 * TOKENS, HIDDEN), dtype=torch.bfloat16, device="npu")
        dist.all_gather_into_tensor(global_hidden, hidden, group=group)
        dist.barrier()
        partial = torch.zeros((2 * TOKENS, HIDDEN), dtype=torch.float32, device="npu")
        for expert in MAIN_EXPERTS[rank]:
            indices, weights = [], []
            for sender in (0, 1):
                for token, row in enumerate(logical_rows_by_rank[sender]):
                    if expert in row:
                        idx = sender * TOKENS + token
                        slot = row.index(expert)
                        indices.append(idx)
                        weights.append(weights_by_rank[sender][token][slot])
            idx_t = torch.tensor(indices, dtype=torch.int64, device="npu")
            w_t = torch.tensor(weights, dtype=torch.float32, device="npu").unsqueeze(1)
            out = chains[(layer, expert)](global_hidden.index_select(0, idx_t))
            partial.index_add_(0, idx_t, out.float() * w_t)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial[rank * TOKENS : (rank + 1) * TOKENS].contiguous()

    def actual_tasks(hidden_rows, counts, local_base, manifests):
        row_comm = []
        for i, c in enumerate(counts):
            row_comm.extend([local_base + i] * int(c))
        tags = hidden_rows[:, 0].detach().cpu().tolist()
        out = []
        for i, comm in enumerate(row_comm):
            tag = int(round(tags[i])); sender, token = divmod(tag, 16)
            slots = [s for s, v in enumerate(manifests[sender]["comm_rows"][token]) if v == comm]
            if len(slots) != 1:
                raise RuntimeError(f"task identity slot ambiguity {(sender,token,comm,slots)}")
            out.append((sender, token, slots[0], int(comm)))
        return out, row_comm

    def run_native(layer, hidden, comm_rows, weight_rows, manifests, mode):
        topk = StandardTopKOutput(torch.tensor(weight_rows, dtype=torch.float32, device="npu"), torch.tensor(comm_rows, dtype=torch.int32, device="npu"), None)
        dispatch_out = dispatcher.dispatch(hidden, topk)
        recv_hidden = dispatch_out.hidden_states[0] if isinstance(dispatch_out.hidden_states, tuple) else dispatch_out.hidden_states
        counts = [int(x) for x in dispatch_out.num_recv_tokens_per_expert]
        expected, _ = actual_tasks(recv_hidden, counts, BASE_COMM[rank], manifests)
        expert_out = torch.zeros_like(recv_hidden)
        pos = 0
        replica_tasks = 0
        for local_idx, count in enumerate(counts):
            if count <= 0:
                continue
            comm = BASE_COMM[rank] + local_idx
            rows = recv_hidden[pos : pos + count]
            if comm == EMPTY_COMM:
                raise RuntimeError("empty comm id received tokens")
            if comm == REPLICA_COMM:
                replica_tasks += count
                if rank != 1 or replica_chain is None or lease is None:
                    raise RuntimeError("replica task reached rank without ready slot")
                with pool.use(lease):
                    expert_out[pos : pos + count].copy_(replica_chain(rows))
            else:
                logical = comm_to_logical(comm)
                key = (layer, logical)
                if key not in chains:
                    raise RuntimeError(f"rank {rank} received nonlocal logical expert {logical}")
                expert_out[pos : pos + count].copy_(chains[key](rows))
            pos += count
        if pos != recv_hidden.shape[0]:
            raise RuntimeError(f"count sum mismatch {pos} vs {recv_hidden.shape[0]}")
        output = dispatcher.combine(DeepEPNormalCombineInput(expert_out, None, None))
        return output, counts, expected, replica_tasks

    # Build legal tasks and independent references for both layers.
    inputs = {}
    manifests_all = {}
    references = {}
    if rank == 0:
        flat = torch.empty(state_bytes, dtype=torch.uint8, device="npu")
    for layer in LAYERS:
        hidden, logical_rows, comm_rows, weight_rows = make_case(layer, rank, "all")
        manifest = {"rank": rank, "logical_rows": logical_rows, "comm_rows": comm_rows, "weights": weight_rows}
        path = OUT_DIR / f"{args.tag}_manifest_l{layer}_rank{rank}.json"
        path.write_text(json.dumps(manifest))
        dist.barrier()
        manifests = [json.loads((OUT_DIR / f"{args.tag}_manifest_l{layer}_rank{i}.json").read_text()) for i in (0,1)]
        inputs[layer] = (hidden, comm_rows, weight_rows)
        manifests_all[layer] = manifests
        ref = reference_for(layer, hidden, [m["logical_rows"] for m in manifests], [m["weights"] for m in manifests])
        references[layer] = ref
        torch.npu.synchronize()
        result.setdefault("memory_after_reference", {})[str(layer)] = memory_snapshot()

    # P2 layer49 correctness: no replica, partial replica, all replica.
    install(LAYERS[0])
    for mode in ("none", "partial", "all"):
        hidden, base_comm, weights = inputs[LAYERS[0]]
        _, logical_rows, _, _ = make_case(LAYERS[0], rank, mode)
        comm_rows = [[logical_to_comm(x) for x in row] for row in logical_rows]
        for token in range(TOKENS):
            for slot, logical in enumerate(logical_rows[token]):
                if logical == 43 and (mode == "all" or (mode == "partial" and token % 2 == 0)):
                    comm_rows[token][slot] = REPLICA_COMM
        manifests = [{"rank": i, "logical_rows": json.loads((OUT_DIR / f"{args.tag}_manifest_l{LAYERS[0]}_rank{i}.json").read_text())["logical_rows"], "comm_rows": json.loads((OUT_DIR / f"{args.tag}_manifest_l{LAYERS[0]}_rank{i}.json").read_text())["comm_rows"]} for i in (0,1)]
        # For partial/all, update local manifest comm ids for task identity.
        local_manifest = {"rank": rank, "logical_rows": logical_rows, "comm_rows": comm_rows}
        (OUT_DIR / f"{args.tag}_manifest_case_{mode}_rank{rank}.json").write_text(json.dumps(local_manifest))
        dist.barrier()
        manifests = [json.loads((OUT_DIR / f"{args.tag}_manifest_case_{mode}_rank{i}.json").read_text()) for i in (0,1)]
        out, counts, actual, replica_tasks = run_native(LAYERS[0], hidden, comm_rows, weights, manifests, mode)
        dist.barrier()
        ref = references[LAYERS[0]]
        err = float((out.float() - ref).abs().max().item())
        rel = err / max(float(ref.abs().max().item()), 1e-9)
        expected_total = sum(sum(1 for c in m["comm_rows"][t] if c in set(range(BASE_COMM[rank], BASE_COMM[rank]+129))) for m in manifests for t in range(TOKENS))
        result["native_runs"].append({
            "layer": LAYERS[0], "mode": mode, "shape": list(out.shape), "counts": counts,
            "replica_tasks": replica_tasks, "empty_tasks": counts[EMPTY_COMM] if rank == 0 and EMPTY_COMM < len(counts) else 0,
            "max_abs_vs_independent_reference": err, "relative_error": rel,
            "within_tolerance": err <= ATOL and rel <= RTOL,
            "expected_local_tasks": expected_total, "actual_local_tasks": len(actual),
            "task_multiset_equal": sorted(actual) == sorted([(m["rank"], t, s, c) for m in manifests for t,row in enumerate(m["comm_rows"]) for s,c in enumerate(row) if c in set(range(BASE_COMM[rank], BASE_COMM[rank]+129))]),
            "memory": memory_snapshot(),
        })
        dist.barrier()

    # P3: three 49->50->49 cycles, all expert43 tasks to the same slot.
    for round_id in range(3):
        for layer in (50, 49):
            inst = install(layer)
            hidden, _, weights = inputs[layer]
            _, logical_rows, _, _ = make_case(layer, rank, "all")
            comm_rows = [[REPLICA_COMM if x == 43 else logical_to_comm(x) for x in row] for row in logical_rows]
            local_manifest = {"rank": rank, "logical_rows": logical_rows, "comm_rows": comm_rows}
            (OUT_DIR / f"{args.tag}_manifest_reuse_{round_id}_{layer}_rank{rank}.json").write_text(json.dumps(local_manifest))
            dist.barrier()
            manifests = [json.loads((OUT_DIR / f"{args.tag}_manifest_reuse_{round_id}_{layer}_rank{i}.json").read_text()) for i in (0,1)]
            out, counts, actual, replica_tasks = run_native(layer, hidden, comm_rows, weights, manifests, "all")
            dist.barrier()
            ref = references[layer]
            err = float((out.float() - ref).abs().max().item())
            result["reuse_runs"].append({
                "round": round_id, "layer": layer, "generation": inst.get("generation"),
                "install_hash": inst.get("target_hash", inst.get("source_hash")),
                "slot_storage_ptr": inst.get("slot_storage_ptr"), "slot_storage_nbytes": inst.get("slot_storage_nbytes"),
                "counts": counts, "replica_tasks": replica_tasks,
                "empty_tasks": counts[EMPTY_COMM] if rank == 0 and EMPTY_COMM < len(counts) else 0,
                "max_abs_vs_independent_reference": err,
                "within_tolerance": err <= ATOL,
                "memory": memory_snapshot(),
            })
            dist.barrier()

    if rank == 1:
        # Slot inventory and final safe release.
        slot_view = pool.slot_buffer(lease)
        result["slot_storage"] = {"count": 1, "bytes": int(slot_view.untyped_storage().nbytes()), "storage_ptr": int(slot_view.untyped_storage().data_ptr()), "generation": lease.generation, "owner": lease.owner_rank, "local_slot": lease.local_slot}
        pool.release(lease)
    else:
        result["slot_storage"] = {"count": 0, "bytes": 0, "owner": 1, "note": "the single persistent slot exists only on rank1"}
    result["memory_final"] = memory_snapshot()
    out_path = OUT_DIR / f"{args.tag}_rank{rank}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"REAL_NATIVE_{args.tag}_RANK{rank} " + json.dumps(result, default=lambda x: x.item() if hasattr(x,'item') else str(x)))
    dist.barrier()
    destroy_model_parallel(); destroy_distributed_environment()
    ok = all(x["within_tolerance"] and x["task_multiset_equal"] and x["empty_tasks"] == 0 for x in result["native_runs"]) and all(x["within_tolerance"] and x["empty_tasks"] == 0 for x in result["reuse_runs"])
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
