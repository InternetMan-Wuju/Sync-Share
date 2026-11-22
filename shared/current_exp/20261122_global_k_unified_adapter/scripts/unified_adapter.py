#!/usr/bin/env python3
"""Single execution path for native 258 placeholder K=1 prefill."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist

from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import (
    LAYER_ATTRS, PARTS, ExpertLayer, RuntimeChain, alloc,
    external_state_from_bank, load_raw_state, memory_snapshot,
    slot_state_views, state_layout,
)

EMPTY_COMM = 128
REPLICA_COMM = 257
BASE_COMM = {0: 0, 1: 129}
PADDING_TAG_BASE = 200
TOP_K = 8


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


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class UnifiedK1Adapter:
    def __init__(self, *, rank, group, dispatcher, bank_chains, main_experts_by_rank, layers, pool, layout, state_bytes, flat, method, matmul):
        self.rank = rank
        self.group = group
        self.dispatcher = dispatcher
        self.bank_chains = bank_chains
        self.main_experts_by_rank = main_experts_by_rank
        self.layers = tuple(layers)
        self.pool = pool
        self.layout = layout
        self.state_bytes = state_bytes
        self.flat = flat
        self.method = method
        self.matmul = matmul
        self.lease = None
        self.replica_chain = None
        self.dispatch_calls = 0
        self.slot_install_events = []
        self.memory_events = [{"stage": "adapter_init", "rank": rank, "memory": memory_snapshot()}]

    def _sync(self):
        torch.npu.synchronize()

    def _shared_peer_flag(self, value):
        t = torch.tensor([int(value)], dtype=torch.int32, device="npu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.group)
        return int(t.item())

    def install(self, layer, out_dir: Path, tag: str):
        started = time.perf_counter_ns()
        info = {"rank": self.rank, "layer": layer, "tag": tag}
        if self.rank == 0:
            source = external_state_from_bank(self.bank_chains[(layer, 43)].layer)
            for name, off, nbytes, _, _ in self.layout:
                self.flat[off:off+nbytes].copy_(source[name].view(torch.uint8).reshape(-1))
            self._sync()
            dist.send(self.flat, dst=1, group=self.group)
            info["source_hash"] = sha(self.flat)
        else:
            if self.lease is not None:
                self.pool.release(self.lease)
            self.lease = self.pool.acquire(request_id=tag, layer_id=layer, expert_id=43)
            view = self.pool.slot_buffer(self.lease, allow_reserved=True)
            dist.recv(view, src=0, group=self.group)
            self.pool.mark_ready(self.lease)
            with self.pool.use(self.lease) as active:
                info.update({
                    "generation": self.lease.generation,
                    "ready": True,
                    "owner": self.lease.owner_rank,
                    "local_slot": self.lease.local_slot,
                    "layer": self.lease.layer_id,
                    "expert": self.lease.expert_id,
                    "target_hash": sha(active),
                    "slot_storage_ptr": int(active.untyped_storage().data_ptr()),
                    "slot_storage_bytes": int(active.untyped_storage().nbytes()),
                })
                ext_layer = type("ReplicaLayer", (), {})()
                views = slot_state_views(active, self.layout)
                slot_formats = []
                for part in PARTS:
                    t = views[part]
                    try: fs = torch_npu.get_npu_format(t)
                    except Exception as ex: fs = f"ERROR:{type(ex).__name__}:{ex}"
                    try: fc = int(torch.ops.npu.get_npu_format(t))
                    except Exception as ex: fc = f"ERROR:{type(ex).__name__}:{ex}"
                    st = t.untyped_storage()
                    slot_formats.append({"name": part, "shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype), "format_string": fs, "format_code": fc, "storage_ptr": int(st.data_ptr()), "storage_bytes": int(st.nbytes()), "data_ptr": int(t.data_ptr()), "slot_storage_ptr": int(active.untyped_storage().data_ptr()), "aliases_slot_storage": int(st.data_ptr()) == int(active.untyped_storage().data_ptr())})
                    setattr(ext_layer, LAYER_ATTRS[part], t)
                info["slot_formats"] = slot_formats
                self.replica_chain = RuntimeChain(self.method, self.matmul, ext_layer, layer)
        dist.barrier(group=self.group)
        info_path = out_dir / f"{tag}_install_rank{self.rank}.json"
        info_path.write_text(json.dumps(info, indent=2))
        dist.barrier(group=self.group)
        other = json.loads((out_dir / f"{tag}_install_rank{1-self.rank}.json").read_text())
        if self.rank == 0:
            info["peer_target_hash"] = other.get("target_hash")
            info["hash_equal"] = info.get("source_hash") == other.get("target_hash")
        else:
            info["source_rank_hash"] = other.get("source_hash")
            info["hash_equal"] = other.get("source_hash") == info.get("target_hash")
        info["install_ms"] = (time.perf_counter_ns() - started) / 1e6
        self.slot_install_events.append(info)
        self.memory_events.append({"stage": f"installed_l{layer}", "rank": self.rank, "memory": memory_snapshot()})
        return info

    def slot_info(self):
        if self.rank != 1 or self.lease is None:
            return None if self.lease is None else {
                "ready": True, "layer": self.lease.layer_id, "expert": self.lease.expert_id,
                "generation": self.lease.generation, "owner": self.lease.owner_rank,
                "local_slot": self.lease.local_slot,
            }
        return {
            "ready": True, "layer": self.lease.layer_id, "expert": self.lease.expert_id,
            "generation": self.lease.generation, "owner": self.lease.owner_rank,
            "local_slot": self.lease.local_slot,
        }

    def execute(self, *, case_id, layer, hidden, logical_rows, weights, mode="none", slot_info=None, inject_invalid_comm=False, expected_generation=None, trace=None):
        """One and only execution entry.

        mode: none/partial/all for expert43 -> comm257 replacement.
        slot_info: must describe an already-ready exact lease when replica tasks exist.
        inject_invalid_comm: after mapping, replace one legal task with empty comm128.
        """
        dispatch_before = self.dispatch_calls
        reasons = []
        rows = int(hidden.shape[0]) if hidden is not None else 0
        if rows != len(logical_rows) or rows != len(weights):
            reasons.append("token_count_mismatch")
        for row in logical_rows:
            if len(row) != TOP_K or len(set(row)) != TOP_K or any(not (0 <= int(e) < 256) for e in row):
                reasons.append("invalid_top8")
                break
        # Establish global legal-token count and replica demand before any dispatch.
        local_tokens = torch.tensor([rows], dtype=torch.int32, device="npu")
        dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM, group=self.group)
        global_tokens = int(local_tokens.item())
        comm_rows = []
        for token, row in enumerate(logical_rows):
            comm_row = [logical_to_comm(int(e)) for e in row]
            if mode == "all":
                comm_row = [REPLICA_COMM if e == 43 else c for e, c in zip(row, comm_row)]
            elif mode == "partial":
                comm_row = [REPLICA_COMM if (e == 43 and token % 2 == 0) else c for e, c in zip(row, comm_row)]
            comm_rows.append(comm_row)
        if inject_invalid_comm:
            if not comm_rows:
                reasons.append("cannot_inject_without_token")
            else:
                comm_rows[0][0] = EMPTY_COMM
        replica_local = any(REPLICA_COMM in row for row in comm_rows)
        replica_flag = torch.tensor([int(replica_local)], dtype=torch.int32, device="npu")
        dist.all_reduce(replica_flag, op=dist.ReduceOp.MAX, group=self.group)
        replica_required = int(replica_flag.item()) == 1
        if replica_required:
            if slot_info is None or not slot_info.get("ready", False):
                reasons.append("replica_not_ready")
            else:
                if int(slot_info.get("layer", -1)) != int(layer):
                    reasons.append("replica_wrong_layer")
                if int(slot_info.get("expert", -1)) != 43:
                    reasons.append("replica_wrong_expert")
                if expected_generation is not None and int(slot_info.get("generation", -1)) != int(expected_generation):
                    reasons.append("replica_stale_generation")
        for row in comm_rows:
            for comm in row:
                if comm == EMPTY_COMM:
                    reasons.append("empty_comm_without_padding")
                elif not (0 <= comm < 258):
                    reasons.append("comm_out_of_range")
        # Cross-rank decision. All ranks participate even when local validation failed.
        bad = self._shared_peer_flag(bool(reasons))
        if bad:
            return {
                "case_id": case_id, "rank": self.rank, "status": "REJECTED",
                "reasons": sorted(set(reasons)), "dispatch_called": False,
                "dispatch_delta": self.dispatch_calls - dispatch_before,
                "global_tokens": global_tokens, "replica_required": replica_required,
            }
        if global_tokens == 0:
            self.memory_events.append({"stage": f"{case_id}_early_exit", "rank": self.rank, "memory": memory_snapshot()})
            return {
                "case_id": case_id, "rank": self.rank, "status": "PASS_EMPTY",
                "reasons": [], "dispatch_called": False,
                "dispatch_delta": self.dispatch_calls - dispatch_before,
                "global_tokens": 0, "output_shape": [0, hidden.shape[1]],
                "padding_sent": 0, "replica_tasks": 0, "empty_legal_tasks": 0,
            }
        padding_sent = 0
        if rows == 0:
            hidden = torch.zeros((1, hidden.shape[1]), dtype=torch.bfloat16, device="npu")
            hidden[0, 0] = PADDING_TAG_BASE + self.rank
            logical_rows = [tuple()]
            weights = [[0.0] * TOP_K]
            comm_rows = [[EMPTY_COMM] * TOP_K]
            padding_sent = 1
            rows = 1
        else:
            hidden = hidden.clone()
            hidden[:, 0] = torch.arange(rows, dtype=torch.bfloat16, device="npu") + self.rank * 16
            hidden[:, 1] = 0
        weight_t = torch.tensor(weights, dtype=torch.float32, device="npu")
        comm_t = torch.tensor(comm_rows, dtype=torch.int32, device="npu")
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput
        self.dispatch_calls += 1
        dispatch_out = self.dispatcher.dispatch(hidden, StandardTopKOutput(weight_t, comm_t, None))
        recv_hidden = dispatch_out.hidden_states[0] if isinstance(dispatch_out.hidden_states, tuple) else dispatch_out.hidden_states
        counts = [int(x) for x in dispatch_out.num_recv_tokens_per_expert]
        row_comm = []
        for i, c in enumerate(counts):
            row_comm.extend([BASE_COMM[self.rank] + i] * c)
        tags = recv_hidden[:, 0].detach().cpu().tolist()
        row_padding = [float(t) >= PADDING_TAG_BASE for t in tags]
        expert_out = torch.zeros_like(recv_hidden)
        pos = 0
        padding_expert_calls = 0
        padding_rows_skipped = 0
        replica_tasks = 0
        for local_idx, count in enumerate(counts):
            if count <= 0:
                continue
            comm = BASE_COMM[self.rank] + local_idx
            p = row_padding[pos:pos+count]
            if comm == EMPTY_COMM:
                if not all(p):
                    reasons.append("empty_comm_nonpadding_row")
                padding_rows_skipped += count
            elif comm == REPLICA_COMM:
                replica_tasks += count
                if self.rank != 1 or self.lease is None or self.replica_chain is None:
                    reasons.append("replica_execution_wrong_rank_or_slot")
                else:
                    with self.pool.use(self.lease):
                        expert_out[pos:pos+count].copy_(self.replica_chain(recv_hidden[pos:pos+count]))
            else:
                logical = comm_to_logical(comm)
                key = (layer, logical)
                if key not in self.bank_chains:
                    reasons.append(f"nonlocal_logical_{logical}")
                else:
                    expert_out[pos:pos+count].copy_(self.bank_chains[key](recv_hidden[pos:pos+count]))
            pos += count
        bad_after = self._shared_peer_flag(bool(reasons))
        if bad_after:
            return {
                "case_id": case_id, "rank": self.rank, "status": "FAIL_AFTER_DISPATCH",
                "reasons": sorted(set(reasons)), "dispatch_called": True,
                "dispatch_delta": self.dispatch_calls - dispatch_before,
                "global_tokens": global_tokens, "replica_required": replica_required,
            }
        combined = self.dispatcher.combine(DeepEPNormalCombineInput(expert_out, None, None))
        valid_rows = combined[: len(logical_rows) - padding_sent]
        # Task identity: explicit padding tags are excluded; legal comm128 is illegal.
        actual = []
        for i, comm in enumerate(row_comm):
            tag = int(round(tags[i]))
            if tag >= PADDING_TAG_BASE:
                continue
            sender, token = divmod(tag, 16)
            actual.append((sender, token, comm))
        empty_legal = sum(1 for (_, _, c) in actual if c == EMPTY_COMM)
        returned = {
            "case_id": case_id, "rank": self.rank, "status": "PASS",
            "reasons": [], "dispatch_called": True,
            "dispatch_delta": self.dispatch_calls - dispatch_before,
            "global_tokens": global_tokens, "replica_required": replica_required,
            "output_shape": list(valid_rows.shape), "padding_sent": padding_sent,
            "padding_received": sum(row_padding), "padding_expert_calls": padding_expert_calls,
            "padding_rows_skipped": padding_rows_skipped,
            "replica_tasks": replica_tasks, "empty_legal_tasks": empty_legal,
            "task_count": len(actual), "actual_task_vector": actual,
            "counts": counts, "memory": memory_snapshot(),
        }
        self.memory_events.append({"stage": f"{case_id}_complete", "rank": self.rank, "memory": memory_snapshot()})
        return returned, valid_rows

    def release_slot(self):
        if self.rank == 1 and self.lease is not None:
            self.pool.release(self.lease)
            self.lease = None
            self.replica_chain = None

    def format_snapshot(self):
        def desc(t, label, module_layer=None, expert=None):
            try: fmt_str = torch_npu.get_npu_format(t)
            except Exception as exc: fmt_str = f"ERROR:{type(exc).__name__}:{exc}"
            try: fmt_code = int(torch.ops.npu.get_npu_format(t))
            except Exception as exc: fmt_code = f"ERROR:{type(exc).__name__}:{exc}"
            st = t.untyped_storage()
            return {"label":label,"layer":module_layer,"expert":expert,"shape":list(t.shape),"stride":list(t.stride()),"dtype":str(t.dtype),"format_string":fmt_str,"format_code":fmt_code,"data_ptr":int(t.data_ptr()),"storage_ptr":int(st.data_ptr()),"storage_bytes":int(st.nbytes()),"storage_offset":int(t.storage_offset())}
        rows=[]
        for (layer, expert), chain in self.bank_chains.items():
            module=chain.layer
            for attr in LAYER_ATTRS.values():
                t=getattr(module,attr,None)
                if t is not None: rows.append(desc(t,f"bank/{attr}",layer,expert))
        return rows
