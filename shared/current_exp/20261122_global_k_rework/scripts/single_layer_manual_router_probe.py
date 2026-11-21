#!/usr/bin/env python3
"""Actual layer49 router + manual external-slot routing for one expert.

This is a single-layer routed-expert subchain integration. It uses the actual
GLM router weight, correction bias, SGLang native top-k function, actual W8A8
expert weights, and the actual NPU GroupedMatmul path. It deliberately computes
only expert43's contribution; other routed experts and shared experts are out
of scope, so this is not a full layer output.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

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
    install_state,
    load_raw_state,
    memory_snapshot,
    slot_state_views,
    state_equal,
    state_layout,
)

MODEL = os.environ.get("GLM_MODEL", "/models/glm-q")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/runexp/artifacts"))
LAYER_ID = LAYERS[0]


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def tensor_from_index(index, name):
    with safe_open(str(Path(MODEL) / index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    import sglang.srt.layers.quantization  # break package import cycle
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW8A8Int8MoEMethod,
    )
    from sglang.srt.layers.moe.topk import fused_topk_torch_native

    index = json.load(
        open(Path(MODEL) / "quant_model_weights.safetensors.index.json")
    )["weight_map"]
    gate_name = f"model.layers.{LAYER_ID}.mlp.gate.weight"
    bias_name = f"model.layers.{LAYER_ID}.mlp.gate.e_score_correction_bias"
    gate_cpu = tensor_from_index(index, gate_name).to(torch.float32)
    bias_cpu = tensor_from_index(index, bias_name).to(torch.float32)
    gate = gate_cpu.npu()
    bias = bias_cpu.npu()

    torch.manual_seed(20261122)
    base = torch.randn((64, 6144), dtype=torch.bfloat16, device="npu")
    direction = (gate[EXPERT] / gate[EXPERT].float().norm()).to(torch.bfloat16)
    # Synthetic hidden is deliberately biased along the actual router row for
    # expert43 so enough real router tasks exist for a manual split.
    hidden = (base + 8.0 * direction.unsqueeze(0)).to(torch.bfloat16)
    router_logits = torch.nn.functional.linear(hidden.to(torch.float32), gate, None)
    topk_weights, topk_ids = fused_topk_torch_native(
        hidden_states=hidden,
        gating_output=router_logits,
        topk=8,
        renormalize=True,
        correction_bias=bias,
        scoring_func="sigmoid",
    )
    topk_ids = topk_ids.to(torch.int32)
    torch.npu.synchronize()

    total_tasks = int(topk_ids.numel())
    per_expert = torch.bincount(topk_ids.reshape(-1).long(), minlength=256).cpu()
    task_positions = (topk_ids == EXPERT).nonzero(as_tuple=False)
    task_tokens = task_positions[:, 0]
    task_weights = topk_weights[
        task_positions[:, 0], task_positions[:, 1]
    ]
    task_hidden = hidden.index_select(0, task_tokens)
    task_count = int(task_positions.shape[0])
    if task_count < 2:
        raise RuntimeError(f"real router produced only {task_count} expert43 tasks")

    bank_state, _, offsets, _, slot_hashes = load_raw_state(LAYER_ID)
    bank_layer = ExpertLayer(bank_state)
    method = NPUW8A8Int8MoEMethod()
    matmul = GroupedMatmul()
    for name in PARTS:
        attr = LAYER_ATTRS[name]
        getattr(bank_layer, attr).data = getattr(bank_layer, attr).data.npu()
    method.process_weights_after_loading(bank_layer, "w13")
    method.process_weights_after_loading(bank_layer, "w2")
    bank_chain = RuntimeChain(method, matmul, bank_layer, LAYER_ID)
    external_state = external_state_from_bank(bank_layer)
    torch.npu.synchronize()
    layout, state_bytes = state_layout(external_state)

    pool = GlobalSlotPool(
        [1],
        state_bytes,
        alloc,
        name="manual-router-external-slot",
        synchronize=torch.npu.synchronize,
    )
    lease = pool.acquire(
        request_id="layer49-router-expert43", layer_id=LAYER_ID, expert_id=EXPERT
    )
    slot = pool.slot_buffer(lease, allow_reserved=True)
    install_state(pool, lease, slot, external_state, layout)
    with pool.use(lease) as active_slot:
        state_fields = state_equal(active_slot, external_state, layout)
        ext_layer = type("ExternalExpertLayer", (), {})()
        views = slot_state_views(active_slot, layout)
        for name in PARTS:
            setattr(ext_layer, LAYER_ATTRS[name], views[name])
        ext_chain = RuntimeChain(method, matmul, ext_layer, LAYER_ID)

        reference_output = bank_chain(task_hidden)
        torch.npu.synchronize()
        # Alternate routed tasks between the main bank and one external replica.
        bank_sel = torch.arange(0, task_count, 2, dtype=torch.int64, device="npu")
        ext_sel = torch.arange(1, task_count, 2, dtype=torch.int64, device="npu")
        refilled = torch.empty_like(reference_output)
        bank_out = bank_chain(task_hidden.index_select(0, bank_sel))
        ext_out = ext_chain(task_hidden.index_select(0, ext_sel))
        refilled.index_copy_(0, bank_sel, bank_out)
        refilled.index_copy_(0, ext_sel, ext_out)
        torch.npu.synchronize()
        expert_diff = float(
            (reference_output.float() - refilled.float()).abs().max().item()
        )
        weighted_reference = reference_output.float() * task_weights.float().unsqueeze(1)
        weighted_refilled = refilled.float() * task_weights.float().unsqueeze(1)
        weighted_diff = float(
            (weighted_reference - weighted_refilled).abs().max().item()
        )
        memory = memory_snapshot()

    pool.release(lease)
    final_stats = pool.stats()
    router_payload = {
        "gate_sha256": hashlib.sha256(
            gate_cpu.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "bias_sha256": hashlib.sha256(
            bias_cpu.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "token_count": int(hidden.shape[0]),
        "top_k": 8,
        "total_tasks": total_tasks,
        "expert43_tasks": task_count,
        "expert_task_histogram_max": int(per_expert.max()),
        "expert_task_histogram_argmax": int(per_expert.argmax()),
        "task_ids_sha256": hashlib.sha256(
            topk_ids.detach().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "task_weights_sha256": hashlib.sha256(
            topk_weights.detach().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "input_kind": "synthetic hidden biased along actual expert43 router row",
    }
    result = {
        "status": "PASS" if expert_diff == 0.0 and weighted_diff == 0.0 else "FAIL",
        "scope": (
            "actual layer49 router + actual W8A8 expert43 subchain; routed "
            "expert contribution only, not full layer/service output"
        ),
        "layer_id": LAYER_ID,
        "logical_expert": EXPERT,
        "router": router_payload,
        "routing_contract": {
            "router_weight": gate_name,
            "correction_bias": bias_name,
            "topk_function": "sglang.srt.layers.moe.topk.fused_topk_torch_native",
            "scoring_func": "sigmoid",
            "renormalize": True,
        },
        "task_identity": {
            "fields": ["layer_id", "token_position", "topk_position", "logical_expert", "router_weight"],
            "total_router_tasks": total_tasks,
            "logical_expert_tasks": task_count,
            "bank_tasks": int(bank_sel.numel()),
            "external_tasks": int(ext_sel.numel()),
            "task_count_conserved": int(bank_sel.numel() + ext_sel.numel()) == task_count,
            "all_tasks_for_same_logical_expert": task_count
            == int((topk_ids == EXPERT).sum()),
        },
        "correctness": {
            "state_fields_equal": state_fields,
            "expert_output_max_abs": expert_diff,
            "weighted_contribution_max_abs": weighted_diff,
        },
        "offsets_loaded": offsets,
        "slot_state_sha256": slot_hashes,
        "memory": memory,
        "final_pool_stats": final_stats,
        "not_done": [
            "other 255 routed experts",
            "shared expert",
            "full top-k combine into one layer output",
            "dispatch/combine framework integration",
        ],
    }
    (OUT_DIR / "single_layer_manual_router_result.json").write_text(
        json.dumps(result, indent=2, default=json_default)
    )
    print("SINGLE_LAYER_MANUAL_ROUTER " + json.dumps(result, default=json_default))
    torch.npu.synchronize()
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
