#!/usr/bin/env python3
"""S1 single-NPU validation for W8A8 expert copies, split dispatch and merge.

The script uses the real audited layer-49 expert tensors and production-compatible
Ascend primitives (dynamic quant, grouped matmul, SwiGLU).  It does not launch the
full GLM model and does not claim to exercise the complete ModelSlim MoE kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

from sglang.srt.hardware_backend.npu.utils import npu_format_cast

DTYPE_MAX = 127.0
REFERENCE_NRMSE_LIMIT = 0.02
MERGE_ATOL = 2e-5


def sync() -> None:
    torch.npu.synchronize()


def tensor_sha256(tensor: torch.Tensor) -> str:
    sync()
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.int16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    sync()
    c = candidate.detach().float()
    r = reference.detach().float()
    diff = c - r
    denom = max(float(torch.linalg.vector_norm(r).item()), 1e-12)
    return {
        "bitwise_equal": bool(torch.equal(candidate, reference)),
        "allclose_1e-5": bool(torch.allclose(c, r, rtol=1e-5, atol=1e-5)),
        "max_abs_error": float(torch.max(torch.abs(diff)).item()) if diff.numel() else 0.0,
        "nrmse": float(torch.linalg.vector_norm(diff).item() / denom),
        "nonfinite_count": int((~torch.isfinite(candidate)).sum().item()),
        "candidate_sha256": tensor_sha256(candidate),
        "reference_sha256": tensor_sha256(reference),
    }


def restore_order(values: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return values[inverse]


def split_plan(size: int, fraction: float, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    if fraction <= 0.0:
        first = 0
    elif fraction >= 1.0:
        first = size
    else:
        first = max(1, min(size - 1, int(math.ceil(size * fraction))))
    permutation = torch.cat(
        [
            torch.arange(0, first, dtype=torch.int64, device=device),
            torch.arange(first, size, dtype=torch.int64, device=device),
        ]
    )
    if fraction == 0.5 and size >= 2:
        # Interleave the two branches rather than using two contiguous halves.
        permutation = torch.cat(
            [
                torch.arange(0, size, 2, dtype=torch.int64, device=device),
                torch.arange(1, size, 2, dtype=torch.int64, device=device),
            ]
        )
        first = (size + 1) // 2
    return permutation, [first, size - first]


class Expert:
    def __init__(self, expert_id: int, tensors: dict[str, torch.Tensor]):
        self.expert_id = expert_id
        self.source = tensors
        self.raw = {name: value.npu() for name, value in tensors.items()}
        self.weight_scale = {
            proj: self.raw[f"{proj}.weight_scale"].flatten().to(torch.bfloat16).contiguous()
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        self.weight_offset = {
            proj: self.raw[f"{proj}.weight_offset"].flatten().contiguous()
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        # ModelSlim processes [out, in] as a transposed [in, out] operand.
        self.weight_t = {
            proj: npu_format_cast(self.raw[f"{proj}.weight"].transpose(0, 1).contiguous())
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        # Two independently allocated replicas, then the same layout processing
        # the production grouped path applies to an expert weight tensor.
        self.weight_t2 = {
            proj: npu_format_cast(
                torch.stack([native, native.clone()], dim=0).contiguous()
            )
            for proj, native in self.weight_t.items()
        }
        self.scale2 = {
            proj: scale.unsqueeze(0).repeat(2, 1).contiguous()
            for proj, scale in self.weight_scale.items()
        }

    def grouped_matmul(
        self,
        projection: str,
        quantized_x: torch.Tensor,
        per_token_scale: torch.Tensor,
        counts: list[int],
    ) -> torch.Tensor:
        group_list = torch.tensor(counts, dtype=torch.int64, device=quantized_x.device)
        return torch.ops.npu.npu_grouped_matmul(
            x=[quantized_x],
            weight=[self.weight_t2[projection]],
            scale=[self.scale2[projection]],
            per_token_scale=[per_token_scale.flatten().contiguous()],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]

    def grouped_chain(
        self,
        quantized_x: torch.Tensor,
        x_scale: torch.Tensor,
        counts: list[int],
    ) -> dict:
        gate = self.grouped_matmul("gate_proj", quantized_x, x_scale, counts)
        up = self.grouped_matmul("up_proj", quantized_x, x_scale, counts)
        activation = torch.nn.functional.silu(gate) * up
        quantized_activation, activation_scale = torch_npu.npu_dynamic_quant(activation)
        output = self.grouped_matmul(
            "down_proj", quantized_activation, activation_scale, counts
        )
        return {
            "gate": gate,
            "up": up,
            "activation": activation,
            "quantized_activation": quantized_activation,
            "activation_scale": activation_scale,
            "output": output,
        }

    def dense_chain(self, x: torch.Tensor) -> dict:
        quantized_x, x_scale = torch_npu.npu_dynamic_quant(x)
        gate = torch_npu.npu_quant_matmul(
            quantized_x,
            self.weight_t["gate_proj"],
            self.weight_scale["gate_proj"],
            pertoken_scale=x_scale.flatten(),
            output_dtype=torch.bfloat16,
        )
        up = torch_npu.npu_quant_matmul(
            quantized_x,
            self.weight_t["up_proj"],
            self.weight_scale["up_proj"],
            pertoken_scale=x_scale.flatten(),
            output_dtype=torch.bfloat16,
        )
        activation = torch.nn.functional.silu(gate) * up
        quantized_activation, activation_scale = torch_npu.npu_dynamic_quant(activation)
        output = torch_npu.npu_quant_matmul(
            quantized_activation,
            self.weight_t["down_proj"],
            self.weight_scale["down_proj"],
            pertoken_scale=activation_scale.flatten(),
            output_dtype=torch.bfloat16,
        )
        return {
            "quantized_x": quantized_x,
            "x_scale": x_scale,
            "gate": gate,
            "up": up,
            "activation": activation,
            "quantized_activation": quantized_activation,
            "activation_scale": activation_scale,
            "output": output,
        }


def load_expert(model: Path, shard: str, expert_id: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.49.mlp.experts.{expert_id}."
    result = {}
    with safe_open(model / shard, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                result[suffix] = handle.get_tensor(key)
    expected = {
        f"{proj}.{suffix}"
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale", "weight_offset")
    }
    if set(result) != expected:
        raise RuntimeError(f"expert {expert_id} incomplete: {sorted(set(result) ^ expected)}")
    return result


def dequant_reference(expert: Expert, x: torch.Tensor, quantized: dict) -> dict:
    """FP32 dequantized reference for the fixed quantized input path."""
    qx = quantized["quantized_x"].cpu().float()
    x_scale = quantized["x_scale"].cpu().float().flatten().unsqueeze(1)
    x_dequant = qx * x_scale
    weights = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        weights[proj] = (
            expert.source[f"{proj}.weight"].float()
            * expert.source[f"{proj}.weight_scale"].float().flatten().unsqueeze(1)
        )
    gate = torch.mm(x_dequant, weights["gate_proj"].T)
    up = torch.mm(x_dequant, weights["up_proj"].T)
    activation = torch.nn.functional.silu(gate) * up
    scale = torch.amax(torch.abs(activation), dim=1, keepdim=True) / DTYPE_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q_activation = torch.clamp(torch.round(activation / scale), -128, 127)
    activation_dequant = q_activation * scale
    output = torch.mm(activation_dequant, weights["down_proj"].T)
    return {
        "output": output,
        "activation_quantized": q_activation.to(torch.int8),
        "activation_scale": scale,
    }


def copy_independence_test(expert: Expert) -> dict:
    source = expert.raw["gate_proj.weight"]
    source_hash = tensor_sha256(source)
    replica_a = source.clone()
    replica_b = source.clone()
    ptrs = {
        "source": int(source.untyped_storage().data_ptr()),
        "replica_a": int(replica_a.untyped_storage().data_ptr()),
        "replica_b": int(replica_b.untyped_storage().data_ptr()),
    }
    a_hash = tensor_sha256(replica_a)
    b_hash = tensor_sha256(replica_b)
    old = replica_b[0, 0].item()
    replica_b[0, 0] = torch.tensor(
        old + 1 if old < 127 else old - 1, dtype=replica_b.dtype, device=replica_b.device
    )
    source_unchanged = tensor_sha256(source) == source_hash
    replica_a_unchanged = tensor_sha256(replica_a) == a_hash
    replica_b_changed = tensor_sha256(replica_b) != b_hash
    # Restore through a separate element so the result hash is exact again.
    replica_b[0, 0] = torch.tensor(old, dtype=replica_b.dtype, device=replica_b.device)
    return {
        "source_sha256": source_hash,
        "replica_a_sha256": a_hash,
        "replica_b_sha256": b_hash,
        "storage_ptrs": ptrs,
        "all_storage_distinct": len(set(ptrs.values())) == 3,
        "same_contents_before_mutation": source_hash == a_hash == b_hash,
        "source_unchanged_after_replica_mutation": source_unchanged,
        "other_replica_unchanged_after_replica_mutation": replica_a_unchanged,
        "mutated_replica_changed": replica_b_changed,
        "replica_restored": tensor_sha256(replica_b) == b_hash,
    }


def merge_semantics_test(device: torch.device) -> dict:
    torch.manual_seed(20261103)
    token_count, top_k, expert_count, width = 257, 8, 8, 64
    topk_ids = (torch.arange(token_count, device=device)[:, None] + torch.arange(top_k, device=device)[None, :]) % expert_count
    topk_ids = topk_ids.to(torch.int32)
    raw_weights = torch.rand((token_count, top_k), device=device, dtype=torch.float32) + 0.1
    topk_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)
    expert_outputs = torch.randn(
        (expert_count, token_count, width), device=device, dtype=torch.float32
    )
    reference = torch.zeros((token_count, width), device=device, dtype=torch.float32)
    for k in range(top_k):
        ids = topk_ids[:, k].long()
        reference += topk_weights[:, k, None] * expert_outputs[ids, torch.arange(token_count, device=device)]

    row_idx = (
        torch.arange(token_count * top_k, dtype=torch.int32, device=device)
        .view(top_k, token_count)
        .permute(1, 0)
        .contiguous()
    )
    x = torch.zeros((token_count, 1), dtype=torch.float32, device=device)
    _, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        x, row_idx=row_idx, expert_idx=topk_ids, active_num=token_count
    )
    source_index = torch.argsort(expanded_row_idx).to(torch.int64)
    source_tokens = source_index % token_count
    expanded_outputs = expert_outputs[expanded_expert_idx.long(), source_tokens]

    # Simulate two independently allocated replicas per logical expert.  Each
    # expanded source task is consumed exactly once; the branch carrying it does
    # not create another finalize-routing contribution.
    branch_counts = []
    cursor = {}
    for expert_id in expanded_expert_idx.cpu().tolist():
        cursor[expert_id] = cursor.get(expert_id, 0) + 1
    for expert_id in range(expert_count):
        n = cursor.get(expert_id, 0)
        branch_counts.append([(n + 1) // 2, n // 2])
    finalize_output = torch_npu.npu_moe_finalize_routing(
        expanded_outputs,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=0,
    )
    finalize_comparison = compare(finalize_output, reference)

    task_count = int(expanded_row_idx.numel())
    unique_task_count = int(torch.unique(expanded_row_idx).numel())
    duplicate_count = task_count - unique_task_count
    return {
        "kind": "synthetic_task_level_merge_semantics",
        "token_count": token_count,
        "top_k": top_k,
        "expert_count": expert_count,
        "task_count": task_count,
        "expected_task_count": token_count * top_k,
        "unique_task_count": unique_task_count,
        "duplicate_task_count": duplicate_count,
        "expert_task_counts": {
            str(expert_id): int((expanded_expert_idx == expert_id).sum().item())
            for expert_id in range(expert_count)
        },
        "replica_branch_counts": branch_counts,
        "branch_total": int(sum(map(sum, branch_counts))),
        "finalize_routing_comparison": finalize_comparison,
        "pass": (
            task_count == token_count * top_k
            and unique_task_count == task_count
            and int(sum(map(sum, branch_counts))) == task_count
            and finalize_comparison["max_abs_error"] <= MERGE_ATOL
            and finalize_comparison["nonfinite_count"] == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/glm-q")
    parser.add_argument("--out", required=True)
    parser.add_argument("--experts", default="40,43,60")
    parser.add_argument("--shapes", default="1,17,257")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    experts = [int(value) for value in args.experts.split(",")]
    shapes = [int(value) for value in args.shapes.split(",")]
    shards = {40: "quant_model_weights-00107-of-00181.safetensors",
              43: "quant_model_weights-00107-of-00181.safetensors",
              60: "quant_model_weights-00108-of-00181.safetensors"}

    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError(f"expected exactly one NPU, got {torch.npu.device_count()}")
    device = torch.device("npu:0")
    sync()
    free_before, total_hbm = torch.npu.mem_get_info()
    torch.npu.reset_peak_memory_stats()

    loaded = {expert_id: Expert(expert_id, load_expert(Path(args.model), shards[expert_id], expert_id))
              for expert_id in experts}

    storage = copy_independence_test(loaded[43])
    quant_state = {}
    for expert_id, expert in loaded.items():
        quant_state[str(expert_id)] = {
            "all_weight_offsets_zero": all(
                bool(torch.count_nonzero(offset).item() == 0)
                for offset in expert.weight_offset.values()
            ),
            "weight_offset_abs_max": {
                proj: float(offset.abs().max().item()) for proj, offset in expert.weight_offset.items()
            },
            "weight_dtype": str(expert.raw["gate_proj.weight"].dtype),
            "weight_scale_dtype_source": str(expert.source["gate_proj.weight_scale"].dtype),
            "weight_scale_dtype_kernel": str(expert.weight_scale["gate_proj"].dtype),
            "raw_shape": list(expert.raw["gate_proj.weight"].shape),
            "kernel_shape": list(expert.weight_t["gate_proj"].shape),
            "replica_group_shape": list(expert.weight_t2["gate_proj"].shape),
        }

    generator = torch.Generator(device="cpu").manual_seed(20261103)
    cases = []
    for expert_id, expert in loaded.items():
        for rows in shapes:
            x = torch.randn((rows, 6144), generator=generator, dtype=torch.float32).to(torch.bfloat16).npu()
            baseline = expert.grouped_chain(
                *torch_npu.npu_dynamic_quant(x), counts=[rows, 0]
            )
            full_q, full_scale = torch_npu.npu_dynamic_quant(x)
            for fraction in (0.0, 1.0, 0.5, 0.9):
                permutation, counts = split_plan(rows, fraction, device)
                permuted_x = x[permutation]
                # Actual dynamic quantization on each branch: verify per-token
                # scales and int8 values are independent of task grouping.
                pieces_q, pieces_s = [], []
                for start, count in ((0, counts[0]), (counts[0], counts[1])):
                    if count:
                        q, s = torch_npu.npu_dynamic_quant(permuted_x[start:start + count])
                    else:
                        q = torch.empty((0, 6144), dtype=torch.int8, device=device)
                        s = torch.empty((0,), dtype=torch.float32, device=device)
                    pieces_q.append(q)
                    pieces_s.append(s)
                regrouped_q = torch.cat(pieces_q, dim=0)
                regrouped_s = torch.cat(pieces_s, dim=0)
                quant_comparison = compare(regrouped_q, full_q[permutation])
                scale_comparison = compare(regrouped_s, full_scale[permutation])

                # Fixed quantized input, split between two independent replicas.
                candidate = expert.grouped_chain(full_q[permutation], full_scale[permutation], counts)
                restored_output = restore_order(candidate["output"], permutation)
                restored_activation_q = restore_order(candidate["quantized_activation"], permutation)
                restored_activation_s = restore_order(candidate["activation_scale"], permutation)
                output_comparison = compare(restored_output, baseline["output"])
                activation_q_comparison = compare(restored_activation_q, baseline["quantized_activation"])
                activation_s_comparison = compare(restored_activation_s, baseline["activation_scale"])
                passed = (
                    output_comparison["bitwise_equal"]
                    and activation_q_comparison["bitwise_equal"]
                    and activation_s_comparison["bitwise_equal"]
                    and quant_comparison["bitwise_equal"]
                    and scale_comparison["bitwise_equal"]
                    and output_comparison["nonfinite_count"] == 0
                )
                cases.append(
                    {
                        "expert": expert_id,
                        "rows": rows,
                        "split_fraction_first_replica": fraction,
                        "branch_counts": counts,
                        "permutation_sha256": tensor_sha256(permutation),
                        "quantized_input_comparison": quant_comparison,
                        "input_scale_comparison": scale_comparison,
                        "restored_output_comparison": output_comparison,
                        "activation_quantized_comparison": activation_q_comparison,
                        "activation_scale_comparison": activation_s_comparison,
                        "pass": passed,
                    }
                )

    # Twenty independent invocations of the same fixed A/A case, as required by
    # the evidence plan; keep all hashes rather than only a summary count.
    aa_rows = 17
    aa_x = torch.randn((aa_rows, 6144), generator=torch.Generator().manual_seed(7), dtype=torch.float32).to(torch.bfloat16).npu()
    aa_q, aa_s = torch_npu.npu_dynamic_quant(aa_x)
    aa_hashes = []
    for _ in range(20):
        aa_out = loaded[43].grouped_chain(aa_q, aa_s, [9, 8])["output"]
        aa_hashes.append(tensor_sha256(aa_out))
    aa_pass = len(set(aa_hashes)) == 1

    reference_case = next(c for c in cases if c["expert"] == 43 and c["rows"] == 17 and c["split_fraction_first_replica"] == 0.5)
    dense = loaded[43].dense_chain(
        torch.randn((17, 6144), generator=torch.Generator().manual_seed(7), dtype=torch.float32).to(torch.bfloat16).npu()
    )
    reference = dequant_reference(loaded[43], None, dense)
    reference_comparison = compare(dense["output"].cpu(), reference["output"])
    reference_pass = reference_comparison["nrmse"] <= REFERENCE_NRMSE_LIMIT and reference_comparison["nonfinite_count"] == 0

    merge = merge_semantics_test(device)
    sync()
    free_after, _ = torch.npu.mem_get_info()
    all_numeric_pass = all(case["pass"] for case in cases) and aa_pass and reference_pass and all(
        value["all_weight_offsets_zero"] for value in quant_state.values()
    )
    storage_pass = all(
        storage[key]
        for key in (
            "all_storage_distinct",
            "same_contents_before_mutation",
            "source_unchanged_after_replica_mutation",
            "other_replica_unchanged_after_replica_mutation",
            "mutated_replica_changed",
            "replica_restored",
        )
    )
    result = {
        "stage": "S1",
        "kind": "single_npu_expert_copy_and_numerics",
        "status": "PASS" if all_numeric_pass and storage_pass and merge["pass"] else "FAIL",
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": int(torch.npu.device_count()),
            "device_name": torch.npu.get_device_name(0),
            "device": "npu:0",
            "cann_tbe_import": "checked separately in container preflight",
            "seed": 20261103,
        },
        "scope": {
            "layer": 49,
            "experts": experts,
            "hot_experts": [43, 40],
            "cold_expert": 60,
            "fixed_quantized_input": True,
            "real_dynamic_quant_pre_grouping": True,
            "production_compatible_primitives": [
                "npu_dynamic_quant",
                "npu_grouped_matmul with two independent replicas",
                "SwiGLU",
            ],
            "not_claimed": [
                "full ModelSlim MoE kernel parity",
                "full GLM model correctness",
                "cross-rank communication or EP16 end-to-end performance",
            ],
        },
        "memory": {
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "total_hbm_bytes": int(total_hbm),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        },
        "quant_state": quant_state,
        "npu_replica_storage": storage,
        "split_dispatch_cases": cases,
        "aa_20_repeats": {
            "sha256": aa_hashes,
            "unique_hash_count": len(set(aa_hashes)),
            "pass": aa_pass,
        },
        "dequantized_reference": {
            "expert": 43,
            "rows": 17,
            "description": "FP32 dequantized input/weight and explicit per-token activation re-quantization",
            "nrmse_limit": REFERENCE_NRMSE_LIMIT,
            "comparison": reference_comparison,
            "pass": reference_pass,
        },
        "eight_expert_top8_merge": merge,
        "pass_criteria": {
            "all_offsets_zero": True,
            "split_outputs_bitwise_equal": True,
            "dynamic_quant_q_and_scale_bitwise_equal_after_restore": True,
            "aa_20_unique_hash_count": 1,
            "dequant_reference_nrmse_max": REFERENCE_NRMSE_LIMIT,
            "merge_max_abs_error_max": MERGE_ATOL,
        },
        "summary": {
            "split_case_count": len(cases),
            "split_cases_passed": sum(int(case["pass"]) for case in cases),
            "storage_independence_pass": storage_pass,
            "numeric_pass": all_numeric_pass,
            "merge_pass": merge["pass"],
        },
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "status": result["status"],
        "cases": f"{sum(int(c['pass']) for c in cases)}/{len(cases)}",
        "aa_20_unique_hashes": len(set(aa_hashes)),
        "reference_nrmse": reference_comparison["nrmse"],
        "merge_max_abs_error": merge["finalize_routing_comparison"]["max_abs_error"],
        "storage_pass": storage_pass,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
