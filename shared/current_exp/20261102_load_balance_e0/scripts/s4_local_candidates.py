#!/usr/bin/env python3
"""S4: bounded local candidates and channel-split numerical counterexample."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_local_cost import ExpertBank, benchmark_call, load_experts, make_case, sync  # noqa: E402
from sglang.srt.hardware_backend.npu.utils import npu_format_cast  # noqa: E402

LOCAL_THRESHOLD = 0.05
SEED = 20261103


def compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    sync()
    c = candidate.float()
    r = reference.float()
    diff = c - r
    return {
        "bitwise_equal": bool(torch.equal(candidate, reference)),
        "max_abs_error": float(diff.abs().max().item()),
        "nrmse": float(torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(r).item(), 1e-12)),
        "nonfinite_count": int((~torch.isfinite(candidate)).sum().item()),
    }


def subgroup_weights(bank: ExpertBank, indices: list[int], duplicate: bool = False) -> dict:
    def gather(values):
        pieces = [values[index] for index in indices]
        if duplicate:
            pieces = [pieces[0], pieces[0]]
        return npu_format_cast(torch.stack(pieces, dim=0).contiguous())
    scale_indices = indices * 2 if duplicate else indices
    return {
        "w13": gather(bank.w13),
        "down": gather(bank.down),
        "w13_scale": torch.stack([bank.w13_scale[index] for index in scale_indices], dim=0).contiguous(),
        "down_scale": torch.stack([bank.down_scale[index] for index in scale_indices], dim=0).contiguous(),
    }


def subchain(weights: dict, quantized_x: torch.Tensor, x_scale: torch.Tensor, counts: list[int]) -> torch.Tensor:
    group_list = torch.tensor(counts, dtype=torch.int64, device="npu:0")
    gate_up = torch.ops.npu.npu_grouped_matmul(
        x=[quantized_x],
        weight=[weights["w13"]],
        scale=[weights["w13_scale"]],
        per_token_scale=[x_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    hidden = weights["w13"].shape[-1] // 2
    activation = torch.nn.functional.silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
    q_activation, activation_scale = torch_npu.npu_dynamic_quant(activation)
    return torch.ops.npu.npu_grouped_matmul(
        x=[q_activation],
        weight=[weights["down"]],
        scale=[weights["down_scale"]],
        per_token_scale=[activation_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/glm-q")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    if torch.npu.device_count() != 1:
        raise RuntimeError(f"expected one NPU, got {torch.npu.device_count()}")
    sync()
    free_before, total_hbm = torch.npu.mem_get_info()
    torch.npu.reset_peak_memory_stats()

    ids = [43, 40, 192, 67, 82, 172, 112, 79]
    bank = ExpertBank("s4_top8", ids, load_experts(Path(args.model), ids))
    hot_weights = subgroup_weights(bank, [0])
    rest_weights = subgroup_weights(bank, list(range(1, 8)))
    row_weights = subgroup_weights(bank, [0], duplicate=True)

    real_counts = [4062, 3978, 1321, 1014, 841, 818, 594, 561]
    distribution_plans = {
        1024: [315, 309, 103, 79, 65, 63, 46, 44],
        4096: [1262, 1235, 410, 315, 261, 254, 185, 174],
        13189: real_counts,
    }
    cases = []
    csv_rows = []
    for rows, counts in distribution_plans.items():
        case = make_case(bank, counts, f"hot_separate_m{rows}", "real_top8")
        hot_count = counts[0]
        rest_counts = counts[1:]

        def baseline():
            return bank.chain(case)

        def hot_separate():
            qx, x_scale = torch_npu.npu_dynamic_quant(case["x"])
            qx = qx[case["permutation"]]
            x_scale = x_scale[case["permutation"]]
            hot = subchain(hot_weights, qx[:hot_count], x_scale[:hot_count], [hot_count])
            rest = subchain(rest_weights, qx[hot_count:], x_scale[hot_count:], rest_counts)
            return torch.cat([hot, rest], dim=0)[case["inverse"]]

        ref = baseline()
        candidate = hot_separate()
        numerical = compare(candidate, ref)
        base_timing = benchmark_call(baseline)
        cand_timing = benchmark_call(hot_separate)
        reduction = 1.0 - cand_timing["device_ms_median"] / base_timing["device_ms_median"]
        cases.append({
            "candidate": "hot_expert_separate_gmm",
            "rows": rows,
            "counts": counts,
            "hot_expert_index": 0,
            "baseline_timing": base_timing,
            "candidate_timing": cand_timing,
            "device_ms_reduction": reduction,
            "meets_5pct_local_threshold": reduction >= LOCAL_THRESHOLD,
            "numerical": numerical,
        })
        csv_rows.append({
            "candidate": "hot_expert_separate_gmm",
            "rows": rows,
            "baseline_ms": base_timing["device_ms_median"],
            "candidate_ms": cand_timing["device_ms_median"],
            "reduction": reduction,
            "numerical_nrmse": numerical["nrmse"],
            "bitwise_equal": numerical["bitwise_equal"],
        })

    # Fixed-quantization row grouping: duplicate the same expert into two groups
    # and process two row ranges sequentially in one grouped call.  The dynamic
    # activation boundary remains per token and is not redefined by the grouping.
    for rows in (1024, 4096):
        counts = [rows] + [0] * 7
        case = make_case(bank, counts, f"row_group_m{rows}", "single_expert")
        first = rows // 2
        split_counts = [first, rows - first]

        def baseline():
            return bank.chain(case)

        def row_grouped():
            qx, x_scale = torch_npu.npu_dynamic_quant(case["x"])
            qx = qx[case["permutation"]]
            x_scale = x_scale[case["permutation"]]
            output = subchain(row_weights, qx, x_scale, split_counts)
            return output[case["inverse"]]

        ref = baseline()
        candidate = row_grouped()
        numerical = compare(candidate, ref)
        base_timing = benchmark_call(baseline)
        cand_timing = benchmark_call(row_grouped)
        reduction = 1.0 - cand_timing["device_ms_median"] / base_timing["device_ms_median"]
        cases.append({
            "candidate": "fixed_quant_two_row_groups",
            "rows": rows,
            "counts": counts,
            "split_counts": split_counts,
            "baseline_timing": base_timing,
            "candidate_timing": cand_timing,
            "device_ms_reduction": reduction,
            "meets_5pct_local_threshold": reduction >= LOCAL_THRESHOLD,
            "numerical": numerical,
        })
        csv_rows.append({
            "candidate": "fixed_quant_two_row_groups",
            "rows": rows,
            "baseline_ms": base_timing["device_ms_median"],
            "candidate_ms": cand_timing["device_ms_median"],
            "reduction": reduction,
            "numerical_nrmse": numerical["nrmse"],
            "bitwise_equal": numerical["bitwise_equal"],
        })

    # Intermediate-channel split correctness counterexample.  The first path
    # keeps the original full-row activation quantization scale; the second path
    # requantizes each channel half independently and exposes scale-boundary
    # drift.
    channel_rows = 257
    channel_case = make_case(bank, [channel_rows] + [0] * 7, "channel_split_m257", "single_expert")
    qx, x_scale = torch_npu.npu_dynamic_quant(channel_case["x"])
    qx = qx[channel_case["permutation"]]
    x_scale = x_scale[channel_case["permutation"]]
    group_list = torch.tensor([channel_rows], dtype=torch.int64, device="npu:0")
    gate_up = torch.ops.npu.npu_grouped_matmul(
        x=[qx], weight=[bank.w13[:1]], scale=[bank.w13_scale[:1]], per_token_scale=[x_scale],
        split_item=2, group_list_type=1, group_type=0, group_list=group_list, output_dtype=torch.bfloat16,
    )[0]
    hidden = bank.w13.shape[-1] // 2
    activation = torch.nn.functional.silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
    q_activation, activation_scale = torch_npu.npu_dynamic_quant(activation)
    full_output = torch_npu.npu_quant_matmul(
        q_activation, bank.down[0], bank.down_scale[0], pertoken_scale=activation_scale.flatten(), output_dtype=torch.bfloat16
    )
    preserved_partials = []
    independent_partials = []
    for start, end in ((0, 1024), (1024, 2048)):
        preserved_partials.append(torch_npu.npu_quant_matmul(
            q_activation[:, start:end],
            bank.down[0][start:end, :].contiguous(),
            bank.down_scale[0],
            pertoken_scale=activation_scale.flatten(),
            output_dtype=torch.bfloat16,
        ))
        local_q, local_scale = torch_npu.npu_dynamic_quant(activation[:, start:end])
        independent_partials.append(torch_npu.npu_quant_matmul(
            local_q,
            bank.down[0][start:end, :].contiguous(),
            bank.down_scale[0],
            pertoken_scale=local_scale.flatten(),
            output_dtype=torch.bfloat16,
        ))
    preserved_sum = torch.stack(preserved_partials, dim=0).sum(dim=0)
    independent_sum = torch.stack(independent_partials, dim=0).sum(dim=0)
    channel_result = {
        "rows": channel_rows,
        "chunks": [[0, 1024], [1024, 2048]],
        "preserved_original_activation_scale": compare(preserved_sum, full_output),
        "independent_chunk_requantization": compare(independent_sum, full_output),
    }
    channel_result["preserved_scale_pass"] = channel_result["preserved_original_activation_scale"]["nrmse"] <= 0.02
    channel_result["independent_requant_is_not_equivalent"] = (
        channel_result["independent_chunk_requantization"]["nrmse"] > channel_result["preserved_original_activation_scale"]["nrmse"] * 2
    )

    sync()
    free_after, _ = torch.npu.mem_get_info()
    result = {
        "stage": "S4",
        "status": "COMPLETE",
        "kind": "bounded_local_candidates",
        "local_threshold": LOCAL_THRESHOLD,
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_count": int(torch.npu.device_count()),
            "device_name": torch.npu.get_device_name(0),
        },
        "candidate_cases": cases,
        "channel_split_correctness": channel_result,
        "decisions": {
            "hot_expert_separate_gmm": {
                "local_threshold_met_at_any_tested_shape": any(
                    case["meets_5pct_local_threshold"] for case in cases
                    if case["candidate"] == "hot_expert_separate_gmm"
                ),
                "interpretation": "Sequential split on one NPU measures added local launches; it does not represent placing the hot expert on another rank.",
            },
            "fixed_quant_two_row_groups": {
                "local_threshold_met_at_any_tested_shape": any(
                    case["meets_5pct_local_threshold"] for case in cases
                    if case["candidate"] == "fixed_quant_two_row_groups"
                ),
                "interpretation": "Numerical equivalence is expected under fixed per-token quant boundaries; local speedup is not.",
            },
        },
        "memory": {
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "total_hbm_bytes": int(total_hbm),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
        },
        "scope_limits": [
            "Single-NPU sequential local timing only; no concurrent replica execution or HCCL.",
            "The hot-separate candidate is a negative local-cost control, not a multi-rank load-balancing result.",
            "The channel-split test is numerical evidence, not an optimized kernel implementation.",
        ],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "cases": [
            {
                "candidate": case["candidate"],
                "rows": case["rows"],
                "baseline_ms": case["baseline_timing"]["device_ms_median"],
                "candidate_ms": case["candidate_timing"]["device_ms_median"],
                "reduction": case["device_ms_reduction"],
                "nrmse": case["numerical"]["nrmse"],
                "bitwise": case["numerical"]["bitwise_equal"],
            }
            for case in cases
        ],
        "channel_split": channel_result,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
