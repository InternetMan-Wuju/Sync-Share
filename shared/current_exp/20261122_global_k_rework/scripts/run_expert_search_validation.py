#!/usr/bin/env python3
"""Run deterministic CPU-search counterexamples and write structured evidence."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from expert_search import Slot, oracle_search, search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
REPEATS = 20
COST = 2.0
EPS = 1e-9

CASES = [
    {
        "name": "q4_candidate_omission",
        "counts": ((12,), (4,)),
        "slots": (Slot(1),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
    },
    {
        "name": "multi_action_hot_pair",
        "counts": ((40,), (39,), (1,), (1,)),
        "slots": (Slot(2), Slot(3)),
        "layer": 50,
        "max_actions": 2,
        "bounded_beam": 4,
    },
    {
        "name": "beam_pruning_pair",
        "counts": ((40,), (39,), (1,), (1,)),
        "slots": (Slot(2), Slot(3)),
        "layer": 50,
        "max_actions": 2,
        "bounded_beam": 1,
    },
    {
        "name": "wrong_expert",
        "counts": ((12,), (4,)),
        "slots": (Slot(1, expert=9, layer=49, ready=True),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
    },
    {
        "name": "busy_slot",
        "counts": ((9,), (8,), (1,)),
        "slots": (Slot(2, expert=0, layer=49, ready=True, busy=True),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
    },
    {
        "name": "partial_rank_owners",
        "counts": ((40,), (39,), (1,), (1,)),
        "slots": (Slot(2), Slot(3)),
        "layer": 49,
        "max_actions": 2,
        "bounded_beam": 32,
    },
    {
        "name": "zero_profit",
        "counts": ((1,), (1,)),
        "slots": (Slot(1),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
    },
    {
        "name": "existing_replica",
        "counts": ((40,), (1,), (1,)),
        "slots": (Slot(1, expert=0, layer=49, ready=True),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
    },
    {
        "name": "same_expert_different_layer_49",
        "counts": ((12,), (4,)),
        "slots": (Slot(1, expert=0, layer=48, ready=True),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
    },
    {
        "name": "same_expert_different_layer_50",
        "counts": ((12,), (4,)),
        "slots": (Slot(1, expert=0, layer=49, ready=True),),
        "layer": 50,
        "max_actions": 1,
        "bounded_beam": 64,
    },
]


def score(state):
    return state.max_time + COST * state.copies


def action_dict(action):
    return {
        "src_rank": action.src_rank,
        "expert": action.expert,
        "dst_slot": action.dst_slot,
        "q": action.q,
        "copy": action.copy,
    }


def legality(case, state):
    initial_total = sum(sum(row) for row in case["counts"])
    errors = []
    if state.task_count != initial_total:
        errors.append("task_conservation")
    if any(n < 0 for row in state.counts for n in row):
        errors.append("negative_main_count")
    if any(n < 0 for n in state.slot_counts):
        errors.append("negative_slot_count")
    if len({a.dst_slot for a in state.actions}) != len(state.actions):
        errors.append("slot_reused_within_plan")
    for action in state.actions:
        if action.q <= 0:
            errors.append("non_positive_q")
        if state.slots[action.dst_slot].owner == action.src_rank:
            errors.append("same_rank_replica")
        if state.slots[action.dst_slot].layer != case["layer"]:
            errors.append("wrong_final_layer")
    return {"legal": not errors, "errors": errors}


def timed(fn, repeats=REPEATS):
    values = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        output = fn()
        values.append((time.perf_counter_ns() - start) / 1e6)
    values.sort()
    p50 = statistics.median(values)
    idx = min(len(values) - 1, int(0.95 * (len(values) - 1)))
    return output, {"median_ms": p50, "p95_ms": values[idx], "raw_ms": values}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    planner_times = []
    invalid = 0
    candidate_misses = 0
    pruning_misses = 0
    max_gap = 0.0
    for case in CASES:
        counts = case["counts"]
        slots = case["slots"]
        kwargs = {
            "target_layer": case["layer"],
            "max_actions": case["max_actions"],
            "cost_per_copy": COST,
        }
        bounded, bounded_time = timed(
            lambda: search(
                counts,
                slots,
                beam_width=case["bounded_beam"],
                quota_mode="limited",
                **kwargs,
            )
        )
        limited_exhaustive, limited_exhaustive_time = timed(
            lambda: search(
                counts,
                slots,
                exhaustive=True,
                quota_mode="limited",
                **kwargs,
            ),
            repeats=max(3, REPEATS // 4),
        )
        complete, complete_time = timed(
            lambda: search(
                counts,
                slots,
                exhaustive=True,
                quota_mode="all",
                **kwargs,
            ),
            repeats=max(2, REPEATS // 10),
        )
        oracle, oracle_time = timed(
            lambda: oracle_search(counts, slots, **kwargs),
            repeats=max(2, REPEATS // 10),
        )
        states = {
            "baseline": None,
            "oracle": oracle,
            "complete": complete,
            "limited_exhaustive": limited_exhaustive,
            "bounded": bounded,
        }
        # Baseline is constructed from the same immutable inputs.
        from expert_search import State

        baseline = State(
            case["layer"], tuple(counts), slots, tuple(0 for _ in slots), (), 0
        )
        states["baseline"] = baseline
        legality_records = {name: legality(case, state) for name, state in states.items()}
        invalid += sum(not item["legal"] for item in legality_records.values())
        base_score = score(baseline)
        complete_score = score(complete)
        limited_exhaustive_score = score(limited_exhaustive)
        bounded_score = score(bounded)
        oracle_score = score(oracle)
        candidate_miss = limited_exhaustive_score > complete_score + EPS
        pruning_miss = bounded_score > limited_exhaustive_score + EPS
        candidate_misses += int(candidate_miss)
        pruning_misses += int(pruning_miss)
        gap = max(0.0, bounded_score - oracle_score)
        max_gap = max(max_gap, gap)
        planner_times.extend(bounded_time["raw_ms"])
        records.append(
            {
                "name": case["name"],
                "layer": case["layer"],
                "counts": counts,
                "slots": [
                    {
                        "owner": s.owner,
                        "expert": s.expert,
                        "layer": s.layer,
                        "ready": s.ready,
                        "busy": s.busy,
                    }
                    for s in slots
                ],
                "baseline": {
                    "score": base_score,
                    "rank_loads": baseline.rank_loads,
                    "actions": [],
                    "legal": legality_records["baseline"],
                },
                "oracle": {
                    "score": oracle_score,
                    "rank_loads": oracle.rank_loads,
                    "actions": [action_dict(a) for a in oracle.actions],
                    "legal": legality_records["oracle"],
                    "time": oracle_time,
                },
                "complete_candidate_space": {
                    "score": complete_score,
                    "rank_loads": complete.rank_loads,
                    "actions": [action_dict(a) for a in complete.actions],
                    "legal": legality_records["complete"],
                    "time": complete_time,
                },
                "limited_exhaustive": {
                    "score": limited_exhaustive_score,
                    "rank_loads": limited_exhaustive.rank_loads,
                    "actions": [action_dict(a) for a in limited_exhaustive.actions],
                    "legal": legality_records["limited_exhaustive"],
                    "time": limited_exhaustive_time,
                },
                "bounded_search": {
                    "score": bounded_score,
                    "rank_loads": bounded.rank_loads,
                    "actions": [action_dict(a) for a in bounded.actions],
                    "legal": legality_records["bounded"],
                    "time": bounded_time,
                },
                "missed_profit_vs_oracle": max(0.0, bounded_score - oracle_score),
                "candidate_omission": candidate_miss,
                "pruning_omission": pruning_miss,
                "oracle_matches_complete_score": abs(oracle_score - complete_score) <= EPS,
            }
        )
    planner_times.sort()
    planner_p50 = statistics.median(planner_times)
    planner_p95 = planner_times[
        min(len(planner_times) - 1, int(0.95 * (len(planner_times) - 1)))
    ]
    summary = {
        "samples": len(records),
        "bounded_runs": len(planner_times),
        "invalid_states": invalid,
        "candidate_omissions": candidate_misses,
        "pruning_omissions": pruning_misses,
        "max_bounded_gap_vs_oracle": max_gap,
        "planner_cpu_ms": {
            "p50": planner_p50,
            "p95": planner_p95,
            "min": min(planner_times),
            "max": max(planner_times),
        },
        "cost_per_copy_toy_units": COST,
        "note": "Toy integer scores only; not milliseconds, NPU times, or production costs.",
    }
    payload = {"summary": summary, "cases": records}
    (OUT / "cpu_expert_search_oracle.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        f"samples={summary['samples']} invalid={invalid} "
        f"candidate_omissions={candidate_misses} pruning_omissions={pruning_misses}",
        f"planner_cpu_ms p50={planner_p50:.6f} p95={planner_p95:.6f}",
        "case\tbaseline\toracle\tcomplete\tlimited_exhaustive\tbounded\t"
        "gap\tcandidate_miss\tpruning_miss",
    ]
    for record in records:
        lines.append(
            "\t".join(
                [
                    record["name"],
                    f"{record['baseline']['score']:.4f}",
                    f"{record['oracle']['score']:.4f}",
                    f"{record['complete_candidate_space']['score']:.4f}",
                    f"{record['limited_exhaustive']['score']:.4f}",
                    f"{record['bounded_search']['score']:.4f}",
                    f"{record['missed_profit_vs_oracle']:.4f}",
                    str(record["candidate_omission"]),
                    str(record["pruning_omission"]),
                ]
            )
        )
    text = "\n".join(lines) + "\n"
    (OUT / "cpu_expert_search_oracle.txt").write_text(text)
    print(text, end="")
    if invalid or candidate_misses == 0 or pruning_misses == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
