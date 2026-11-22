#!/usr/bin/env python3
"""Per-case CPU search regression and independent integer-oracle comparison."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from expert_search import Slot, State, oracle_search, search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
REPEATS = 12
COST = 2.0
EPS = 1e-9

CASES = [
    {
        "name": "q4_candidate_fix",
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
        "name": "zero_net_saving_tie",
        "counts": ((4,), (0,)),
        "slots": (Slot(1),),
        "layer": 49,
        "max_actions": 1,
        "bounded_beam": 64,
        "active_expert_cost": 0.0,
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
        started = time.perf_counter_ns()
        output = fn()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return output, {"raw_ms": values, "median_ms": statistics.median(values)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    invalid = 0
    candidate_misses = 0
    pruning_misses = 0
    legacy_misses = 0
    max_fixed_gap = 0.0
    for case in CASES:
        active_cost = case.get("active_expert_cost", 0.25)
        common = {
            "target_layer": case["layer"],
            "max_actions": case["max_actions"],
            "cost_per_copy": COST,
            "active_expert_cost": active_cost,
        }
        fixed, fixed_time = timed(
            lambda: search(
                case["counts"],
                case["slots"],
                beam_width=case["bounded_beam"],
                quota_mode="intersection",
                return_stats=True,
                **common,
            )
        )
        fixed_state, fixed_stats = fixed
        fixed_exhaustive, fixed_exhaustive_time = timed(
            lambda: search(
                case["counts"],
                case["slots"],
                exhaustive=True,
                quota_mode="intersection",
                **common,
            ),
            repeats=3,
        )
        complete, complete_time = timed(
            lambda: search(
                case["counts"],
                case["slots"],
                exhaustive=True,
                quota_mode="all",
                **common,
            ),
            repeats=2,
        )
        legacy, legacy_time = timed(
            lambda: search(
                case["counts"],
                case["slots"],
                beam_width=case["bounded_beam"],
                quota_mode="legacy",
                **common,
            ),
            repeats=3,
        )
        oracle, oracle_time = timed(
            lambda: oracle_search(
                case["counts"], case["slots"], **common
            ),
            repeats=2,
        )
        baseline = State(
            case["layer"],
            tuple(case["counts"]),
            case["slots"],
            tuple(0 for _ in case["slots"]),
            active_expert_cost=active_cost,
        )
        states = {
            "baseline": baseline,
            "oracle": oracle,
            "complete": complete,
            "fixed_exhaustive": fixed_exhaustive,
            "fixed_bounded": fixed_state,
            "legacy_bounded": legacy,
        }
        legality_records = {name: legality(case, state) for name, state in states.items()}
        invalid += sum(not item["legal"] for item in legality_records.values())
        base_score = score(baseline)
        complete_score = score(complete)
        fixed_exhaustive_score = score(fixed_exhaustive)
        fixed_score = score(fixed_state)
        oracle_score = score(oracle)
        candidate_miss = fixed_exhaustive_score > complete_score + EPS
        pruning_miss = fixed_score > fixed_exhaustive_score + EPS
        legacy_miss = score(legacy) > complete_score + EPS
        candidate_misses += int(candidate_miss)
        pruning_misses += int(pruning_miss)
        legacy_misses += int(legacy_miss)
        gap = max(0.0, fixed_score - oracle_score)
        max_fixed_gap = max(max_fixed_gap, gap)
        records.append(
            {
                "name": case["name"],
                "layer": case["layer"],
                "counts": case["counts"],
                "slots": [
                    {
                        "owner": slot.owner,
                        "expert": slot.expert,
                        "layer": slot.layer,
                        "ready": slot.ready,
                        "busy": slot.busy,
                    }
                    for slot in case["slots"]
                ],
                "active_expert_cost": active_cost,
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
                "fixed_exhaustive": {
                    "score": fixed_exhaustive_score,
                    "rank_loads": fixed_exhaustive.rank_loads,
                    "actions": [action_dict(a) for a in fixed_exhaustive.actions],
                    "legal": legality_records["fixed_exhaustive"],
                    "time": fixed_exhaustive_time,
                },
                "fixed_bounded": {
                    "score": fixed_score,
                    "rank_loads": fixed_state.rank_loads,
                    "actions": [action_dict(a) for a in fixed_state.actions],
                    "legal": legality_records["fixed_bounded"],
                    "time": fixed_time,
                    "stats": fixed_stats,
                },
                "legacy_bounded": {
                    "score": score(legacy),
                    "rank_loads": legacy.rank_loads,
                    "actions": [action_dict(a) for a in legacy.actions],
                    "legal": legality_records["legacy_bounded"],
                    "time": legacy_time,
                },
                "missed_profit_vs_oracle": gap,
                "candidate_omission": candidate_miss,
                "pruning_omission": pruning_miss,
                "legacy_omission": legacy_miss,
                "oracle_matches_complete_score": abs(oracle_score - complete_score) <= EPS,
            }
        )
    summary = {
        "cases": len(records),
        "invalid_states": invalid,
        "candidate_omissions_fixed": candidate_misses,
        "pruning_omissions_fixed": pruning_misses,
        "legacy_misses": legacy_misses,
        "max_fixed_gap_vs_oracle": max_fixed_gap,
        "cost_per_copy_toy_units": COST,
        "note": (
            "Toy integer scores only. Search time is reported per case below; "
            "no cross-case production-latency p95 is claimed."
        ),
    }
    payload = {"summary": summary, "cases": records}
    (OUT / "cpu_expert_search_oracle.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        f"cases={len(records)} invalid={invalid} "
        f"candidate_omissions_fixed={candidate_misses} "
        f"pruning_omissions_fixed={pruning_misses} "
        f"legacy_misses={legacy_misses}",
        "case\tbaseline\toracle\tcomplete\tfixed_exhaustive\tfixed_bounded\t"
        "legacy\tgap\tcandidate_miss\tpruning_miss\tsearch_median_ms\tdedup",
    ]
    for record in records:
        lines.append(
            "\t".join(
                [
                    record["name"],
                    f"{record['baseline']['score']:.4f}",
                    f"{record['oracle']['score']:.4f}",
                    f"{record['complete_candidate_space']['score']:.4f}",
                    f"{record['fixed_exhaustive']['score']:.4f}",
                    f"{record['fixed_bounded']['score']:.4f}",
                    f"{record['legacy_bounded']['score']:.4f}",
                    f"{record['missed_profit_vs_oracle']:.4f}",
                    str(record["candidate_omission"]),
                    str(record["pruning_omission"]),
                    f"{record['fixed_bounded']['time']['median_ms']:.6f}",
                    str(record["fixed_bounded"]["stats"]["deduplicated"]),
                ]
            )
        )
    text = "\n".join(lines) + "\n"
    (OUT / "cpu_expert_search_oracle.txt").write_text(text)
    print(text, end="")
    if invalid or candidate_misses or pruning_misses:
        # Non-zero gaps are valid evidence for bounded search; only structural
        # invalidity is a failure. q4 and tie cases have dedicated assertions.
        if invalid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
