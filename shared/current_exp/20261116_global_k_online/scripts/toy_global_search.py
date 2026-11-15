#!/usr/bin/env python3
"""CPU-only multi-action search validation.

This is a load-vector toy, not an NPU timing claim. It validates:
- K global slots with fixed owners;
- multiple actions in one plan;
- integer quota conservation;
- full/reuse/no-copy candidates;
- a case where an individual action is not profitable but a pair is.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Optional


@dataclass(frozen=True)
class Slot:
    owner: int
    expert: int = -1
    ready: bool = False


@dataclass(frozen=True)
class Action:
    source: int
    dest: int
    q: int
    slot_index: int
    copy: bool


@dataclass(frozen=True)
class Plan:
    loads: tuple[int, ...]
    actions: tuple[Action, ...]
    copies: int

    @property
    def max_load(self):
        return max(self.loads)


def apply_action(loads, slots, used, action):
    assert 0 <= action.source < len(loads) and 0 <= action.dest < len(loads)
    assert action.slot_index not in used
    assert action.dest == slots[action.slot_index].owner
    assert 0 < action.q <= loads[action.source]
    x = list(loads)
    x[action.source] -= action.q
    x[action.dest] += action.q
    return tuple(x)


def quota_values(n, exhaustive=False):
    if exhaustive:
        return range(1, n + 1)
    return sorted({1, n // 4, n // 3, n // 2, n} - {0})


def candidate_actions(loads, slots, used=(), exhaustive=False):
    out = []
    used = set(used)
    for si, slot in enumerate(slots):
        if si in used:
            continue
        for src in range(len(loads)):
            if loads[src] <= 0 or src == slot.owner:
                continue
            for q in quota_values(loads[src], exhaustive=exhaustive):
                out.append(Action(src, slot.owner, q, si, copy=not slot.ready))
    return out


def score(plan, copy_ms=2):
    return plan.max_load + copy_ms * plan.copies

def score_key(plan, copy_ms=2):
    return (score(plan, copy_ms), sum(x*x for x in plan.loads), len(plan.actions), tuple((a.source,a.dest,a.q,a.slot_index,a.copy) for a in plan.actions))


def search(loads, slots, max_actions=2, beam_width=32, exhaustive=False, copy_ms=2):
    start = Plan(tuple(loads), (), 0)
    if exhaustive:
        frontier = [start]
        best = start
        for depth in range(max_actions):
            nxt = []
            for p in frontier:
                nxt.append(p)
                used = {a.slot_index for a in p.actions}
                for a in candidate_actions(p.loads, slots, used, exhaustive=True):
                    x = apply_action(p.loads, slots, used, a)
                    cand = Plan(x, p.actions + (a,), p.copies + int(a.copy))
                    nxt.append(cand)
            frontier = nxt
            # keep all for exact reference
            for p in frontier:
                if score_key(p, copy_ms) < score_key(best, copy_ms): best = p
        return best
    frontier = [start]
    best = start
    for _ in range(max_actions):
        nxt = []
        for p in frontier:
            used = {a.slot_index for a in p.actions}
            for a in candidate_actions(p.loads, slots, used, exhaustive=False):
                x = apply_action(p.loads, slots, used, a)
                nxt.append(Plan(x, p.actions + (a,), p.copies + int(a.copy)))
        # Do not prune solely by "must improve"; keep best-width partial states.
        nxt.sort(key=lambda p: score_key(p, copy_ms))
        frontier = nxt[:beam_width]
        for p in frontier:
            if score_key(p, copy_ms) < score_key(best, copy_ms): best = p
    return best
