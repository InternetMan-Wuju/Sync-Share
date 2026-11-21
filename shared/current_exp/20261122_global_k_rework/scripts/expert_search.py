#!/usr/bin/env python3
"""Expert-aware CPU search and an independent bounded integer oracle.

This is a toy objective in explicit units, not an NPU timing model. The state
tracks tasks per rank/logical expert plus tasks assigned to external slots.
Slot identity is (owner, target layer, logical expert), and a slot is reusable
without copy only when all three match and the slot is ready.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional


@dataclass(frozen=True)
class Slot:
    owner: int
    expert: Optional[int] = None
    layer: Optional[int] = None
    ready: bool = False
    busy: bool = False


@dataclass(frozen=True)
class Action:
    src_rank: int
    expert: int
    dst_slot: int
    q: int
    copy: bool


@dataclass(frozen=True)
class State:
    layer: int
    counts: tuple[tuple[int, ...], ...]
    slots: tuple[Slot, ...]
    slot_counts: tuple[int, ...] = ()
    actions: tuple[Action, ...] = ()
    copies: int = 0

    @property
    def rank_loads(self):
        out = [sum(row) for row in self.counts]
        for i, slot in enumerate(self.slots):
            out[slot.owner] += self.slot_counts[i]
        return tuple(out)

    @property
    def active_slot_count(self):
        out = [sum(1 for n in row if n > 0) for row in self.counts]
        for i, slot in enumerate(self.slots):
            if self.slot_counts[i] > 0:
                out[slot.owner] += 1
        return tuple(out)

    def rank_times(self):
        return tuple(
            self.rank_loads[r] + 0.25 * self.active_slot_count[r]
            for r in range(len(self.counts))
        )

    @property
    def max_time(self):
        return max(self.rank_times())

    @property
    def task_count(self):
        return sum(sum(row) for row in self.counts) + sum(self.slot_counts)


def _expected_copy(slot: Slot, expert: int, layer: int) -> bool:
    same = (
        slot.ready
        and slot.expert == expert
        and slot.layer == layer
        and not slot.busy
    )
    return not same


def apply_action(state: State, action: Action) -> Optional[State]:
    """Apply one action after validating identity, capacity, and task bounds."""
    if not (0 <= action.src_rank < len(state.counts)):
        return None
    if not (0 <= action.expert < len(state.counts[action.src_rank])):
        return None
    if not (0 <= action.dst_slot < len(state.slots)):
        return None
    slot = state.slots[action.dst_slot]
    if slot.busy or action.q <= 0:
        return None
    if action.q > state.counts[action.src_rank][action.expert]:
        return None
    if slot.owner == action.src_rank:
        return None
    if action.copy != _expected_copy(slot, action.expert, state.layer):
        return None

    same = not action.copy
    rows = [list(row) for row in state.counts]
    rows[action.src_rank][action.expert] -= action.q
    slot_counts = list(state.slot_counts or [0] * len(state.slots))
    slot_counts[action.dst_slot] += action.q
    slots = list(state.slots)
    slots[action.dst_slot] = Slot(
        owner=slot.owner,
        expert=action.expert,
        layer=state.layer,
        ready=True,
        busy=True,
    )
    return State(
        layer=state.layer,
        counts=tuple(tuple(row) for row in rows),
        slots=tuple(slots),
        slot_counts=tuple(slot_counts),
        actions=state.actions + (action,),
        copies=state.copies + int(not same),
    )


def limited_candidates(state: State):
    """Historical finite candidate set; deliberately omits arbitrary q."""
    used = {a.dst_slot for a in state.actions}
    out = []
    for slot_index, slot in enumerate(state.slots):
        if slot_index in used or slot.busy:
            continue
        for src_rank, row in enumerate(state.counts):
            if src_rank == slot.owner:
                continue
            for expert, count in enumerate(row):
                if count <= 0:
                    continue
                for q in sorted({1, count // 2, count}):
                    out.append(
                        Action(
                            src_rank,
                            expert,
                            slot_index,
                            q,
                            _expected_copy(slot, expert, state.layer),
                        )
                    )
    return out


def all_integer_candidates(state: State):
    """Complete candidate space: every legal q for every legal slot/source/expert."""
    used = {a.dst_slot for a in state.actions}
    out = []
    for slot_index, slot in enumerate(state.slots):
        if slot_index in used or slot.busy:
            continue
        for src_rank, row in enumerate(state.counts):
            if src_rank == slot.owner:
                continue
            for expert, count in enumerate(row):
                for q in range(1, count + 1):
                    out.append(
                        Action(
                            src_rank,
                            expert,
                            slot_index,
                            q,
                            _expected_copy(slot, expert, state.layer),
                        )
                    )
    return out


def _key(state: State, cost_per_copy: float):
    return (
        state.max_time + cost_per_copy * state.copies,
        state.max_time,
        state.copies,
        len(state.actions),
        tuple((a.src_rank, a.expert, a.dst_slot, a.q, a.copy) for a in state.actions),
    )


def search(
    counts,
    slots,
    *,
    target_layer: int,
    max_actions=2,
    beam_width=64,
    cost_per_copy=2,
    quota_mode="limited",
    exhaustive=False,
):
    """Bounded search. Set exhaustive/beam_width=None for complete enumeration."""
    start = State(
        layer=int(target_layer),
        counts=tuple(tuple(row) for row in counts),
        slots=tuple(slots),
        slot_counts=tuple(0 for _ in slots),
    )
    if not exhaustive and beam_width is None:
        raise ValueError("bounded search requires a finite beam_width")
    generator = all_integer_candidates if quota_mode == "all" else limited_candidates
    frontier = [start]
    best = start
    for _ in range(max_actions):
        nxt = list(frontier)  # complete bypass at every depth
        for state in frontier:
            for action in generator(state):
                candidate = apply_action(state, action)
                if candidate is not None:
                    nxt.append(candidate)
        if exhaustive:
            frontier = nxt
        else:
            nxt.sort(key=lambda s: _key(s, cost_per_copy))
            frontier = nxt[:beam_width]
        best = min([best] + frontier, key=lambda s: _key(s, cost_per_copy))
    return best


def _oracle_transition(
    counts, slots, slot_counts, target_layer, action, copies, depth
):
    """Independent transition used only by the brute-force integer oracle."""
    src, expert, slot_index, q, copy = action
    if depth <= 0 or q <= 0 or q > counts[src][expert]:
        return None
    slot = slots[slot_index]
    if slot.busy or slot.owner == src:
        return None
    matching = slot.ready and slot.layer == target_layer and slot.expert == expert
    if copy == matching:
        return None
    new_counts = [list(row) for row in counts]
    new_counts[src][expert] -= q
    new_slot_counts = list(slot_counts)
    new_slot_counts[slot_index] += q
    new_slots = list(slots)
    new_slots[slot_index] = Slot(
        slot.owner, expert, target_layer, ready=True, busy=True
    )
    return (
        tuple(tuple(row) for row in new_counts),
        tuple(new_slots),
        tuple(new_slot_counts),
        copies + int(copy),
    )


def _oracle_score(counts, slots, slot_counts, copies, cost_per_copy):
    loads = [sum(row) for row in counts]
    active = [sum(n > 0 for n in row) for row in counts]
    for i, slot in enumerate(slots):
        loads[slot.owner] += slot_counts[i]
        active[slot.owner] += int(slot_counts[i] > 0)
    max_time = max(loads[r] + 0.25 * active[r] for r in range(len(loads)))
    return max_time + cost_per_copy * copies


def oracle_search(counts, slots, *, target_layer: int, max_actions=2, cost_per_copy=2):
    """Exhaustive integer oracle independent of the finite candidate generator."""
    counts = tuple(tuple(row) for row in counts)
    slots = tuple(slots)
    initial_slot_counts = tuple(0 for _ in slots)
    best = {
        "score": _oracle_score(
            counts, slots, initial_slot_counts, 0, cost_per_copy
        ),
        "actions": (),
        "counts": counts,
        "slots": slots,
        "slot_counts": initial_slot_counts,
        "copies": 0,
    }

    def recurse(cur_counts, cur_slots, cur_slot_counts, copies, actions):
        nonlocal best
        score = _oracle_score(
            cur_counts, cur_slots, cur_slot_counts, copies, cost_per_copy
        )
        tie_key = (score, sum(sum(r) for r in cur_counts), copies, len(actions), actions)
        best_key = (
            best["score"],
            sum(sum(r) for r in best["counts"]),
            best["copies"],
            len(best["actions"]),
            best["actions"],
        )
        if tie_key < best_key:
            best = {
                "score": score,
                "actions": actions,
                "counts": cur_counts,
                "slots": cur_slots,
                "slot_counts": cur_slot_counts,
                "copies": copies,
            }
        if len(actions) >= max_actions:
            return
        used_slots = {a[2] for a in actions}
        for slot_index, slot in enumerate(cur_slots):
            if slot_index in used_slots or slot.busy:
                continue
            for src, row in enumerate(cur_counts):
                if src == slot.owner:
                    continue
                for expert, count in enumerate(row):
                    matching = (
                        slot.ready
                        and slot.layer == target_layer
                        and slot.expert == expert
                    )
                    for q in range(1, count + 1):
                        action = (src, expert, slot_index, q, not matching)
                        transition = _oracle_transition(
                            cur_counts,
                            cur_slots,
                            cur_slot_counts,
                            target_layer,
                            action,
                            copies,
                            depth=1,
                        )
                        if transition is None:
                            continue
                        recurse(*transition, actions + (action,))

    recurse(counts, slots, initial_slot_counts, 0, ())
    state = State(
        layer=target_layer,
        counts=best["counts"],
        slots=best["slots"],
        slot_counts=best["slot_counts"],
        actions=tuple(Action(*a) for a in best["actions"]),
        copies=best["copies"],
    )
    return state
