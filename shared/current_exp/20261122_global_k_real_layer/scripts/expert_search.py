#!/usr/bin/env python3
"""Expert-aware CPU search scaffold (not an NPU timing model).

State has:
  counts[source_rank][logical_expert] = tasks still on the main bank
  slot_counts[slot] = tasks routed to a replica slot

A slot records its owner/layer/expert/ready/busy identity. A wrong identity
requires a copy; a matching ready slot is reusable without a new copy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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
    counts: tuple[tuple[int, ...], ...]
    slots: tuple[Slot, ...]
    slot_counts: tuple[int, ...] = ()
    actions: tuple[Action, ...] = ()
    copies: int = 0

    @property
    def rank_loads(self):
        out=[sum(row) for row in self.counts]
        for i,s in enumerate(self.slots):
            out[s.owner] += self.slot_counts[i]
        return tuple(out)

    @property
    def active_slot_count(self):
        out=[sum(1 for n in row if n>0) for row in self.counts]
        for i,s in enumerate(self.slots):
            if self.slot_counts[i]>0: out[s.owner]+=1
        return tuple(out)

    def rank_times(self):
        return tuple(self.rank_loads[r] + 0.25*self.active_slot_count[r] for r in range(len(self.counts)))

    @property
    def max_time(self): return max(self.rank_times())


def apply_action(state: State, a: Action) -> Optional[State]:
    if not (0 <= a.src_rank < len(state.counts) and 0 <= a.expert < len(state.counts[a.src_rank])): return None
    if not (0 <= a.dst_slot < len(state.slots)): return None
    slot=state.slots[a.dst_slot]
    if slot.busy or a.q<=0 or a.q>state.counts[a.src_rank][a.expert] or slot.owner==a.src_rank: return None
    same=slot.ready and slot.expert==a.expert and slot.layer==0
    if a.copy != (not same): return None
    rows=[list(x) for x in state.counts]
    rows[a.src_rank][a.expert]-=a.q
    slot_counts=list(state.slot_counts or [0]*len(state.slots))
    slot_counts[a.dst_slot]+=a.q
    slots=list(state.slots); slots[a.dst_slot]=Slot(slot.owner,a.expert,0,True,True)
    return State(tuple(tuple(r) for r in rows),tuple(slots),tuple(slot_counts),state.actions+(a,),state.copies+int(not same))


def candidates(state: State):
    used={a.dst_slot for a in state.actions}
    out=[]
    for si,slot in enumerate(state.slots):
        if si in used or slot.busy: continue
        for src,row in enumerate(state.counts):
            if src==slot.owner: continue
            for e,n in enumerate(row):
                if n<=0: continue
                for q in sorted({1,n//2,n}):
                    same=slot.ready and slot.expert==e and slot.layer==0
                    out.append(Action(src,e,si,q,not same))
    return out


def search(counts, slots, max_actions=2, beam_width=64, cost_per_copy=2):
    start=State(tuple(tuple(r) for r in counts),tuple(slots),tuple(0 for _ in slots))
    frontier=[start]; best=start
    def key(s): return (s.max_time+cost_per_copy*s.copies,s.max_time,tuple((a.src_rank,a.expert,a.dst_slot,a.q,a.copy) for a in s.actions))
    for _ in range(max_actions):
        nxt=[]
        for s in frontier:
            nxt.append(s)
            for a in candidates(s):
                ns=apply_action(s,a)
                if ns is not None: nxt.append(ns)
        nxt.sort(key=key); frontier=nxt[:beam_width]
        best=min([best]+frontier,key=key)
    return best
