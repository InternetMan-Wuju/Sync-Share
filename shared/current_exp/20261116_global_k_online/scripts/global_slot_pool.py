#!/usr/bin/env python3
"""Logical EP-group pool with exactly K physical slots.

Physical slots are owned by a fixed subset of ranks. Ranks with zero owned
slots receive no backing store. K does not need to be divisible by EP size.
"""
from __future__ import annotations
import threading, time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional
from slot_pool import SlotPool, SlotPoolExhausted


@dataclass(frozen=True)
class GlobalLease:
    owner_rank: int
    local_slot: int
    generation: int
    request_id: str
    layer_id: int
    expert_id: int


class GlobalSlotPool:
    def __init__(self, owner_ranks, slot_bytes, allocator, *, name='global-k-pool'):
        if not owner_ranks:
            raise ValueError('K must be positive')
        if any(r < 0 for r in owner_ranks):
            raise ValueError('owner ranks must be non-negative')
        self.owner_ranks = tuple(int(r) for r in owner_ranks)
        self.capacity = len(self.owner_ranks)
        self.slot_bytes = int(slot_bytes)
        self._counts = Counter(self.owner_ranks)
        self._pools = {
            rank: SlotPool(count, self.slot_bytes, allocator, name=f'{name}-rank{rank}')
            for rank, count in sorted(self._counts.items())
        }
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pool_id = f'{name}:{id(self):x}'

    @property
    def committed_bytes(self):
        return self.capacity * self.slot_bytes

    def per_rank_capacity(self):
        return dict(self._counts)

    def per_rank_allocated_bytes(self):
        return {r: c * self.slot_bytes for r, c in self._counts.items()}

    def available_owners(self):
        out = []
        with self._lock:
            for rank, pool in self._pools.items():
                if pool.stats().free > 0:
                    out.append(rank)
        return out

    def acquire(self, *, request_id, layer_id, expert_id, allowed_ranks=None, timeout_s=None, wait=False):
        allowed = None if allowed_ranks is None else set(allowed_ranks)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while True:
                # Deterministic owner order; no uniform per-rank dummy slot.
                for rank in sorted(self._pools):
                    if allowed is not None and rank not in allowed:
                        continue
                    pool = self._pools[rank]
                    try:
                        lease = pool.acquire(layer_id, expert_id, request_id, wait=False)
                    except SlotPoolExhausted:
                        continue
                    return GlobalLease(rank, lease.slot_id, lease.generation, str(request_id), int(layer_id), int(expert_id))
                if not wait:
                    raise SlotPoolExhausted('no free global expert slot among owner ranks')
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise SlotPoolExhausted('timed out waiting for a global expert slot')
                self._condition.wait(remaining)

    def slot_buffer(self, lease: GlobalLease, *, allow_reserved=False):
        return self._pools[lease.owner_rank].slot_buffer(self._local(lease), allow_reserved=allow_reserved)

    def mark_ready(self, lease: GlobalLease):
        pool = self._pools[lease.owner_rank]
        pool.mark_ready(self._local(lease))

    def release(self, lease: GlobalLease):
        pool = self._pools[lease.owner_rank]
        pool.release(self._local(lease))
        with self._condition:
            self._condition.notify_all()

    def use(self, lease: GlobalLease):
        return self._pools[lease.owner_rank].use(self._local(lease))

    def _local(self, lease):
        from slot_pool import SlotLease
        return SlotLease(self._pools[lease.owner_rank].pool_id, lease.local_slot, lease.generation, lease.layer_id, lease.expert_id, lease.request_id)

    def stats(self):
        local = {r: p.stats() for r, p in self._pools.items()}
        return {
            'K': self.capacity,
            'slot_bytes': self.slot_bytes,
            'committed_bytes': self.committed_bytes,
            'per_rank_capacity': self.per_rank_capacity(),
            'per_rank_allocated_bytes': self.per_rank_allocated_bytes(),
            'active_bytes': sum(s.slot_bytes * (s.reserved + s.ready + s.in_use) for s in local.values()),
            'cached_free_bytes': sum(s.slot_bytes * s.free for s in local.values()),
            'local': local,
        }
