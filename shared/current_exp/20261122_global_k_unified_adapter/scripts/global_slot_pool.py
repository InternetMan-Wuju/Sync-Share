#!/usr/bin/env python3
"""Logical EP-group pool with exactly K physical expert slots.

Physical slots are owned by a fixed subset of ranks. Ranks with zero owned
slots receive no backing store. K need not be divisible by EP size. The global
lease is a capability: every public entry point checks pool, owner, generation,
request, layer, and expert identity before touching another pool's memory.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from slot_pool import SlotPool, SlotPoolExhausted, InvalidLease


@dataclass(frozen=True)
class GlobalLease:
    pool_id: str
    owner_rank: int
    local_slot: int
    generation: int
    request_id: str
    layer_id: int
    expert_id: int


class GlobalSlotPool:
    def __init__(
        self,
        owner_ranks,
        slot_bytes,
        allocator,
        *,
        name="global-k-pool",
        synchronize: Callable[[], None] | None = None,
    ):
        if not owner_ranks:
            raise ValueError("K must be positive")
        if any(r < 0 for r in owner_ranks):
            raise ValueError("owner ranks must be non-negative")
        self.owner_ranks = tuple(int(r) for r in owner_ranks)
        self.capacity = len(self.owner_ranks)
        self.slot_bytes = int(slot_bytes)
        self._counts = Counter(self.owner_ranks)
        self._pools = {
            rank: SlotPool(
                count, self.slot_bytes, allocator, name=f"{name}-rank{rank}"
            )
            for rank, count in sorted(self._counts.items())
        }
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pool_id = f"{name}:{id(self):x}"
        self._synchronize = synchronize

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

    def acquire(
        self,
        *,
        request_id,
        layer_id,
        expert_id,
        allowed_ranks=None,
        timeout_s=None,
        wait=False,
    ):
        allowed = None if allowed_ranks is None else set(allowed_ranks)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while True:
                for rank in sorted(self._pools):
                    if allowed is not None and rank not in allowed:
                        continue
                    pool = self._pools[rank]
                    try:
                        lease = pool.acquire(
                            layer_id, expert_id, request_id, wait=False
                        )
                    except SlotPoolExhausted:
                        continue
                    return GlobalLease(
                        self._pool_id,
                        rank,
                        lease.slot_id,
                        lease.generation,
                        str(request_id),
                        int(layer_id),
                        int(expert_id),
                    )
                if not wait:
                    raise SlotPoolExhausted(
                        "no free global expert slot among owner ranks"
                    )
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise SlotPoolExhausted(
                        "timed out waiting for a global expert slot"
                    )
                self._condition.wait(remaining)

    def _validated_local(self, lease: GlobalLease):
        """Validate the global capability and return (local pool, local lease)."""
        if not isinstance(lease, GlobalLease):
            raise InvalidLease("expected GlobalLease")
        if lease.pool_id != self._pool_id:
            raise InvalidLease("lease belongs to another global pool")
        if lease.owner_rank not in self._pools:
            raise InvalidLease("lease owner rank has no physical slot")
        pool = self._pools[lease.owner_rank]
        # Construct the real local lease after all identity checks. SlotPool
        # repeats pool/generation/request/layer/expert validation.
        from slot_pool import SlotLease

        local = SlotLease(
            pool.pool_id,
            lease.local_slot,
            lease.generation,
            lease.layer_id,
            lease.expert_id,
            lease.request_id,
        )
        return pool, local

    def _synchronize_or_keep_reserved(self, lease: GlobalLease, stage: str):
        if self._synchronize is None:
            return
        try:
            self._synchronize()
        except Exception as exc:
            # No state transition: the RESERVED/READY lease remains owned and
            # cannot be acquired by another request until a future safe retry.
            raise RuntimeError(f"device synchronize failed during {stage}") from exc

    def slot_buffer(self, lease: GlobalLease, *, allow_reserved=False):
        pool, local = self._validated_local(lease)
        return pool.slot_buffer(local, allow_reserved=allow_reserved)

    def mark_ready(self, lease: GlobalLease):
        pool, local = self._validated_local(lease)
        self._synchronize_or_keep_reserved(lease, "mark_ready")
        try:
            pool.mark_ready(local)
        except Exception:
            # The install itself failed before publication. Synchronize again
            # before recycling the slot so a partially submitted write cannot
            # land in the next generation.
            try:
                self._synchronize_or_keep_reserved(lease, "mark_ready cleanup")
            except Exception:
                raise
            pool.abort(local)
            with self._condition:
                self._condition.notify_all()
            raise

    def release(self, lease: GlobalLease):
        pool, local = self._validated_local(lease)
        # Gate first: no new use can enter after this point. Existing users from
        # other threads are drained before the required device synchronization.
        pool.begin_release(local)
        try:
            self._synchronize_or_keep_reserved(lease, "release")
        except Exception:
            # Keep the slot in RELEASING. A later safe retry may finish release;
            # it is never returned to the free list after a failed synchronization.
            raise
        pool.finish_release(local)
        with self._condition:
            self._condition.notify_all()

    def use(self, lease: GlobalLease):
        pool, local = self._validated_local(lease)
        return pool.use(local)

    def abort(self, lease: GlobalLease):
        pool, local = self._validated_local(lease)
        # Mandatory before recycling: an asynchronous write/HCCL operation may
        # still target the reserved bytes. If synchronization fails, leave the
        # lease RESERVED rather than allowing generation N+1 to reuse it.
        self._synchronize_or_keep_reserved(lease, "abort")
        pool.abort(local)
        with self._condition:
            self._condition.notify_all()

    def stats(self):
        local = {r: p.stats() for r, p in self._pools.items()}
        return {
            "K": self.capacity,
            "slot_bytes": self.slot_bytes,
            "committed_bytes": self.committed_bytes,
            "per_rank_capacity": self.per_rank_capacity(),
            "per_rank_allocated_bytes": self.per_rank_allocated_bytes(),
            "active_bytes": sum(
                s.slot_bytes
                * (s.reserved + s.ready + s.in_use + s.releasing)
                for s in local.values()
            ),
            "cached_free_bytes": sum(s.slot_bytes * s.free for s in local.values()),
            "local": local,
        }
