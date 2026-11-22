#!/usr/bin/env python3
"""Fixed-capacity, thread-safe expert replica slot pool.

The pool owns one flat backing store with exactly ``capacity * slot_bytes``
bytes. It has no per-layer backing storage. Leases bind pool identity,
generation, request, layer, and expert. All public lease operations validate
all of those fields before touching the backing store.

Asynchronous callers must synchronize device work before ``mark_ready`` and
before ``abort``/``release`` return a slot for reuse. The global pool wrapper
enforces that contract with an explicit synchronize callback.
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional


class SlotPoolError(RuntimeError):
    pass


class SlotPoolExhausted(SlotPoolError):
    pass


class InvalidLease(SlotPoolError):
    pass


class SlotState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    READY = "ready"
    IN_USE = "in_use"
    RELEASING = "releasing"


@dataclass(frozen=True)
class SlotLease:
    pool_id: str
    slot_id: int
    generation: int
    layer_id: int
    expert_id: int
    request_id: str


@dataclass
class _SlotRecord:
    state: SlotState = SlotState.FREE
    generation: int = 0
    layer_id: Optional[int] = None
    expert_id: Optional[int] = None
    request_id: Optional[str] = None
    users: int = 0
    users_by_thread: dict[int, int] = field(default_factory=dict)
    reserved_at_ns: Optional[int] = None
    ready_at_ns: Optional[int] = None


@dataclass
class PoolStats:
    pool_id: str
    capacity: int
    slot_bytes: int
    allocated_bytes: int
    free: int
    reserved: int
    ready: int
    in_use: int
    releasing: int
    acquisitions: int
    releases: int
    reuses: int
    max_concurrent_leases: int
    max_concurrent_uses: int
    distinct_layers_seen: int
    distinct_experts_seen: int
    backing_reallocations: int = 0
    invariant_violations: int = 0


class SlotPool:
    """A fixed-capacity pool of expert-state slots.

    ``allocator(capacity, slot_bytes)`` must return an indexable object
    supporting ``obj[slot_id] -> writable slot view``. CPU tests use a list of
    bytearrays; NPU uses a ``torch.uint8`` tensor shaped
    ``[capacity, slot_bytes]``.
    """

    def __init__(
        self,
        capacity: int,
        slot_bytes: int,
        allocator: Callable[[int, int], Any],
        *,
        name: str = "expert-slot-pool",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        self.pool_id = f"{name}:{uuid.uuid4().hex[:12]}"
        self.capacity = int(capacity)
        self.slot_bytes = int(slot_bytes)
        self._storage = allocator(self.capacity, self.slot_bytes)
        if len(self._storage) != self.capacity:
            raise ValueError("allocator returned wrong capacity")
        self._records = [_SlotRecord() for _ in range(self.capacity)]
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._acquisitions = 0
        self._releases = 0
        self._reuses = 0
        self._active_leases = 0
        self._active_uses = 0
        self._max_concurrent_leases = 0
        self._max_concurrent_uses = 0
        self._layers_seen: set[int] = set()
        self._experts_seen: set[tuple[int, int]] = set()

    @property
    def allocated_bytes(self) -> int:
        return self.capacity * self.slot_bytes

    def _validate_locked(self, lease: SlotLease) -> _SlotRecord:
        if not isinstance(lease, SlotLease):
            raise InvalidLease("expected SlotLease")
        if lease.pool_id != self.pool_id:
            raise InvalidLease("lease belongs to another pool")
        if not 0 <= lease.slot_id < self.capacity:
            raise InvalidLease("slot id out of range")
        rec = self._records[lease.slot_id]
        if rec.generation != lease.generation:
            raise InvalidLease("stale lease generation")
        if rec.request_id != lease.request_id:
            raise InvalidLease("lease request does not own slot")
        if rec.layer_id != lease.layer_id:
            raise InvalidLease("lease layer identity does not match slot")
        if rec.expert_id != lease.expert_id:
            raise InvalidLease("lease expert identity does not match slot")
        return rec

    def _find_free_locked(self) -> Optional[int]:
        for idx, rec in enumerate(self._records):
            if rec.state is SlotState.FREE:
                return idx
        return None

    def acquire(
        self,
        layer_id: int,
        expert_id: int,
        request_id: str,
        *,
        timeout_s: Optional[float] = None,
        wait: bool = True,
    ) -> SlotLease:
        """Reserve one free slot; concurrent callers never receive the same one."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._closed:
                    raise SlotPoolError("pool is closed")
                slot_id = self._find_free_locked()
                if slot_id is not None:
                    break
                if not wait:
                    raise SlotPoolExhausted("no free expert slot")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise SlotPoolExhausted("timed out waiting for a free expert slot")
                self._condition.wait(remaining)

            rec = self._records[slot_id]
            generation = rec.generation + 1
            rec.state = SlotState.RESERVED
            rec.generation = generation
            rec.layer_id = int(layer_id)
            rec.expert_id = int(expert_id)
            rec.request_id = str(request_id)
            rec.users = 0
            rec.users_by_thread = {}
            rec.reserved_at_ns = time.monotonic_ns()
            rec.ready_at_ns = None

            self._acquisitions += 1
            self._active_leases += 1
            self._max_concurrent_leases = max(
                self._max_concurrent_leases, self._active_leases
            )
            if generation > 1:
                self._reuses += 1
            self._layers_seen.add(int(layer_id))
            self._experts_seen.add((int(layer_id), int(expert_id)))

            return SlotLease(
                pool_id=self.pool_id,
                slot_id=slot_id,
                generation=generation,
                layer_id=int(layer_id),
                expert_id=int(expert_id),
                request_id=str(request_id),
            )

    def slot_buffer(self, lease: SlotLease, *, allow_reserved: bool = False) -> Any:
        """Return the backing view if and only if all lease identity fields match."""
        with self._lock:
            rec = self._validate_locked(lease)
            allowed = {SlotState.READY, SlotState.IN_USE}
            if allow_reserved:
                allowed.add(SlotState.RESERVED)
            if rec.state not in allowed:
                raise InvalidLease(f"slot is {rec.state.value}, not readable/writable")
            return self._storage[lease.slot_id]

    def mark_ready(self, lease: SlotLease) -> None:
        """Publish a RESERVED slot after the caller synchronized device writes."""
        with self._condition:
            rec = self._validate_locked(lease)
            if rec.state is not SlotState.RESERVED:
                raise InvalidLease(f"cannot mark ready from {rec.state.value}")
            rec.state = SlotState.READY
            rec.ready_at_ns = time.monotonic_ns()
            self._condition.notify_all()

    @contextmanager
    def use(self, lease: SlotLease) -> Iterator[Any]:
        """Enter a read/use critical section; release is rejected while active."""
        with self._condition:
            rec = self._validate_locked(lease)
            if rec.state not in (SlotState.READY, SlotState.IN_USE):
                raise InvalidLease(f"cannot use a {rec.state.value} slot")
            if rec.state is SlotState.READY:
                rec.state = SlotState.IN_USE
            rec.users += 1
            owner = threading.get_ident()
            rec.users_by_thread[owner] = rec.users_by_thread.get(owner, 0) + 1
            self._active_uses += 1
            self._max_concurrent_uses = max(
                self._max_concurrent_uses, self._active_uses
            )
            view = self._storage[lease.slot_id]
        try:
            yield view
        finally:
            with self._condition:
                rec = self._validate_locked(lease)
                rec.users -= 1
                owner = threading.get_ident()
                own_count = rec.users_by_thread.get(owner, 0)
                if own_count <= 1:
                    rec.users_by_thread.pop(owner, None)
                else:
                    rec.users_by_thread[owner] = own_count - 1
                self._active_uses -= 1
                if rec.users == 0 and rec.state is SlotState.IN_USE:
                    rec.state = SlotState.READY
                self._condition.notify_all()

    def begin_release(self, lease: SlotLease) -> None:
        """Gate a slot for release and wait for users from other threads.

        The state becomes RELEASING before waiting, so no new use can enter while
        device completion is being established. A same-thread release inside an
        active use context is rejected rather than deadlocking that thread.
        """
        with self._condition:
            rec = self._validate_locked(lease)
            if rec.state not in (
                SlotState.READY,
                SlotState.IN_USE,
                SlotState.RELEASING,
            ):
                raise InvalidLease(f"cannot release from {rec.state.value}")
            owner = threading.get_ident()
            if rec.users_by_thread.get(owner, 0):
                raise InvalidLease("cannot release while this thread has an active use")
            rec.state = SlotState.RELEASING
            while rec.users != 0:
                self._condition.wait()
            self._condition.notify_all()

    def finish_release(self, lease: SlotLease) -> None:
        """Free a slot already gated by begin_release()."""
        with self._condition:
            rec = self._validate_locked(lease)
            if rec.state is not SlotState.RELEASING:
                raise InvalidLease(f"cannot finish release from {rec.state.value}")
            if rec.users != 0:
                raise InvalidLease("cannot finish release with active users")
            rec.state = SlotState.FREE
            rec.layer_id = None
            rec.expert_id = None
            rec.request_id = None
            rec.users = 0
            rec.users_by_thread = {}
            rec.reserved_at_ns = None
            rec.ready_at_ns = None
            self._active_leases -= 1
            self._releases += 1
            self._condition.notify_all()

    def release(self, lease: SlotLease) -> None:
        """Gate, wait, and return a ready slot to the free list."""
        self.begin_release(lease)
        self.finish_release(lease)

    def abort(self, lease: SlotLease) -> None:
        """Abort a RESERVED lease without publishing it."""
        with self._condition:
            rec = self._validate_locked(lease)
            if rec.state is not SlotState.RESERVED:
                raise InvalidLease(f"cannot abort from {rec.state.value}")
            if rec.users != 0:
                raise InvalidLease("cannot abort while slot has users")
            rec.state = SlotState.FREE
            rec.layer_id = None
            rec.expert_id = None
            rec.request_id = None
            rec.users = 0
            rec.reserved_at_ns = None
            rec.ready_at_ns = None
            self._active_leases -= 1
            self._condition.notify_all()

    def stats(self) -> PoolStats:
        with self._lock:
            counts = {s: 0 for s in SlotState}
            for rec in self._records:
                counts[rec.state] += 1
            return PoolStats(
                pool_id=self.pool_id,
                capacity=self.capacity,
                slot_bytes=self.slot_bytes,
                allocated_bytes=self.allocated_bytes,
                free=counts[SlotState.FREE],
                reserved=counts[SlotState.RESERVED],
                ready=counts[SlotState.READY],
                in_use=counts[SlotState.IN_USE],
                releasing=counts[SlotState.RELEASING],
                acquisitions=self._acquisitions,
                releases=self._releases,
                reuses=self._reuses,
                max_concurrent_leases=self._max_concurrent_leases,
                max_concurrent_uses=self._max_concurrent_uses,
                distinct_layers_seen=len(self._layers_seen),
                distinct_experts_seen=len(self._experts_seen),
            )

    def close(self) -> None:
        with self._condition:
            if any(rec.state is not SlotState.FREE for rec in self._records):
                raise SlotPoolError("cannot close pool with live slots")
            self._closed = True
            self._condition.notify_all()
