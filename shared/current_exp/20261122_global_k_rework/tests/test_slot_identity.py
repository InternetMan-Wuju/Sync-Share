#!/usr/bin/env python3
import sys
import threading
import types
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from global_slot_pool import GlobalSlotPool
from slot_pool import InvalidLease, SlotLease, SlotPoolExhausted


def alloc(capacity, nbytes):
    return [bytearray(nbytes) for _ in range(capacity)]


class SlotIdentityTests(unittest.TestCase):
    def make_pair(self):
        calls = {"a": [], "b": []}
        a = GlobalSlotPool(
            [1], 8, alloc, name="a", synchronize=lambda: calls["a"].append("sync")
        )
        b = GlobalSlotPool(
            [1], 8, alloc, name="b", synchronize=lambda: calls["b"].append("sync")
        )
        return a, b, calls

    def test_foreign_reserved_and_ready_lease_rejected_by_every_entry(self):
        a, b, calls = self.make_pair()
        foreign_reserved = a.acquire(request_id="ra", layer_id=7, expert_id=11)
        own_reserved = b.acquire(request_id="rb", layer_id=7, expert_id=11)
        b_bytes = b.slot_buffer(own_reserved, allow_reserved=True)
        b_bytes[:] = b"BBBBBBBB"

        operations = [
            lambda: b.slot_buffer(foreign_reserved, allow_reserved=True),
            lambda: b.mark_ready(foreign_reserved),
            lambda: b.use(foreign_reserved),
            lambda: b.release(foreign_reserved),
            lambda: b.abort(foreign_reserved),
        ]
        for op in operations:
            with self.assertRaises(InvalidLease):
                op()
        self.assertEqual(bytes(b_bytes), b"BBBBBBBB")
        self.assertEqual(calls["b"], [])
        self.assertEqual(b.stats()["local"][1].reserved, 1)

        a.mark_ready(foreign_reserved)
        operations_ready = [
            lambda: b.slot_buffer(foreign_reserved),
            lambda: b.mark_ready(foreign_reserved),
            lambda: b.use(foreign_reserved),
            lambda: b.release(foreign_reserved),
            lambda: b.abort(foreign_reserved),
        ]
        for op in operations_ready:
            with self.assertRaises(InvalidLease):
                op()
        self.assertEqual(bytes(b_bytes), b"BBBBBBBB")
        a.release(foreign_reserved)
        b.abort(own_reserved)

    def test_identity_validation_mutation_is_detected_if_bypassed(self):
        """Mutation control: the old implementation shape must fail this test."""
        a, b, _ = self.make_pair()
        lea = a.acquire(request_id="same", layer_id=3, expert_id=5)
        leb = b.acquire(request_id="same", layer_id=3, expert_id=5)
        target = b.slot_buffer(leb, allow_reserved=True)
        target[:] = b"ORIGINAL"

        def unsafe_local(self, lease):
            pool = self._pools[lease.owner_rank]
            return pool, SlotLease(
                pool.pool_id,
                lease.local_slot,
                lease.generation,
                lease.layer_id,
                lease.expert_id,
                lease.request_id,
            )

        b._validated_local = types.MethodType(unsafe_local, b)
        foreign_view = b.slot_buffer(lea, allow_reserved=True)
        foreign_view[:] = b"FOREIGN!"
        self.assertEqual(bytes(target), b"FOREIGN!")
        a.abort(lea)
        b.abort(leb)

    def test_forged_request_layer_and_expert_rejected(self):
        pool = GlobalSlotPool([0], 8, alloc, name="identity")
        lease = pool.acquire(request_id="r", layer_id=10, expert_id=20)
        forged = [
            replace(lease, request_id="other"),
            replace(lease, layer_id=11),
            replace(lease, expert_id=21),
            replace(lease, generation=lease.generation + 1),
        ]
        for item in forged:
            with self.assertRaises(InvalidLease):
                pool.slot_buffer(item, allow_reserved=True)
        view = pool.slot_buffer(lease, allow_reserved=True)
        view[:] = b"ABCDEFGH"
        pool.mark_ready(lease)
        self.assertEqual(bytes(pool.slot_buffer(lease)), b"ABCDEFGH")

    def test_unready_read_rejected(self):
        pool = GlobalSlotPool([0], 8, alloc, name="unready")
        lease = pool.acquire(request_id="r", layer_id=1, expert_id=2)
        with self.assertRaises(InvalidLease):
            pool.slot_buffer(lease)
        with self.assertRaises(InvalidLease):
            with pool.use(lease):
                pass
        pool.abort(lease)

    def test_release_while_in_use_rejected_and_no_overwrite(self):
        pool = GlobalSlotPool([0], 8, alloc, name="in-use")
        lease = pool.acquire(request_id="r", layer_id=1, expert_id=2)
        view = pool.slot_buffer(lease, allow_reserved=True)
        view[:] = b"INUSE!!!"
        pool.mark_ready(lease)
        with pool.use(lease) as used_view:
            with self.assertRaises(InvalidLease):
                pool.release(lease)
            with self.assertRaises(SlotPoolExhausted):
                pool.acquire(request_id="next", layer_id=2, expert_id=3)
            self.assertEqual(bytes(used_view), b"INUSE!!!")
        pool.release(lease)
        self.assertEqual(pool.stats()["active_bytes"], 0)

    def test_stale_generation_rejected_after_reuse(self):
        pool = GlobalSlotPool([0], 8, alloc, name="stale")
        old = pool.acquire(request_id="old", layer_id=1, expert_id=2)
        pool.mark_ready(old)
        pool.release(old)
        new = pool.acquire(request_id="new", layer_id=3, expert_id=4)
        operations = {
            "slot_buffer": lambda: pool.slot_buffer(old, allow_reserved=True),
            "mark_ready": lambda: pool.mark_ready(old),
            "release": lambda: pool.release(old),
            "abort": lambda: pool.abort(old),
            "use": lambda: pool.use(old).__enter__(),
        }
        for name, op in operations.items():
            with self.subTest(operation=name):
                with self.assertRaises(InvalidLease):
                    op()
        pool.abort(new)

    def test_failed_mark_ready_cleanup_syncs_and_frees(self):
        calls = []
        pool = GlobalSlotPool(
            [0], 8, alloc, name="failed-install", synchronize=lambda: calls.append("sync")
        )
        lease = pool.acquire(request_id="bad", layer_id=1, expert_id=2)
        local_pool = pool._pools[0]
        original = local_pool.mark_ready

        def fail_install(_lease):
            raise RuntimeError("simulated HCCL/install failure")

        local_pool.mark_ready = fail_install
        try:
            with self.assertRaises(RuntimeError):
                pool.mark_ready(lease)
        finally:
            local_pool.mark_ready = original
        self.assertEqual(calls, ["sync", "sync"])
        self.assertEqual(pool.stats()["active_bytes"], 0)
        replacement = pool.acquire(request_id="good", layer_id=3, expert_id=4)
        self.assertGreater(replacement.generation, lease.generation)
        pool.abort(replacement)

    def test_abort_waits_for_delayed_write_completion(self):
        pool = None
        completed = threading.Event()
        abort_done = threading.Event()

        def delayed_sync():
            self.assertTrue(completed.wait(timeout=2.0), "test synchronization timed out")

        pool = GlobalSlotPool(
            [0], 8, alloc, name="delayed-write", synchronize=delayed_sync
        )
        lease = pool.acquire(request_id="delayed", layer_id=1, expert_id=2)

        def do_abort():
            pool.abort(lease)
            abort_done.set()

        thread = threading.Thread(target=do_abort)
        thread.start()
        self.assertFalse(abort_done.wait(timeout=0.10))
        self.assertEqual(pool.stats()["local"][0].reserved, 1)
        completed.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(pool.stats()["active_bytes"], 0)

    def test_abort_sync_failure_quarantines_until_safe_retry(self):
        fail = {"value": True}

        def sync():
            if fail["value"]:
                raise RuntimeError("device busy")

        pool = GlobalSlotPool([0], 8, alloc, name="abort-failure", synchronize=sync)
        lease = pool.acquire(request_id="r", layer_id=1, expert_id=2)
        with self.assertRaises(RuntimeError):
            pool.abort(lease)
        self.assertEqual(pool.stats()["local"][0].reserved, 1)
        with self.assertRaises(SlotPoolExhausted):
            pool.acquire(request_id="next", layer_id=2, expert_id=3)
        fail["value"] = False
        pool.abort(lease)
        self.assertEqual(pool.stats()["active_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
