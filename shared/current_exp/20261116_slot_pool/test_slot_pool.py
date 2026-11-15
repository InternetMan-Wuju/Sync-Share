#!/usr/bin/env python3
import threading
import time
import unittest
from contextlib import contextmanager

from slot_pool import InvalidLease, SlotPool


def bytearray_allocator(capacity, slot_bytes):
    return [bytearray(slot_bytes) for _ in range(capacity)]


class SlotPoolTests(unittest.TestCase):
    def test_cross_layer_lifecycle_and_fixed_allocation(self):
        calls = []

        def allocator(capacity, nbytes):
            calls.append((capacity, nbytes))
            return bytearray_allocator(capacity, nbytes)

        pool = SlotPool(2, 32, allocator, name='cross-layer-test')
        for layer in range(75):
            lease = pool.acquire(layer, layer % 16, f'layer-{layer}')
            slot = pool.slot_buffer(lease, allow_reserved=True)
            slot[:] = bytes([layer % 256]) * 32
            pool.mark_ready(lease)
            with pool.use(lease) as used:
                self.assertEqual(used[0], layer % 256)
                self.assertEqual(len(used), 32)
            pool.release(lease)

        s = pool.stats()
        self.assertEqual(calls, [(2, 32)])
        self.assertEqual(s.allocated_bytes, 64)  # independent of 75 layers
        self.assertEqual(s.free, 2)
        self.assertEqual(s.acquisitions, 75)
        self.assertEqual(s.releases, 75)
        self.assertGreater(s.reuses, 0)
        self.assertEqual(s.distinct_layers_seen, 75)
        pool.close()

    def test_stale_lease_and_ready_gate(self):
        pool = SlotPool(1, 8, bytearray_allocator, name='lease-test')
        lease = pool.acquire(1, 2, 'r1')
        with self.assertRaises(InvalidLease):
            with pool.use(lease):
                pass
        pool.mark_ready(lease)
        with pool.use(lease):
            pass
        pool.release(lease)

        lease2 = pool.acquire(2, 3, 'r2')
        with self.assertRaises(InvalidLease):
            pool.mark_ready(lease)
        with self.assertRaises(InvalidLease):
            with pool.use(lease):
                pass
        with self.assertRaises(InvalidLease):
            pool.release(lease)
        pool.abort(lease2)
        pool.close()

    def test_concurrent_acquire_never_conflicts(self):
        pool = SlotPool(2, 16, bytearray_allocator, name='concurrency-test')
        barrier = threading.Barrier(6)
        active = set()
        active_guard = threading.Lock()
        max_active = 0
        seen_requests = set()
        seen_guard = threading.Lock()
        errors = []

        def worker(i):
            nonlocal max_active
            try:
                barrier.wait(timeout=2)
                lease = pool.acquire(0, i, f'r{i}', timeout_s=5)
                with active_guard:
                    if lease.slot_id in active:
                        raise AssertionError(f'slot {lease.slot_id} handed out twice')
                    active.add(lease.slot_id)
                    max_active = max(max_active, len(active))
                    seen_requests.add(lease.request_id)
                pool.mark_ready(lease)
                with pool.use(lease) as slot:
                    slot[0] = i
                    time.sleep(0.04)
                pool.release(lease)
                with active_guard:
                    active.remove(lease.slot_id)
            except Exception as exc:  # collect for the main thread
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertFalse(any(t.is_alive() for t in threads), 'worker hung')
        self.assertEqual(errors, [])
        self.assertEqual(max_active, 2)
        s = pool.stats()
        self.assertEqual(s.acquisitions, 6)
        self.assertEqual(s.releases, 6)
        self.assertGreaterEqual(s.reuses, 4)
        self.assertEqual(s.max_concurrent_leases, 2)
        self.assertEqual(s.free, 2)

    def test_nonblocking_exhaustion_and_abort_notification(self):
        pool = SlotPool(1, 4, bytearray_allocator, name='exhaustion-test')
        first = pool.acquire(10, 1, 'first')
        with self.assertRaises(Exception):
            pool.acquire(11, 2, 'second', wait=False)

        result = []

        def waiter():
            lease = pool.acquire(11, 2, 'second', timeout_s=2)
            pool.mark_ready(lease)
            pool.release(lease)
            result.append(True)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        pool.abort(first)
        t.join(timeout=3)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [True])
        pool.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
