#!/usr/bin/env python3
"""Small real-NPU correctness test for the fixed-capacity slot pool.

This validates the pool lifecycle and backing storage on an actual Ascend NPU.
It does not yet replace the model's per-layer 17th slot; full dispatch integration
is a separate step.
"""
import argparse
import json
import time

import torch
import torch_npu

from slot_pool import SlotPool


def alloc(capacity, slot_bytes):
    return torch.empty((capacity, slot_bytes), dtype=torch.uint8, device='npu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slot-bytes', type=int, default=37_810_176)
    ap.add_argument('--capacity', type=int, default=2)
    ap.add_argument('--cycles', type=int, default=8)
    args = ap.parse_args()

    torch.npu.set_device(0)
    torch.npu.synchronize()
    print('NPU_TEST_START', torch.npu.current_device(), args.capacity, args.slot_bytes, args.cycles, flush=True)

    pool = SlotPool(args.capacity, args.slot_bytes, alloc, name='npu-slot-pool')
    started = time.perf_counter()
    leases = []
    # Two live slots at once: prove capacity is fixed, not layer-count dependent.
    for i in range(args.capacity):
        lease = pool.acquire(100 + i, i, f'concurrent-{i}')
        leases.append(lease)
        slot = pool.slot_buffer(lease, allow_reserved=True)
        slot.fill_(0x40 + i)
        torch.npu.synchronize()
        pool.mark_ready(lease)
    for i, lease in enumerate(leases):
        with pool.use(lease) as slot:
            got = int(slot[0].item())
            assert got == 0x40 + i, (i, got)
    for lease in leases:
        pool.release(lease)

    # Cross-layer reuse: same fixed pool serves different logical layers.
    for layer in range(args.cycles):
        lease = pool.acquire(200 + layer, layer % 16, f'reuse-{layer}')
        slot = pool.slot_buffer(lease, allow_reserved=True)
        slot.fill_(layer & 0xFF)
        torch.npu.synchronize()
        pool.mark_ready(lease)
        with pool.use(lease) as used:
            assert int(used[0].item()) == (layer & 0xFF)
            assert int(used[-1].item()) == (layer & 0xFF)
        pool.release(lease)

    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    s = pool.stats()
    result = {
        'device': int(torch.npu.current_device()),
        'capacity': s.capacity,
        'slot_bytes': s.slot_bytes,
        'allocated_bytes': s.allocated_bytes,
        'cycles': args.cycles,
        'acquisitions': s.acquisitions,
        'releases': s.releases,
        'reuses': s.reuses,
        'max_concurrent_leases': s.max_concurrent_leases,
        'distinct_layers_seen': s.distinct_layers_seen,
        'free': s.free,
        'elapsed_s': elapsed,
        'status': 'PASS',
    }
    print(json.dumps(result, indent=2), flush=True)
    # Keep pool allocation alive until stats; drop it after synchronize.
    del pool
    torch.npu.synchronize()


if __name__ == '__main__':
    main()
