#!/usr/bin/env python3
"""NPU check: pack a synthetic expert state through a pool and install it back."""
import json
import torch
import torch_npu
from slot_pool import SlotPool
from expert_slot_state import StateSpec, pack_and_mark_ready, install_and_release


class Layer:
    def __init__(self, seed=0):
        self.w13_weight = torch.full((2, 4, 4), seed + 1, dtype=torch.int8, device='npu')
        self.w2_weight = torch.full((2, 4, 4), seed + 2, dtype=torch.int8, device='npu')
        self.w13_weight_scale = torch.full((2, 4), seed + 3, dtype=torch.float16, device='npu')
        self.w2_weight_scale = torch.full((2, 4), seed + 4, dtype=torch.float16, device='npu')


def alloc(capacity, slot_bytes):
    return torch.empty((capacity, slot_bytes), dtype=torch.uint8, device='npu')


def main():
    torch.npu.set_device(0)
    source = Layer(10)
    target = Layer(90)
    spec = StateSpec.from_layer(source)
    pool = SlotPool(2, spec.total_bytes, alloc, name='npu-expert-state-test')
    lease = pool.acquire(49, 0, 'state-copy')
    pack_and_mark_ready(pool, lease, source, 0, spec)
    install_and_release(pool, lease, target, 0, spec)
    torch.npu.synchronize()
    checks = {}
    for name in ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale'):
        checks[name] = bool(torch.equal(getattr(source, name)[0], getattr(target, name)[0]))
    ok = all(checks.values())
    print(json.dumps({
        'status': 'PASS' if ok else 'FAIL',
        'slot_bytes': spec.total_bytes,
        'allocated_bytes': pool.stats().allocated_bytes,
        'checks': checks,
        'acquisitions': pool.stats().acquisitions,
        'releases': pool.stats().releases,
    }, indent=2))
    if not ok:
        raise SystemExit(1)
    del pool
    torch.npu.synchronize()


if __name__ == '__main__':
    main()
