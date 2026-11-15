#!/usr/bin/env python3
import json
import torch
import torch_npu
from global_slot_pool import GlobalSlotPool


def alloc(capacity, nbytes):
    return torch.empty((capacity, nbytes), dtype=torch.uint8, device='npu')


def main():
    torch.npu.set_device(0)
    # K=3, not a multiple of the 16-rank EP group. Rank 0 has no slot.
    p = GlobalSlotPool([2, 2, 5], 37_810_176, alloc, name='npu-global-k-probe')
    leases = [p.acquire(request_id=f'r{i}', layer_id=10+i, expert_id=i) for i in range(3)]
    for i, lease in enumerate(leases):
        b = p.slot_buffer(lease, allow_reserved=True)
        b.fill_(0x30 + i)
        torch.npu.synchronize()
        p.mark_ready(lease)
    for i, lease in enumerate(leases):
        with p.use(lease) as b:
            assert int(b[0].item()) == 0x30+i
            assert int(b[-1].item()) == 0x30+i
    for lease in leases:
        p.release(lease)
    # Cross-layer reuse of the same global K slots.
    for layer in range(8):
        lease = p.acquire(request_id=f'reuse-{layer}', layer_id=layer, expert_id=layer % 5)
        b = p.slot_buffer(lease, allow_reserved=True)
        b.fill_(layer & 0xFF)
        torch.npu.synchronize()
        p.mark_ready(lease)
        with p.use(lease) as used:
            assert int(used[0].item()) == (layer & 0xFF)
        p.release(lease)
    torch.npu.synchronize()
    torch.npu.synchronize()
    s = p.stats()
    out = {
        'status': 'PASS',
        'K': s['K'],
        'slot_bytes': s['slot_bytes'],
        'committed_bytes': s['committed_bytes'],
        'per_rank_capacity': s['per_rank_capacity'],
        'per_rank_allocated_bytes': s['per_rank_allocated_bytes'],
        'active_bytes_after_release': s['active_bytes'],
        'cached_free_bytes_after_release': s['cached_free_bytes'],
        'torch_memory_allocated_after_release': int(torch.npu.memory_allocated()),
        'torch_memory_reserved_after_release': int(torch.npu.memory_reserved()),
        'torch_max_memory_allocated': int(torch.npu.max_memory_allocated()),
        'local_reuses': {str(r): v.reuses for r,v in s['local'].items()},
    }
    print(json.dumps(out, indent=2))
    assert s['K'] == 3
    assert s['committed_bytes'] == 3 * 37_810_176
    assert s['per_rank_capacity'] == {2: 2, 5: 1}
    assert s['active_bytes'] == 0
    assert s['cached_free_bytes'] == s['committed_bytes']
    del p
    torch.npu.synchronize()


if __name__ == '__main__':
    main()
