#!/usr/bin/env python3
"""Real-NPU W8A8 probe: external slot weight/scale participates in GMM."""
import json
import torch
import torch_npu
from global_slot_pool import GlobalSlotPool


def pool_alloc(capacity, nbytes):
    return torch.empty((capacity, nbytes), dtype=torch.uint8, device='npu')


def gmm(x, weights, scales, group_list):
    return torch.ops.npu.npu_grouped_matmul(
        x=[x], weight=weights, scale=scales,
        per_token_scale=[torch.ones((x.shape[0],), dtype=torch.float32, device='npu')],
        group_list=group_list, split_item=2, group_type=0,
        group_list_type=1, output_dtype=torch.bfloat16,
    )[0]


def main():
    torch.npu.set_device(0)
    torch.manual_seed(7)
    x = torch.randint(-8, 8, (6, 64), dtype=torch.int8, device='npu')
    main_w = torch.randint(-8, 8, (2, 64, 64), dtype=torch.int8, device='npu')
    main_s = torch.ones((2, 64), dtype=torch.bfloat16, device='npu')
    ext_w = torch.randint(-8, 8, (1, 64, 64), dtype=torch.int8, device='npu')
    ext_s = torch.ones((1, 64), dtype=torch.bfloat16, device='npu')
    gl = torch.tensor([2, 2, 2], dtype=torch.int64, device='npu')

    # K=1, owner rank 1 only. Rank 0 has no backing allocation.
    wbytes = ext_w.numel() * ext_w.element_size()
    sbytes = ext_s.numel() * ext_s.element_size()
    pool = GlobalSlotPool([1], wbytes + sbytes, pool_alloc, name='w8a8-external-slot')
    lease = pool.acquire(request_id='probe', layer_id=49, expert_id=2)
    slot = pool.slot_buffer(lease, allow_reserved=True)
    slot[:wbytes].copy_(ext_w.view(torch.uint8).reshape(-1))
    slot[wbytes:wbytes+sbytes].copy_(ext_s.view(torch.uint8).reshape(-1))
    torch.npu.synchronize()
    pool.mark_ready(lease)
    with pool.use(lease) as used:
        ext_w_slot = used[:wbytes].view(torch.int8).reshape(64, 64)
        ext_s_slot = used[wbytes:wbytes+sbytes].view(torch.bfloat16).reshape(64)
        bank = torch.cat([main_w, ext_w], dim=0)
        bank_s = torch.cat([main_s, ext_s], dim=0)
        baseline = gmm(x, [bank], [bank_s], gl)
        split = gmm(x, [main_w[0], main_w[1], ext_w_slot],
                    [main_s[0], main_s[1], ext_s_slot], gl)
        torch.npu.synchronize()
        diff = float((baseline.float() - split.float()).abs().max().item())
        out = {
            'status': 'PASS' if diff == 0.0 else 'FAIL',
            'K': pool.capacity,
            'owner_ranks': list(pool.owner_ranks),
            'per_rank_capacity': pool.per_rank_capacity(),
            'committed_bytes': pool.stats()['committed_bytes'],
            'output_shape': list(split.shape),
            'max_abs_diff': diff,
            'external_slot_used': True,
        }
    pool.release(lease)
    print(json.dumps(out, indent=2))
    if diff != 0.0:
        raise SystemExit(1)
    del pool
    torch.npu.synchronize()


if __name__ == '__main__':
    main()
