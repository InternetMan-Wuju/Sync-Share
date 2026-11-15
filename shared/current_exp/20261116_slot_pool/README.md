# Fixed-capacity expert replica slot pool

## Scope

This is a first implementation and correctness prototype for a per-rank fixed-capacity expert replica slot pool. It is deliberately separate from the model's per-layer 17th slot, so it can be tested without claiming that the production MoE dispatch already reads from the pool.

Lifecycle:

```text
acquire(layer_id, expert_id, request_id)
  -> RESERVED
  -> caller packs expert bytes into the slot
  -> mark_ready()
  -> use() context manager
  -> release()
  -> FREE
  -> next acquire() reuses the same fixed slot
```

Every lease contains a slot generation. Stale leases cannot be marked ready,
used, or released after a slot has been reused. Concurrent callers are
serialized under one condition lock and can never receive the same live slot.

## Files

| File | Purpose |
|---|---|
| `slot_pool.py` | Fixed-capacity thread-safe slot lifecycle |
| `expert_slot_state.py` | Pack/install expert state tensors into/out of a slot |
| `test_slot_pool.py` | CPU lifecycle, stale-lease, exhaustion, concurrency tests |
| `test_npu_slot_pool.py` | Real Ascend NPU pool lifecycle/reuse test |
| `test_npu_expert_state.py` | Real NPU state pack/install test |

## Backing storage

The allocator is called exactly once:

```text
allocated_bytes = capacity * slot_bytes
```

No layer dimension is present in the backing store. The 75-layer CPU test and
the 75-cycle NPU test both reuse the same two slots.

For this test, `slot_bytes=37,810,176`, matching the previously measured GLM
expert-state size. With capacity 2:

```text
2 x 37,810,176 = 75,620,352 bytes (about 75.6 MB)
```

The byte capacity is independent of the number of MoE layers. This prototype
does not yet remove the model's existing per-layer spare tensor; full dispatch
integration is required before claiming end-to-end memory savings.

## Reproduce

CPU:

```bash
python3 test_slot_pool.py
```

NPU on `npu1-19`, device 0:

```bash
sudo -n docker run --rm --name cyt-slot-pool-n19 --net host --shm-size 4g \
  --security-opt seccomp=unconfined \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /home/chengyetai/workspace/current_exp/20261116_slot_pool:/runexp:ro \
  -e PYTHONPATH=/runexp \
  --entrypoint bash quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3 -lc \
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
   source /usr/local/Ascend/nnal/atb/set_env.sh && \
   python3 /runexp/test_npu_slot_pool.py --capacity 2 --cycles 75'
```

For the state pack/install test, run `test_npu_expert_state.py` in the same
container environment.

## Integration boundary

The current model's fused MoE path reads weights from layer-owned tensors
(`w13_weight`, `w2_weight`, scales, offsets, ...). A pool slot is only useful
for the real EPLB path after one of these is implemented:

1. an external-slot-aware dispatch/MoE kernel that accepts a pool slot handle;
2. a layer weight view/indirection that can point at a pool slot without copying
   into a per-layer 17th slot.

Until then, this code validates the pool/lease/state-copy machinery only. It
does not claim that the existing model's extra per-layer memory has already
been reduced.
