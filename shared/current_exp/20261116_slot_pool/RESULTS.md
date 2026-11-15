# Slot pool correctness results

## Implemented

- Per-rank fixed-capacity slot pool.
- Flat backing allocation: `capacity * slot_bytes`, allocated once.
- States: `RESERVED -> READY -> IN_USE -> FREE`.
- Lease generation validation prevents stale-handle reuse.
- Concurrent acquire/release with condition-variable waiting.
- Cross-layer reuse.
- Expert state tensor packing/install bridge.
- No per-layer allocation in the pool itself.

## CPU tests

Command:

```bash
python3 test_slot_pool.py
```

Result: **4 tests passed**.

Key concurrent test:

- capacity: 2
- concurrent workers: 6
- max concurrent leases: 2
- acquisitions/releases: 6/6
- reuses: at least 4
- no duplicate live slot
- no hung worker

Cross-layer test:

- logical layers: 75
- capacity: 2
- backing allocator calls: 1
- allocated bytes: 64 bytes for the tiny test
- acquisitions/releases: 75/75

## NPU tests on `npu1-19`

Device: `davinci0` (Ascend 910)
Image: `quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`

### Pool lifecycle and reuse

- slot capacity: 2
- slot size: **37,810,176 bytes**
- allocated bytes: **75,620,352 bytes**
- logical layers exercised: 75
- acquisitions: 77
- releases: 77
- reuses: 75
- max concurrent leases: 2
- free at end: 2
- elapsed: **0.416 s**
- status: **PASS**

This shows that the backing allocation stayed at two expert-sized slots while
serving 75 logical layer cycles.

### Expert state pack/install

A synthetic multi-field expert state was packed into a pool slot, marked ready,
installed into a target layer slot, and compared field by field.

- status: **PASS**
- checked fields: `w13_weight`, `w2_weight`, `w13_weight_scale`, `w2_weight_scale`
- acquisitions/releases: 1/1

## What is not yet validated

- Full model dispatch using a pool slot as the actual MoE compute slot.
- End-to-end removal of the model's per-layer 17th tensor.
- Distributed ready/barrier semantics for a shared pool slot across ranks.
- Concurrent model requests with real MoE dispatch and pool leases.
- Performance/copy overlap under production scheduling.

The current result is a correctness prototype, not a production memory-saving
claim.
