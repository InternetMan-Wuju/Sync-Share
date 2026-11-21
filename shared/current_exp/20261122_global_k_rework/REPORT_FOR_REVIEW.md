# Global-K Safety, Cost Rework, and Oracle Validation

- Repository: `Sync-Share`
- Review baseline: `57f0e30`
- Execution base: `main@3b97ab7` (contains required `4133859`)
- Host: `npu1-23`
- Devices: real Ascend 910C `davinci0`, `davinci1`
- Image: `quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- Model: `/data_lib/data/models/glm-q`, ModelSlim W8A8 int8
- Scope: expert-child replay plus one routed-expert subchain
- Not measured: full MoE layer, full service throughput, TTFT, 32-card execution

This report treats only NPU-observed values as measurements. CPU oracle scores are toy integer units, not milliseconds. Old 1.1 ms and 0.36/0.45 ms numbers are not reused.

## Executive result

The correctness and safety work passed. The two-rank complete-boundary result is negative:

| Load | Mode | Median | p95 | Range |
|---|---:|---:|---:|---:|
| 6 tokens, 3+3 quota | original | **0.9836 ms** | 1.0271 ms | 0.9347-1.0363 ms |
| 6 tokens, 3+3 quota | K=1 cold replica | **2.5919 ms** | 2.6396 ms | 2.5422-2.9119 ms |
| 6 tokens, 3+3 quota | K=1 existing replica | **1.7343 ms** | 1.7781 ms | 1.6914-1.8763 ms |
| 512 tokens, 256+256 quota | original | **1.0156 ms** | 1.0276 ms | 0.9736-1.0360 ms |
| 512 tokens, 256+256 quota | K=1 cold replica | **2.8723 ms** | 3.1121 ms | 2.8447-3.1188 ms |
| 512 tokens, 256+256 quota | K=1 existing replica | **2.0183 ms** | 2.0462 ms | 1.9609-2.0550 ms |

Paired median differences (`original - replica`; negative means the replica path is slower):

| Load | Original-cold | Original-existing |
|---|---:|---:|
| 6 tokens | **-1.6242 ms** | **-0.7503 ms** |
| 512 tokens | **-1.8602 ms** | **-0.9957 ms** |

The destination was otherwise idle. These numbers show that for one 37.8 MB expert and the implemented serial control/transfer/refill path, the added cost is much larger than any observed useful split. They do **not** estimate a complete MoE layer or service result. No net benefit is claimed.

## 1. R1: slot isolation and failure recovery

Fixed in `scripts/slot_pool.py` and `scripts/global_slot_pool.py`:

- Every global lease entry validates pool identity, owner, request, generation, layer, and expert before touching slot storage.
- `slot_buffer()`, `mark_ready()`, `use()`, `release()`, and `abort()` reject foreign/stale/forged leases.
- `abort()` synchronizes device work before recycling the slot. If synchronization fails, the lease remains reserved and cannot be reacquired.
- Failed `mark_ready` cleanup synchronizes again before abort.
- Release/abort while in use is rejected. A reserved slot cannot be acquired twice.

`tests/test_slot_identity.py` has 9 passing regression tests covering:

- foreign reserved and ready leases against all public entry points;
- no mutation of the foreign pool's backing bytes;
- deliberate bypass of the central identity check, which makes the foreign access mutate the target pool, proving the test detects that regression;
- stale generation, forged request/layer/expert identity;
- unready read, in-use release, failed install cleanup;
- delayed asynchronous write completion and synchronization-failure quarantine.

The mutation-control test intentionally replaces the validation path and demonstrates that the old failure mode is observable. It is not a broad exception catch.

Evidence: `artifacts/cpu_slot_identity.txt`.

## 2. R2: single-card timing with a valid lease

The single-card probe uses the image's actual:

- `NPUW8A8Int8MoEMethod.process_weights_after_loading`;
- `GroupedMatmul.forward`;
- actual layer49/50 expert43 weights, scales and offsets;
- actual NZ bank and an external ND slot produced by `npu_format_cast(..., ND)`.

State contract:

- payload: **37,810,176 bytes**;
- `w13`: int8 `[1,6144,4096]`, 25,165,824 bytes;
- `w2`: int8 `[1,2048,6144]`, 12,582,912 bytes;
- `s13`: bf16 `[1,4096]`, 8,192 bytes;
- `s2`: bf16 `[1,6144]`, 12,288 bytes;
- `o13`: float32 `[1,4096]`, 16,384 bytes;
- `o2`: float32 `[1,6144]`, 24,576 bytes.

All loaded offset tensors are exactly zero. The installed 0.5.20 W8A8 runtime passes scale and per-token scale; it does not pass `weight_offset` in this path. The state still includes and verifies offsets.

Correctness:

| Check | Result |
|---|---:|
| all six state fields equal installed slot | PASS |
| full external chain vs actual bank | 0.0 max_abs |
| 3+3 split and refill | 0.0 max_abs |
| 6+0 split | 0.0 max_abs |
| 0+6 split / zero original tasks | 0.0 max_abs |
| task count conserved | PASS |

Timing used a still-valid lease for all modes, 20 warmups, 5 blocks, and adaptive block lengths around 100 ms. Raw records are retained.

| Mode | Median/call | Min | Max | Calls/block |
|---|---:|---:|---:|---:|
| bank | 0.2195 ms | 0.2132 ms | 0.2551 ms | 333 |
| external slot | 0.2113 ms | 0.2065 ms | 0.2534 ms | 343 |
| 3+3 split + refill | 0.6311 ms | 0.5569 ms | 0.6756 ms | 141 |

The split path is slower because it launches two expert chains and performs refill. The bank and external single-chain paths are indistinguishable at this scale.

Initial compile/format observations are separate from steady-state timing:

- first actual bank call for layer49: 200.40 ms;
- first layer50 bank call: 0.514 ms;
- first external full call after install: 1.340 ms;
- first split+refill call: 8.963 ms.

Single-card memory included the source bank, one K=1 slot, the actual runtime workspace and probe buffers:

- timing snapshot: allocated 113,958,400 B, reserved 192,937,984 B, peak allocated 132,635,136 B;
- no stale views were used after release; cross-layer measurements obtained a new valid lease/view.

Evidence: `artifacts/single_card_rework.json`, `artifacts/single_card_rework.log`.

## 3. R3: two-rank K=1 complete-boundary paired timing

Both ranks run the actual ModelSlim/NPU grouped-matmul path. The source rank has the actual processed bank; the destination rank allocates only one 37,810,176-byte external slot. No second permanent full-model copy is loaded on the destination.

The coordinator clock starts before the go notification and ends only after:

- cold NZ→ND conversion, packing and weight transfer when applicable;
- slot acquire, receive and ready;
- activation preparation/transfer;
- source/main compute and destination/external compute;
- result return and index refill;
- slot release, device synchronization and acknowledgement.

The first group for each load is an unrecorded warmup. Then 10 formal groups per load, with six rotating mode orders, were recorded. The same input fingerprint and quota are used for the three modes in each group. No formal samples were deleted.

### R3.1 Unified time

Tables above. Raw per-sample records are in `artifacts/two_rank_paired_summary.json`.

Paired sample counts:

- 10 groups × 3 modes × 2 load cases = **60 formal samples**;
- all **60** output comparisons have `max_abs = 0.0`;
- all quotas conserve tasks.

### R3.2 Representative cold-path segments

Medians below are separately synchronized stages; they need not sum to the unified time because of dependent gaps and collective control.

| Segment | 6 tokens | 512 tokens |
|---|---:|---:|
| NZ→ND conversion + pack | 0.363 ms | 0.360 ms |
| weight send | 0.473 ms | 0.461 ms |
| destination acquire + weight receive + ready | 0.976 ms | 0.980 ms |
| activation send | 0.228 ms | 0.246 ms |
| source main compute | 0.348 ms | 0.381 ms |
| destination activation receive | 0.174 ms | 0.188 ms |
| destination external compute | 0.312 ms | 0.351 ms |
| source result receive | 0.283 ms | 0.333 ms |
| refill | 0.173 ms | 0.378 ms |
| release + done send | 0.420 ms | 0.613 ms |
| done receive | 0.239 ms | 0.226 ms |

The 37.8 MB weight transfer and control/ready path are the dominant cold-copy cost. Existing-replica still pays activation dispatch, result return, release/ack, and two serial expert-chain launches.

### R3.3 Memory

Source rank:

- initial allocated/reserved: 75,623,936 / 138,412,032 B;
- sample-max allocated/reserved: 110,234,624 / 184,549,376 B;
- peak allocated: 148,976,128 B.

Destination rank:

- initial allocated/reserved: 37,811,200 / 41,943,040 B;
- sample-max allocated/reserved: 44,109,824 / 111,149,056 B;
- peak allocated: 82,327,552 B;
- persistent K=1 slot committed: **37,810,176 B**.

Source has an additional input/flat/workspace budget; its persistent external staging copy was removed by converting the processed bank to ND at each cold copy. Torch allocator reserved memory is not the whole device usage; communication-library buffers outside torch are not fully represented here.

Evidence: `artifacts/two_rank_paired_rank0.json`, `artifacts/two_rank_paired_rank1.json`, `artifacts/two_rank_paired_summary.json`, `artifacts/two_rank_paired.log`.

## 4. R4: independent integer oracle and bounded search

`scripts/expert_search.py` now has:

- an independent exhaustive integer oracle that does not use the finite candidate generator;
- complete finite-candidate enumeration;
- limited candidate enumeration;
- bounded beam search;
- explicit target layer plus slot owner/layer/expert/ready/busy identity;
- no-op and existing-replica plans.

The required `[12,4]` case:

| Baseline | Oracle | Complete candidates | Limited candidates | Bounded result |
|---:|---:|---:|---:|---:|
| 12.25 | **10.50** | 10.50 | 12.25 | 12.25 |

Oracle optimum is `q=4`, loads `[8,8]`, copy cost 2. The limited `{1,n//2,n}` candidates miss it. This is a candidate-set omission, not beam pruning.

Validation summary:

- 10 deterministic cases;
- 200 bounded-search timing runs;
- invalid final states: **0**;
- candidate omissions: **8**;
- pruning omissions: **1**;
- max bounded gap versus oracle: **15.0 toy units**;
- planner CPU p50/p95 over bounded runs: **0.1894 / 8.1890 ms** (CPU implementation timing, not production planning latency).

The `[40,39,1,1]` double-action case is retained. A beam width of 1 returns no action while exhaustive limited search finds the profitable pair, separating pruning from candidate omission. Busy slots, wrong experts, partial owner sets, over-allocation, zero-profit, existing replicas, and same expert at layers 49/50 are covered.

Evidence: `artifacts/cpu_expert_search_oracle.json`, `artifacts/cpu_expert_search_oracle.txt`, `artifacts/cpu_expert_search_tests.txt`, `scripts/run_expert_search_validation.py`.

## 5. R5: real cross-layer reuse

The same fixed K=1 slot executed ten complete `layer49 -> layer50 -> layer49` round trips:

- 30 actual state installs;
- same slot id `0`;
- lease generations 2 through 31;
- every install converted the actual processed bank back to ND;
- all six state fields matched each install;
- all 30 complete expert-chain outputs matched the corresponding bank output exactly;
- output checksums and task request identities were recorded;
- no output or state-field failure occurred.

After the first post-warm cycle, torch allocated memory was constant at **114,032,640 B** and reserved memory at **192,937,984 B** across all 30 round-trip installs/releases. The K=1 pool remained 37,810,176 B. This is evidence that the slot storage does not grow with layer transitions, not proof for an entire model or every communication buffer.

Evidence: `artifacts/single_card_rework.json`, section `roundtrip`.

## 6. Conditional R5.2: actual router + manual expert split

A partial single-layer integration was completed:

- actual layer49 gate weight and correction bias;
- actual SGLang `fused_topk_torch_native`, sigmoid scoring, top-8, renormalization;
- 64 synthetic hidden rows deliberately biased along the actual expert43 router direction;
- 512 real router tasks, 63 of which select logical expert43;
- 63 tasks split 32 bank / 31 external;
- actual W8A8 expert43 chain and actual `GroupedMatmul.forward`;
- task identities include layer, token position, top-k position, logical expert and router weight;
- expert output difference: **0.0**;
- weighted routed contribution difference: **0.0**;
- task conservation: PASS.

This demonstrates manual routing of one logical expert into an external slot with the real router and real quantized compute. It does **not** produce a complete layer output: the other 255 routed experts, shared expert, full top-k combine, dispatcher protocol and service scheduling remain unverified.

Evidence: `artifacts/single_layer_manual_router_result.json`, `artifacts/single_layer_manual_router.log`.

## 7. What was not done

- No full MoE layer output or full-service TTFT/throughput.
- No 32-card service.
- No real K=2 multi-action NPU experiment.
- No production automatic gate/filter integration.
- No cross-node test.
- No claim that the current two-rank path is profitable.
- No claim that the old 6.415 ms / 0.5 ms historical constants are still valid.

## 8. Evidence classification

| Item | Classification |
|---|---|
| R1 identity/abort tests | CPU regression tests |
| q4 and multi-action oracle cases | CPU simulation / independent integer oracle |
| layer49/50 state bytes, W8A8 output, cross-layer reuse | NPU measurement, real model shards |
| two-rank cold/existing/original times | NPU measurement, complete local coordination boundary, expert-child scope |
| actual router top-k identities and split | NPU measurement, partial routed-expert scope |
| full layer, service TTFT, production p95 | Not measured |

## 9. Reproduction

From `shared/current_exp/20261122_global_k_rework`:

```bash
# CPU safety, candidate/oracle validation
./reproduce.sh cpu

# One NPU: actual bank/slot equivalence, valid-lease timing, 10 layer round trips
./reproduce.sh single

# One NPU: actual router + manual expert split
./reproduce.sh router

# Two NPUs: 10 paired groups for 6-token and 512-token loads
./reproduce.sh two-rank

# All NPU probes, including result combination
./reproduce.sh all-npu
```

Exact container command pattern is in `reproduce.sh`; runtime source hashes are in `artifacts/runtime_sources.sha256`.

## 10. Short answer to the two original questions

1. **Does the online split save compute?** On these expert-child loads, no. The cold replica costs about 2.59-2.87 ms median and the existing replica about 1.73-2.02 ms, versus about 0.98-1.02 ms for the original path. The measured 37.8 MB copy and coordination path is not covered. A complete net-benefit claim remains impossible without full layer/service costs.
2. **How much communication would EP enlargement require?** This round did not run the old decode six-cell experiment; the core request was R1-R5. The R3 result establishes that for one real expert-child task, communication/control overhead is larger than the measured compute split even with an existing replica. The full-layer and real multi-card calibration remains the next blocking measurement.
