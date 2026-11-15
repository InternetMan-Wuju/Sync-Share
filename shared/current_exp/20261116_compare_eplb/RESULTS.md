# EPLB comparison on npu1-19

## Scope

- Host: `npu1-19`
- Devices: local Ascend devices 0-15
- Image: `quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`
- Model: `/data_lib/data/models/glm-q`, W8A8, TP16/EP16/DP1
- Input: one human-readable SHA-pinned ShareGPT-derived prompt
- Prompt: 8,176 tokenizer tokens (target-model tokenizer), output limit 32
- TTFT definition: HTTP request send to first non-empty streamed content token
- Each run: 1 warmup + 3 measured requests, concurrency 1, radix/prefix cache disabled
- Four runs used the same prompt SHA and serving memory/context settings

This is a single 8K request comparison. It is not a production throughput benchmark.

## Measured TTFT

| Arm | Configuration | TTFT median | Min-max | Delta vs off |
|---|---|---:|---:|---:|
| off | No EPLB, 0 redundant experts | **2.6022 s** | 2.5929-2.6030 s | baseline |
| EPLB | EPLB enabled, 0 redundant experts, no rearrangement triggered | **2.6073 s** | 2.6061-2.6121 s | +5.11 ms (+0.20%) |
| EPLB + redundancy | EPLB enabled, 16 redundant experts, no rearrangement triggered | **2.6432 s** | 2.6122-2.6558 s | +41.00 ms (+1.58%) |
| ours | Current-batch online decision, 17th slot layout, 8K gate | **2.6211 s** | 2.6046-2.6417 s | +18.92 ms (+0.73%) |

The small differences are close to run-to-run variation and are not evidence of a
latency speedup. In the `ours` 8K arm, the profitability gate rejected copying;
therefore this arm measured the no-copy online path, not copy/remap compute.

## Forced EPLB rearrangement

A separate run used:

```text
--enable-eplb
--ep-num-redundant-experts 0
--eplb-rebalance-num-iterations 2
--expert-distribution-recorder-buffer-size 1
context 8192, max-total-tokens 8192, mem-fraction-static 0.80
```

Result: **failed during rearrangement**.

- All 16 rank copies printed `[EPLBManager] rebalance start` at `2026-11-15 21:23:57`.
- The first NPU OOM appeared at `2026-11-15 21:24:01.659`, after about **4.7 s**.
- Temporary allocation requests were **194 MiB or 386 MiB**.
- At failure, each affected NPU had about **50-200 MiB free** while 54.3-54.7 GiB was already allocated.
- No `[EPLBManager] rebalance end` line was emitted.
- Therefore the 4.7 s is a failure window, **not** a completed rearrangement-wait measurement.
- The same model/config without rearrangement starts successfully, so the failure is in the EPLB update staging path, not ordinary prefill.

This is a real limitation on this 61 GiB-per-device node: the model weights plus
KV/cache reserve leave insufficient HBM for the existing EPLB temp buffer. A
larger-HBM node, a lighter model, or a redesigned low-memory updater is required
to measure completed rearrangement TTFT on this arm.

## Interpretation

- The `off`, EPLB-no-rearrangement, EPLB+redundancy-no-rearrangement and `ours`
  no-copy TTFTs are all around 2.6 s; no arm demonstrates a statistically
  meaningful TTFT improvement in this single-prompt comparison.
- EPLB no-rearrangement and EPLB+redundancy TTFT do **not** include the
  rearrangement wait.
- Forced EPLB rearrangement cannot complete on this node due to the 386 MiB
  temporary allocation; no valid `rearrange end` / TTFT-after-rearrangement
  number is reported.
- The requested detailed rearrangement-wait breakdown is therefore only
  available as the failed-window evidence above, not as a successful timing.

## Raw evidence

- Request results: `results/*_8k_ttft*.json`
- Forced-rebalance log: `logs/eplb0_20261116_0620/server.log`
- No-rearrangement EPLB logs: `logs/eplb0_20261116_0700/server.log`,
  `logs/eplb16_20261116_0710/server.log`
- Our online log: `logs/ours_20261116_0650/server.log`
- Off matched baseline log: `logs/off_20261116_0720/server.log`
- Prompt metadata: `prompts/prompt_8k.meta.json`
