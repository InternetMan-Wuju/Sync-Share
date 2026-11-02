# E3 DeepEP 通信策略与 dispatch capacity

状态：**完成**。默认 fused low-latency 路径最佳；capacity 16 略优于 64；`ops` 不可用，`alltoall` 严重退化。

## 1. 对比配置

全部为单机 TP16 / EP16 / DP1、256 专家槽、`mem=0.88`，只改变 DeepEP low-latency 实现或 dispatch capacity。

| 配置 | Prefill TTFT p50 | Decode TPOT p50 | Mixed E2E p50 | Mixed 输出 tok/s |
|---|---:|---:|---:|---:|
| default, cap64 | 2.573 s | 48.00 ms | 4.180 s | 59.50 |
| default, cap32 | 2.596 s | 47.48 ms | 4.171 s | 60.00 |
| default, cap16 | 2.595 s | 47.53 ms | 4.153 s | 60.99 |
| alltoall, cap16 | 2.602 s | 257.02 ms | 17.348 s | 14.50 |

说明：

- cap64 为 E0 基线，3 轮；其他为 5 轮。
- cap32/cap16 的 prefill 略慢约 0.8%，但 decode 和 mixed 有约 1–2.5% 改善。
- 差异较小，不能宣称大幅收益，但方向稳定且 cap16 通过了并发 16 压测。

## 2. Capacity 16 安全性

并发 16、每请求 128→16 token：

- 成功：16/16
- E2E：约 6.61 s
- 没有 overflow、报错或请求失败

因此在本配置 `max-running-requests=16`、decode graph max batch=16 下，capacity 16 是安全的最小容量。

## 3. E3 路线结论

### default fused low-latency

- **推荐继续使用**
- capacity 从 64 降到 16，减少 padding/通信工作，decode/mixed 小幅改善。

### DEEP_USE_MODE=ops

- 服务可启动，基础探针可通过。
- 首次真实 8K prefill 触发 `runtime result = 507011`，模型 stream 执行失败。
- **当前 A3 + GLM-5.2 + graph 组合下不可用。**

### DEEP_USE_MODE=alltoall

- 可运行且结果正确。
- Decode TPOT 约 257 ms，是默认路径约 5.4 倍。
- Mixed 吞吐约 14.5 tok/s，是默认路径约 1/4。
- **不可作为默认路径。**

## 4. 最终结论

1. decode 侧 MoE low-latency dispatch/combine 确实占关键路径约 15.5%，但现有 default fused op 已明显优于替代实现。
2. 可行的低风险调优是把 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 从 64 改为 16（与 max-running-requests / graph batch 对齐）。
3. 当前收益约 1–2.5%，属于小优化；仍需要更长压测和真实 SLO 验证才能作为生产默认。
4. 旧 E1 落点负结果不能单独关闭 E2/E4；后续单卡 S0–S3 已给出需多卡验证的修正布局与配额候选。E5 仍没有可验收候选。
5. 后续真正值得投入的是 prefill 的 DSA SparseFlashAttention/LightningIndexer，以及 decode 的调度和 fused op 内部实现，而不是继续扩大专家副本。

## 5. 证据

- `results/E3_EVIDENCE.json`
- `results/e3_ops_failure.json`
- `results/e3_cap16_concurrency16.json`
- `logs/e3_cap32_*/bench_baseline_warm.json`
- `logs/e3_cap16_*/bench_baseline_warm.json`
- `logs/e3_alltoall_*/bench_baseline_warm.json`
