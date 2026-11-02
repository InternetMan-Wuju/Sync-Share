# 关键路径 Profile 与 E3 通信策略结果（2026-11-02）

## 1. Prefill profile

单请求：8192 input → 1 output，16 个 rank 全部采样。

| 指标 | 结果 |
|---|---:|
| step 时间中位数 | 2.654 s |
| step 最小 / 最大 | 2.653 / 2.654 s |
| Attention 相关 | 67.2% |
| MoE 相关 | 14.4% |
| 集合通信 | 2.6% |
| 其他 | 15.9% |

Top kernels（16-rank 中位数）：

| Kernel | 时间 | 占 step |
|---|---:|---:|
| SparseFlashAttention | 1.594 s | 60.1% |
| CamMoeCombineNormal | 229.9 ms | 8.7% |
| QuantBatchMatmulV3 | 104.1 ms | 3.9% |
| LightningIndexer | 81.7 ms | 3.1% |
| GroupedMatmul | 74.7 ms | 2.8% |
| CamMoeDispatchNormal | 57.7 ms | 2.2% |

结论：8K prefill 明显由 GLM DSA 的稀疏 attention/indexer 主导，而不是专家计算或 expert dispatch。因此继续优化 EPLB/专家复制无法改善该 prefill 路径。

## 2. Decode profile

单请求：128 input → 32 output。

Profile 有额外采样开销，绝对时间不能直接当作生产 TPOT；只看相对构成。

Top kernels：

| Kernel | 时间 | 占采集 step |
|---|---:|---:|
| QuantBatchMatmulV3 | 343.9 ms | 14.1% |
| GroupedMatmul | 238.8 ms | 9.8% |
| MoeLowLatencyCombineV2 | 195.4 ms | 8.0% |
| MoeLowLatencyDispatchV2 | 182.8 ms | 7.5% |
| NotifyDispatch | 88.2 ms | 3.6% |
| AscendQuantV2 | 80.5 ms | 3.3% |
| CamMoeCombineNormal | 72.2 ms | 3.0% |
| SparseFlashAttention | 70.6 ms | 2.9% |

MoE 相关合计约 29%。其中 low-latency dispatch + combine 约 378 ms，占采集 step 约 15.5%。decode 侧比 prefill 更值得研究 MoE 通信，但首先要确认现有 fused op 是否已经按 rank 去重。

## 3. E3：DEEP_USE_MODE=ops

配置：

```text
TP16 / EP16 / DP1
DEEP_USE_MODE=ops
mem-fraction-static=0.88
```

服务启动和基础探针可以通过，但第一轮真实 8K prefill 时触发：

```text
rtDeviceSynchronizeWithTimeout execution failed
reason=the model stream execute failed
runtime result = 507011
```

多个 scheduler 随后进入 defunct 状态。结论：

> 当前 A3 + GLM-5.2 + DeepEP + decode graph 组合下，`DEEP_USE_MODE=ops` 不可用。保留默认 DeepEP low-latency 路径。

## 4. 决策

1. 停止继续做专家复制/在线搬运。
2. prefill 优化应先针对 DSA SparseFlashAttention / LightningIndexer 和调度、CPU 开销。
3. decode 优化可以继续研究 MoE low-latency dispatch/combine，但必须先确认 fused op 的去重和容量策略，不能直接重写通信内核。
4. 原始 16-rank trace 已清理；保留每类 rank0 的 op/通信统计和 16-rank 聚合摘要。

## 5. 证据

- `results/profile_summary.json`
- `results/profile_decode_summary.json`
- `results/profiles/prefill_summary.json`
- `results/profiles/decode_summary.json`
- `results/profiles/prefill_rank0/`
- `results/profiles/decode_rank0/`
- `results/e3_ops_failure.json`
- `scripts/analyze_profile.py`
