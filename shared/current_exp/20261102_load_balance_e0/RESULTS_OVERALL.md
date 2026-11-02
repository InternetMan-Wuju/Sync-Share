# GLM-5.2 / Ascend 910C 单机负载均衡实验总结

日期：2026-11-02  
环境：npu1-23，单机 16×Ascend 910C

## 完成了什么

### E0：可信基线（PASS）

- 使用 publisher 恢复权重，182 文件、2912 条加载审计、0 错误。
- TP16/EP16/DP1，DeepEP auto，decode graph 开启。
- 两次独立路由采集：prefill 任务数精确守恒。
- 基线：
  - 8K prefill TTFT p50 2.573 s
  - decode 128→256 TPOT p50 48.0 ms
  - mixed 4 并发 59.50 tok/s

### E1：专家复制（负结果）

- layer 49 top-2 和 layer 49/36/54 top-2 预装副本。
- dynamic dispatch 确认 expert 43/40 可 50/50 分流到副本。
- layer 49 层内 rank 负载代理从 4.95 降到 2.99（−39.6%）。
- 但端到端：
  - prefill TTFT 不变
  - decode TPOT 慢约 1–3%
  - mixed 吞吐不变或下降
- 结论：层内 token 失衡不是当前端到端关键路径，**停止 C/D 在线复制**。

### Profile：真正瓶颈

- Prefill 16-rank 时间线高度均衡（2.653–2.654 s）。
- Prefill 构成：
  - DSA attention 相关约 67.2%
  - MoE 相关约 14.4%
  - 集合通信约 2.6%
- 最大单 kernel：SparseFlashAttention 约 1.594 s，占 60.1%。
- Decode 中 MoE 相关约 29%；MoeLowLatency dispatch+combine 约 15.5%。

### E3：通信策略与容量（完成）

| 配置 | Decode TPOT p50 | Mixed tok/s |
|---|---:|---:|
| default cap64 | 48.00 ms | 59.50 |
| default cap32 | 47.48 ms | 60.00 |
| default cap16 | 47.53 ms | 60.99 |
| alltoall cap16 | 257.02 ms | 14.50 |
| ops | 507011 执行失败 | — |

- default fused low-latency 是最佳可用路径。
- capacity 16 与 max-running-requests / graph max batch 对齐，减少 padding，decode/mixed 小幅改善。
- capacity 16 通过 16 并发压力测试，16/16 成功。
- `ops` 不可用；`alltoall` 严重退化。

## 最终建议

1. 保留 default DeepEP low-latency。
2. 单机 TP16/EP16、max-running-requests=16 时，将
   `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 从 64 调整为 16。
3. 不继续做专家复制/EPLB 在线重排。
4. Prefill 后续优化重心是 DSA SparseFlashAttention / LightningIndexer。
5. Decode 后续优化重心是调度、量化 GEMM 和 fused dispatch/combine 内部实现。
6. E2/E4 被 E1 负结果门控关闭；E5 没有可验收的候选策略。

## 证据入口

- `RESULTS_E0.md`
- `RESULTS_E1.md`
- `RESULTS_PROFILE.md`
- `RESULTS_E3.md`
- `results/E0_EVIDENCE.json`
- `results/E1_EVIDENCE.json`
- `results/E3_EVIDENCE.json`
- `results/profile_summary.json`
- `results/profile_decode_summary.json`
