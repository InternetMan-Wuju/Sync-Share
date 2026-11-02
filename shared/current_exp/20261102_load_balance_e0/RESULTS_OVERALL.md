# GLM-5.2 / Ascend 910C 负载均衡实验总结

日期：2026-11-02 至 2026-11-03
环境：npu1-23；早期基线使用单机 16×Ascend 910C；2026-11-03 补测按修订方案只使用 1 张 NPU。

## 1. 结论先说

### 已确认

1. E0 健康基线和权重审计通过。
2. Prefill 的主要时间是 DSA/SparseFlashAttention，不是 MoE 专家不均衡。
3. Decode 中 MoE 约占 29%，fused low-latency dispatch+combine 约占 15.5%。
4. `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16` 与 graph batch/max-running-requests 对齐，是本机已验证的小幅优化。
5. 旧 E1 映射实际把 e43/e40 副本放在 rank15，不是 rank0/1；旧负结果不能否定修正后的副本位置。
6. 单卡上，真实 W8A8 专家副本在固定量化边界下可以逐位一致；两副本分流、中间 re-quant 和 8 专家 top-8 合并均通过 S1。
7. 局部链成本不是 token 线性：小 M 主要受固定开销影响；同样 1024 行从 1 个活跃专家增至 8 个约慢 43%。
8. 在 layer 49 单卡局部回放中，考虑其他负载的时间配额把校准/留出集的 rank-local 最大时间分别降低 45.1% / 47.8%，优于 50/50 和纯行数 minimax。
9. “热点单独 GMM”和“固定 quant 边界的 token 行分组”两个顺序局部候选都是负收益；通道二分片若独立重量化会破坏原 scale 边界。
10. shared expert 是真实工作量，4096 行本地双 stream 可部分重叠；单槽覆盖和版本检查通过。

### 尚未证明

- 修正副本布局和 S3 配额能否在真实 16 rank、HCCL、dispatch/combine 下带来净收益。
- 完整模型质量、TTFT/ITL/SLO 和同 HBM 预算下的服务收益。
- shared expert 在多卡 decode/prefill 时间线中是否真的有足够可移动空档。
- 任何 EP64 或跨节点结论。

## 2. E0：可信基线（PASS）

- publisher 恢复权重：182 文件、2912 条加载审计、0 错误。
- TP16/EP16/DP1，DeepEP auto，decode graph 开启。
- 两次独立路由采集：prefill 任务数精确守恒。
- 8K prefill TTFT p50 2.573 s。
- decode 128→256 TPOT p50 48.0 ms。
- mixed 4 并发 59.50 tok/s。

证据：`RESULTS_E0.md`、`results/E0_EVIDENCE.json`。

## 3. E1：旧负结果与映射修正

### 原始实验

- layer 49 top-2，以及 layer 49/36/54 top-2 预装副本。
- dynamic dispatch 确认 expert 43/40 可分流。
- 特定旧配置下：prefill TTFT 无明显变化，decode TPOT 慢约 1–3%，mixed 吞吐不变或下降。

### S0 修正

旧 272 槽映射按连续槽规则：

```text
272 / 16 = 17 slots/rank
slot 256, 257 → rank15
```

因此旧实验实际是 e43/e40 复制到 rank15，不是原来说的 rank0/1。运行态 dump 与旧 mapping JSON 一致，数字可复查，但解释必须修正。

修正布局采用每个 rank 16 primary + 1 extra：

```text
e43: rank2 primary + rank0 replica
e40: rank2 primary + rank1 replica
```

layer49 rank max/mean proxy：

| 布局 | calibration | holdout |
|---|---:|---:|
| 原始 256 槽 | 4.951 | 5.044 |
| 旧 272 映射（rank15） | 3.061 | 3.046 |
| 修正映射（rank0/1） | 2.988 | 3.082 |

结论：旧 E1 只能说明“旧落点的特定配置没有显示明显 TTFT 改善”，不能关闭 E4，也不能否定修正落点或配额策略。

证据：`RESULTS_S0.md`、`results/S0_MAPPING_AUDIT.json`。

## 4. Profile：真正瓶颈

- Prefill 16-rank 时间线高度均衡（2.653–2.654 s）。
- Prefill 构成：DSA attention 约 67.2%，MoE 约 14.4%，集合通信约 2.6%。
- 最大单 kernel：SparseFlashAttention 约 1.594 s，占 60.1%。
- Decode：MoE 约 29%；MoE low-latency dispatch+combine 约 15.5%。

这决定了：MoE 局部优化即使有效，也不能直接外推为 TTFT 大幅改善。

证据：`RESULTS_PROFILE.md`、`results/profile_summary.json`、`results/profile_decode_summary.json`。

## 5. E3：通信策略与容量（完成）

| 配置 | Decode TPOT p50 | Mixed tok/s |
|---|---:|---:|
| default cap64 | 48.00 ms | 59.50 |
| default cap32 | 47.48 ms | 60.00 |
| default cap16 | 47.53 ms | **60.99** |
| alltoall cap16 | 257.02 ms | 14.50 |
| ops | 507011 失败 | — |

- default fused low-latency 是当前最佳可用路径。
- capacity 16 略优；16 并发压力测试 16/16 成功。
- `alltoall` 严重退化，`ops` 不可用。

证据：`RESULTS_E3.md`、`results/E3_EVIDENCE.json`。

## 6. S1-S4：单卡补测

### S1：量化与副本正确性（PASS）

- 27/27 个张量完整，e43/e40/e60，总计 113.49 MB。
- `weight_offset` 全为 0；源 `scale` float32，kernel 使用 bfloat16。
- 36/36 分流用例逐位一致；动态量化 q/scale 随分组重排不变。
- 20 次 A/A 仅 1 个哈希。
- 对 FP32 反量化参考 NRMSE 1.248%。
- 8 专家 top-8 合并逐位一致，2056/2056 唯一任务。

证据：`RESULTS_S1.md`、`results/S1_NUMERICS.json`。

### S2：局部成本曲线

- 1 个活跃专家：8–256 行约 0.19 ms，1024 行 0.317 ms，4096 行 0.975 ms。
- 1024 行 8 个活跃专家约 0.452–0.486 ms，比单专家慢约 43–53%。
- 两个 bank 共 604 MB 预驻留时，交替/重复权重耗时比 0.9998；这不是冷缓存测试。

证据：`RESULTS_S2.md`、`results/S2_LOCAL_COST.csv`。

### S3：配额优化（局部支持，多卡未验证）

固定两个额外副本，时间感知 minimax 配额：

| split | e43→rank0 | e40→rank1 | owner-only max | 优化后 max | 局部降幅 |
|---|---:|---:|---:|---:|---:|
| calibration | 3392 / 4062 | 2646 / 3978 | 2.688 ms | 1.475 ms | **45.1%** |
| holdout | 3387 / 4066 | 2883 / 3972 | 2.756 ms | 1.437 ms | **47.8%** |

注意：这是 rank0/1/2 精确权重回放与局部链时间，不包含跨 rank 通信。它是恢复多卡后最值得进入 M1 的候选。

证据：`RESULTS_S3.md`、`results/S3_ALLOCATION.csv`。

### S4：局部算子候选（均未过门槛）

| 候选 | rows | 相对 baseline |
|---|---:|---:|
| 热点专家单独 GMM | 1024 | −63.2% |
| 热点专家单独 GMM | 4096 | −27.9% |
| 热点专家单独 GMM | 13189 | −19.2% |
| 固定量化边界两行分组 | 1024 | −36.3% |
| 固定量化边界两行分组 | 4096 | −12.5% |

两个候选数值逐位一致，但局部更慢。通道二分片若保持原 activation scale，NRMSE 0.285%；独立分片重量化升到 1.922%。

证据：`RESULTS_S4.md`、`results/S4_LOCAL_CANDIDATES.json`。

## 7. P2：共享专家与单槽

- layer49 shared expert 有完整 9 个 W8A8_DYNAMIC 张量，约 37.77 MB。
- shared expert 与单 routed expert 的局部链成本同量级（1024 行约 0.320 vs 0.323 ms）。
- 1024 行双 stream 并发慢 31.7%；4096 行并发比顺序和节省 17.3%，理想上限为 50%。
- 单槽交替装入 layer49 shared 和 layer48 expert0：版本、旧读者等待、输出逐位一致性均 PASS。
- 单专家 NPU→NPU 复制 0.117 ms；H2D 1.806 ms；pointer switch 0.0003 ms，但 pointer switch 不是权重搬运。

证据：`RESULTS_P2.md`、`results/P2_LOCAL_CHECKS.json`。

## 8. 最终建议

1. 生产参数先保留已验证的 `capacity=16`，不要重新启用 `alltoall` 或 `ops`。
2. 恢复多卡后，优先做 M1：
   - A：原布局；
   - A_adapter：同副本但禁止额外槽接任务；
   - B：修正布局 + 时间感知配额；
   - 固定真实输入、相同副本和相同 logical top-k。
3. 同时记录实际 HCCL payload、dispatch/combine、统计/规划、rank-local GMM 和完整层时间。S3 的 45–48% 不能直接作为端到端预测。
4. 顺序“热点单独 GMM”和 token 行拆小方向在当前实现下停止。
5. 通道分片只有在能保持原 activation scale并计入跨分片 max/同步成本后再考虑。
6. shared expert waterfill 只在多卡时间线证明存在可移动空档时推进；prefill 大 batch 已有部分双 stream 重叠。
7. Prefill 长期优化仍应优先 DSA attention / SparseFlashAttention / LightningIndexer。
8. 不把 S1-S4 的单卡结果写成完整模型正确性或服务收益。

## 9. 证据入口

- `README.md`
- `RESULTS_E0.md`
- `RESULTS_E1.md`
- `RESULTS_PROFILE.md`
- `RESULTS_E3.md`
- `RESULTS_S0.md`
- `RESULTS_S1.md`
- `RESULTS_S2.md`
- `RESULTS_S3.md`
- `RESULTS_S4.md`
- `RESULTS_P2.md`
- `results/` 下对应 summary JSON/CSV
- `scripts/s0_mapping_audit.py` 至 `scripts/p2_local_checks.py`
