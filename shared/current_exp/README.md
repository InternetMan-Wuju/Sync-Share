# current_exp

当前实验状态和远端同步入口。

- 总体总结：[`20261102_load_balance_e0/RESULTS_OVERALL.md`](20261102_load_balance_e0/RESULTS_OVERALL.md)
- 单卡复现说明：[`20261102_load_balance_e0/README_LOCAL.md`](20261102_load_balance_e0/README_LOCAL.md)

## 20261103_3h_online_decode

- run：`20261103_051202`，单张 Ascend NPU，eager 局部专家链代理。
- Online：960/64 → 512/512 的本地计算代理下降 21.0%；planner 1.03 µs，本地 state copy 0.06145 ms，format prep 0.10950 ms。
- Decode：batch1 无正计算收益；batch8 的 EP32/64 有约 0.040–0.041 ms/层（15.3%–15.6%）通信回本预算。
- 真实 dump 契约已核验，但缺 token top8/router weight，故六格使用固定种子人工路由。
- 入口：[`20261103_3h_online_decode/RESULTS.md`](20261103_3h_online_decode/20261103_051202/RESULTS.md)

## 20261102_load_balance_e0

### 16 卡服务实验

- E0：**PASS**。健康基线、权重审计、两组路由采集、三轮性能基线。
- E1：原始实验为负结果，但 S0 已纠正实际副本落点为 rank15；不能据此否定修正落点。
- Profile：prefill 由 DSA attention 主导（约 67%）；decode MoE LL dispatch/combine 约占 15.5%。
- E3：**完成**。default fused low-latency 最佳；capacity 16 略优并通过并发 16；ops 不可用；alltoall 严重退化。

### 单卡补测

- S0：[映射纠错与守恒](20261102_load_balance_e0/RESULTS_S0.md)
- S1：[完整量化状态与副本数值验证 PASS](20261102_load_balance_e0/RESULTS_S1.md)
- S2：[局部专家链成本曲线](20261102_load_balance_e0/RESULTS_S2.md)
- S3：[固定副本预算配额与精确 rank 回放](20261102_load_balance_e0/RESULTS_S3.md)
- S4：[局部候选均未过 5% 门槛](20261102_load_balance_e0/RESULTS_S4.md)
- P2：[shared expert 与单槽生命周期](20261102_load_balance_e0/RESULTS_P2.md)

关键数字：S3 时间感知配额在 layer49 单卡局部回放中把 rank-local 最大时间降低 45.1%（calibration）/47.8%（holdout）；这不是端到端收益，仍必须做多卡 M1 与 HCCL 验证。

- 证据与清单：[`20261102_load_balance_e0/MANIFEST.md`](20261102_load_balance_e0/MANIFEST.md)
