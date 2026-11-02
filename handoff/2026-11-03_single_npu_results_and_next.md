# 单卡补测完成与多卡后续（2026-11-03）

## 已完成

- S0：确认旧 E1 的 slot 256/257 属于 rank15，生成本轮修正映射。
- S1：layer49 e43/e40/e60 的 27 个量化张量完整；36/36 分流用例逐位一致；A/A 20 次只有 1 个哈希；top-8 合并逐位一致。
- S2：实测单卡 W8A8 局部链成本，证明小 M 固定开销和活跃专家数成本显著。
- S3：固定两个额外副本，时间感知 minimax 配额在 rank0/1/2 精确权重回放中：
  - calibration 2.688 → 1.475 ms（−45.1%）
  - holdout 2.756 → 1.437 ms（−47.8%）
- S4：热点单独 GMM 和 token 行分组均为负收益；通道独立重量化改变 scale 边界。
- P2：shared expert 成本同量级；4096 行本地双 stream 有部分重叠；单槽生命周期 PASS。

## 不能由此得出的结论

- 不能说完整 GLM 质量通过。
- 不能说 EP16/EP64 完整层或端到端 TTFT/TPOT 改善。
- 不能把 S3 的 45–48% 直接写入服务收益。
- 不能把单卡顺序拆分或 pointer switch 当作跨 rank 方案。

## 下一步：恢复多卡后

1. 先做 M0：短 prefill 与真实 decode batch，核对实际 graph 桶、padding、全 rank 任务/接收计数。
2. 做 M1：
   - A 原布局
   - A_adapter 同副本但禁止额外槽接任务
   - B 修正布局 + S3 时间感知配额
   - 固定真实输入、相同 logical top-k、相同量化路径
3. 同轮采集 rank-local GMM、dispatch、combine、HCCL payload、通信等待和完整层时间。
4. S3 配额只有在完整层/服务 SLO 和同 HBM 预算下仍有净收益，才进入 E5。
5. 顺序热点拆分停止；shared expert waterfill 仅在时间线证明有可移动空档时继续。
6. 多卡权重搬运必须独立计量，不能用本机 NPU→NPU copy 0.117 ms 代表 HCCL。

## 入口

- `shared/current_exp/20261102_load_balance_e0/RESULTS_OVERALL.md`
- `shared/current_exp/20261102_load_balance_e0/RESULTS_S3.md`
- `shared/current_exp/20261102_load_balance_e0/MANIFEST.md`
- 本地：`/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/`
