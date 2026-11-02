# Share：本地代理与远程高级模型协作仓库

本仓库用于在本地环境与远程高级模型之间同步：

- 实验总结、结论和分析文档
- 关键实验代码、启动脚本、补丁与配置
- 小型汇总表、重要图表和交接说明

大型权重、原始日志、数据集和原始运行结果不提交到 Git。

## 当前内容

### `shared/history_exp/`

`history_exp` 的远端精选镜像，包含 01–09 全部实验批次的 Markdown 文档、关键脚本/补丁、汇总表与主要图表。

- 总索引：[`shared/history_exp/README.md`](shared/history_exp/README.md)
- 收录范围和文件统计：[`shared/history_exp/MANIFEST.md`](shared/history_exp/MANIFEST.md)
- 文件完整性校验：[`shared/history_exp/SHA256SUMS`](shared/history_exp/SHA256SUMS)

截至 2026-11-02：

- 本地完整归档：22,708 个文件，约 17.6GiB
- Git 精选镜像：652 个文件，约 14.9MiB
- 未同步内容仍保存在本地 `/data_lib/cyt/workspace/history_exp/`

### `shared/current_exp/`

当前实验状态和远端同步约定。活跃实验的原始数据保留在本地，阶段性成果按同样规则精选到 Git。

- [实验总体总结](shared/current_exp/20261102_load_balance_e0/RESULTS_OVERALL.md)：E0/E1/Profile/E3 与单卡 S0–S4/P2 的统一结论。
- [E0 健康基线](shared/current_exp/20261102_load_balance_e0/RESULTS_E0.md)：权重审计、路由守恒和性能基线。
- [E1 原始负结果](shared/current_exp/20261102_load_balance_e0/RESULTS_E1.md)：保留原始数据；S0 已纠正实际副本落点为 rank15。
- [S0 映射纠错](shared/current_exp/20261102_load_balance_e0/RESULTS_S0.md)：272 槽 slot→rank、修正映射与任务守恒。
- [S1 量化与副本正确性](shared/current_exp/20261102_load_balance_e0/RESULTS_S1.md)：27 张量、36/36 分流、A/A、top-8 合并。
- [S2 局部成本曲线](shared/current_exp/20261102_load_balance_e0/RESULTS_S2.md)：小 M 固定开销、活跃专家数量和局部链时间。
- [S3 配额与回放](shared/current_exp/20261102_load_balance_e0/RESULTS_S3.md)：时间感知配额在单卡局部回放中降低 45–48%，待多卡验证。
- [S4 局部候选](shared/current_exp/20261102_load_balance_e0/RESULTS_S4.md)：两个候选均未过 5% 门槛；通道独立重量化破坏 scale 边界。
- [P2 共享专家与单槽](shared/current_exp/20261102_load_balance_e0/RESULTS_P2.md)：shared expert 成本、部分双 stream 重叠和单槽生命周期。
- [Profile](shared/current_exp/20261102_load_balance_e0/RESULTS_PROFILE.md) / [E3](shared/current_exp/20261102_load_balance_e0/RESULTS_E3.md)：DSA attention 主导 prefill；default low-latency + capacity 16 已验证。

### `shared/current_exp/20261103_3h_online_decode/`

单卡 3 小时 Online / Decode 补测（run `20261103_051202`）：

- [RESULTS](shared/current_exp/20261103_3h_online_decode/20261103_051202/RESULTS.md)：online 配额与本地状态开销、decode EP16/32/64 × batch1/8 六格。
- [INPUTS_PLANS](shared/current_exp/20261103_3h_online_decode/20261103_051202/INPUTS_PLANS.json)：完整 16/8/4 槽向量、配额和输入身份。
- [TIMINGS](shared/current_exp/20261103_3h_online_decode/20261103_051202/TIMINGS.csv)：正式计时块原始记录。

### `handoff/`

本地代理与远程高级模型之间的阶段交接、待办、问题定位和下一步计划。

- [Online与大EP Decode结果复核（2026-11-03）](handoff/2026-11-03_online_decode_review.md)：125423a的局部证据、确定的复用问题，以及两项主任务尚缺的验证。
- [3小时 Online 与 Decode EP 执行卡（2026-11-03）](handoff/2026-11-03_glm52_3h_online_decode.md)：本轮服务器执行入口，复用S0–S4/P2，只补online开销账本和decode六格分析，180分钟硬截止。

- [GLM-5.2 / 昇腾 910C 负载均衡实验方案（2026-11-02）](handoff/2026-11-02_glm52_ascend_load_balance_plan.md)：前置核验、单层复制对照、共享缓冲、decode 通信优化及未决问题。
- [单卡补测完成与多卡后续（2026-11-03）](handoff/2026-11-03_single_npu_results_and_next.md)：S0–S4/P2 结果、证据边界和恢复多卡后的 M0–M3 顺序。

## 更新历史实验镜像

历史归档变化后，在仓库根目录执行：

```bash
python3 tools/sync_history_snapshot.py
git add shared/history_exp
git commit -m "Update history_exp snapshot"
git push
```

脚本会重新生成精选镜像、`MANIFEST.md` 和 `SHA256SUMS`。

## 收录规则

- 包含全部 `.md` 文档。
- 包含实验脚本、分析脚本、shell 启动脚本、补丁和配置。
- 包含小型 summary/report/index/status/coverage/汇总/指标/结果类 JSON、CSV。
- 包含顶层 `tables/`、`figs/`、`figures/` 和分析目录中的关键图表。
- 排除 `results/` 原始结果、`logs/`、`metrics/`、`datasets/`、`weights/`、`.pt`、`.safetensors`、大型 manifest、缓存和上游源码完整快照。

## Git 工作流

开始工作前：

```bash
git pull --rebase
```

完成一小块工作后：

```bash
git add <files>
git commit -m "描述本次改动"
git push
```

不要提交 SSH 私钥、token、密码、模型权重或 GB 级数据。
