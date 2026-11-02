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

- [单机负载均衡 E0（2026-11-02，PASS）](shared/current_exp/20261102_load_balance_e0/RESULTS_E0.md)：健康基线、权重审计、两组路由采集和三轮基线性能。
- [单机负载均衡 E1（2026-11-02，负结果）](shared/current_exp/20261102_load_balance_e0/RESULTS_E1.md)：layer 49 与 49/36/54 top-2 预装副本未改善端到端，停止在线复制方向。

### `handoff/`

本地代理与远程高级模型之间的阶段交接、待办、问题定位和下一步计划。

- [GLM-5.2 / 昇腾 910C 负载均衡实验方案（2026-11-02）](handoff/2026-11-02_glm52_ascend_load_balance_plan.md)：前置核验、单层复制对照、共享缓冲、decode 通信优化及未决问题；E0 已完成；E1 已完成并得到负结果，转入关键路径 profile。

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
