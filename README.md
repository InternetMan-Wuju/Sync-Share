# Share：本地代理与远程高级模型协作仓库

本仓库用于在本地环境与远程高级模型之间同步：

- 实验总结、结论和分析文档
- 关键实验代码、启动脚本、补丁与配置
- 小型汇总表、重要图表和交接说明

大型权重、原始日志、数据集和原始运行结果不提交到 Git。

## 当前内容

- [EP32 单次 DeepEP 改道与在线 EPLB（2026-11-23）](handoff/2026-11-23_ep32_native_reroute_and_online_eplb.md)是当前执行入口。正式比较原版、带兼容补丁且完成在线重排的 EPLB +32、我方可配置 K 真改道。
- [EP32 原生单次改道与在线 EPLB 结果](shared/current_exp/20261123_ep32_native_reroute_and_online_eplb/REPORT_FOR_REVIEW.md)：`3512cae` 基线；我方针 K=1/K=2 使用同一轮 DeepEP dispatch，EP32 K=2 三档完成；EPLB +32 在双节点完成在线重排后三档通过。8K K=2 存在有界数值 token 差异，16K/32K 输出 ID 一致。
- [EP32 三组重复性能对比](shared/current_exp/20261123_ep32_perf_repeat/REPORT_FOR_REVIEW.md)：`bd80f52` 基线；三组 × 三档各 10 个正式请求，三轮启动顺序轮换；EPLB formal 样本全部含在线重排，K=2 使用 quiet 正式路径。
- [EP32 在线 EPLB +32：1000 迭代复测](shared/current_exp/20261123_ep32_eplb1000/REPORT_FOR_REVIEW.md)：`bf94982` 基线；每轮 496 条独立校准流量后完成重排，正式 30 条样本未再触发重排；原版和 K=2 沿用上一批数据。
- [Native-dispatch service 修复与 EP32 三组验收报告](shared/current_exp/20261123_global_k_native_service_dispatch/REPORT_FOR_REVIEW.md)：`22d65f2` 基线；32K 槽复用挂起已修复，EP16 K=1/K=2 和 EP32 原版/EPLB 静态/K=2 功能矩阵通过；额外 DeepEP 副本编号在 combine 处失败，服务 K 结果明确标为 local-backfill fallback。
- [可配置 K 服务原型与 EP32 结果](shared/current_exp/20261123_global_k_service_ep32/REPORT_FOR_REVIEW.md)记录 `db1ce2d` 的槽复用、32K 挂起和静态 EPLB 三档结果。
- [EP16 回放与服务障碍报告](shared/current_exp/20261123_prefill_eplb_three_way/REPORT_FOR_REVIEW.md)记录 `c8f8961` 的单层回放、静态 EPLB 和原生启动参数的失败证据。
- [K=1 原型修正报告](shared/current_exp/20261123_global_k_adapter_repair/REPORT_FOR_REVIEW.md)记录 `7bfa44b` 的输入保真、owner 状态和显存验证。
- [统一adapter实验报告](shared/current_exp/20261122_global_k_unified_adapter/REPORT_FOR_REVIEW.md)：3efa0a6；小样本真实计算/槽复用通过，复核发现输入改写、槽说明未绑定真实状态及显存基线问题，后续以当前修正计划为准。
- [Prefill K=1：统一执行路径、格式与显存验收（2026-11-22）](handoff/2026-11-22_global_k_unified_adapter.md)：上一轮方案及验收背景。
- [空编号K=1小范围可行性报告](shared/current_exp/20261122_global_k_placeholder_proof/REPORT_FOR_REVIEW.md)：607121e；原生alltoall与真实专家/跨层复用通过，有限主库、合成输入；统一边界处理和格式审计待补，性能未测。
- [Prefill K=1：空编号路线的最小可行性证明（2026-11-22）](handoff/2026-11-22_global_k_placeholder_feasibility.md)：上一轮方案及验收背景。
- [原生接口与持续驻留实验报告](shared/current_exp/20261122_global_k_native_dispatch/REPORT_FOR_REVIEW.md)：595428c；默认布局要求均匀分片，驻留条件未稳定回本；当前方案进一步验证空编号和真实存储解耦。
- [Prefill K=1：原生分发可行性与已有副本路径诊断（2026-11-22）](handoff/2026-11-22_global_k_native_dispatch_diagnosis.md)：上一轮方案及验收背景。
- [Prefill K=1 回本区间实测报告](shared/current_exp/20261122_global_k_breakeven/REPORT_FOR_REVIEW.md)：d0c4a3f；324 次执行、108 个配对组，临时复制全部变慢，已有副本尚未稳定回本；后续接口及驻留核查见595428c报告。
- [Prefill 全组 K=1：正式计时与回本区间实验（2026-11-22）](handoff/2026-11-22_global_k_prefill_breakeven.md)：上一轮方案及验收背景。
- [全组 K 重做与两卡配对报告](shared/current_exp/20261122_global_k_rework/REPORT_FOR_REVIEW.md)：b8320e5；真实专家子链和跨层复用正确性已有进展，6/512-token 的现有两卡计时均为负收益，后续执行入口见上。
- [全组固定 K 槽：安全修正与完整成本补测（2026-11-22）](handoff/2026-11-22_global_k_correctness_cost_rework.md)：上一轮任务及验收背景。
- [真实专家与两卡 K=1 上轮报告](shared/current_exp/20261122_global_k_real_layer/REPORT_FOR_REVIEW.md)：真实状态和手工分流已有进展；槽安全与计时存在待修问题，性能数字须按当前交接补测。
- [全组固定 K 槽原型完整报告](shared/current_exp/20261116_global_k_online/FULL_REPORT_FOR_REVIEW.md)：固定 owner 槽、小尺寸 W8A8 和 16-rank 1KB 传输已验证，真实 GLM external-slot dispatch 与 global-K TTFT 尚未完成。
- [当前批次复制决策：论文调研、CPU模拟与23节点16卡实测](shared/current_exp/20261109_decision/20261109_074727/RESULTS.md)：64K分块层回放每次冷复制仍净省约19%；8K避免负优化，32K门槛偏保守；未证明整机请求加速。

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

- [本地独立复核与会议报告](handoff/2026-11-03_online_decode_review.md)：复算b7ec624，解释21.5%计算空间、净收益不确定性和人工decode六格的适用边界。
- [审阅报告](shared/current_exp/20261103_3h_online_decode/20261103_051202/REPORT_FOR_REVIEW.md)：面向本地 AI 的完整证据分级、结果、边界和审阅检查点。
- [RESULTS](shared/current_exp/20261103_3h_online_decode/20261103_051202/RESULTS.md)：online 配额与本地状态开销、decode EP16/32/64 × batch1/8 六格。
- [INPUTS_PLANS](shared/current_exp/20261103_3h_online_decode/20261103_051202/INPUTS_PLANS.json)：完整 16/8/4 槽向量、配额和输入身份。
- [TIMINGS](shared/current_exp/20261103_3h_online_decode/20261103_051202/TIMINGS.csv)：正式计时块原始记录。

### `shared/current_exp/20261103_complex_moe_sim/`

真实 trace 驱动的 75 层 MoE 复杂模拟：

- [RESULTS](shared/current_exp/20261103_complex_moe_sim/RESULTS.md)：EP16/32/64、online 临时复制 K 扫描、远端复制回本线。
- [全量输出](shared/current_exp/20261103_complex_moe_sim/SIMULATION_RESULT.json) / [清单](shared/current_exp/20261103_complex_moe_sim/MANIFEST.json)

### `shared/current_exp/20261103_prefill_copy_aware/`

Prefill 当前路由复制与精确配额 M0：

- [实现与源码接入点](shared/current_exp/20261103_prefill_copy_aware/20261103_0935/IMPLEMENTATION.md)
- [验证记录](shared/current_exp/20261103_prefill_copy_aware/20261103_0935/VALIDATION.md)
- [默认 off 补丁](shared/current_exp/20261103_prefill_copy_aware/20261103_0935/patches/prefill_copy_aware_m0.patch)

### `shared/current_exp/20261108_prefill_copy_aware_external/`

外挂式 Prefill copy-aware M0：

- [实现说明](shared/current_exp/20261108_prefill_copy_aware_external/20261108_1000c/IMPLEMENTATION.md)
- [NPU smoke 结果](shared/current_exp/20261108_prefill_copy_aware_external/20261108_1000c/artifacts/npu_smoke.json)
- [验证记录](shared/current_exp/20261108_prefill_copy_aware_external/20261108_1000c/VALIDATION.md)
- [完整本地专家链结果](shared/current_exp/20261108_prefill_copy_aware_external/20261108_1000c/artifacts/full_local_smoke.json)

### `shared/current_exp/20261109_ep16_online/`

23节点16个NPU设备的整机验证，固定8192输入 / 32输出；外挂接入layer49，使用真实机内HCCL通信：

- [结果与breakdown](shared/current_exp/20261109_ep16_online/20261109_061333/RESULTS.md)：区分完整请求、同输入并行子链、冷复制与已就绪副本；输入为AgentX块身份驱动的合成token。
- [外挂实现与边界](shared/current_exp/20261109_ep16_online/20261109_061333/IMPLEMENTATION.md)
- [16设备正确性证据](shared/current_exp/20261109_ep16_online/20261109_061333/VALIDATION.json)
- [计时记录](shared/current_exp/20261109_ep16_online/20261109_061333/TIMINGS.csv)

### `handoff/`

本地代理与远程高级模型之间的阶段交接、待办、问题定位和下一步计划。

- [EP32 单次 DeepEP 改道与在线 EPLB（2026-11-23）](handoff/2026-11-23_ep32_native_reroute_and_online_eplb.md)是当前实现和全量验证入口。
- [全组 K 槽服务修复与 DeepEP 真实改道（2026-11-23）](handoff/2026-11-23_global_k_native_service_dispatch.md)记录上一轮 32K 修复方案。
- [可配置 K 服务接入与 EP32 三组验收（2026-11-23）](handoff/2026-11-23_global_k_service_ep32_execution.md)记录上一轮服务原型方案。
- [Prefill 三组对照（2026-11-23）](handoff/2026-11-23_prefill_eplb_three_way_comparison.md)记录上一轮方案与对照口径。
- [Prefill K=1 原型修正（2026-11-23）](handoff/2026-11-23_global_k_adapter_repair.md)是上一轮正确性与资源验证方案。
- [Prefill K=1：统一执行路径、格式与显存验收（2026-11-22）](handoff/2026-11-22_global_k_unified_adapter.md)：上一轮方案及验收背景。
- [Prefill K=1：空编号路线的最小可行性证明（2026-11-22）](handoff/2026-11-22_global_k_placeholder_feasibility.md)：上一轮方案及验收背景。
- [Prefill K=1：原生分发可行性与已有副本路径诊断（2026-11-22）](handoff/2026-11-22_global_k_native_dispatch_diagnosis.md)：上一轮方案及验收背景。
- [Prefill 全组 K=1：正式计时与回本区间实验（2026-11-22）](handoff/2026-11-22_global_k_prefill_breakeven.md)：上一轮方案及验收背景。
- [全组固定 K 槽：安全修正与完整成本补测（2026-11-22）](handoff/2026-11-22_global_k_correctness_cost_rework.md)：上一轮任务背景，执行范围以最新交接为准。
- [上一轮真实 GLM 单层闭环交接](handoff/2026-11-22_global_k_real_layer_execution.md)：历史任务背景与固定 K 合同，执行顺序以当前交接为准。
- [昇腾910C Prefill专家复制与精确配额实现方案（2026-11-03）](handoff/2026-11-03_ascend_prefill_copy_aware_implementation.md)：历史实现方案；实际SGLang接线、完整量化状态复制、显存与通信约束可参考，执行范围以最新交接为准。
- [Online与大EP Decode结果复核（2026-11-03）](handoff/2026-11-03_online_decode_review.md)：最新b7ec624的证据复核、会议结论及最小补齐顺序。
- [3小时 Online 与 Decode EP 执行卡（2026-11-03）](handoff/2026-11-03_glm52_3h_online_decode.md)：已结束轮次的原执行要求；当前状态见结果复核，不自动续跑。

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
