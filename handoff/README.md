# handoff

在这里放置本地代理与远程高级模型之间的阶段交接说明。

## 当前执行入口

- [可配置 K 服务接入与 EP32 三组验收（2026-11-23）](2026-11-23_global_k_service_ep32_execution.md)是当前执行入口。基于 `c8f8961` 完成真实服务接线，再用两台机器共 32 die、AgentX 8K/16K/32K 验收。
- [Prefill 三组对照（2026-11-23）](2026-11-23_prefill_eplb_three_way_comparison.md)记录上一轮方案和三组对照口径。
- [Prefill K=1 原型修正（2026-11-23）](2026-11-23_global_k_adapter_repair.md)记录上一轮的输入保真、owner 状态和显存验收。
- [Prefill K=1：统一执行路径、格式与显存验收（2026-11-22）](2026-11-22_global_k_unified_adapter.md)：上一轮方案；3efa0a6完成小样本测试，复核发现输入保真、实际槽身份与内存基线待修；性能未测。
- [Prefill K=1：空编号路线的最小可行性证明（2026-11-22）](2026-11-22_global_k_placeholder_feasibility.md)：上一轮方案；607121e已证明有限主库下的原生alltoall、真实W8A8和唯一槽复用，统一边界处理与格式审计待补。
- [Prefill K=1：原生分发可行性与已有副本路径诊断（2026-11-22）](2026-11-22_global_k_native_dispatch_diagnosis.md)：历史方案；595428c发现默认均匀分片限制，驻留副本未稳定回本。
- [Prefill 全组 K=1：正式计时与回本区间实验（2026-11-22）](2026-11-22_global_k_prefill_breakeven.md)：上一轮方案；d0c4a3f 已完成，所测点未发现稳定回本，后续以最新交接为准。
- [全组固定 K 槽：安全修正与完整成本补测（2026-11-22）](2026-11-22_global_k_correctness_cost_rework.md)：上一轮任务，结果见 b8320e5；真实子链已有进展，正式计时、搜索漏解及释放竞争的后续以最新方案为准。
- [全组固定 K 槽：真实 GLM 单层闭环执行交接（2026-11-22）](2026-11-22_global_k_real_layer_execution.md)：历史任务及原始验收背景，执行顺序以最新交接方案为准。
- [昇腾910C Prefill专家复制与精确配额实现方案（2026-11-03）](2026-11-03_ascend_prefill_copy_aware_implementation.md)：历史阶段方案；源码调研、接入合同与 M0 验收可参考，其中暂停实验和每 rank 容量等旧阶段约束以最新交接为准。
- [Online与大EP Decode结果复核（2026-11-03）](2026-11-03_online_decode_review.md)：最新b7ec624的独立复核与会议报告；21.5%计算空间、净收益不确定性、人工六格和真实直方图补充方向。
- [3小时 Online 与 Decode EP 验证执行卡（2026-11-03）](2026-11-03_glm52_3h_online_decode.md)：已结束轮次的原执行要求；当前状态见结果复核，不自动续跑。
- [已有单卡结果与后续边界](2026-11-03_single_npu_results_and_next.md)：S0–S4/P2复用入口，不作为本轮新采样。

建议每个交接文件包含：

1. 目标与当前状态
2. 已完成内容
3. 关键文件与路径
4. 未决问题
5. 下一步具体动作
6. 验证命令与结果

命名示例：`2026-11-02_workspace-storage.md`。
