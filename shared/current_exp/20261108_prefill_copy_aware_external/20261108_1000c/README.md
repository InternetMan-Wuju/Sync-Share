# 20261108 外挂式 Prefill Copy-Aware M0

- [实现说明](IMPLEMENTATION.md)
- [验证记录](VALIDATION.md)
- [单 NPU 正收益结果](RESULTS.md)
- [启动与撤销](APPLY.md)
- [Plan 示例](PLAN_EXAMPLES.json)
- [环境信息](ENVIRONMENT.json)
- [外挂代码](m0_external/)
- [测试](tests/)
- [NPU smoke 结果](artifacts/npu_smoke.json)
- [NPU 正收益结果](artifacts/positive_case.json)

状态：**CPU PASS；单 NPU top-k/Plan smoke PASS；单 NPU 真实 e43 完整状态复制与
本地专家链 PASS；固定输入 `4096/64 -> 2080/2080` 在扣除 planner 和本地完整
state copy 后为正收益；同机跨卡 HCCL/P2P、完整模型和生产 graph 尚未验证。**

外挂实现不修改 `/sgl-workspace` 源码；默认 `off`，通过运行时符号包装接入。
两个 rank 的后续验证需要同一台机器上的两张 NPU，不需要多机；多机只在跨节点
EP64/网络拓扑成为实验对象时才必要。
