# 20261108 外挂式 Prefill Copy-Aware M0

- [实现说明](IMPLEMENTATION.md)
- [验证记录](VALIDATION.md)
- [启动与撤销](APPLY.md)
- [Plan 示例](PLAN_EXAMPLES.json)
- [环境信息](ENVIRONMENT.json)
- [外挂代码](m0_external/)
- [测试](tests/)
- [NPU smoke 结果](artifacts/npu_smoke.json)

状态：**CPU PASS；单 NPU top-k/Plan smoke PASS；单 NPU 真实 e43 完整状态复制与本地专家链 PASS；跨卡 HCCL/完整模型/生产 graph 未完成。**
