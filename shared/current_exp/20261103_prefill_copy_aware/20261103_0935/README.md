# 20261103 Prefill Copy-Aware M0

- [实现说明](IMPLEMENTATION.md)
- [环境与源码哈希](ENVIRONMENT.json)
- [验证记录](VALIDATION.md)
- [补丁应用/撤销](APPLY.md)
- [Plan 示例](PLAN_EXAMPLES.json)
- [补丁](patches/prefill_copy_aware_m0.patch)
- [CPU 测试](tests/test_cpu_logic.py)
- [补丁静态测试](tests/test_patch_integration.py)
- [NPU 验证入口（本轮未执行）](tests/npu_validate_prefill_copy_aware.py)

状态：M0 代码和 CPU 测试完成；NPU、HCCL、生产 graph 未验证。
