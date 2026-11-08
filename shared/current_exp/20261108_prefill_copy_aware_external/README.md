# 20261108 Prefill Copy-Aware 外挂实现

- run：`20261108_1000c`
- 服务器源码：未修改，使用运行时函数包装。
- CPU：12 个逻辑测试 + 3 个外挂安装测试通过。
- NPU：单张 Ascend910_9382 上 `NPU_PASS_TEST_ONLY` 和 `NPU_FULL_LOCAL_PASS_TEST_ONLY`。
- 完整本地链：真实 e43 运行态复制 37,810,176 B，top-8 输入，Plan q=256，e43 加权贡献逐位一致。
- 跨卡 HCCL、完整模型、生产 graph：未完成。

入口：[`20261108_1000c/IMPLEMENTATION.md`](20261108_1000c/IMPLEMENTATION.md)
