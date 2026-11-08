# 20261108 Prefill Copy-Aware 外挂实现

- run：`20261108_1000b`
- 服务器源码：未修改，使用运行时函数包装。
- CPU：12 个逻辑测试 + 3 个外挂安装测试通过。
- NPU：单张 Ascend910_9382 上 `NPU_PASS_TEST_ONLY`；真实调用镜像内 `fused_topk_npu`。
- 完整量化状态/HCCL/生产 graph：未完成。

入口：[`20261108_1000b/IMPLEMENTATION.md`](20261108_1000b/IMPLEMENTATION.md)
