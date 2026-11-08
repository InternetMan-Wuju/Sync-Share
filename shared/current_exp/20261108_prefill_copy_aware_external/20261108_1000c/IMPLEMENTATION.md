# M0 外挂实现：Prefill 当前路由复制与配额分流

状态：**外部 runtime hook 实现完成；CPU 测试通过；单 NPU test-only smoke 通过。**

- 未修改 `/sgl-workspace` 下的原始服务器文件。
- 未加载完整 GLM 模型。
- 未启动或重启模型服务。

## 1. 实际环境和接入点

- 镜像：`quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`
- 镜像 ID：`sha256:26a020117564da514f33c4e72353d07020bc4bfe176f4e61858b4abeac6d3e9b`
- 实际基础文件：`sglang/srt/hardware_backend/npu/moe/topk.py`
- 基础文件 SHA256：`f82f25bb343892b7b8b1eb833a4e6ca376a510ac74a3a935e0a1997dbd072f75`
- 实际模型类：`sglang.srt.models.glm4_moe.GlmMoeDsaForCausalLM`
- 接入函数：`fused_topk_npu`
- 逻辑→物理边界：`topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)`

## 2. 外挂机制

`m0_external/install.py` 在进程启动后包装运行时符号：

1. 保留原 `fused_topk_npu` reference；
2. 用 `_fused_wrapper(*args, **kwargs)` 设置当前 `layer_id` ContextVar 后调用原函数；
3. 保留原 `topk_ids_logical_to_physical`；
4. 将 NPU topk 模块和相关调用模块中的 mapping alias 替换成 `_mapping_wrapper`；
5. `_mapping_wrapper` 在得到逻辑 top-k IDs 后调用控制器：
   - `off`：立即调用原 mapping；
   - `shadow`：只统计/规划，调用原 mapping；
   - `active`：只有显式注册运行时 controller 时，返回 Plan 生成的物理 ID；否则回退原 mapping。

`m0_external/launch_server.py` 先安装外挂，再通过 `runpy` 执行原 SGLang 启动模块：

```python
install()
runpy.run_module("sglang.launch_server", run_name="__main__")
```

因此不需要覆盖原文件。

## 3. 已实现逻辑

- 逻辑路由统计：排除 padding、invalid ID、fused shared 列；
- 多输入来源 quota：6+4、q=7 → 6+1；
- hot expert 候选：`n[e] > 2 * total/E`；
- 有限整数 quota：`{0, q0/2, q0}`；
- cold/cache、planner、prepare、dispatch 和 epsilon 门槛；
- logical→physical remap：原 router weight 不变；
- 槽位生命周期：DRAINING / FILLING / PREPARING / READY / FAILED；
- reader 未 drain 时拒绝替换；
- epoch/identity 不匹配拒绝 dispatch；
- 默认 `off` 不做新增统计；`shadow` 不复制、不改 ID。

## 4. 开关

```text
SGLANG_PREFILL_COPY_AWARE_MODE=off       # 默认
SGLANG_PREFILL_COPY_AWARE_MODE=shadow
SGLANG_PREFILL_COPY_AWARE_MODE=active
SGLANG_PREFILL_COPY_AWARE_LAYER=49
```

active 只允许测试 runtime 显式注册；当前仅适配 layer49、Prefill extend、static mapping、expert-TP=1、DP1、eager。

## 5. NPU 实机 smoke 结果

使用单张 `Ascend910_9382` 直接调用镜像内真实 `fused_topk_npu`：

| 检查 | 结果 |
|---|---|
| off 目标槽计数 | 0 |
| shadow 输出与 off 相同 | PASS |
| shadow transport 调用 | 0 |
| active Plan quota | 448 |
| active 物理槽 33 计数 | 448 |
| active transport 调用 | 1 |
| top-k weights finite | PASS |
| slot epoch/identity/READY | PASS |
| 服务启动 | false |
| 完整模型加载 | false |

状态：`NPU_PASS_TEST_ONLY`。
单次 active 调用事件时间约 **113.16 ms**；这是 test-only smoke，不是性能基准。

## 6. 明确未完成

- 真实完整量化专家状态复制；
- ND/NZ staging、接收 buffer、格式恢复；
- HCCL/P2P 和 EP 全组状态提交；
- 全局计数交换；
- 真实 DeepEP dispatch/combine 完整链；
- 生产 graph 接线；
- 额外槽真实 HBM；
- 完整模型精度、TTFT、TPOT、吞吐和 SLO。

不能把本次 smoke 写成完整 M1/M2 或生产加速通过。


## 7. 完整本地应用链（本轮新增）

`m0_external/full_local_smoke.py` 已在同一单 NPU 上运行：

- 通过 publisher 恢复层加载真实 e43/e40 权重；
- 外部 top-k wrapper 生成 top-8 路由；
- Plan 产生 q=256；
- 将 e43 的完整运行态复制到 rank1 独立临时槽；
- 使用真实 W8A8 本地 e43 链处理分流后的两个 token 集合；
- 与原 top-k 权重对应贡献比较。

结果：

| 项目 | 值 |
|---|---|
| status | `NPU_FULL_LOCAL_PASS_TEST_ONLY` |
| top-k | 8 |
| tokens | 512 |
| e0 off / active primary / active destination | 512 / 256 / 256 |
| Plan q | 256 |
| copy bytes | 37,810,176 |
| copy/权重/scale/offset 状态一致 | 全部逐位一致 |
| e43 加权贡献对比 | 逐位一致，NRMSE 0 |
| transport 调用 | 1 |

该结果证明“顶层路由→Plan→真实本地状态复制→真实量化专家链→原权重贡献恢复”的外部本地路径；仍未证明完整模型、跨卡 HCCL、完整 dispatch/combine 或服务 SLO。
