# M0 实现说明：Prefill 当前路由复制与配额分流

状态：**M0 代码与 CPU 测试完成；NPU/HCCL/生产 graph 未验证**。
日期：2026-11-03
本轮未启动 NPU 实验，未重启模型服务。

## 1. 实际服务器环境

- 节点：`npu1-23`
- 镜像：`quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`
- 镜像 ID：`sha256:26a020117564da514f33c4e72353d07020bc4bfe176f4e61858b4abeac6d3e9b`
- SGLang：0.5.18
- torch：2.10.0+cpu
- torch_npu：2.10.0
- sgl-kernel-npu：2026.6.1
- 实际模型入口：`sglang/srt/models/glm4_moe.py::GlmMoeDsaForCausalLM`
- 实际 MoE 块：`sglang/srt/models/glm4_moe.py::Glm4MoeSparseMoeBlock`
- 目标层：layer 49（沿用现有 `layer_id` 约定）

## 2. 实际调用链

```text
Glm4MoeSparseMoeBlock.forward_deepep
  -> ExpertLocationDispatchInfo.init_new(layer_id=49)
  -> self.topk(...)
  -> TopK.forward_npu
  -> hardware_backend/npu/moe/topk.py::fused_topk_npu
  -> 原始逻辑 top-k IDs
  -> [本轮 hook] prefill_copy_aware.maybe_apply_prefill_copy_aware
  -> 若未应用：原 topk_ids_logical_to_physical
  -> DeepEPDispatcher / _DeepEPDispatcherImplNormal
  -> get_dispatch_layout / dispatch
  -> 原 MoE runner / combine
```

关键服务器源码：

| 文件 | 行/函数 | SHA256 |
|---|---|---|
| `sglang/srt/models/glm4_moe.py` | `GlmMoeDsaForCausalLM`/`Glm4MoeSparseMoeBlock` | `da66fe6314fa84e7401900cf7e7a05d7dced72b39477595c3a67fda735e24256` |
| `sglang/srt/hardware_backend/npu/moe/topk.py` | `fused_topk_npu`，逻辑→物理边界 | `f82f25bb343892b7b8b1eb833a4e6ca376a510ac74a3a935e0a1997dbd072f75` |
| `sglang/srt/eplb/expert_location_dispatch.py` | `topk_ids_logical_to_physical` | `c7f6b68fb9c6262ab781de8c31514580f36046732ecf75868e3fdc0c6a9e11b0` |
| `sglang/srt/layers/moe/token_dispatcher/deepep.py` | `_DeepEPDispatcherImplNormal`，`get_dispatch_layout` | `201827c6ffb2289392b9e1b36908ca524009c864d07e19388603f435efd76d2d` |
| `sglang/srt/layers/quantization/modelslim/modelslim.py` | ModelSlim runner 构造 | `c981304e11d625b0d58d2f24c1be576aadecf011603213cd0e92ae164de854cd` |
| `sglang/srt/layers/moe/fused_moe_triton/layer.py` | FusedMoE 权重/布局 | `dcb2620e96f23e1004e1439914a807e74293d3c1080db929f9aa542e742e5f05` |

## 3. 补丁内容

补丁：`patches/prefill_copy_aware_m0.patch`

包含：

1. 新文件 `sglang/srt/eplb/prefill_copy_aware.py`；
2. 修改 `sglang/srt/hardware_backend/npu/moe/topk.py`，在逻辑 top-k 之后、标准物理映射之前调用 hook。

`topk.py` 的行为：

```python
if expert_location_dispatch_info is not None:
    copy_aware_applied = False
    if get_is_extend_in_batch():
        topk_ids, copy_aware_applied = maybe_apply_prefill_copy_aware(...)
    if not copy_aware_applied:
        topk_ids = topk_ids_logical_to_physical(...)
```

因此：

- 默认 `off`：不改 ID，不做新增计数/通信；
- `shadow`：统计逻辑路由并生成计划候选，但不复制、不改变物理 ID；
- `active`：必须显式注册测试 runtime controller；未注册时自动回退原路径；
- 当前 active 仅允许 `static` 物理映射、layer49、Prefill extend 路径；
- `lp`/其他映射只允许后续验证，不在 M0 active 范围。

## 4. 已实现的 CPU 逻辑

`prefill_copy_aware.py` 包含：

| 模块 | 功能 |
|---|---|
| `collect_routing_stats` | 排除 padding、无效 ID、shared 列，按输入来源计数 |
| `RoutingStats.quota_by_source` | 全局整数配额按来源稳定前缀分配 |
| `Planner` | hot_factor、有限候选、q0/q0/2/0、成本/误差判定 |
| `remap_logical_to_physical` | 只改物理 ID；逻辑 expert 和 router weight 不变 |
| `SlotManager` | EMPTY/DRAINING/FILLING/PREPARING/READY/FAILED、reader 计数、epoch/身份检查 |
| `RuntimeController` | 从实际逻辑 top-k 统计到 Plan，再到可选活跃 remap |
| `maybe_apply_prefill_copy_aware` | 实际 NPU top-k hook，默认 off |

## 5. 默认开关

```text
SGLANG_PREFILL_COPY_AWARE_MODE=off       # 默认；不做新增统计
SGLANG_PREFILL_COPY_AWARE_MODE=shadow    # 只统计/计划
SGLANG_PREFILL_COPY_AWARE_MODE=active    # 未注册 runtime 时自动回退
SGLANG_PREFILL_COPY_AWARE_LAYER=49       # 默认 layer49
```

其他 planner 参数存在于 `RuntimeController`/`ServiceRuntimeConfig`，不在 M0 中从服务启动参数自动开启。
**没有 `register_runtime_controller` 和对应测试依赖时，active 不会生效。**

## 6. 尚未实现/未验证

以下均明确为未完成，不得写成已通过：

1. 实际跨 EP 的计数 all-reduce/控制通信；
2. 真实临时槽的分配、显存预算和层生命周期；
3. 真实模型运行态量化状态 manifest；
4. NZ→ND、独立接收 buffer、格式恢复和全状态 copy；
5. HCCL source/target 通信组、Work 完成语义、故障回退；
6. 真实 dispatch/combine 完整链；
7. active 服务接线与生产 graph 等价性；
8. 实际额外槽是否全层生效及 HBM 峰值；
9. 真实精度、TTFT、吞吐和 SLO。

本轮只做无 NPU 设备的源码检查和无设备容器；没有启动模型服务，没有读取 NPU 状态。

## 7. 执行结论

> **实现完成度：M0 代码与 CPU 逻辑完成；硬件验证未开始。**
>
> 可以进入 M1：在单 NPU 上使用测试专用 cost table 和 fake/sentinel transport，串行验证 Plan→copy/staging→精确分流→原专家计算→恢复；但真实跨卡复制、HCCL 和生产 graph 仍未完成。
