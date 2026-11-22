# Prefill 空编号 + 全组 K=1：最小可行性证明

**最终判定：GO（范围受限）。** 256 原生阳性对照、257 实际失败、258 原生 dispatch/combine、真实 ModelSlim W8A8 部分/全部分流、空编号零专家状态、全组唯一槽以及 layer49→50→49 三轮复用均通过。

GO 的边界是：

- 真实主专家只加载了有限活跃集合：每 rank 每层 4 个主专家，共 8 个活跃逻辑专家；
- 没有验证完整 128/128 主专家库实际分配；
- 没有启动完整模型服务、TTFT/吞吐、Decode、K=2、跨节点或并发请求；
- 本轮性能未测，因此“接法可行”不等于“已经获得性能收益”。

- 方案提交：`f7e2638`
- 上一轮基线：`595428ca960c4c04da5e5dde0c44e372cd5b8377`
- 主机：`npu1-23`
- 设备：两张空闲 Ascend 910C，`davinci0`、`davinci1`
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 模型：`/data_lib/data/models/glm-q`
- 层：真实 layer49 / layer50
- 副本：真实 expert43，唯一外部槽，37,810,176 bytes

## 1. 实际接口路线

首次使用镜像默认 DeepEP normal strategy 做 256 阳性对照时，真实调用在：

```text
Buffer.get_dispatch_layout
→ deep_ep/strategies/normal_strategy.py
→ RuntimeError: aclnnDispatchLayout or aclnnDispatchLayoutGetWorkspaceSize
  not in libopapi.so
```

这说明默认 normal 通信算子在该镜像不可用。按方案允许的 Python 适配，改用安装包中已有的 DeepEP **NORMAL / alltoall strategy**：

```text
DeepEPDispatcher(NORMAL)
→ DEEP_USE_MODE=alltoall
→ torch_npu.npu_moe_init_routing_v2
→ torch.distributed all_to_all_single / all_to_allv
→ 真实专家计算
→ torch_npu.npu_moe_finalize_routing
→ DeepEP combine
```

这不是自写 P2P：任务分发和合并仍通过安装镜像的原生 DeepEP normal dispatcher 与 NPU routing/finalize 算子完成。

## 2. P0：实际 256 阳性与 257 失败

### 2.1 256 原生阳性

两 rank 使用：

```text
num_experts=256
num_local_experts=128
top_k=8
native DeepEP NORMAL/alltoall
```

结果：

- rank0：expected/actual task 96/96，任务多重集一致；
- rank1：expected/actual task 32/32，任务多重集一致；
- 专家类别标签最大误差约 `0.0129`；
- token 位置最大误差 `0.0`；
- dispatch/combine 均真实执行。

因此不是“256 也失败后归因于 257”。

### 2.2 实际 257 调用

使用相同原生 dispatcher 实际构造 `num_experts=257`：

```text
constructor: PASS
dispatch layout reshape:
shape '[2, 128]' is invalid for input of size 257
```

两个 rank 都捕获到同一真实异常。失败位置在 alltoall layout 把 257 个专家按 2 rank 重排为 `[2,128]` 时，说明 257 无法进入均匀通信编号布局。

原始栈：`artifacts/interface_run_257_rank{0,1}.json`。

## 3. P1：258 原生 dispatch/combine

映射严格使用方案：

| owner | 本地编号 | 通信编号 | 内容 |
|---|---|---|---|
| rank0 | 0..127 | 0..127 | 逻辑专家 0..127 |
| rank0 | 128 | 128 | 空编号 |
| rank1 | 0..127 | 129..256 | 逻辑专家 128..255 |
| rank1 | 128 | 257 | expert43 唯一副本 |

### 3.1 任务与合并

258 协议测试结果：

- rank0：90/90 本地任务，任务多重集完全一致；
- rank1：38/38 本地任务，任务多重集完全一致；
- 逻辑专家 0、43、127、128、255 每个均有 16 个实际任务；
- replica comm257 收到 6 个测试任务；
- 空 comm128 合法任务数：0；
- 专家类别标签误差约 `0.0129`；
- token 位置误差 `0.0`；
- 合并权重只应用一次，输出与独立期望一致。

### 3.2 空编号拒绝

故意把一个通讯任务改到 comm128 后：

- 两个 rank 先做全局布尔 all-reduce；
- 两 rank 均观察到非法空编号；
- 在进入 dispatch 前一致拒绝；
- dispatch 未调用。

### 3.3 零合法任务 rank

真实 zero-task 场景中，rank1 不发送合法 token。直接 zero-shape alltoallv 在 HCCL 上实际失败：

```text
RunAicpuRpcSrvLaunchV2_alltoallv
kernel param invalid
ACL stream synchronize failed: 507018
```

随后按方案允许的 padding 方式验证：

- rank1 只发送 1 个零权重 padding token 到空编号128；
- rank1 合法发送任务数 0、合法接收任务数 0；
- rank0 收到 8 个 empty padding row，不做专家计算；
- rank0 的 64 个合法任务和合并标签正确；
- 空编号 padding 不进入专家状态、不增加专家任务。

结论：零合法任务 rank 可以参与协议，但 alltoallv 需要显式空的 padding 编号；真正 zero shape 的 alltoallv 在当前 HCCL 不可用。

## 4. P2：真实 W8A8 与原生 258 通信

真实活跃逻辑专家集合：

```text
[0, 1, 43, 127, 128, 129, 130, 255]
```

主专家 owner：

- rank0：0、1、43、127；
- rank1：128、129、130、255；
- rank1 唯一副本：layer当前版本的 expert43，通讯编号257。

每个 rank 每层加载 4 个真实主专家；layer49 和 layer50 共两组。没有为 layer49/layer50 预建完整备用 bank。

真实 dispatch/combine 结果：

| 模式 | replica tasks | 空编号任务 | 最大误差 vs 独立参考 | 任务身份 |
|---|---:|---:|---:|---|
| 无分流 | 0 | 0 | ≈0.00045 | PASS |
| 部分分流 | 8 | 0 | ≈0.00045 | PASS |
| 全部分流 | 16 | 0 | ≈0.00045 | PASS |

预定义容差为绝对误差 `0.02`、相对误差 `0.02`；所有运行均通过。参考路径独立于 258 映射：

- 两 rank 交换原始 token；
- 每个 owner 只用真实主专家链计算自己负责的任务；
- 路由权重在本地累加；
- 通过标准 HCCL all-reduce 得到完整参考输出。

因此不是拿待测编号映射生成自己的正确答案。

真实计算链为：

```text
真实 hidden [8,6144]
→ 原生 DeepEP 258 dispatch
→ 真实 ModelSlim W8A8 w13 GMM
→ SwiGLU / dynamic quant
→ 真实 w2 GMM
→ 原生 DeepEP combine
```

## 5. P3：存储审计与跨层复用

### 5.1 实际主权重容量

每 rank 实际加载：

- layer49：4 个主专家；
- layer50：4 个主专家；
- 每 rank 去重后主专家 storage：`302,481,408 bytes`；
- `full_128_per_rank_main_library = false`。

没有把128个主专家先分配再删除。

### 5.2 全组唯一额外槽

- rank0 持久槽数量：0，字节0；
- rank1 持久槽数量：1；
- rank1 槽 storage：`37,810,176 bytes`；
- 全组额外持久专家状态：**1 份 / 37,810,176 bytes**；
- 空编号持久专家状态：**0 bytes**；
- 临时 flat/transfer/benchmark buffer 单独列在 `STORAGE_AUDIT.json`，不冒充额外专家副本。

### 5.3 layer49 → 50 → 49

同一槽完成 3 个完整往返，共 7 次安装：

- lease generation：1..7；
- 槽底层 storage pointer 全程相同：`20618833559552`；
- slot bytes 全程 `37,810,176`；
- layer49 hash：`4ff89f8d...ec8821`；
- layer50 hash：`9926f215...29b8f`；
- 每次源卡发送 hash 与目标槽接收 hash 完全一致；
- 每轮 layer50/layer49 的真实 W8A8 输出均与独立参考一致；
- 每次 all-replica 运行接收16个 replica tasks；
- 空编号任务始终0；
- 最终槽安全释放。

因此权重确实切换、输出正确、storage 复用成立，额外持久专家容量不随层切换增长。

## 6. 修改了哪些代码

1. 使用 DeepEP 安装包已有的 NORMAL/alltoall strategy，绕过镜像缺失的 `aclnnDispatchLayout`。
2. 增加 258 通信映射和空编号 guard：
   - rank1 原专家通信编号整体偏移；
   - comm128 为空；
   - comm257 为 expert43 唯一副本。
3. 增加外部槽 runner 适配：
   - 主专家按真实 local-expert 分组执行；
   - comm257 直接读取唯一槽；
   - 空 comm128 不进入任何权重计算；
   - 输出恢复原生 combine 顺序。
4. 增加独立参考：
   - 不调用 258 映射计算参考；
   - 使用真实主专家分片计算并通过 HCCL all-reduce 汇总。
5. 增加 storage 去重审计和跨层槽复用记录。

## 7. 性能

**本轮性能未测。**

必测功能全部通过后，性能仍需要：

- 使用固定主点 `8192/2048`；
- 统一包含原生 mapping、dispatch、真实计算、combine、同步和 use 引用归还；
- 单独计算首次安装和跨层驱逐；
- 不能用本轮小规模16-token功能测试的时间代表性能。

## 8. 复现

```bash
cd shared/current_exp/20261122_global_k_placeholder_proof

./reproduce.sh interface
./reproduce.sh protocol
./reproduce.sh real
./reproduce.sh memory
./reproduce.sh reuse

# 全流程
./reproduce.sh all
```

硬件要求：两张空闲 Ascend 910C；镜像和模型路径见 `artifacts/environment.json`。

主要证据：

- `artifacts/INTERFACE_RUNS.json`
- `artifacts/TASK_IDENTITY.json`
- `artifacts/STORAGE_AUDIT.json`
- `artifacts/LAYER_REUSE.json`
- `artifacts/VERDICT.json`
- 原始 rank JSON 与日志
- `artifacts/runtime_sources.sha256`
- `artifacts/model_metadata.sha256`

## 9. 未完成项

- 完整 128/128 主专家库分配；
- 完整 MoE 层和 shared expert；
- 完整模型 service；
- TTFT/吞吐；
- K=2；
- Decode；
- 跨节点；
- 多请求并发；
- 性能对照。

本轮结论只覆盖：**有限活跃主专家集合下，258 原生 all-to-all 通信、真实 ModelSlim W8A8、全组唯一 K=1 槽和两层复用可行。**
