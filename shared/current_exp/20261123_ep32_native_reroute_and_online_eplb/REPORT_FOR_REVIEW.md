# EP32 native single-dispatch reroute and patched online EPLB

## 结论先说

本轮在 `3512cae` 基线上完成了两项关键验证：

1. **我方可配置 K 已经从“本地回填”推进到同一轮原生 DeepEP 分发改道。** 任务在 dispatch 前改写通信目的地，主专家与副本共用一次 DeepEP NORMAL dispatch 和对应的 combine。EP16 K=1/K=2、同层槽复用、跨层复用和 q=0 均通过；EP32 K=2 三档服务请求均完成。
2. **EPLB +32 在线组已挂载适配到 v0.5.20 的 NPU 搬运补丁并真实重排。** 两台节点都加载了相同哈希的补丁；每个 rank 完成 10 次 `rebalance end`，布局发生变化，重排后的正式 AgentX 8K/16K/32K 请求全部完成。

EP32 九格正式矩阵全部完成：

- 原版：8K/16K/32K PASS。
- 在线 EPLB +32：8K/16K/32K PASS，输出 token IDs 与原版一致。
- 我方 K=2：8K/16K/32K PASS；16K/32K 输出 token IDs 与原版一致，8K 输出 ID 不同但每层数值验证有限且有界。

最终状态：

**`NATIVE_SINGLE_DISPATCH_PASS_EPLB_ONLINE_PASS_MATRIX_PASS_8K_TOKEN_DIFF`**

本轮没有把单次 TTFT 当作生产 p95，也没有把 8K 的 token ID 差异说成 token-exact。

## 1. 环境与拓扑

- 基线：`3512cae`
- 主机：`npu1-23`、`npu1-25`
- 每节点 16 个 Ascend 910 die，共 32 die，组成一个 `TP32/EP32/DP1` 组
- rank 0-15：node23 local die 0-15
- rank 16-31：node25 local die 0-15
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 镜像 ID：`sha256:618e95de519b050855cbc2ed469f608fae43f9fbecb74b6c237fc15d9d3f4e0d`
- SGLang：v0.5.20；torch_npu：2.10.0
- 模型：`/data_lib/data/models/glm-q`，ModelSlim W8A8
- 生产格式：**UNKNOWN_NO_ACTIVE_SERVICE**

输入是 AgentX 块身份经确定性物化得到的 token IDs，不是原始人类文本：

| 输入 | token 数 | 指纹 |
|---|---:|---|
| 8K | 8,192 | `86240a227b4f194e7f0e3ab5661942b09c0aa5a44cdcc610f503f54d9e3f3d84` |
| 12K 校准/回归 | 12,288 | `d496e85dff7d63228f4b2e678d33389dee5e58e9e72756d122d055c8c7b76bc9` |
| 16K | 16,384 | `ef1b6259ebc4a272df50b9e905887909e9acab82924231e72cff73c981b217c9` |
| 32K | 32,768 | `cd222160c6815f5ce8db984894f40d40880333dfed62139a23b4ffa0464fafe1` |

来源 SHA256：`8b1fc2a11506d1642de5cb851c92c402d4f1e0569a88d90197fd728913bb2ae7`。32K 来自同一 trace 两个 16K 片段的确定性拼接，这一点在输入元数据和本报告中单独标明。

## 2. 我方路径：一次 dispatch 内完成改道

### 2.1 实现

代码入口：`plugin/k_service.py`。

对于每个目标层：

```text
真实 router top-k
→ 选每个 hot expert 的前 1/2 任务
→ 将选中 top-k 位置的通信编号改成 owner 的外部槽编号
→ 一次 DeepEP NORMAL dispatch
→ 每个本地 slot 执行真实 W8A8 主专家或外部槽链
→ 一次对应的 DeepEP combine
→ 返回 routed MoE 结果
```

细节：

- 逻辑专家和 router 权重不改变；只改变 dispatch 目的地。
- EP16：`num_experts=272`、每 rank 17 个本地通信位置；EP32：`num_experts=288`、每 rank 9 个本地通信位置。
- 主专家仍使用原专家的真实权重；owner 外部槽使用实际复制到 slot 的 W8A8 状态。
- INT8 dispatch 的 per-token scale 被传入 W8A8 链；combine 的输入按 BF16 `[received_rows, hidden]` 准备，这是修复早期 `Invalid attr=601` 的关键。
- 不再把主路径权重置零，也不另构造只发副本的 dispatcher；主专家和副本来自同一轮 dispatch。
- owner 状态仍在收发前通过全组 gather/MIN 决定复用、重装或回退；槽 storage、layer、expert、generation 和 ready 都进入身份检查。

### 2.2 早期失败与修复

第一版把 INT8 dispatch 返回的 scale 当成不支持的状态，诊断后改为把 scale 送入本地 W8A8 链。

第二版将 W8A8 结果直接写回继承 INT8 dtype 的 buffer，`aclnnCamMoeCombineNormal` 报：

```text
Invalid_Argument EE1001, Invalid attr=601
```

最终修复为：保留 INT8 输入和 scale，但 combine 行使用 BF16 输出 buffer。失败阶段、原因和修复记录在：

`artifacts/failures/EP16_SINGLE_DISPATCH_INCIDENTS.json`

这不是把旧 local-backfill 结果冒充原生改道；最终的 `EP16_K*_NATIVE_TASK_ACCOUNTING.json` 和 `EP32_K2_NATIVE_TASK_ACCOUNTING.json` 都来自实际 `call_single_dispatch` 记录。

## 3. EP16 小规模验证

### 3.1 K=1

8K AgentX 输入，layer49/50：

| 指标 | layer49 | layer50 |
|---|---:|---:|
| hot expert 逻辑任务 | 418 | 75 |
| 改道任务 | 205 | 33 |
| 原专家实际计算 | 213 | 42 |
| owner 收到副本任务 | 205 | 33 |
| layer 最大绝对误差 | 0.0068359375 | 0.0068359375 |
| 最大 NRMSE | 0.010568 | 0.011055 |

`task_conservation=true`，原专家主路径计算数等于 `逻辑任务数 - 改道数`，owner 接收数等于改道数。

证据：`artifacts/EP16_K1_NATIVE_TASK_ACCOUNTING.json`。

### 3.2 K=2

12,288-token 两 chunk 请求，layer49，两个 owner：

| 指标 | chunk 1 | chunk 2 |
|---|---:|---:|
| 改道任务 | 210 | 104 |
| owner0 接收 | 205 | 104 |
| owner1 接收 | 5 | 0 |
| 原专家 163 计算 | 213 | 112 |
| 原专家 133 计算 | 6 | 1 |
| 最大绝对误差 | 0.0068359375 | 0.0078125 |
| 最大 NRMSE | 0.010568 | 0.010332 |

同一层第二个 chunk 在 rank0、rank1、rank10 都记录了 owner0 和 owner1 的 `reuse`，没有重新安装。跨层配置中，两个 owner 的 storage pointer 保持不变，generation 按层和请求递增；q=0 配置没有 `install`，128 次 `call_noop`，输出 ID 与 EP16 原版完全一致。

证据：

- `artifacts/EP16_SLOT_CONSISTENCY.json`
- `artifacts/EP16_K2_NATIVE_TASK_ACCOUNTING.json`
- `artifacts/ep16_k2_q0_8192.json`

EP16 的 K=1/K=2 输出与 EP16 原版的 token ID 不完全相同；本报告只声明数值有限和有界，不声明 EP16 token-exact。

## 4. EP32 K=2 原生改道

### 4.1 32K 回归

同一份 32,768-token AgentX 输入：

| 场景 | 结果 | TTFT |
|---|---|---:|
| 32K 隔离单请求 | PASS，四个 prefill chunk 完成，output IDs 与原版一致 | 16.0811 s |
| 8K→16K→32K 连续请求 | 三档全部 PASS | 2.9835 / 6.0231 / 12.4303 s |

32K 隔离和连续请求的 output ID SHA 都是：

`c6c9dd5a573bee55ecc81ad4ad9ae398d74b32c2cfb447ac4ffeca1653be042c`

与 EP32 原版相同。

### 4.2 任务账本

`EP32_K2_NATIVE_TASK_ACCOUNTING.json` 聚合了 704 个 rank/层/seq 调用：

- 每次 dispatch 的真实通信编号数：288
- 每 rank 本地通信位置：9
- 32 个 rank 全部出现
- `task_conservation=true`
- 所有层输出 finite
- 32K 隔离四个 chunk 的改道任务：layer49 累计 948，layer50 累计 1,206
- 32K 连续四个 chunk：layer49 累计 948，layer50 累计 1,206

以 32K 隔离的首个 chunk 为例：

| expert | 逻辑任务 | 改道 | 原专家计算 | owner 接收 |
|---|---:|---:|---:|---:|
| 163 | 383 | 181 | 202 | 181（owner0） |
| 133 | 6 | 2 | 4 | 2（owner1） |

owner0/owner1 的槽分别位于 rank0/rank1，源权重来自 rank20/rank16；两个槽都跨节点复制到 node23。两个 owner 的 storage pointer 在整个 K=2 服务期间保持不变，generation 达到 22。

EP32 K=2 三档输出：

- 8K：`[198, 198, 154842]`，与原版 `[13, 13, 154842]` 不同；最大层绝对误差 `0.0078125`，最大 NRMSE `0.011446`，全部 finite。
- 16K：`[13, 13, 154842]`，与原版一致。
- 32K：`[154827, 154827, 154842]`，与原版一致。

证据：

- `artifacts/EP32_K2_NATIVE_TASK_ACCOUNTING.json`
- `artifacts/ep32_k2_single_32768_isolated.json`
- `artifacts/ep32_k2_single_seq_8192.json`
- `artifacts/ep32_k2_single_seq_16384.json`
- `artifacts/ep32_k2_single_seq_32768.json`

## 5. EP32 在线 EPLB +32

### 5.1 补丁适配

使用目标 v0.5.20 镜像的实际源码作为基线，最小移植：

- `patches/expert_location_updater_v0520.py`
  - NPU 发送前把 FRACTAL_NZ 专家切片物化为 ND；
  - 接收临时 buffer 使用 ND；
  - 按确定性 P2P batch 延迟转换，降低一次性转换峰值；
  - 收完后继续写回原 NZ 权重。
- `patches/torch_npu_distributed_c10d_v0520.py`
  - NPU P2P 使用原子 `batch_isend_irecv`，避免 coalescing manager 的阻塞死锁。

两节点的只读挂载和运行时哈希一致：

| 文件 | SHA256 |
|---|---|
| `expert_location_updater_v0520.py` | `8e10ac923c2c93fac0ba649384ae6ce16431a012a378668ae292ee12bfce5dc7` |
| `torch_npu_distributed_c10d_v0520.py` | `a2003dc906133f52de843426125da344901cfc14c613d8f6b4fe74bed15e2ded` |

证据：`artifacts/EP32_EPLB_PATCHES.json`。

### 5.2 重排完成与布局生效

EP32 配置：

```text
logical routed experts: 256
physical experts: 288
redundant: +32
local physical positions per rank: 9
rebalance iterations: 2
recorder buffer size: 1
```

结果：

- node23 和 node25 每个 rank 都记录 10 次 `rebalance start` 和 10 次 `rebalance end`；每个节点共 160 个 rank-events。
- 观测到的 `rebalance end` 时间约为 4.1–6.6 s/轮。
- layer49 的 `rebalance layout after` 出现 10 个快照，布局发生变化。
- 最终 layer49 的 physical-to-logical map SHA256 为 `7b9d2d4ccee88efe892dd389f71f9477a27a49d0a0f8c78e2bb2f8a327b2944c`。
- 最终布局包含 16 个有副本的逻辑专家、48 个副本物理位置；完整 288 槽 map 保存在 `artifacts/EP32_EPLB_ONLINE_EVIDENCE.json`。

### 5.3 重排后的实际任务

校准输入使用独立的 12,288-token AgentX 派生输入；正式输入为同一批 8K/16K/32K。

正式三档共观测：

- 逻辑任务：`458,752`（8K 1 chunk + 16K 2 chunks + 32K 4 chunks，每 chunk 65,536 个 layer49 任务）
- 落到有副本逻辑专家的实际任务：`162,158`
- 每档：
  - 8K：32,549
  - 16K：31,337 + 30,133
  - 32K：17,541 + 16,921 + 17,030 + 16,647

这些数字来自真实 rank49 observer 路由和实际 physical dispatch 计数，不是静态布局或模拟推算。每个 formal chunk 的活跃 rank 数都是 32，活跃物理位置约 269–280 个（总共 288）。

重排后的 8K/16K/32K output IDs 与原版完全一致：

- 8K/16K：`c228eba5...`
- 32K：`c6c9dd5a...`

证据：

- `artifacts/EP32_EPLB_ONLINE_EVIDENCE.json`
- `artifacts/ep32_eplb_online_calibration_12288.json`
- `artifacts/ep32_eplb_online_8192.json`
- `artifacts/ep32_eplb_online_16384.json`
- `artifacts/ep32_eplb_online_32768.json`

## 6. EP32 九格正式矩阵

| 输入 | 原版 | EPLB online +32 | 我方 K=2 |
|---|---|---|---|
| 8K | PASS，4.9498 s，ID `[13,13,154842]` | PASS，6.7801 s，ID 相同 | PASS，2.9835 s，ID `[198,198,154842]`，层误差有界 |
| 16K | PASS，5.7356 s，ID `[13,13,154842]` | PASS，5.8761 s，ID 相同 | PASS，6.0231 s，ID 相同 |
| 32K | PASS，11.8486 s，ID `[154827,154827,154842]` | PASS，22.7137 s，ID 相同 | PASS，12.4303 s，ID 相同 |

TTFT 为每个 cell 的**单次请求**，没有做重复 block、交错 A/B 或生产 p95；不能据此宣布性能胜负。

`EP32_MATRIX.json` 保存逐格的文件、output IDs 和状态。

## 7. 显存与状态

启动后的 memory-pool 最小可用显存（全部 rank 中最低值）：

- 原版：`27.76 GB`
- 我方 K=2：`27.76 GB`
- 在线 EPLB +32：`25.11 GB`

在线 EPLB 的差值约 `2.65 GB/rank`，是启动快照中的观测值，不等同于 rebalance 瞬时峰值；本轮没有单独采集重排期间峰值。K=2 每个 owner 的外部槽实际为 `37,810,176 bytes`，两个 owner 合计 `75,620,352 bytes`，不是每 rank 分配一个槽。

完整数字：`artifacts/EP32_TIMINGS_MEMORY.json`。

## 8. 正确性边界

- 原版与在线 EPLB +32 的三档 output IDs 完全一致。
- 我方 K=2 的 16K/32K output IDs 与原版一致；8K output IDs 不同，但层输出 finite，最大绝对误差 `0.0078125`，最大 NRMSE `0.011446`，task conservation 通过。
- `EP16` 的 K=1/K=2 也是数值容差通过，不声明 token-exact。
- 本报告没有把单次 TTFT 当作生产 p95，也没有声称端到端生产加速。
- observer 额外开销未用于性能计时；正式矩阵是服务请求记录。
- 生产格式仍为 `UNKNOWN_NO_ACTIVE_SERVICE`。

## 9. 复现入口

```bash
cd shared/current_exp/20261123_ep32_native_reroute_and_online_eplb
./reproduce.sh inputs
./reproduce.sh ep16-k1
./reproduce.sh ep16-k2
./reproduce.sh same-layer
./reproduce.sh q0
./reproduce.sh ep32-original
./reproduce.sh ep32-eplb-online
./reproduce.sh ep32-k2
./reproduce.sh validate
```

`reproduce.sh` 使用 `DEPLOY_ROOT=/data_lib/cyt/ep32_native_reroute_20261123`，启动脚本会把 `plugin/`、`control/`、`layouts/` 和 `patches/` 同步到两台节点。运行时日志和生成的 AgentX token JSON 默认不提交到 Git。

## 10. 交付清单

- `artifacts/RESULTS_SUMMARY.json`：整体结果和证据索引
- `artifacts/VALIDATION.json`：自动验收
- `artifacts/EP32_MATRIX.json`：九格矩阵
- `artifacts/EP32_TIMINGS_MEMORY.json`：耗时和显存
- `artifacts/EP16_SLOT_CONSISTENCY.json`：同层/跨层/ q0
- `artifacts/EP16_K*_NATIVE_TASK_ACCOUNTING.json`：EP16 任务账本
- `artifacts/EP32_K2_NATIVE_TASK_ACCOUNTING.json`：EP32 任务账本
- `artifacts/EP32_EPLB_ONLINE_EVIDENCE.json`：在线重排与 observer 任务证据
- `artifacts/EP32_EPLB_PATCHES.json`：补丁哈希与两节点加载
- `artifacts/failures/EP16_SINGLE_DISPATCH_INCIDENTS.json`：早期失败与修复
