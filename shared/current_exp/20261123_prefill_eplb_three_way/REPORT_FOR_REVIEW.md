# Prefill 三组对照：原版、原生 EPLB 冗余、全组 K=1

## 结论先说

本轮没有完成三组都能运行的服务级正式对照，但把阻塞定位到了可复现的实际调用：

- **原版**：EP16 服务与 layer49 routed-MoE replay 均跑通。
- **原生 EPLB + 冗余**：原生算法布局的静态服务与 layer49 replay 跑通；**在线重排失败**，失败点为内部格式 storage offset 和 410/206 MiB NPU OOM，没有 `rebalance end`。
- **我们的全组 K=1**：layer49 routed-MoE replay 跑通，全组只有一个 `37,810,176 bytes` 的真实外部槽；但 **EP16 服务无法启动**，因为原生物理布局要求 `num_physical_experts % ep_size == 0`，全组 K=1 给出 257 个物理专家，`257 % 16 != 0`。

因此正式三组服务级 TTFT/吞吐判定为：

**`INCONCLUSIVE_THREE_WAY_BLOCKED_UNIFORM_LAYOUT`**

已经完成的 EP16 layer49 replay 结论是：三组输出逐位一致；8K、16K、32K 上没有稳定、可重复的延迟优势。中位数差为零点几到几毫秒，EPLB 与 K1 的相对符号随档位变化，并存在孤立的大抖动。不能据此宣布任何一组更快。

- 基线：`738e2be4711e8de132c242a3f30b7c6d4931d8b1`
- 主机：`npu1-23`；`npu1-25` 已检查为 16 NPU 空闲，但未建立 EP32 正式三组
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
  - digest：`sha256:0756b534a0379863c435ec8b7896ba17a18fccda2c8889326fb1dc9453de9be4`
- 模型：`/data_lib/data/models/glm-q`，ModelSlim W8A8
- 生产格式：**UNKNOWN_NO_ACTIVE_SERVICE**
- 本轮 replay 实际格式：base/ND，format code 2

## 1. 输入与拓扑

### 输入

使用已有 ShareGPT human-only 文本，固定 agent 输入：

| 档位 | 实际 tokenizer token | 文件 SHA-256 | 服务输入 |
| --- | ---: | --- | --- |
| 8K | 8,176 | `85370921bcd6ce6c7cf146802ceb5bfe9dffde971ef3f4098eade85df6e49425` | 同一文本 |
| 16K | 16,372 | `8399167a9995509574b166ee2c58c61548699865595698ad8a60c0e47f334f2c` | 同一文本 |
| 32K | 32,729 | `ddfffc791ce5a35d4d0b3ca627c455ffab3ece15a8a59ffbbafb7141847a0561` | 同一文本 |

8K 输入在 layer49 出现明显热点：logical expert 163 有 1,671/65,536 个 top8 任务（2.55%），均匀值约 0.39%，约 6.4 倍。前 16 个热点专家是：

```text
163, 133, 200, 123, 77, 159, 41, 115,
26, 244, 88, 209, 103, 231, 39, 211
```

缺口：没有额外准备同长度的“普通”文本对；本轮每个长度只有一组自然 human-only 输入。因此“热点 vs 普通”的完整 2×3 矩阵未完成。

### EP16 replay 输入

在一次原版 EP16 服务请求中，用只读 observer 捕获 layer49 的真实 `hidden_states`、`topk_ids`、`topk_weights`，按 16 个 TP/EP rank 分别保存。replay 保持：

- EP16、top8、同一 hidden 和 router 输出；
- 同样的 8,192 chunk 配置；
- 同一 layer49 完整主专家权重；
- 原生 DeepEP dispatch/combine；
- 真实 ModelSlim W8A8 grouped expert 链。

验证轮次不是正式性能轮次；observer 不参与 replay 计时。

| 档位 | TP padding 后全局行数 | 每 rank 行数 | top8 任务 |
| --- | ---: | ---: | ---: |
| 8K | 8,192 | 512 | 65,536 |
| 16K | 16,384 | 1,024 | 131,072 |
| 32K | 32,752 | 2,047 | 262,016 |

32K 的 32,752 行包含 TP padding；有效输入仍是上表的 32,729 token。

## 2. 三组实际执行方式

### 原版

- `num_experts=256`，每 rank 16 个 routed expert；
- layout trivial；
- 无额外物理专家；
- 真实原生 dispatch/combine 与真实 W8A8 计算。

### 原生 EPLB + 冗余

原生算法来自目标镜像：

```text
sglang.srt.eplb.eplb_algorithms.deepseek.rebalance_experts
```

EP16 配置：

```text
num_logical_experts = 256
num_physical_experts = 272
num_local_physical_experts = 17
redundant = +16
```

用 8K 校准 trace 的 16-rank logical counts 调用上述原生算法，生成 78 层布局；只有 layer49 是数据驱动布局，其余层填 trivial 占位，并在报告中严格标为 **layer49 原生策略布局回放**。

服务静态组使用：

```text
--ep-num-redundant-experts 16
--ep-dispatch-algorithm static
--init-expert-location /runexp/layouts/eplb16_static.json
```

服务日志确认：

```text
init_expert_location from init_by_mapping using ServerArgs.init_expert_location
```

因此布局确实被服务加载，而不是只生成 JSON。

原生在线更新另跑一组：

```text
--enable-eplb
--ep-num-redundant-experts 16
--eplb-algorithm deepseek_hierarchical
--eplb-rebalance-num-iterations 2
--expert-distribution-recorder-mode stat
--expert-distribution-recorder-buffer-size 1
```

它进入了所有 rank 的 `[EPLBManager] rebalance start`，但没有完成。

### 我们的全组 K=1

layer49 replay 中：

- 通信编号仍为 272，每 rank 17 个 local comm id；
- 每 rank 只有 16 个主专家；
- 每 rank 的 placeholder comm id 不接合法任务；
- 全组只有 rank0 的一个真实外部槽；
- 槽大小 `37,810,176 bytes`；
- hot logical expert 163 从 rank10 复制到 rank0；
- 每个 rank 把本 rank hot expert 任务的一半移到 replica comm id 16；
- 原执行位置的对应任务被删除，不双算。

K1 稳态安装单独重复 5 次：

| 阶段 | 中位数 |
| --- | ---: |
| source prepare / ND 打包 | 0.394 ms |
| 37,810,176-byte HCCL copy | 1.209 ms |
| 总安装 | **1.584 ms** |

该数字不含首次 HCCL lazy-init 和模型加载。首次冷安装出现 2-6 s 的一次性初始化延迟，没有拿来当稳态成本。

## 3. EP16 layer49 replay 结果

统一完成边界：

```text
hidden/topk 已就绪
→ dispatch
→ 所有本地专家计算
→ combine
→ NPU synchronize
```

每个档位两组独立进程加载相同层权重，10 个正式重复，三组顺序轮换。每个样本保留原始行数、任务数、replica 任务数和各 rank 任务分布。

### 延迟

表中是每轮“最慢 rank 的完整 MoE replay 路径”的组级耗时；一个请求包含 8K 的 1 个 chunk、16K 的 2 个 chunk、32K 的 4 个 chunk。

| 档位 | 原版 median / p95 | EPLB median / p95 | ours K1 median / p95 |
| --- | ---: | ---: | ---: |
| 8K | 9.704 / 10.245 ms | 9.924 / 11.686 ms | 9.903 / 11.170 ms |
| 16K | 20.646 / 21.529 ms | 21.589 / 22.552 ms | 21.018 / 68.524 ms |
| 32K | 40.020 / 47.898 ms | 41.844 / 96.692 ms | 40.247 / 42.399 ms |

逐配对中位数差：

| 档位 | 原版 − EPLB | 原版 − ours | 说明 |
| --- | ---: | ---: | --- |
| 8K | −0.319 ms | +0.024 ms | EPLB 略慢，ours 基本持平 |
| 16K | −0.672 ms | −0.194 ms | 两组略慢 |
| 32K | +0.418 ms | +2.091 ms | EPLB/ours 中位数略快，但 p95 和大抖动不支持稳定结论 |

p95 受 60-270 ms 的孤立抖动影响，尤其 16K ours 和 32K EPLB。结论是：**本轮范围内无法区分三组稳态计算优势。**

### 正确性

- 三档、两个新组相对原版：
  - max absolute error = **0**
  - max NRMSE = **0**
  - finite = true
- 每个 chunk 的本地 task count 严格等于 `local_rows × 8`；
- 三组总 top8 任务数完全相同；
- 没有合法的 placeholder/empty 任务；
- K1 slot 只出现在 rank0，其他 rank slot bytes=0。

### 各 rank 任务负载

| 档位 | 原版 max rank / min rank | EPLB max / min | K1 max / min |
| --- | ---: | ---: | ---: |
| 8K | 5,428 / 3,092 | 4,316 / 3,866 | 5,428 / 3,092 |
| 16K | 9,222 / 6,563 | 9,953 / 7,038 | 9,222 / 6,563 |
| 32K | 20,108 / 13,697 | 18,869 / 13,513 | 20,108 / 13,697 |

EPLB 在 8K/32K 降低了最重 rank 的任务数，但 8K 延迟没有改善，16K 甚至增加了最重 rank 任务数。K1 的 hot expert 163 不是当前最慢 rank 的决定因素，因此 rank 最大任务数不变，也解释了 K1 replay 没有稳定收益。

### 实际 replica 任务数

定义：EPLB 中“logical expert 有 >1 个物理副本”的任务数；K1 中通过 comm id 16 进入唯一副本的任务数。

| 档位 | EPLB replica tasks | K1 replica tasks |
| --- | ---: | ---: |
| 8K | 12,071 / 65,536 | 831 / 65,536 |
| 16K | 18,111 / 131,072 | 968 / 131,072 |
| 32K | 23,566 / 262,016 | 583 / 262,016 |

服务静态 EPLB 的 8K observer 单独核对为 **12,061 / 65,536** 个 replica 任务，和 replay 的 12,071 接近；差异来自离线校准 route trace 与静态服务实际 topk 的少量差异。

各项 rank 明细：`artifacts/REPLAY_RANK_DETAILS.json`。

## 4. 显存账

### Replay 持久权重

| 对象 | 每 rank bytes |
| --- | ---: |
| 原版 16-expert bank | 604,962,816 |
| EPLB 17-expert bank | 642,772,992 |
| EPLB 相对原版 | +37,810,176 |
| K1 唯一外部槽（仅 rank0） | 37,810,176 |

EP16 的 K1 全组真实额外专家状态是 **一份 37.8 MB 槽**，不是每个 rank 一份。其他 rank 的 placeholder 只有通信编号，没有权重 storage。

### 服务级显存

同 MEM_FRACTION=0.88：

| 组 | 每 rank 模型权重 | 可用 max_total_tokens |
| --- | ---: | ---: |
| 原版 | 47.97 GiB | 53,632 |
| EPLB +16 静态 | 50.63 GiB | 27,392 |
| 差值 | **+2.66 GiB/rank** | **−26,240 token capacity** |

EPLB 每 rank 多一个完整物理 expert 的所有层权重，因此不是全组 O(1)。K1 的 replay 槽是单层 37.8 MB，但缺少跨 78 层服务接入，不能把 replay 槽大小冒充全模型服务显存。

## 5. 服务级诊断

由于 K1 服务无法启动，下面不是正式三组 TTFT。它是原版与原生 EPLB 静态布局的独立服务诊断，使用相同 human-only prompt、TP16/EP16/DP1、chunk 8,192、output 32、prefix cache disabled、并发 1。

| 档位 | 原版 TTFT median / p95 | EPLB 静态 TTFT median / p95 | EPLB − 原版 |
| --- | ---: | ---: | ---: |
| 8K | 2.5686 / 2.5748 s | 2.5528 / 2.5635 s | −15.8 ms |
| 16K | 5.2319 / 5.2382 s | 5.2390 / 5.2516 s | +7.1 ms |
| 32K | 10.9026 / 10.9189 s | 未运行 | 上下文容量阻塞 |

32K 静态 EPLB 实际失败：

```text
Input length (32741 tokens) exceeds the maximum allowed length (27386 tokens).
```

因为 EPLB +16 的额外权重把可用 KV token capacity 从 53,632 降到 27,392。不是调度参数写错，而是显存容量后果。

服务日志中的 Prefill chunk 仍按 8,192 token 运行；8K 为 1 chunk，16K 为 2 chunks，32K 原版为 4 chunks。

## 6. 已验证的阻塞

### A. K1 服务统一布局断言

配置：

```text
EP16
num_logical_experts=256
ep_num_redundant_experts=1
num_physical_experts=257
```

实际失败：

```text
sglang.srt.eplb.expert_location._init_common
assert num_physical_experts % ep_size == 0
AssertionError
```

所有 rank 在模型初始化阶段失败，服务未进入 Prefill。下一验证入口应修改物理布局契约，让 257 个真实/占位 physical ID 能合法映射；不能只改 argparse 或把 257 四舍五入成 272。

### B. 原生在线 EPLB 重排

16 个 rank均打印 `[EPLBManager] rebalance start`，随后：

```text
RuntimeError: For a tensor of internal format, it's storage_offset must be 0
```

同时出现：

```text
NPU out of memory. Tried to allocate 410.00 MiB
216.30 MiB free
NPU out of memory. Tried to allocate 206.00 MiB
```

没有 `[EPLBManager] rebalance end`。因此：

- 不能报告“重排完成耗时”；
- 不能把静态 layout 的 TTFT 当作包含在线重排的 TTFT；
- 静态组只能标为 **原生策略布局回放**。

### C. EP32

`npu1-25` 已检查为 16 个空闲 NPU，但 EP16 已证明 K1 的统一物理布局契约阻塞。没有修改该契约前，EP32 的正式三组服务对照同样无法成立。本轮没有把 EP16 replay 或 EP16 服务数字外推成 EP32。

## 7. 三组结论

| 组 | 已完成 | 未完成 / 阻塞 |
| --- | --- | --- |
| 原版 | EP16 layer49 replay 8/16/32K；EP16 服务 TTFT 8/16/32K | EP32 未运行 |
| 原生 EPLB + 冗余 | 原生算法 272 布局；静态服务 8/16K；layer49 replay 8/16/32K；实际 replica 任务核对 | 在线重排内部格式错误 + OOM；32K 服务容量不足；EP32 未运行 |
| 我们的全组 K=1 | EP16 layer49 replay 8/16/32K；全组唯一 37.8 MB 真槽；实际分流任务 831/968/583；稳态安装 1.584 ms | 服务启动被 257%16 断言拒绝；EP32 未运行 |

没有稳定正收益。8K 服务中 EPLB 静态布局比原版 TTFT 低 15.8 ms，但 16K 高 7.1 ms，且这不是完成在线重排后的正式三组结果。layer49 replay 的三组中位数差方向也不一致。最诚实的结论是：

**当前实现下，原生 EPLB 和 K1 的收益都在测量噪声或阻塞范围内；没有证据支持服务加速。**

## 8. 产物与复现

主要证据：

- `artifacts/RESULTS_SUMMARY.json`
- `artifacts/REPLAY_RANK_DETAILS.json`
- `artifacts/REPLAY_TIMINGS.csv`
- `artifacts/SERVICE_TTFT.csv`
- `artifacts/K1_INSTALL.json`
- `artifacts/FAILURE_EVIDENCE.json`
- `artifacts/ENVIRONMENT.json`
- `artifacts/INPUTS.json`
- `artifacts/VALIDATION.json`
- `layouts/eplb16_static.json`
- `artifacts/calibration/eplb16_layer49_summary.json`
- 原始 per-rank replay、hidden、日志保留在实验端；`RAW_REPLAY_HASHES.json` 记录哈希

复现入口：

```bash
cd shared/current_exp/20261123_prefill_eplb_three_way

# 生成真实 routing 捕获、原生布局、三档 EP16 replay、服务诊断与验收
./reproduce.sh all

# 单独运行
./reproduce.sh observe
./reproduce.sh layout
./reproduce.sh replay
./reproduce.sh service-off
./reproduce.sh service-eplb
./reproduce.sh k1-blocker
./reproduce.sh online-rebalance-failure
./reproduce.sh analyze
```

所有服务命令使用 `npu1-23` 的 16 个 NPU；EP32 未启动，避免在统一布局阻塞未解决时消耗第二台机器。
