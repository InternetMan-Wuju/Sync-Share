# Global-K service repair and native-dispatch diagnosis

## 结论先说

本轮完成了上一轮 32K 挂起的修复，并跑完两台机器全部 32 die 的三组 AgentX 三档服务矩阵：

- **原版**：8K/16K/32K 全部 PASS。
- **原始 SGLang EPLB + 冗余**：静态原生布局三档全部 PASS；在线重排仍 FAIL，原因是内部格式 storage offset，未产生 `rebalance end`。
- **我们的可配置 K=2**：8K/16K/32K 全部 PASS；32K 的隔离单请求和 8K→16K→32K 连续请求都返回完整响应。

32K 挂起已确认由各 rank 槽复用分支不一致引起，现已修复。新的全组决定流程是：

```text
owner 上报真实槽身份/状态
→ all-gather 到所有 rank
→ 全组对 reuse/install 做同一个 MIN 决定
→ 只有决定 install 时才按统一顺序收发
```

修复后，同层第二个 chunk 在 owner、非 owner 和 source rank 上一致记录 `reuse`，没有 rank 分歧；32K 四轮 8192-token chunk 完整结束。

但真实 DeepEP 副本改道仍受阻：本轮新增的 top-k=1 与 top-k=8 replica dispatcher 都在 `aclnnCamMoeCombineNormal` 处失败，错误为 `Invalid attr=601`。因此服务 K 主矩阵采用明确标注的 **local backfill fallback**：真实槽、真实 W8A8、真实分流和真实回填成立，但副本任务没有经过额外 DeepEP 通信编号。

最终状态：

**`K_SLOT_FUNCTIONAL_MATRIX_PASS_NATIVE_REPLICA_DISPATCH_BLOCKED`**

## 1. 环境和拓扑

- 基线：`22d65f2`
- 主机：`npu1-23`、`npu1-25`
- 每节点 16 个 Ascend 910 die
- 两节点共 32 die，组成一个 `TP32/EP32/DP1` 组
- rank 0-15：`npu1-23` local die 0-15
- rank 16-31：`npu1-25` local die 0-15
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 模型：`/data_lib/data/models/glm-q`，ModelSlim W8A8
- 生产格式：**UNKNOWN_NO_ACTIVE_SERVICE**

32 个 rank 都有分布式初始化证据：`artifacts/EP32_DEVICE_MAP.json`。

## 2. 槽复用一致性修复

### 2.1 新决定流程

`ensure_slots()` 不再让 owner 和各 rank 独立猜测状态。每个槽按固定顺序：

1. owner 提供真实 `(owner_rank, local_slot_id)`、layer、expert、generation、ready、storage pointer；
2. `all_gather_object` 将 owner 状态送到所有 rank；
3. owner 先检查真实 buffer storage pointer；
4. 所有 rank 对“是否可复用”做 `MIN` 全组归约；
5. 全组同时 `reuse`，或全组同时进入相同顺序的安装发送/接收；
6. owner 安装并发布新 generation 后再次广播真实状态。

启动时还校验：

- `K == len(owners) == len(hot_experts)`；
- 首版一 owner 一槽；
- 槽身份是 `(owner, local_slot=0)`。

### 2.2 同层连续命中

用 K=2、layer49 单独启用、12,288-token 两 chunk 请求验证：

- 第一个 chunk：两个 owner 安装，generation=1；
- 第二个 chunk：owner rank0、owner/source rank1、source rank10 都记录 `reuse`；
- 所有 rank 没有进入 send/recv；
- 输出成功，chunk 层误差：
  - max abs `0.001953125`
  - max NRMSE `0.00057698`
- q=0 诊断：安装次数为 0，完全走原路径，输出与基线 SHA 相同。

证据：`artifacts/SLOT_CONSISTENCY.json`。

### 2.3 32K 修复结果

上一轮卡在三轮 8192-token chunk 后，最后一轮缺失。本轮同一 32,768-token AgentX 派生输入：

| 场景 | 结果 |
|---|---|
| 32K 隔离请求 | **PASS**，TTFT 16.1495 s，四轮 chunk 完成 |
| 8K→16K→32K 连续请求 | **全部 PASS** |
| 32K 连续请求 | TTFT 12.4128 s |
| 32K 输出 ID SHA 与原版 | 相同：`c6c9dd5a...` |

没有再用心跳或 HTTP 200 单独冒充成功；以完整响应和 output ids 为完成依据。

## 3. DeepEP 真实副本改道诊断

### 3.1 已接线但未通过的路径

新增服务适配路径（只作为失败诊断保留，服务矩阵没有启用它）：

- 逻辑 top-k 和原权重保持不变；
- 选中的 hot-expert task 通过全组整数配额决定；
- 原路径对应 top-k weight 置零；
- 设计意图是通过独立通信编号把副本任务交给 owner；
- owner 从当前真实槽执行 W8A8 专家链；
- 结果按原 token 顺序回填，再与原 routed 输出相加；
- shared expert 路径保持原样。

独立 dispatcher 使用：

```text
EP16: 272 个通信编号，每 rank 17
EP32: 288 个通信编号，每 rank 9
```

只有配置的 K 个额外通信编号拥有真实槽。其余是协议 padding，真实任务不能落到这些位置。

### 3.2 失败证据

top-k=1 replica dispatcher：

```text
aclnnCamMoeCombineNormal failed
Invalid_Argument EE1001
Invalid attr=601
```

随后改成 top-k=8，仿照服务原路径：

```text
第一列 = 真实 owner replica comm
其余 7 列 = 显式零权重 placeholder
```

dispatch 阶段完成，但在相同的 native combine 处仍失败：

```text
aclnnCamMoeCombineNormal failed
Invalid attr=601
```

因此本轮没有把“额外通信编号的副本任务”标为完成。证据：`artifacts/NATIVE_DISPATCH_FAILURE.json`。

服务功能矩阵使用 local backfill fallback，并明确记录：

- 真实槽；
- 真实 W8A8；
- 真实任务分流；
- 原路径零权重；
- 回填和 combine 输出；
- 但没有额外 DeepEP replica dispatch/combine 成功证据。

## 4. AgentX 输入

来源：

```text
/data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json
```

只有长度和 64-token 块哈希，没有原文本。按历史规则做确定性 token 物化，标为 **AgentX-derived**，不是原始人类文本。

| 输入 | token 数 | 输入指纹 |
|---|---:|---|
| 8K | 8,192 | `86240a227b4f194e7f0e3ab5661942b09c0aa5a44cdcc610f503f54d9e3f3d84` |
| 12K 同层复用测试 | 12,288 | `d496e85dff7d63228f4b2e678d33389dee5e58e9e72756d122d055c8c7b76bc9` |
| 16K | 16,384 | `ef1b6259ebc4a272df50b9e905887909e9acab82924231e72cff73c981b217c9` |
| 32K | 32,768 | `cd222160c6815f5ce8db984894f40d40880333dfed62139a23b4ffa0464fafe1` |

三组使用同一模型、镜像、32-die 拓扑、chunk 8192、并发 1、关闭 radix cache、temperature 0、max_new_tokens 2。表中 12K 只用于 EP16 同层复用诊断，不进入 EP32 三组矩阵。表内耗时是单次请求记录，不据此下性能结论。

## 5. EP32 三组 × 三档

| 输入 | 原版 | EPLB 静态布局 | 我们的 K=2 |
|---|---|---|---|
| 8K | PASS，4.9461 s | PASS，5.0055 s | PASS，2.9569 s |
| 16K | PASS，5.7489 s | PASS，5.8717 s | PASS，6.0152 s |
| 32K | PASS，11.8509 s | PASS，12.0839 s | PASS，12.4128 s |

32K K=2 另有隔离请求：PASS，TTFT 16.1495 s。

输出：

- 原版和 EPLB 静态三档 output IDs SHA 完全一致；
- K=2 的 16K/32K output IDs 与原版一致；
- K=2 的 8K output IDs 与原版不同，但层数值最大绝对误差 `0.001953125`、最大 NRMSE `0.00098393`，全部有限；
- 不把 8K greedy token 差异说成 token-exact。

## 6. 我们的 K=2 服务事实

EP16 小规模：

| 项目 | 结果 |
|---|---|
| K | 2 |
| owner | rank0/expert163，rank1/expert133 |
| 两个槽同层都接任务 | PASS |
| layer49 任务 | slot0 18，slot1 29 |
| layer50 任务 | slot0 15，slot1 5 |
| 同槽跨层复用 | PASS，storage pointer 不变，generation 1→2 |
| 层误差 | max abs 0.0029297，max NRMSE 0.00091977 |

EP32：

- 两个真实槽，owner rank0/rank1；
- 槽大小各 `37,810,176 bytes`；
- 来源 rank20/node25 → owner0/node23，rank16/node25 → owner1/node23；
- 同一 owner storage pointer 跨多次层安装保持固定，generation 增长到 22；
- 32K 隔离实际 replica 任务 61；
- 32K 连续请求实际 replica 任务 61；
- 8K/16K 连续请求 replica 任务 17/38；
- K=2 32K 输出完整返回，并与原版 output IDs 相同。

证据：

- `artifacts/EP16_SERVICE_ACCEPTANCE.json`
- `artifacts/EP32_K2_ACCEPTANCE.json`
- `artifacts/K2_NATIVE_DISPATCH_FALLBACK_TASKS.json`

## 7. 原始 EPLB

### 静态布局

EP32 配置：

```text
logical routed experts: 256
physical experts: 288
redundant: +32
local physical experts per rank: 9
```

原生 DeepSeek EPLB 算法生成布局，静态服务确认：

```text
init_expert_location from init_by_mapping
```

三档均 PASS，输出 IDs 与原版完全相同。8K 固定输入实测 `5,098 / 65,536` 个任务落到有多副本的逻辑专家；本次矩阵服务重跑了三档，逐任务 observer 数据沿用同布局、同输入的前一轮固定布局证据并已标 provenance。

### 在线重排

EP32 在线原生 updater 进入重排后失败：

```text
RuntimeError: For a tensor of internal format, its storage_offset must be 0
```

没有 `rebalance end`。在线布局未生效，未报告在线重排完成时间。静态布局结果单独列出。

## 8. 主要阻塞

### A. Native replica combine

- 独立 replica dispatcher 的 dispatch 可以进入；
- top-k 1 和 top-k 8 两种布局都在 `aclnnCamMoeCombineNormal` 被拒绝；
- 失败早于返回 combine 输出；
- 服务 K 功能路径因此降级为 local backfill。

下一步应检查该 NPU combine 算子对独立 dispatcher buffer、`topk_ids/topk_weights` 元数据和 receiver row count 的约束，不能把 local fallback 当 native 改道。

### B. Hash-first 输入

AgentX 源只有内容块哈希，没有原文本或模型 token id。32K 来自两个同 trace 的 16K 片段，属于 AgentX 派生输入；报告没有称其为原始人类文本。

## 9. 复现

完整入口和阶段说明见 `MANIFEST.json` 与 `reproduce.sh`。以下命令在本轮服务器上执行；`logs/` 和生成的 AgentX token JSON 默认不提交。

```bash
cd shared/current_exp/20261123_global_k_native_service_dispatch

# 生成 AgentX-derived 输入（8K/12K/16K/32K）
python3 scripts/make_agentx_inputs.py \
  --source /data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json \
  --outdir inputs

# 推荐直接按阶段运行（每个阶段会启动、请求、停止并清理）
./reproduce.sh inputs
./reproduce.sh ep16-k0
./reproduce.sh ep16-k1
./reproduce.sh ep16-k2
./reproduce.sh same-layer
./reproduce.sh q0

# EP16 手工等价命令示例
KCONFIG=$PWD/control/k1_verify.json MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 \
  ./scripts/serve_ep16.sh k1 k1_final
KCONFIG=$PWD/control/k2_verify.json MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 \
  ./scripts/serve_ep16.sh k2 k2_final

# 同层复用和 q=0
KCONFIG=$PWD/control/k2_l49.json MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 \
  ./scripts/serve_ep16.sh k2 same_layer
KCONFIG=$PWD/control/k2_q0.json MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 \
  ./scripts/serve_ep16.sh k2 q0

# EP32 三组
./reproduce.sh ep32-off
./reproduce.sh ep32-eplb-static
./reproduce.sh ep32-k2
# 预期失败诊断：在线 EPLB 重排
./reproduce.sh ep32-eplb-online

# 手工等价命令
./scripts/serve_ep32.sh start off ep32_off_new
for s in 8192 16384 32768; do
  python3 scripts/request_agentx.py --url http://127.0.0.1:18023 \
    --input inputs/agentx_${s}.json --out artifacts/ep32_off_new_${s}.json --max-new-tokens 2
done
./scripts/serve_ep32.sh stop

LAYOUT=$PWD/layouts/eplb32_static.json ./scripts/serve_ep32.sh start eplb_static ep32_eplb_new
for s in 8192 16384 32768; do
  python3 scripts/request_agentx.py --url http://127.0.0.1:18023 \
    --input inputs/agentx_${s}.json --out artifacts/ep32_eplb_static_new_${s}.json --max-new-tokens 2
done
./scripts/serve_ep32.sh stop

KCONFIG=$PWD/control/k2_ep32.json ./scripts/serve_ep32.sh start k2 native_ep32_k2_1
python3 scripts/request_agentx.py --url http://127.0.0.1:18023 \
  --input inputs/agentx_32768.json --out artifacts/native_ep32_k2_32768_isolated.json --max-new-tokens 2
for s in 8192 16384 32768; do
  python3 scripts/request_agentx.py --url http://127.0.0.1:18023 \
    --input inputs/agentx_${s}.json --out artifacts/native_ep32_k2_seq_${s}.json --max-new-tokens 2
done
./scripts/serve_ep32.sh stop
```

## 10. 验收汇总

| 项 | 状态 |
|---|---|
| 同层连续命中、全组 reuse 一致 | PASS |
| q=0 不安装槽、走原路径 | PASS |
| 32K 隔离请求 | PASS |
| 8K→16K→32K 连续请求 | PASS |
| EP16 K=1 | PASS，token-exact with q0 baseline |
| EP16 K=2 双槽同层任务 | PASS |
| EP16 K=2 跨层复用 | PASS |
| EP32 原版三档 | PASS |
| EP32 EPLB 静态三档 | PASS |
| EP32 EPLB 在线重排 | FAIL，无 end |
| EP32 K=2 三档 | PASS |
| Extra DeepEP replica dispatch/combine | FAIL，combine attr=601 |

所有本轮服务容器和 NPU 进程已清理。
