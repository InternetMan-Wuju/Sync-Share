# Global-K service integration and two-node EP32 validation

## 结论先说

本轮做了两条线：

1. 在真实 SGLang Prefill 服务中接入可配置 K 的外部专家槽，完成 EP16 `K=0/K=1/K=2`、layer49+50、固定 owner 的实际运行。K=1 和 K=2 都有真实槽、实际分流任务和层输出数值验证；K=2 两个槽在同层都接到任务，并验证同一槽跨层复用。
2. 在两台机器的全部 32 个 die 上组成同一个 `TP32/EP32/DP1` 组，原版完成 AgentX 派生 8K/16K/32K；原生 EPLB 静态布局也完成三档，输出与原版完全一致；我们的 K=2 完成 8K/16K，32K 在可见 Prefill chunk 完成后卡住，未返回响应。

最终状态：**`PARTIAL_MATRIX_READY_K2_32K_HANG`**。

最重要的限制：本轮服务版 K 路径采用 **weight-masked local backfill**。它把 owner 本 rank 上选中的热点任务从原路径权重中置零，用真实外部槽计算后回填到同一最终输出；它没有把副本任务送进额外的 DeepEP 通信编号。通信编号与真实槽分离已在上一轮 replay 中验证，但尚未完整接入服务 dispatch。因此本轮应标为“真实服务功能原型”，不能标成完整通信协议接入。

## 1. 环境

- 仓库基线：`6893f7a`
- 主机：`npu1-23`、`npu1-25`
- 每节点 16 个 Ascend 910 die，两节点共 32 die
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 模型：`/data_lib/data/models/glm-q`，ModelSlim W8A8
- 生产格式：**UNKNOWN_NO_ACTIVE_SERVICE**
- 本轮实际服务格式：ModelSlim 运行时 ND/NZ 混合；服务适配验证数值，不声明生产格式兼容性

EP32 rank → 节点 → die：

```text
rank 0-15  -> npu1-23, local die 0-15
rank 16-31 -> npu1-25, local die 0-15
```

32 个 rank 均记录到 `Init torch distributed ends`，两节点同属一个 EP32 组。证据：`artifacts/EP32_DEVICE_MAP.json`。

## 2. 可配置 K 服务实现

### 2.1 启动配置

`control/k1.json`、`control/k2.json`、`control/k1_ep32.json`、`control/k2_ep32.json` 读取：

- `K`：全组真实额外槽总数；
- `owners`：每个槽的固定 owner rank；
- `hot_experts`：该槽承载的逻辑专家；
- `layers`：启用层，本轮为 layer49、layer50；
- `split_num/split_den`：固定分流比例，本轮为 1/2；
- `ep`、`main_per_rank`、日志目录和 verify 开关。

启动时检查 `len(owners)==len(hot_experts)==K`，每个 owner 只创建一个真实 flat slot。原生 `--ep-num-redundant-experts` 始终为 0，因此不借用每 rank 原生冗余来伪装 K 槽。

### 2.2 执行方式

每个启用层：

```text
logical top-k / router weights
→ 选择 owner 本 rank 上热点专家的一部分任务
→ 把该任务在原路径的 top-k weight 置 0
→ 原 grouped MoE 路径执行剩余任务
→ 从真实外部槽执行 W8A8 专家链
→ 用原 top-k weight 回填
→ 与 shared expert / 原 combine 输出相加
```

这是一条真实服务内的数值路径，但原路径仍会以零权重计算被搬走的 token；这是第一个可运行原型，不是零冗余开销实现。

### 2.3 槽安装与跨层复用

slot owner 读取真实池记录（owner、槽号、layer、expert、generation、ready、storage pointer），复制完成后才发布 ready。同层的任务在 copy 完成后才使用槽。

EP16 K=1：

- 全组 1 个真实槽；
- owner rank0，expert163，来源 rank10；
- 槽大小 `37,810,176 bytes`；
- layer49 generation1，layer50 generation2；
- 两层使用同一 storage pointer；
- 实际分流任务：layer49 `18`，layer50 `15`。

EP16 K=2：

- 全组 2 个真实槽；
- owner rank0/expert163，owner rank1/expert133；
- 两个槽在同层都收到任务；
- layer49：slot0 `18`，slot1 `29`；
- layer50：slot0 `15`，slot1 `5`；
- 每个 owner 的 storage pointer 跨层不变，generation 从 1 增到 2；
- 层输出相对原路径：max abs `0.0029296875`，max NRMSE `0.00091977`，全部有限。

K=2 确实覆盖了两个槽同时接到任务，也覆盖了同槽跨层复用。证据：`artifacts/EP16_SERVICE_ACCEPTANCE.json`。

## 3. EP16 服务验收

请求：8,176-token human-readable ShareGPT prompt，输出上限 2，temperature 0，固定 1/2 owner-local quota。

| 配置 | 请求 | 真实槽数 | 实际任务 | TTFT | 输出 SHA | 结论 |
|---|---|---:|---:|---:|---|---|
| K=0 | 成功 | 0 | 0 | 4.6680 s | `c977dbe...` | PASS |
| K=1 | 成功 | 1 | 33（两层合计） | 5.3759 s | `c977dbe...` | PASS，最终文本与 K=0 完全一致 |
| K=2 | 成功 | 2 | 67（两层合计） | 5.8505 s | `10c5fe5...` | 数值容差 PASS；greedy 最终文本与 K=0 不同 |

K=2 的最终文本不同，但 MoE 层原始误差很小；这说明最终 token 选择可能因微小数值扰动改变。报告不把 K=2 说成 token-exact，只把层数值和请求完成作为通过依据。

## 4. AgentX 输入

使用历史入口：

```text
/data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json
```

该数据只有 64-token 块哈希和长度，不包含原文本；按历史规则确定性物化 token ID，并明确标为 **AgentX-derived**，不是原始人类文本。

| 输入 | token 数 | SHA-256 |
|---|---:|---|
| 8K | 8,192 | `86240a227b4f194e7f0e3ab5661942b09c0aa5a44cdcc610f503f54d9e3f3d84` |
| 16K | 16,384 | `ef1b6259ebc4a272df50b9e905887909e9acab82924231e72cff73c981b217c9` |
| 32K | 32,768 | `cd222160c6815f5ce8db984894f40d40880333dfed62139a23b4ffa0464fafe1` |

三档都通过服务 `/generate` 发送 token IDs，temperature0，max_new_tokens2，同一模型、同一 32-die 拓扑、chunk 8,192、并发 1、prefix cache 关闭。

## 5. EP32 三组结果

| 输入 | 原版 | 原生 EPLB 静态布局 | 我们的 K=2 |
|---|---|---|---|
| 8K | PASS，TTFT 4.9628 s | PASS，TTFT 5.0861 s | PASS，TTFT 6.3633 s |
| 16K | PASS，TTFT 5.7508 s | PASS，TTFT 5.8521 s | PASS，TTFT 5.9519 s |
| 32K | PASS，TTFT 11.8583 s | PASS，TTFT 12.0921 s | **FAIL_HANG**，可见 chunk 结束后无响应 |

### 5.1 原版

- 两节点全 32 die 加入同一个 EP32；
- 三档请求全部成功；
- 8K/16K 输出 ID SHA 相同：`c228eba5...`；
- 32K 输出 ID SHA：`c6c9dd5a...`。

### 5.2 原生 EPLB + 冗余

配置为 EP32、全组每层 +32 个真实冗余位置、288 个物理专家、每 rank 9 个物理槽。

- 使用目标镜像原生 deepseek 算法，从 8K 校准计数生成布局；
- 静态服务通过 `init_expert_location from init_by_mapping` 生效；
- 8K/16K/32K 请求都成功；
- 三档输出 ID 与原版完全一致；
- 8K 服务实际观测到 `5,098 / 65,536` 个任务落到有多个副本的逻辑专家；
- 这是静态布局数据，不是在线重排后的数据。

在线重排单独尝试：EP32 进入原生 updater 后失败：

```text
RuntimeError: For a tensor of internal format, its storage_offset must be 0
```

没有 `rebalance end`。因此在线重排完成耗时和在线布局收益均未得到有效结果。静态结果单列，不能代替在线组。

### 5.3 我们的 K=2

- EP32 真实槽：2 个，owner rank0/rank1；
- hot experts：163、133；
- 权重源：rank20（node25）→ owner0（node23），rank16（node25）→ owner1（node23），有实际跨节点槽复制；
- 同一 owner 的槽跨 layer49/layer50 复用，storage pointer 保持不变，generation递增到 11；
- 8K：实际 replica 任务 17；PASS；
- 16K：实际 replica 任务 38；PASS；
- 32K：layer49/50 的可见 chunk 有实际任务和有限数值验证，但请求最终卡住：detokenizer heartbeat 过期，HTTP 无响应；FAIL_HANG；
- 8K 输出 ID 与原版不同，16K 输出 ID 与原版相同；K=2 的层数值最大绝对误差 `0.001953125`，NRMSE `0.0009839`。

K=2 32K 不是提前拒绝，而是已经完成可见 Prefill 任务后挂起。准确失败位置是请求完成/返回阶段，而不是槽 identity 或任务守恒检查。

## 6. 各档验收状态

| 项 | 状态 | 说明 |
|---|---|---|
| K=0 与原版一致 | PASS | 服务直通；K=1 最终输出与 K=0 token-exact |
| K=1 槽数/任务/输出 | PASS | 1 槽、33 个真实分流任务、层误差在容差内 |
| K=2 槽数/双槽任务 | PASS | 2 槽；同层两个槽都有任务 |
| 跨层同槽复用 | PASS | storage pointer 不变，generation 递增 |
| EP32 原版三档 | PASS | 32 die、同一 EP32、同一 AgentX 输入 |
| EP32 EPLB 静态三档 | PASS | 布局生效，输出与原版一致 |
| EP32 原生在线重排 | FAIL | internal-format storage offset；无 end |
| EP32 K=2 8K/16K | PASS | 真实槽、真实分流、层数值通过 |
| EP32 K=2 32K | FAIL_HANG | visible chunks 后无响应 |
| 通信编号与真实槽完整分离 | NOT_COMPLETE | 服务当前为 local backfill，没有副本任务走额外 DeepEP 编号 |

## 7. 显存与成本基线

- EP16 K=1 槽：每 owner `37,810,176 bytes`；全组一个。
- EP16 K=2：两个 owner 各一个 `37,810,176 bytes` 槽。
- EP32 K=2：两个 owner 各一个 `37,810,176 bytes` 槽；跨层复用同一 storage。
- EPLB EP32 静态：288 个物理专家，每 rank 9 个，额外 32 个全模型 routed expert 位置。
- EP16 K=1 首次槽传输约 570 ms；同槽后续跨层重装约 7.6-9.9 ms。首次包含 HCCL/格式准备成本，不能作为稳态成本。
- EP32 K=2 槽重装约 11-16 ms；不同层重复使用同一槽。
- 本轮没有完成统一性能重复；TTFT 是功能基线，不声称胜负。

## 8. 阻塞与解释

### 8.1 服务通信编号还未完整接入

当前服务实现把 owner 本 rank 上的热点任务通过“原路径零权重 + 外部槽回填”完成，因此：

- 真实槽、真实 W8A8、真实分流、真实回填均成立；
- 但没有把副本任务编码成额外 DeepEP 通信编号；
- 原路径仍承担这些 token 的零权重计算。

这是下一步最重要的实现缺口。上一轮 EP16 replay 已证明 272 编号、一个真实槽和任务分流可以正确运行，需要把该 dispatcher 路径搬入服务。

### 8.2 EP32 K=2 32K 挂起

失败证据：

- 32K 请求的超时等待约 679 s；
- 可见 Prefill chunk 已进入 layer49/50，槽任务和数值误差正常；
- 最终响应没有返回；
- detokenizer heartbeat 过期，HTTP health 失败。

需要下一步用 profiler/完整 rank 日志定位返回阶段卡点，并复测更长 context。

### 8.3 原生 EPLB 在线重排

EP32 在线重排在第一阶段即遇到内部格式 storage offset，随后没有进入可完成的权重安装阶段。静态布局服务可用，但静态数据不能冒充在线重排完成。

## 9. 复现

```bash
cd shared/current_exp/20261123_global_k_service_ep32

# 生成 AgentX-derived 三档输入
./scripts/make_agentx_inputs.py \
  --source /data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json \
  --outdir inputs

# EP16 K0/K1/K2 service
KCONFIG=$PWD/control/k1.json MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 ./scripts/serve_ep16.sh k1 k1_verify
python3 scripts/request_service.py --url http://127.0.0.1:18023 --prompt prompts/prompt_8k.txt --out artifacts/k1_verify_8k.json --max-tokens 2

KCONFIG=$PWD/control/k2_verify.json MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 ./scripts/serve_ep16.sh k2 k2_verify
python3 scripts/request_service.py --url http://127.0.0.1:18023 --prompt prompts/prompt_8k.txt --out artifacts/k2_verify_8k.json --max-tokens 2

# EP32 baseline / static EPLB / K2
./scripts/serve_ep32.sh start off ep32_off_1
for s in 8192 16384 32768; do python3 scripts/request_agentx.py --url http://127.0.0.1:18023 --input inputs/agentx_${s}.json --out artifacts/ep32_off_${s}.json --max-new-tokens 2; done
./scripts/serve_ep32.sh stop

LAYOUT=$PWD/layouts/eplb32_static.json ./scripts/serve_ep32.sh start eplb_static ep32_eplb_static_1
for s in 8192 16384 32768; do python3 scripts/request_agentx.py --url http://127.0.0.1:18023 --input inputs/agentx_${s}.json --out artifacts/ep32_eplb_static_${s}.json --max-new-tokens 2; done
./scripts/serve_ep32.sh stop

KCONFIG=$PWD/control/k2_ep32.json ./scripts/serve_ep32.sh start k2 ep32_k2_1
for s in 8192 16384 32768; do python3 scripts/request_agentx.py --url http://127.0.0.1:18023 --input inputs/agentx_${s}.json --out artifacts/ep32_k2_${s}.json --max-new-tokens 2; done
./scripts/serve_ep32.sh stop
```

单次启动前均使用 `serve_*.sh stop`；上一轮服务、EP32 两节点容器均已清理。

## 10. 关键产物

- `artifacts/EP16_SERVICE_ACCEPTANCE.json`
- `artifacts/EP32_DEVICE_MAP.json`
- `artifacts/EP32_MATRIX.json`
- `artifacts/EP32_K2_ACCEPTANCE.json`
- `artifacts/EP32_EPLB_REPLICA_TASKS.json`
- `artifacts/EP32_SERVICE_EVIDENCE.json`
- `artifacts/EP32_K2_TASKS.json`
- `artifacts/EP32_OFF_LOG_EXTRACT.json`
- `artifacts/k0_8k.json`、`k1_verify_8k.json`、`k2_verify_8k.json`
- `artifacts/ep32_{off,eplb_static,k2}_{8192,16384,32768}.json`
- `scripts/k_service.py`、`scripts/serve_ep16.sh`、`scripts/serve_ep32.sh`
- `control/*.json`、`layouts/eplb32_static.json`、`inputs/agentx_*.meta.json`
