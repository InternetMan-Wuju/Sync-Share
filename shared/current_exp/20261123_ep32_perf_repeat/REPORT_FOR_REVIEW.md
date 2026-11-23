# EP32 三组重复性能对比

## 结论先说

本轮基于 `bd80f52`，在 node23 + node25 的全部 32 个 die 上组成一个 EP32 组，完成了原版、在线 EPLB +32 和我方 K=2 真改道的重复测量。

- 每个长度、每组完成 **10 次正式请求**。
- 三轮启动顺序轮换：
  - round 1：original → eplb → k2
  - round 2：eplb → k2 → original
  - round 3：k2 → original → eplb
- 每个长度、每组、每轮先跑 1 次独立 warmup；warmup 共 27 条，全部单独记录，不进入正式统计。
- 正式计时关闭 EPLB observer；K=2 使用 `verify=false` 和 quiet 日志，关闭逐任务记录、张量哈希和额外原 MoE 参考计算。
- 本轮结果是**单机双节点 EP32 服务请求测量**，不是生产 p95；所有 formal 样本的中位数、范围和逐请求值均已保存。

## 1. 环境与统一条件

| 项 | 值 |
|---|---|
| 基线 | `bd80f52` |
| 节点 | `npu1-23`、`npu1-25` |
| 拓扑 | 32 die，单 EP32 组；rank 0-15 / 16-31 分别对应 node23 / node25 |
| 镜像 | `quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3` |
| 模型 | `/data_lib/data/models/glm-q`，ModelSlim W8A8 |
| chunked prefill | 8192 |
| max prefill tokens | 32768 |
| context length | 65536 |
| 并发 | 1 |
| cache | radix cache disabled |
| 输出长度 | `max_new_tokens=2` |
| 输入 | 同一批 AgentX-derived 8K/16K/32K token IDs |

输入指纹：

| 输入 | SHA256 |
|---|---|
| 8K | `86240a227b4f194e7f0e3ab5661942b09c0aa5a44cdcc610f503f54d9e3f3d84` |
| 16K | `ef1b6259ebc4a272df50b9e905887909e9acab82924231e72cff73c981b217c9` |
| 32K | `cd222160c6815f5ce8db984894f40d40880333dfed62139a23b4ffa0464fafe1` |

32K 由同一 trace 的两个 16K 片段确定性拼接；输入不是原始人类文本或原始模型 token ID。

## 2. 三组配置

### 原版

- SGLang 原生路径
- EPLB 关闭
- 冗余专家为 0
- 无 observer、无额外验证计算

### 在线 EPLB +32

- `--enable-eplb --ep-num-redundant-experts 32`
- 使用 `bd80f52` 已验证的 v0.5.20 NPU 兼容补丁：
  - `expert_location_updater_v0520.py`
  - `torch_npu_distributed_c10d_v0520.py`
- 每轮启动后先发送独立 12K 校准请求，等待新的 `rebalance end`，再进入 warmup/formal。
- 正式运行保持在线 EPLB 开启；正式计时不挂 observer。
- 当前三轮都在正式请求期间继续发生重排，因此日志中的每个正式 EPLB 请求都标记为与至少一次重排重叠。

### 我方 K=2 真改道

- layer49/50 使用同一轮 DeepEP NORMAL dispatch 和对应 combine。
- 主专家和副本都走真实 W8A8 专家链；不另开副本专用 dispatcher，不把主路径权重置零。
- 正式计时使用 quiet 配置：
  - `verify=false`
  - 不记录逐任务样本
  - 不计算 dispatch/output tensor hash
  - 不执行额外原 MoE 参考
- 另起一次独立的 verbose 验证服务，发送一条 8K 请求，保留原专家少算、副本接手和数值误差证据；该验证不进入性能中位数。

## 3. 性能表

TTFT 为每个 `(arm, size)` 的 10 个正式请求中位数；括号内为范围。

| 输入 | 原版 TTFT | EPLB +32 TTFT（全部含重排） | K=2 TTFT | EPLB 相对原版 | K=2 相对原版 |
|---|---:|---:|---:|---:|---:|
| 8K | **2.8025 s** (2.7981–2.8104) | **3.1255 s** (3.1114–3.1790) | **2.9358 s** (2.9266–2.9574) | **+11.57%** | **+4.72%** |
| 16K | **5.7147 s** (5.7009–5.7280) | **12.4076 s** (9.3569–12.4803) | **5.9785 s** (5.9510–6.1995) | **+117.33%** | **+4.53%** |
| 32K | **11.7875 s** (11.7633–11.8065) | **22.4916 s** (19.5302–22.8064) | **12.3013 s** (12.2448–12.3403) | **+90.81%** | **+4.42%** |

端到端总耗时中位数：

| 输入 | 原版 | EPLB +32 | K=2 | EPLB 相对原版 | K=2 相对原版 |
|---|---:|---:|---:|---:|---:|
| 8K | **3.3236 s** | **3.7060 s** | **3.4774 s** | **+11.43%** | **+4.60%** |
| 16K | **6.2401 s** | **13.1001 s** | **6.4972 s** | **+109.79%** | **+4.07%** |
| 32K | **12.3165 s** | **23.3233 s** | **12.8586 s** | **+89.48%** | **+4.40%** |

相对值是用同一轮、同一 repeat、同一输入配对后的中位百分比差，不是直接拿两个中位数相减。每个 cell 都有 10 个配对样本。

紧凑表也保存为 `artifacts/PERFORMANCE_TABLE.md`。

## 4. EPLB 重排与样本标记

### 校准确认

三轮 EPLB 服务都在正式请求前完成了实际重排：

| round | TP0 rank 观察到 rebalance end | layout snapshots | layout changed |
|---|---:|---:|---|
| 1 | 30 | 30 | 是 |
| 2 | 25 | 25 | 是 |
| 3 | 25 | 25 | 是 |

三次校准请求均返回，输出 ID 与原版一致。完整时间线在 `artifacts/PERFORMANCE_SUMMARY.json` 的 `rebalance_timeline`，布局快照和补丁哈希在 `artifacts/EPLB_CALIBRATION_EVIDENCE.json`。

### 正式请求重叠

正式 EPLB 样本与重排的重叠计数：

| 输入 | 正式样本数 | 与重排重叠 | 不重叠 |
|---|---:|---:|---:|
| 8K | 10 | 10 | 0 |
| 16K | 10 | 10 | 0 |
| 32K | 10 | 10 | 0 |

所以上表 EPLB 的中位数和范围就是**当前在线 EPLB 配置包含重排成本的实际服务时间**，不是去掉重排后的稳态时间。16K/32K 的双峰范围也反映了部分请求正好跨过 4–6 s 的重排窗口。

## 5. 正确性与已知差异

正式输出：

- 原版：三档均稳定返回。
- EPLB +32：三档 10/10 返回，output IDs 与原版完全一致。
- K=2：
  - 16K/32K 的 output IDs 与原版 10/10 一致。
  - 8K 的 output IDs 为 `[198, 198, 154842]`，原版为 `[13, 13, 154842]`；这是已知的 8K 差异，不声明 token-exact。

独立 verbose K=2 验证（不属于正式计时）：

- 8K，layer49：
  - hot expert163：逻辑 383 个任务，改道 181 个，原专家实际计算 202 个；
  - hot expert133：逻辑 6 个任务，改道 2 个，原专家实际计算 4 个；
  - owner0/owner1 接收分别为 181/2 个；
  - 原专家计算数等于“逻辑任务数减改道数”；
  - 最大绝对误差 `0.0078125`，全部 finite。
- layer50：
  - 改道 534 个，owner0/1 接收 28/506 个；
  - `task_conservation=true`。

证据：`artifacts/VALIDATION_K2_NATIVE_TASK_ACCOUNTING.json` 和 `artifacts/raw/validation/k2_native_8k.json`。

## 6. 原始数据与失败记录

- `artifacts/raw/formal/`：90 个正式请求 JSON，每格 10 个。
- `artifacts/raw/warmup/`：27 个 warmup JSON，每 arm × size × round 一条。
- `artifacts/raw/calibration/`：三轮 EPLB 校准请求。
- `artifacts/PERFORMANCE_RAW_REQUESTS.csv`：逐请求 TTFT、端到端耗时、输出 token 数、输出 SHA、重排重叠标记、错误字段。
- `artifacts/PERFORMANCE_SUMMARY.json`：每组统计、配对差值、warmup 统计、重排时间线。
- `artifacts/EPLB_CALIBRATION_EVIDENCE.json`：三轮 `rebalance end`、layout 快照和补丁哈希。

本轮 formal 请求失败数为 0；三轮 EPLB 校准均成功。没有把 warmup 样本混入正式统计。

## 7. 解释与边界

- 本轮只回答“在这套 EP32 服务配置下，这三次实现路径各有多快”。不能外推到其它模型、拓扑、并发或生产 p95。
- EPLB 的在线重排不是免费操作；本轮所有 EPLB formal 样本都至少与一次重排重叠，因此 EPLB 的差值就是该配置下的实际样本差值，而不是稳态计算上限。
- K=2 formal 的额外开销在当前轮中约为原版的 **+4.4%～+4.7% TTFT**；这与上一轮单次结果不同，不能直接比较。
- 8K K=2 的 token 差异仍然存在；16K/32K 本轮 10/10 token IDs 一致。
- 生产格式仍为 `UNKNOWN_NO_ACTIVE_SERVICE`。
