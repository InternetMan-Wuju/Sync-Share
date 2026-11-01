# Q4Exp 数据索引（给分析用）

**实验**：任务 3+4 联合实验。4 组配置 × 2 种负载 = 8 轮，每轮 15 分钟、并发 8、无错误。
**跑批脚本**：`run_all_q4.sh`（序列）/ `run_q4exp.sh`（单轮）；**分析脚本**：`analyze_q4exp.py`

## 1. 四组配置

| 组 | EPLB | 采集 | 冗余 | 说明 |
|---|---|---|---|---|
| A0 | 关 | **关** | 0 | 纯性能基线（不受采集影响） |
| A | 关 | 开 | 0 | 热力基线；布局 = identity |
| B | 开 | 开 | 0 | 负优化主证据 |
| C | 开 | 开 | **64** | 冗余专家（每卡 4→5 槽，物理槽 256→320） |

## 2. 有效的 run 目录（8 轮全部完成，0 错误）

### rank 负载 max/mean（逐周期）

| 组 | agentX（16K/512） | random（8K/64） |
|---|---|---|
| A0（无采集） | — | — |
| A | **1.27~1.38×**（14 窗口全稳） | **1.43~1.45×**（13 窗口全稳） |
| B | 1.33×（重排前）→ **5.63 / 5.68 / 5.63 / 5.69×** | 1.44×（重排前）→ **6.11 / 6.11×** |
| C（冗余64） | **1.26~1.38×** | **1.30~1.37×** |

> **B 组在两种负载下都出现负优化**（agentX 5.6× / random 6.1×）→ 说明热点集中在**与请求内容无关**（是模型/路由本身的性质）。

### 性能（每组 15 分钟，0 错误）

| 组 | 请求数 agentX/random | 吞吐 agentX | 吞吐 random | TTFT p50 agentX | TPOT p50 |
|---|---|---|---|---|---|
| A0 | 168 / 624 | **91 tok/s** | **44 tok/s** | 20.3s | 50ms |
| A | 160 / 608 | 90 | 43 | 20.4s | 51ms |
| B | 160 / 584 | 87 | 41 | 20.3s | 50ms |
| C | 160 / 608 | 88 | 43 | 17.0s | 51ms |

- **采集代价**：A0 vs A ≈ **1~2%**（agentX 91→90，random 44→43）。
- **负优化的代价**：B 比 A0 低 **4%（agentX）/ 7%（random）** 吞吐。

### 目录

| run | 说明 |
|---|---|
| `A0_agentx_20261012_044134` / `A0_random_20261012_060410` | 无窗口（不采集） |
| `A_agentx_20261012_050236` / `A_random_20261012_062417` | 14 / 13 个 60s 窗口 |
| `B_agentx_20261012_052314` / `B_random_20261012_064400` | 6 / 3 个**完整周期**窗口 |
| `C_agentx_20261012_054412` / `C_random_20261012_071801` | 6 / 3 个完整周期窗口（P=320, 每卡 5 槽） |

> `INVALID_*` 前缀目录 = 早期调试中止的轮次，分析时忽略。
> 另：`C_random` 第一次启动失败（瞬态，服务 10 分钟没起来），重跑成功 —— 日志见 `results/C_random_rerun.log`。

## 3. 每组目录里有什么

```
<RUN>/
  windows/*.pt        热度窗口（详见第 4 节）
  metrics/*.txt        每 60 秒抓一次的服务 /metrics 原文（含 TTFT/TPOT/E2E/queue/retract…）
  npu/<时间>_npu<N>.txt 每 60 秒抓一次的四台机器 npu-smi
  bench/summary.json   TTFT/TPOT/E2E 的 p50/p95、吞吐、请求数
  bench/requests.jsonl 每个请求的明细（含 ttft/tpot/e2e/cached_tokens）
  bench/sse/*.jsonl.gz 每个请求的流式原始事件（带时间戳，可画"卡顿带"）
  logs_node_{19,20,23,25}.log  四个节点的服务日志（重排事件在里面）
  rebalance_events.txt 重排次数统计
  orchestrator.log     跑批过程（含每次 dump 的耗时）
```

## 4. 窗口（windows/*.pt）字段 —— 分析的核心

| 字段 | 形状 | 含义 |
|---|---|---|
| `num_layers` | 78 | **前 3 层 dense，无 MoE** → 矩阵里那 3 行恒 0，只有 75 层有数 |
| `num_logical_experts` | 256 | 逻辑专家数（**不随冗余变**） |
| `num_physical_experts` | 256 / **320**(C组) | 物理槽数 |
| `ep_size` | 64 | rank 数；**rank = 槽号 // (物理槽数/ep_size)**（A/B 组每卡 4 槽，C 组 5 槽） |
| `logical_count_prefill` | `[步,78,256]` int32 | prefill 阶段的**逻辑专家**计数 |
| `logical_count_decode` | `[步,78,256]` int32 | decode 阶段同上 |
| `physical_to_logical_map` | `[78,物理槽数]` int64 | 本窗口生效的映射（**正向**）。反向表可由它求逆得到 |

**三条用法**
1. 热度矩阵 = 对"步"维求和：`H = o["logical_count_decode"].float().sum(0).numpy()` → `[78,256]`
2. rank 负载：`rank_load[r] = Σ_l Σ_{p∈rank r} H[l][pmap[l][p]]`，指标 `max/mean`
3. **PD 分开**：prefill / decode 各算一份，再算合计（这是明确要求）

**两个坑**
- `prefill` 与 `decode` 的量级差 3 个数量级（prefill 一步 = 16384×块数 个 token；decode 一步 ≈ 几十个）→ **绝不能按"步"平均**，要按 token/计数加权。
- 计数单位 = token 数 × 75 层 × top-8（例：`29491200 = 3×16384×75×8`）。

## 5. 周期怎么切

- **窗口 `kind`**：`manual_final`（定时 dump，A 组每 60s 一个）/ `eplb_rebalance`（B、C 组在**重排时自动 dump**）。
- 因为 `buffer-size=2000` 且重排间隔也是 2000 步 → **B/C 每个 `eplb_rebalance` 窗口 = 一个完整周期**（159MB/个），不用拼接。
- A 组没有重排 → 14 个 60s 窗口；做"周期"统计时把相邻窗口加起来即可（= 灵活增量统计）。
- 窗口自带 `window_start_timestamp / window_end_timestamp / dump_index`，可与 `/metrics` 和重排事件按时间对齐。

## 6. 已知的数字（供交叉验证）

- **B 组**：第 1 个窗口（重排前）1.33× → 之后每个周期 5.63/5.68/5.63/5.69×；
  PD 分开：prefill **6.2×** / decode **4.0~4.2×**。
- **C 组**：1.26~1.38×（320 槽、每卡 5 槽）；PD：prefill 1.33~1.46× / decode 1.55~1.74×。
- **A 组**：1.27~1.38×（14 窗口全稳）；PD：prefill 1.45~1.50× / decode 1.60~1.61×。
- **重排耗时**（server.log）：B = 1.6s 和 48.7s 两种；C = 2.4s / 11.5s / 47.0s。
- **采集代价**：A0（不采集）91 tok/s vs A（采集）90 tok/s ≈ 1%。
