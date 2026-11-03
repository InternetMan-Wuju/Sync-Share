# 75 层 Trace-Driven MoE 复杂模拟

日期：2026-11-03
性质：CPU 离散模拟；路由来自真实 decode/prefill dump，局部时间曲线来自真实 NPU 实测；网络和 dispatch 是参数扫描，不是实测。

## 1. 输入与方法

- 真实 dump：`20261102_081322/eplb_windows/000000_1793579172.815737.pt`
- 75 个 MoE layer：3–77
- prefill：4 个真实 step，每层 32768 token-expert tasks
- decode：64 个真实 step，每层 128 tasks
- EP：16 / 32 / 64
- expert 映射：连续 logical expert 分配到 EP rank
- 局部时间模型：
  - 单专家曲线来自单 NPU 实测 S2；
  - `rank_time = single_time(total_rows) + 0.020 ms × (active_experts−1)`；
  - 0.020 ms 是 S2 grouped 多专家点拟合的附加专家开销。
- online 策略：
  - 每层从当前最热专家中选择候选；
  - 目标 rank 最多一个临时槽；
  - 每层最多 K=1/2/4/8/16 个副本；
  - 整数迁移量按 owner/target 局部最大时间最小化；
  - 每层模拟 75 次 layer 级复制。

## 2. EP 扩大的计算空间

结果是对 75 层逐层 `max_r(rank_local_time)` 求和后的 MoE-only 计算代理，不是 TTFT/TPOT。

| phase | EP16 | EP32 | EP64 | EP32 节省 | EP64 节省 |
|---|---:|---:|---:|---:|---:|
| prefill | 48.67 ms/step | 34.02 ms/step | 27.20 ms/step | 14.65 ms | **21.47 ms** |
| decode | 19.99 ms/step | 18.31 ms/step | 17.67 ms/step | 1.68 ms | **2.32 ms** |

这些是额外跨 rank 通信/调度能消耗的**上限式计算预算**，不是已实现收益。

## 3. Online 临时复制

EP16，当前计数→top 热专家→临时副本→整数配额。

### prefill

| K/层 | 计算节省/step | 副本数/step | 迁移 tasks/step | 远端 copy 回本线（dispatch=0） | dispatch=0.1µs/task |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.01 ms | 75 | 39,119 | **53.5 µs/expert** | 负，且只有 50% step 为正 |
| 2 | 4.98 ms | 150 | 65,541 | 33.2 µs/expert | 负 |
| 4 | 5.80 ms | 300 | 97,159 | 19.3 µs/expert | 负 |
| 8 | 7.50 ms | 600 | 122,464 | 12.5 µs/expert | 负 |
| 16 | 11.73 ms | 1127 | 136,698 | 10.3 µs/expert | 负 |

### decode

| K/层 | 计算节省/step | 副本数/step | 迁移 tasks/step | 远端 copy 回本线（dispatch=0） | dispatch=0.1µs/task |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.299 ms | 75 | 846 | **4.0 µs/expert** | 2.85 µs，仍为正 |
| 2 | 0.638 ms | 150 | 1465 | 4.25 µs/expert | 3.31 µs |
| 4 | 1.067 ms | 299 | 2176 | 3.57 µs/expert | 2.83 µs |
| 8 | 1.221 ms | 565 | 2723 | 2.17 µs/expert | 1.70 µs |
| 16 | 1.321 ms | 719 | 2887 | 1.85 µs/expert | 1.46 µs |

## 4. 解释

1. prefill 的 online 计算空间比 decode 大一个数量级，但迁移任务也多一个数量级以上。
2. prefill K=1 在零 dispatch 成本时，只有远端复制低于约 **53.5 µs/专家**才能回本；dispatch 达到 0.1 µs/task 后，中位数已经为负。
3. decode 的远端复制回本线只有约 **1.5–4.3 µs/专家**，远低于本地已打包状态复制实测 75 µs。
4. 增加 K 能继续降低计算时间，但复制和迁移任务同步增长，净收益不一定增加。
5. 现有 Ascend EPLB 历史实机路径还出现过 NZ 格式 P2P 崩溃、isend/irecv 死锁；修复后首次 rebalance 约 45–48 s、增量约 1.5 s，说明真实迁移路径远未达到这里的微秒级预算。

## 5. 边界

- 路由是真实的，但 hidden 和执行路径不是完整生产 graph。
- 局部时间模型是 NPU 实测曲线拟合，不是 75 层实测。
- 网络/远端拷贝和 dispatch 参数是扫描值，不是现场标定。
- prefill/decode 的 MoE-only 时间不能当 TTFT/TPOT，也不能乘成服务结论。
- 没有模拟真实源 rank、路由权重、HCCL 拓扑、KV、attention 和请求排队。

## 6. 结论

复杂模拟支持一个比 toy case 更可靠的判断：

> Online 复制的计算空间在 prefill 上明显存在，但只有在“每专家远端复制几十微秒以内、dispatch 近零”的非常苛刻条件下才可能回本；decode 的回本线只有几微秒，现实系统几乎不可能满足。

因此应用落地的主要难点不是 planner，而是**专家状态跨卡迁移、临时槽 ready 语义、dispatch/combine 增量和生产 graph 集成**。
