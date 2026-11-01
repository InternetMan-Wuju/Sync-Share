# Q4Exp 图与表（分析产出索引）

> 由 `Q4Exp/q4exp_batch.py`（八轮总分析）和 `Q4Exp/analyze_q4exp.py`（单轮明细）生成，纯 CPU。
> 重新生成：`cd current_exp && python3 Q4Exp/q4exp_batch.py`

## 口径（看表前必须知道）

1. `rank = 物理槽号 // (物理槽数 / ep_size)`；A/B 组每卡 4 槽，C 组每卡 5 槽。
2. **decode 的计数单位是 prefill 的 8 倍**（实测）→ 任何把两相相加的地方都用 `total = prefill + decode/8`（表/图里标注 corrected）。
3. 周期 = 两次重排之间；B/C 的 `eplb_rebalance` 窗口就是一个完整周期。
4. 78 层里前 3 层是 dense（无 MoE），矩阵那 3 行恒 0，只有 75 层有数。

## 表（`tables/`）

| 文件 | 内容 |
|---|---|
| `perf.csv` | 8 轮性能：吞吐 / TTFT / TPOT / E2E 的 p50、p95、请求数、重排次数与每次耗时 |
| `latency_breakdown.csv` | 每请求平均的服务端分段：收到→进forward / queue_time / prefill 前向 / 首token发送 / decode，以及两相速度 |
| `rank_imbalance_by_window.csv` | **逐窗口** rank 不均衡（prefill / decode / 合计-corrected）+ PD top-8 重合 + 窗口类型 |
| `pooled_summary.csv` | **汇总**：三相不均衡、最热专家落在 rank0 的层比例、PD 重合、50% 集中度、系统级 prefill/decode 速度、槽位配置 |
| `window_stability.csv` | 相邻窗口的逐层形状相关 / top-8 重合（周期稳定性） |

## 图（`figs/`）

| 文件 | 内容 |
|---|---|
| `fig1_perf.png` | 吞吐 / TTFT p50 / TPOT p50（4 组 × 2 负载，斜纹=random） |
| `fig2_rank_imbalance_time.png` | **逐窗口 rank 不均衡曲线**（方块=合计corrected，圆圈=prefill 单看），agentX / random 两个面板 |
| `fig3_rank_load_curve.png` | **每 rank 负载曲线**（归一化）：A（无 EPLB）/ B（EPLB）/ C（EPLB+冗余64） |
| `fig4_layer_expert.png` | **Layer×Expert 热力图**：A(汇聚) / B(重排前) / B(重排后) / C(汇聚) |
| `fig5_layer_rank.png` | **Layer×Rank 热力图**：A / B / C（agentX） |
| `fig6_pd_split.png` | PD 分开：rank 不均衡（prefill/decode/合计）、两相 top-8 重合（对随机基准 0.25）、集中度 |
| `fig7_latency_breakdown.png` | E2E 分段堆叠（收到→forward / prefill / 首token / decode）× 8 轮 |
| `fig8_stability.png` | 周期稳定性：相邻窗口逐层相关、top-8 重合 |
| `fig9_story.png` | **结论图**：尖峰对齐比例（最热专家落 rank0 的层占比）/ 合计不均衡 / 重排耗时 |

## 单轮明细（`runs/<RUN>/`）

6 个带窗口的 run 各有 `figs/`（Layer×Expert、Layer×Rank、rank×窗口、PD 对比、合计口径对比）与 `tables/`（window_summary / rank_load / layer_stats / top_experts）。用于下钻某一轮。

## 关键数字（从表里直接抄）

| run | 吞吐 tok/s | TTFT p50 | TPOT p50 | rank 不均衡 prefill | 合计(corrected) | 最热专家落 rank0 的层占比 | PD top-8 重合 |
|---|---|---|---|---|---|---|---|
| A0_agentX | 90.9 | 20.32s | 50.3ms | — | — | — | — |
| A0_random | 44.0 | 7.07s | 50.2ms | — | — | — | — |
| A_agentX | 89.6 | 20.39s | 50.8ms | 1.48× | 1.45× | 0% | 15% |
| A_random | 43.1 | 7.77s | 49.6ms | 1.49× | 1.48× | 0% | 15% |
| B_agentX | 86.5 | 20.31s | 50.0ms | 6.22× | 6.14× | 84% | 15% |
| B_random | 41.3 | 7.17s | 49.8ms | 6.31× | 6.28× | 92% | 15% |
| C_agentX | 88.1 | 17.0s | 51.4ms | 1.44× | 1.42× | 15% | 15% |
| C_random | 43.1 | 5.72s | 51.2ms | 1.34× | 1.34× | 16% | 15% |

**一句话结论**：A（不开 EPLB）合计不均衡 1.45×，最热专家散布（落 rank0 的层 = 0%）；
B（开 EPLB）→ 6.1×，**84~92% 的层最热专家被搬到 rank0**（= 尖峰对齐）；
C（EPLB+冗余 64）→ 1.42×，回到基线水平。两种负载行为一致。

**代价**：采集代价 A0 vs A ≈ 1~2% 吞吐；EPLB 负优化代价 B vs A0 ≈ 4%（agentX）/ 7%（random）；
重排本身要停 1.5~48s（B: 1.67s & 48.7s；C: 2.5s / 11.6s / 47.1s）。