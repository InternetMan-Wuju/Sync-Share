# ChunkSize 8K / 1K：Prefill、延迟与速度图表

整理日期：**2026-09-28（北京时间）**。本页使用 **2026-09-28 的恢复权重后复测**，从八条原始 SSE 重新计算；两档分别为 `--chunked-prefill-size=8192 / 1024`。本页不混入 9 月 26 日的 54.6K 历史负载。

**本轮结果：8K 的 Prefill 更快，平均首 token 等待更短，插入长 Prefill 时的输出停顿也更短。** 以下均为这次四请求小样本的观察。先给 TTFT/TPOT 表，再给四张图，重点放大首 token 前的 Prefill 阶段。

## 1. 数据和统计口径

| 条件 | 本轮设置 |
|---|---|
| 输入 / 输出 | 每条 **32,768 总输入 / 300 输出**；A/B/C/D 四条输入，两档复用相同输入 |
| 缓存 | 每条实际命中 **19,584 / 32,768 = 59.765625%**；实际新算 **13,184 token** |
| 环境 | 节点 23，单机 TP16/EP16，GLM-5.2 W8A8，固定镜像 `fe694986`；MTP OFF |
| 调度 | `enable_mixed_chunk=False`、`prefill_decode_interval=0`；Graph 实际关闭；服务端最大运行请求 4 |
| 发请求规则 | A 输出至少 20 token 后发 B；B 至少 20 token 后发 C；A 完成且 C 至少 20 token、仍在 Decode 时发 D；实际活跃上限 3 |
| 请求级样本数 | **4 条/Chunk**，先测 8K，再测 1K；不是随机顺序多轮重复 |
| ITL 样本数 | 每条 300 次新增输出事件，得到 299 个间隔；每档 **1,196 个间隔，来自 4 条请求** |
| 成功条件 | 八条均 HTTP 200、输出 300、`num_retractions=0`；本次已从原始流核对 |

原始实验说明：[复测报告](../chunk_redo_20260928/results/20260928_043035/README.md)；实际参数：[8K](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/before_bench/server_info.json)、[1K](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/before_bench/server_info.json)。

## 2. TTFT / TPOT：均值、P95、P99

- **TTFT**：客户端开始请求到首次收到输出。
- **TPOT**：`(客户端 E2E − TTFT) / (300 − 1)`；本轮含首 token 后的生成、等待、输出传输及结束收尾。
- **P95/P99**：采用最近秩法，排序后取第 `ceil(n × p)` 个值；每档 `n=4`，所以 **P95=P99=该档最大值**。保留这些统计满足查看需求，但它们不能估计稳定的业务尾延迟。

| 指标 | Chunk 8K n | 8K 均值 | 8K P95 | 8K P99 | Chunk 1K n | 1K 均值 | 1K P95 | 1K P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTFT（s） | 4 | 5.976 | 6.096 | 6.096 | 4 | 10.110 | 10.265 | 10.265 |
| TPOT（ms/token） | 4 | 492.40 | 505.98 | 505.98 | 4 | 518.35 | 544.26 | 544.26 |

8K 相对 1K：平均 TTFT **降低 40.89%**；平均 TPOT **降低 5.01%**。TPOT 变化来自这次有 Prefill 注入的全过程，不能单独解释为 Decode 内核更快。

## 3. E2E（s）及四段分解

### 3.1 Queue 有记录，IPC 尚无独立测量

**建议用“服务端 Queue → 服务端 Prefill → 首 token 前未归因 → 首 token 后至结束”四段。** Queue 并非完全猜测：最终 SSE 的 `meta_info.queue_time` 给出了该字段。但它只是服务端报告的排队口径，不代表请求在所有环节的排队都已被独立测量。**IPC 没有独立起止时间，不能把剩余时间直接命名为 IPC。**

| 图中阶段 | 计算方式 | 可以说明什么 |
|---|---|---|
| Queue：服务端报告 | `meta_info.queue_time` | 框架记录的排队用时，不是推测值；不是整个请求链路的所有排队 |
| Prefill：服务端阶段 | `prefill_finished_time − forward_entry_time` | 从进入 forward 至 Prefill 结束的阶段区间，未拆出纯算子时间 |
| 首 token 前未归因 | `TTFT − Queue − Prefill` | 可能包含请求收发、分派、交付及计时边界差；**不等于实测 IPC** |
| 首 token 后至结束 | `E2E − TTFT` | 主要是后续生成，也含被其他 Prefill 阻塞、传输、调度与收尾；**不等于纯 NPU Decode** |

四段按定义严格相加为客户端 E2E。这是**账面分解**：Queue/Prefill 来自服务端时间，TTFT/E2E 来自客户端单调时钟；各段不是已证明互不重叠的硬件时间线。图中顺序用于阅读，不证明未归因开销都发生在 Prefill 之后。

### 3.2 完整 E2E、TTFT 与小开销分别放大

![两种 Chunk 的 E2E 四段分解及 Prefill 放大](../chunk_redo_20260928/analysis/prefill_comparison_20260928/01_e2e_breakdown.png)

图 A 为完整 E2E；图 B 放大首 token 前区间。Queue 很短，在整体图中几乎看不见，因此图 C/D 分别显示 Queue 和未归因时间，单位改用 ms，且两图横轴量程不同；未人为拉长堆叠中的短阶段。

| Chunk | Queue 均值（ms） | Prefill 均值（s） | 首 token 前未归因均值（ms） | TTFT 均值（s） | 首 token 后均值（s） | E2E 均值（s） |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 0.654 | 5.646 | 329.198 | 5.976 | 147.229 | 153.205 |
| 1K | 2.295 | 9.758 | 349.302 | 10.110 | 154.988 | 165.098 |

主要差距在 Prefill：1K **9.758 s**，8K **5.646 s**，8K 减少 **4.112 s（42.14%）**。TTFT 的均值差为 **4.134 s**，与 Prefill 差额接近。Queue 是毫秒级，无法解释秒级差距。

E2E 则是 **165.098 → 153.205 s**，8K 降低 **7.20%**。首 token 后仍占大部分时间，所以 Prefill 改善比例不能直接当整请求的改善比例。

### 3.3 八条请求的阶段用时

| Chunk | 请求 | Queue（ms） | Prefill（s） | 未归因（ms） | 首 token 后（s） | E2E（s） |
|---|---|---:|---:|---:|---:|---:|
| 8K | A | 0.641 | 5.640 | 11.947 | 151.287 | 156.939 |
| 8K | B | 0.593 | 5.649 | 446.801 | 150.892 | 156.989 |
| 8K | C | 0.723 | 5.637 | 447.044 | 146.021 | 152.106 |
| 8K | D | 0.658 | 5.659 | 411.001 | 140.714 | 146.785 |
| 1K | A | 5.580 | 10.085 | 29.911 | 162.734 | 172.854 |
| 1K | B | 1.205 | 9.803 | 460.384 | 162.352 | 172.617 |
| 1K | C | 1.140 | 9.621 | 480.940 | 152.815 | 162.918 |
| 1K | D | 1.255 | 9.525 | 425.974 | 142.050 | 152.002 |

## 4. ITL 条形图：常见间隔和稀有停顿都要看

![两种 Chunk 的 ITL 分位、逐请求最大间隔及注入窗口最大间隔](../chunk_redo_20260928/analysis/prefill_comparison_20260928/02_itl_bars.png)

图 A 将每档四条请求的所有 **1,196 个 ITL** 汇合后计算；每请求输出长度相同，各有 299 个间隔。图 B 保留每条请求的最长停顿；图 C 对应三次 Prefill 注入。**图 A 使用 ms，图 B/C 使用 s；每个面板内两档使用相同尺度。**

| Chunk | 请求数 / ITL 数 | ITL 均值（ms） | P50（ms） | P95（ms） | P99（ms） | 全组最大 ITL（s） |
|---|---|---:|---:|---:|---:|---:|
| 8K | 4 / 1196 | 492.40 | 465.00 | 516.80 | 582.51 | 3.170 |
| 1K | 4 / 1196 | 518.35 | 472.27 | 532.02 | 593.61 | 9.620 |

ITL P95/P99 是**间隔级**分位数，不能视为 1,196 个独立请求的尾延迟。两档 P99 只有约 0.58–0.59 s，但 1K 单次最长停顿达到 **9.620 s**；说明稀有长停顿占比不到 1% 时，P99 仍可能看起来接近正常值。

| 正在 Decode / 后来 Prefill | 8K 注入前 P50（ms） | 1K 注入前 P50（ms） | 8K 窗口最大 ITL（s） | 1K 窗口最大 ITL（s） |
|---|---:|---:|---:|---:|
| A 遇到 B | 505.53 | 474.39 | 3.100 | 9.620 |
| B 遇到 C | 475.17 | 475.80 | 3.103 | 9.400 |
| C 遇到 D | 463.51 | 474.72 | 3.170 | 9.318 |

注入窗口从后来请求的发起时刻到其首 token 到达；凡与窗口相交的 ITL 均计入。窗口包含请求传输、排队、Prefill 与首 token 交付，不是纯 Prefill 时段；三次注入也不是三个独立重复实验。

**当前结果支持 8K：**注入前常见间隔约 0.46–0.51 s；注入时 8K 最大停顿约 3.1 s，1K 约 9.3–9.6 s。8K 仍有秒级停顿，不能理解成已经消除了 Prefill 干扰。

## 5. Prefill 速度柱状图

每请求速度采用 **未命中输入 token / 服务端 Prefill 阶段时间**：

```text
Prefill speed = (32768 − 19584) / (prefill_finished_time − forward_entry_time)
              = 13184 / Prefill秒数
```

![两种 Chunk 的逐请求和均值 Prefill 处理速度](../chunk_redo_20260928/analysis/prefill_comparison_20260928/03_prefill_speed_bars.png)

| 请求 | 8K Prefill 速度（token/s） | 1K Prefill 速度（token/s） | 8K / 1K |
|---|---:|---:|---:|
| A | 2337.74 | 1307.35 | 1.788× |
| B | 2333.97 | 1344.85 | 1.735× |
| C | 2338.65 | 1370.32 | 1.707× |
| D | 2329.57 | 1384.21 | 1.683× |
| 均值 | 2334.98 | 1351.68 | 1.727× |

本图均值为 **8K 2334.98 token/s，1K 1351.68 token/s**；8K 为 1K 的 **1.727×**。各请求分子都为 13,184，所以差别来自 Prefill 阶段用时。

**平均方式：**先计算每条速度，再取四条的算术均值。它不是 `总未命中 token / Prefill 时间之和`。后者本轮为 8K **2334.97**、1K **1351.04 token/s**；此前综合报告使用过该口径，轻微数值差异来自平均方式，不是数据变化。所有这些值都是服务端阶段速率，不能称为纯算子速度或服务总吞吐。

## 6. Decode 速度柱状图

为与 TPOT 一致，使用**客户端首 token 后的平均输出速度**：

```text
Decode-observed speed = (300 − 1) / (E2E − TTFT)
                      = 1000 / TPOT(ms)
```

![两种 Chunk 的逐请求和均值客户端 Decode 输出速度](../chunk_redo_20260928/analysis/prefill_comparison_20260928/04_decode_speed_bars.png)

| 请求 | 8K 首 token 后速度（token/s） | 1K 首 token 后速度（token/s） | 8K / 1K |
|---|---:|---:|---:|
| A | 1.9764 | 1.8374 | 1.076× |
| B | 1.9815 | 1.8417 | 1.076× |
| C | 2.0476 | 1.9566 | 1.047× |
| D | 2.1249 | 2.1049 | 1.009× |
| 均值 | 2.0326 | 1.9351 | 1.050× |

均值为 **8K 2.033 token/s，1K 1.935 token/s**，8K 高约 **5.04%**。这也是先计算每请求速度再平均，不等于 `1000 / 全组平均TPOT`。

这段时间含后来 Prefill 造成的阻塞，结果更适合描述“这次混合负载的输出体验”。本轮没有配套的同输入 Decode-only 对照，不能将差额都归为 Decode 内核性能。A/C/D 两档实际输出 ID 不一致，只有 B 完全一致；输入与长度相同不等于生成内容相同。

## 7. 如何解释 Prefill 的差距

两档实际新计算量相同，为 13,184 token。仅按分块上限作工作量示意，8K 约需 2 块，1K 约需 13 块；这不是实际 forward 次数的独立跟踪。当前 `enable_mixed_chunk=False`、`prefill_decode_interval=0`，不能假设切成小块后，Decode 就一定在块间获得执行机会。

实测中，1K 的 Prefill 阶段更长，已有 Decode 遭遇的停顿也更长；这与该调度条件相符。但现有记录没有把 kernel、通信、调度和 IPC 分开计时，不能将差额精确归到其中一项。切换 mixed chunk、Graph、高并发或 PD 分离后应重新比较。

本轮的 TTFT/TPOT 请求级 P95/P99 仅供查看样本最大值；不要用它们承诺生产尾延迟。速度与分解图都保留八条单请求数据或列出明细，方便识别均值背后的样本差异。

## 8. 导出、原始数据与复算

四张图均有 PNG 与 SVG；另有四页 [合并 PDF](../chunk_redo_20260928/analysis/prefill_comparison_20260928/ChunkSize_8K_vs_1K_四图.pdf)。SVG 的文字以路径保存，便于在没有相同中文字体的机器上查看。

| 图 | PNG | SVG |
|---|---|---|
| E2E / Prefill 分解 | [PNG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/01_e2e_breakdown.png) | [SVG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/01_e2e_breakdown.svg) |
| ITL | [PNG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/02_itl_bars.png) | [SVG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/02_itl_bars.svg) |
| Prefill 速度 | [PNG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/03_prefill_speed_bars.png) | [SVG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/03_prefill_speed_bars.svg) |
| Decode 速度 | [PNG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/04_decode_speed_bars.png) | [SVG](../chunk_redo_20260928/analysis/prefill_comparison_20260928/04_decode_speed_bars.svg) |

派生数据：[逐请求阶段与速度 CSV](../chunk_redo_20260928/analysis/prefill_comparison_20260928/per_request_metrics.csv)、[TTFT/TPOT 分位表 CSV](../chunk_redo_20260928/analysis/prefill_comparison_20260928/latency_summary.csv)、[ITL 汇总 CSV](../chunk_redo_20260928/analysis/prefill_comparison_20260928/itl_summary.csv)、[全部 ITL CSV](../chunk_redo_20260928/analysis/prefill_comparison_20260928/itl_intervals.csv)、[注入窗口 CSV](../chunk_redo_20260928/analysis/prefill_comparison_20260928/interference.csv)、[统计与校验 JSON](../chunk_redo_20260928/analysis/prefill_comparison_20260928/summary.json)。

生成代码：[plot_report.py](../chunk_redo_20260928/analysis/prefill_comparison_20260928/plot_report.py)；输入文件摘要：[input_manifest.json](../chunk_redo_20260928/analysis/prefill_comparison_20260928/input_manifest.json)。脚本只读归档，重新生成本目录图表和本页，不调用在线服务。

```bash
python3 /home/chengyetai/workspace/current_exp/chunk_redo_20260928/analysis/prefill_comparison_20260928/plot_report.py
```

原始流：[8K 请求目录](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/bench/requests/)、[1K 请求目录](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/bench/requests/)。实际最终输出采用 `output_ids_final.json`，不使用早期错误拼接累计数组的 `output_ids.json`。

本次校验：八条请求各有 300 个逐 token 输出事件，2392 个相邻间隔；输入 SHA 两档逐对一致；最终 ID 与 SSE 末帧一致；四段之和等于 E2E；与存档请求指标和注入窗口最大 ITL 交叉核对通过。
