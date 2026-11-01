# ChunkSize 单轮实验：以新 Prefill 注入时刻观察 Decode 干扰

整理日期：**2026-09-28（北京时间）**。数据来自今天的 `20260928_043035`：**Chunk 8K、1K 各一轮四条，合计八条正式请求**。本页按用户确认的单轮数据分析，重点是新 Prefill 到来时，新请求自身与正在输出的旧请求分别发生什么。

**速度口径先澄清：图中的约 0.3 token/s 是含长停顿的短窗口交付速率，不是正常 Decode 速度。** 本轮注入前旧请求约 2.0–2.2 token/s；约 6.1 秒窗口只收到 2 token，才得到约 0.33 token/s。

**观察结论：本轮 8K 仍然更合适。** 三次注入中，新请求的 Prefill 都更短；旧请求在 8K 下通常出现两段约 3 秒的停顿，1K 下出现一段约 9.3–9.6 秒的长停顿。但两档在各自“注入→新首 token”窗口里，所观察的旧请求都只交付 **2 个输出 token**：8K 窗口约 6.1 秒，1K 约 10 秒。应同时看停顿形状、窗口长度和输出进度，不能只报一个最大 ITL。

## 1. 本轮怎么发生：三个事件不是三轮

| 注入事件 | 新请求 | 主要观察的旧请求 | 注入前已有 Decode | 触发条件 |
|---|---|---|---|---|
| E1：A 遇 B | B | A | A，共 1 条 | A 已输出至少 20 token，仍在 Decode |
| E2：B 遇 C | C | B | A、B，共 2 条 | B 已输出至少 20 token |
| E3：C 遇 D | D | C | B、C，共 2 条 | A 已完成，C 已输出至少 20 token 且仍在 Decode |

每条总输入 32,768，实际命中 19,584（59.765625%），实际新计算 **13,184 token**，输出 300。同一节点 23，TP16/EP16，MTP OFF，Graph 实际关闭，`enable_mixed_chunk=False`、`prefill_decode_interval=0`。两档复用同一输入，先跑 8K 后跑 1K。以上状态按归档配置核对。

计划与实际发起时刻：[8K](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/bench/dispatches.json)、[1K](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/bench/dispatches.json)；实际配置：[8K](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/before_bench/server_info.json)、[1K](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/before_bench/server_info.json)。

三个事件的背景 Decode 数不同，并且前一个事件会影响后续时间和状态；**比较应以相同事件位置的 8K 对 1K 为主，不把它们当作三次独立重复实验。**

## 2. 主图：把新请求等待与旧请求输出对齐

![按三次注入对齐的旧请求输出进度与 ITL](../chunk_redo_20260928/analysis/injection_focus_20260928/01_注入对齐_输出进度与ITL.png)

读图方法：

1. 每行对应一个注入事件，左列 8K、右列 1K；**横坐标 0 是新请求发起时刻**。
2. 黄色区间从发起至客户端收到新请求首 token，包含客户端发送、服务端处理和交付，**不是单独测得的纯 Prefill 时段**。服务端 Prefill 只在图内列持续时间，没有映射到客户端时钟。
3. 上半图是旧请求相对 t=0 的输出进度；阶梯长时间水平，表示这段时间没有新输出到达。负数表示注入前的输出位置，不是负的生成数量。
4. 下半图一条水平线表示一个完整 ITL，线段横向覆盖该间隔的开始和结束，纵坐标为持续秒数。长横线代表一次长停顿，不是连续测到了多次相同停顿。
5. 全部面板使用相同时间范围和 ITL 纵轴；注入前后局部范围为 −5～14 秒，没有把后续另一次新请求发起混进本次展示窗口。

**最值得注意的形状：**8K 的两段约 3 秒停顿之间还有一次输出到达；1K 在开始阶段短暂输出后，经历一段约 9–10 秒的连续停顿。最长间隔可能跨过黄色右边界，因此其完整长度包括新请求首 token 到达之后的一小段时间。

## 3. 逐事件指标：新请求和旧请求分开统计

![逐事件 Prefill、TTFT、最大 ITL 与窗口输出速率](../chunk_redo_20260928/analysis/injection_focus_20260928/02_逐事件_Prefill与Decode干扰.png)

| 事件 | Chunk | 新请求 Prefill（s） | 新请求 TTFT（s） | 发起→新首 token 窗口（s） | 旧请求相交最大 ITL（s） | 窗口旧输出数 | 窗口旧输出速率（token/s） |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 A遇B | 8K | 5.649 | 6.096 | 6.118 | 3.100 | 2 | 0.327 |
| E1 A遇B | 1K | 9.803 | 10.265 | 10.285 | 9.620 | 2 | 0.194 |
| E2 B遇C | 8K | 5.637 | 6.085 | 6.106 | 3.103 | 2 | 0.328 |
| E2 B遇C | 1K | 9.621 | 10.103 | 10.124 | 9.400 | 2 | 0.198 |
| E3 C遇D | 8K | 5.659 | 6.071 | 6.092 | 3.170 | 2 | 0.328 |
| E3 C遇D | 1K | 9.525 | 9.952 | 9.972 | 9.318 | 2 | 0.201 |

三个指标不要混淆：

- **新请求 TTFT**从实际客户端请求开始计时；**注入窗口**从控制器提交请求计时。两者相差约 20–21 ms，属于控制器提交至实际请求开始的时间差；表中没有把两者写成一个值。
- **相交最大 ITL**取所有与注入窗口相交的完整旧请求输出间隔中的最大值，不把跨边界间隔裁短。它不是整个旧请求的最大值；旧请求可能还遭遇后续注入。
- **窗口输出速率**为 `窗口内实际收到的旧输出 token / 窗口秒数`。本轮分子均为 2，速率差直接来自窗口时长差，不能把它当作独立测得的纯 Decode 算子吞吐。

### 3.1 每个相交间隔，保留原始停顿结构

| 事件 | Chunk | 注入窗口相交的三个完整 ITL（s，按时间顺序） | 新首 token 后等到旧请求再输出（s） |
|---|---|---|---:|
| E1 | 8K | 0.465 / 3.100 / 3.046 | 0.493 |
| E1 | 1K | 0.476 / 0.708 / 9.620 | 0.518 |
| E2 | 8K | 0.464 / 3.103 / 3.003 | 0.464 |
| E2 | 1K | 0.496 / 0.688 / 9.400 | 0.461 |
| E3 | 8K | 0.456 / 3.170 / 2.982 | 0.488 |
| E3 | 1K | 0.469 / 0.686 / 9.318 | 0.473 |

这些完整间隔可能包含窗口前、窗口后时间，因此不能将三个 ITL 直接相加后称作“窗口内的停顿总时间”。

## 4. TTFT/TPOT 的均值与 P95/P99：保留两种请求口径

**全四条 A/B/C/D**便于对照上一版；**后来注入的 B/C/D**更贴近本次问题。旧请求已经进入 Decode 时，它原先的 TTFT 已发生；新 Prefill 对它的影响主要看上面的局部 ITL 和输出进度。

| 请求口径 | 指标 | Chunk | n | 均值 | P95 | P99 |
|---|---|---|---:|---:|---:|---:|
| 全部 A/B/C/D | TTFT（s） | 8K | 4 | 5.976 | 6.096 | 6.096 |
| 全部 A/B/C/D | TTFT（s） | 1K | 4 | 10.110 | 10.265 | 10.265 |
| 全部 A/B/C/D | TPOT（ms/token） | 8K | 4 | 492.40 | 505.98 | 505.98 |
| 全部 A/B/C/D | TPOT（ms/token） | 1K | 4 | 518.35 | 544.26 | 544.26 |
| 后来 B/C/D | TTFT（s） | 8K | 3 | 6.084 | 6.096 | 6.096 |
| 后来 B/C/D | TTFT（s） | 1K | 3 | 10.107 | 10.265 | 10.265 |
| 后来 B/C/D | TPOT（ms/token） | 8K | 3 | 487.88 | 504.66 | 504.66 |
| 后来 B/C/D | TPOT（ms/token） | 1K | 3 | 509.72 | 542.98 | 542.98 |

TPOT = `(客户端 E2E − TTFT)/(300−1)`，是整条请求首 token 后的平均值，可能包含不止一次新请求注入的影响。**这里没有把“窗口秒数 / 收到的 2 个 token”叫 TPOT。**

P95/P99 按最近秩法 `ceil(n×p)` 取值；n=3 或 4 时二者均为样本最大值，**不能视为稳定的业务尾延迟**。B/C/D 的三个样本也不等于三次独立实验。

## 5. 全部四请求的平均 E2E Breakdown

**A/B/C/D 全部纳入，每个 Chunk 各 4 条请求；四个阶段分别取算术均值，图中只画 8K、1K 两条。** Total 为四条请求 E2E 的算术均值，不是整轮墙钟时长。

![包括 A/B/C/D 的 Chunk 8K 和 1K 平均 E2E Breakdown](../chunk_redo_20260928/analysis/injection_focus_20260928/03_新请求_E2E与Prefill分解.png)

图例按展示要求简写为 Queue(几乎没有)、Prefill、Mid(IPC主导)、Decode。**Mid 的数值仍是首 token 前未归因差值，IPC 主导是展示假设，尚未通过独立计时验证；Decode 的数值仍含首 token 后的等待、调度与传输。**

| 阶段 | 数据与解释 |
|---|---|
| Queue | 最终 SSE 的 `queue_time`，是服务端报告字段 |
| Prefill | `prefill_finished_time − forward_entry_time`，服务端阶段区间，不是独立纯算子耗时 |
| 首 token 前未归因 | `TTFT − Queue − Prefill`，不能命名为实测 IPC |
| 首 token 后至结束 | `E2E − TTFT`，含生成、其他 Prefill 阻塞、调度、传输和收尾，不是纯 NPU Decode |

四段按定义相加为 E2E；这是使用服务端区间与客户端时长的**账面分解**，不是已经证明各阶段严格互斥的硬件执行时间线。图中顺序不说明所有未归因开销都发生在 Prefill 之后。

| Chunk | 请求数 | Queue 均值（ms） | Prefill 均值（s） | Mid 均值（ms） | Decode 均值（s） | Total 均值（s） |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 4 | 0.654 | 5.646 | 329.198 | 147.229 | 153.205 |
| 1K | 4 | 2.295 | 9.758 | 349.302 | 154.988 | 165.098 |

图表均值可复核：[e2e_breakdown_means.csv](../chunk_redo_20260928/analysis/injection_focus_20260928/e2e_breakdown_means.csv)。

## 6. Prefill 速度与注入期间的 Decode 输出速度

![新请求 Prefill 速度及旧请求注入前参考与窗口交付速度](../chunk_redo_20260928/analysis/injection_focus_20260928/04_Prefill速度与窗口Decode速度.png)

- **Prefill 速度**：`13184 / 新请求服务端 Prefill 秒数`，分子只含实际未命中输入。
- **旧请求窗口速度**：`窗口内实际收到的输出 token / 注入窗口秒数`，分母含整个新请求首 token 等待。
- **注入前局部参考**：选旧请求在注入前已经完整结束的最近至多 20 个 ITL，速率为 `间隔数 / 这些间隔的时长之和`；E1/E2/E3 实际分别有 19/19/20 个。**它不是另跑的无注入对照**，与窗口交付速率的观察时长和计算方式也不同。

| 事件 | Chunk | 新请求 Prefill（token/s） | 注入前参考 ITL 数 | 注入前参考速度（token/s） | 注入窗口旧输出速度（token/s） |
|---|---|---:|---:|---:|---:|
| E1 | 8K | 2333.97 | 19 | 2.004 | 0.327 |
| E1 | 1K | 1344.85 | 19 | 2.098 | 0.194 |
| E2 | 8K | 2338.65 | 19 | 2.112 | 0.328 |
| E2 | 1K | 1370.32 | 19 | 2.098 | 0.198 |
| E3 | 8K | 2329.57 | 20 | 2.177 | 0.328 |
| E3 | 1K | 1384.21 | 20 | 2.158 | 0.201 |

图中空心菱形在约 2 token/s，窗口柱形只有约 0.19–0.33 token/s，直接体现新 Prefill 到来时旧请求交付输出变稀疏。它不意味着 Decode 算子本身只剩这点计算速度。

### 6.1 完整请求的 Decode 平均速度仍可作为补充

为与原来的 TPOT 图对应，下面保留 `1000/TPOT(ms)`，即 `299/(E2E−TTFT)`。这是完整请求的客户端观察速度，不能全部归因到某一次注入。

| 请求 | 8K 整请求首 token 后速度（token/s） | 1K 整请求首 token 后速度（token/s） |
|---|---:|---:|
| A | 1.9764 | 1.8374 |
| B | 1.9815 | 1.8417 |
| C | 2.0476 | 1.9566 |
| D | 2.1249 | 2.1049 |

## 7. 放回整轮时间线，检查背景条件

![两档单轮四请求的生命周期与三次注入](../chunk_redo_20260928/analysis/injection_focus_20260928/05_单轮四请求时序.png)

这张图用于解释样本结构：E1 只有 A 在 Decode；E2 已有 A/B；E3 时 A 已结束，仍有 B/C。因此 E1 是最直接的“一个 Decode 遇到一个新 Prefill”观察，E2/E3 是已有两个 Decode 时的补充。同一事件位置在两档的已有请求数量一致，但两档的实际发起时刻、输出上下文进度并不完全相同。

## 8. 本轮能得出的判断与边界

1. **8K 的新请求 Prefill 更短，旧请求最长连续停顿也更短。** 三个事件位置都给出相同方向。
2. **平均 TPOT 会稀释短时干扰。** 这次判断更应看按注入时刻对齐的输出进度、完整 ITL 与新请求 Prefill。
3. **两个约 3 秒停顿与一个约 9.6 秒停顿，要连同整个等待窗口理解。** 本轮两档窗口内都只收到 2 个旧输出；8K 更早结束这段等待，不代表它没有 Prefill 阻塞。
4. **只有每档一轮。** 不能推断重复实验的方差或稳定 P99；三次注入有相关性，不能扩充成三轮。
5. 本轮没有独立的 Decode-only 控制组；注入前参考不是无干扰反事实。随机 token、强制 300 输出、Graph 关闭、mixed chunk 关闭限制了外推范围。
6. A/C/D 在两档的最终输出 ID 不同，只有 B 完全相同；同输入与输出长度相同不等于输出内容完全相同。

### 8.1 一般应选什么调度方式

若目标是在线对话的输出流畅性，我建议优先保障已有 Decode，再用剩余 token/计算预算安排 Prefill chunk，或使用能将 Decode 与小段 Prefill 混合的调度。这不是机械地规定每一块后只能执行一次 Decode；应同时约束已有请求 ITL、新请求 TTFT 与整体吞吐。若负载主要是批量输入处理、几乎没有活跃 Decode，较大的 Prefill 批次或更偏 Prefill 的策略可以作为另一种目标下的选择。此处是调优建议，未在本轮做交错调度 A/B。

### 8.2 A 是否完全不动：调度、执行与交付要分开

归档配置是 `enable_mixed_chunk=False`、`prefill_decode_interval=0`。固定镜像的 [scheduler.py](../mtp_full_20260928/results/20260928_061159/source_snapshot/served_image_sources/scheduler.py) 在约 3305 行优先选择存在的新 Prefill batch，约 3486 行继续处理未完成的 chunked 请求。这解释了为什么不能期待每个 1K 块后自动安排旧请求 Decode。

但不能把它加强成“从 B 发起到首 token 的整个窗口，A 连一个 token 都不会动”。原始 SSE 在每个窗口内确实收到观察对象的 2 个 token；同时 `disable_overlap_schedule=False`，源码约 1832 行的 overlap loop 会交错提交当前 batch 与处理前一 batch 结果。因此窗口内到达的 token 可能涉及已提交计算、结果队列与交付延迟，不能仅按到达时刻认定它在该时刻执行 Decode。现有日志 `decode_log_interval=40`，也不是逐 Decode step 的完整轨迹；不能凭中间缺少 Decode 日志就证明完全没有执行。

### 8.3 1K 为什么更慢：已确认的变化与尚未定位的根因

四条请求的服务日志均确认：8K 是 `8192 + 4992`，共 2 个 Prefill batch；1K 是 `12 × 1024 + 896`，共 13 个 Prefill batch。新计算总量都为 13,184 token。模型配置摘要与当时归档一致，模型有 78 层，MoE 路由专家数为 256、每 token 选择 8 个。原始批次证据和配置核对保存在 [scheduler_evidence.json](../chunk_redo_20260928/analysis/injection_focus_20260928/scheduler_evidence.json)。

**两个原因链条应分开：调度优先级解释旧请求为何长时间没有连续输出；Prefill 执行效率解释相同输入为何总用时不同。** 更多小批次意味着更多次跨模型层的批次执行；固定的调度、kernel launch、同步/通信成本以及小矩阵效率损失，是需要验证的候选原因。Graph 实际关闭、TP16/EP16、小规模每专家 token 批量，使这些方向值得优先检查，但本轮没有 kernel/通信/CPU 的 profiler breakdown，不能给任何一项分配确定的秒数或宣称根因已定位。

只按新注入的 B/C/D 看，1K Prefill 为 9.525–9.803 秒，8K 为 5.637–5.659 秒，耗时约为 1.68–1.74 倍；全四条平均为 1.728 倍，即多约 72.8%，不是精确两倍。1K 不是把整份输入重算十三遍；每个 chunk 只推进其新增输入，但要走相应的模型计算与调度路径。

### 8.4 0.3 token/s 为什么看起来离谱

以 E1 为例，8K 的窗口为 6.117528 秒、观察的 A 收到 2 token，故为 `2 / 6.117528 = 0.326929 token/s`；1K 为 `2 / 10.285293 = 0.194452 token/s`。这个数是真实窗口交付率，但其中大部分时间是等待。注入前参考约 2.0–2.2 token/s，完整请求首 token 后均速约 1.84–2.12 token/s；三个量不能混用，任何一个单请求值也不是多请求服务的总输出吞吐。

约 2 token/s 的基线是否还有优化空间是另一个问题。当前是 78 层 MoE、TP16/EP16、MTP OFF、Graph disabled、DeepEP normal、1–3 条活跃请求；这些是可确认的运行条件，并不能单凭配置判断哪项吃掉了最多时间。需要用无注入基线和算子/通信/调度分析来定位，不能把这个低速直接说成硬件正常上限，或保证单开某个开关就会达到某个速度。

## 9. 文件、导出与复算

五张图均有 PNG、SVG，并汇总成 [五页 PDF](../chunk_redo_20260928/analysis/injection_focus_20260928/ChunkSize_单轮注入干扰_五图.pdf)。

| 图 | PNG | SVG |
|---|---|---|
| 主图：按注入对齐 | [PNG](../chunk_redo_20260928/analysis/injection_focus_20260928/01_注入对齐_输出进度与ITL.png) | [SVG](../chunk_redo_20260928/analysis/injection_focus_20260928/01_注入对齐_输出进度与ITL.svg) |
| 逐事件指标 | [PNG](../chunk_redo_20260928/analysis/injection_focus_20260928/02_逐事件_Prefill与Decode干扰.png) | [SVG](../chunk_redo_20260928/analysis/injection_focus_20260928/02_逐事件_Prefill与Decode干扰.svg) |
| 四请求平均 E2E / Prefill | [PNG](../chunk_redo_20260928/analysis/injection_focus_20260928/03_新请求_E2E与Prefill分解.png) | [SVG](../chunk_redo_20260928/analysis/injection_focus_20260928/03_新请求_E2E与Prefill分解.svg) |
| 阶段速度 | [PNG](../chunk_redo_20260928/analysis/injection_focus_20260928/04_Prefill速度与窗口Decode速度.png) | [SVG](../chunk_redo_20260928/analysis/injection_focus_20260928/04_Prefill速度与窗口Decode速度.svg) |
| 整轮时序 | [PNG](../chunk_redo_20260928/analysis/injection_focus_20260928/05_单轮四请求时序.png) | [SVG](../chunk_redo_20260928/analysis/injection_focus_20260928/05_单轮四请求时序.svg) |

数值明细：[注入事件 CSV](../chunk_redo_20260928/analysis/injection_focus_20260928/injection_events.csv)、[全部请求 CSV](../chunk_redo_20260928/analysis/injection_focus_20260928/requests.csv)、[TTFT/TPOT 汇总](../chunk_redo_20260928/analysis/injection_focus_20260928/latency_summary.csv)、[按注入对齐的全部 ITL](../chunk_redo_20260928/analysis/injection_focus_20260928/aligned_itl_intervals.csv)、[统计与校验](../chunk_redo_20260928/analysis/injection_focus_20260928/summary.json)。

生成代码：[plot_injection_report.py](../chunk_redo_20260928/analysis/injection_focus_20260928/plot_injection_report.py)，复用[原始 SSE 校验解析器](../chunk_redo_20260928/analysis/prefill_comparison_20260928/plot_report.py)。输入文件与代码摘要见 [input_manifest.json](../chunk_redo_20260928/analysis/injection_focus_20260928/input_manifest.json)。

```bash
python3 /home/chengyetai/workspace/current_exp/chunk_redo_20260928/analysis/injection_focus_20260928/plot_injection_report.py
```

本次离线复核八条原始 SSE 与最终输出 ID；每条 300 个逐 token 事件，HTTP 200，实际命中 19,584，输出长度完整；跨档输入 SHA 一致。六个注入窗口均核查已有 Decode 数、实际到达 token 数、边界相交间隔和归档最大 ITL，结果一致。脚本读取归档数据生成图表，不调用在线服务。
