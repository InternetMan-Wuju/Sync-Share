# ChunkSize 8K / 1K 与开启 MTP 的效果：综合分析

整理日期：**2026-09-28（北京时间）**。范围：`/home/chengyetai/workspace/current_exp` 中已归档的实验、原始 SSE、配置快照和固定镜像源码。本次离线复核了 **8 条 Chunk 请求、83 条 MTP 请求**，并从原始输出到达时刻重算关键统计；以下百分比使用未四舍五入的数据计算。

## 1. 直接回答：选哪个，MTP 值不值得开

**当前这套已测的单机配置，ChunkSize 8K 比 1K 更好。MTP 在已测的三个低并发负载上明显加快后续生成，但不保证首 token 更快，也还不能直接视为真实业务与高并发下的收益。**

| 决策 | 结论 | 最直接的证据 |
|---|---|---|
| ChunkSize 8K 还是 1K | **优先选 8K**，适用于本轮 `mixed_chunk=False`、Graph 实际关闭的单机条件 | 平均 TTFT：8K **5.976 s**、1K **10.110 s**，8K 降低 **40.89%**；长 Prefill 插入时，正在生成的请求最长停顿约 **3.1 s 对 9.3–9.6 s** |
| 开启 MTP 后生成速度如何 | 在已测负载下有明显收益，值得作为优化方向 | 1K/32K 目标、8K/1K、32K/300 三个场景，45 分钟窗口输出吞吐分别变为 OFF 的 **3.706×、3.148×、2.224×** |
| MTP 是否让所有延迟都更低 | **没有**；收益主要在首 token 之后 | 32K/300 场景 TTFT 从 **5.396 s 增至 6.411 s（+18.79%）**，同时 TPOT 从 **476.53 ms 降到 186.45 ms（−60.87%）** |
| 能否宣布“8K＋MTP”已经验证最优 | **尚不能** | 恢复权重后的 Chunk 对照是 **MTP OFF**；较完整的 MTP 对照固定 **Chunk=16K**。早期虽有 8K＋MTP 探索，但没有同等的语义和权重恢复验证 |
| 是否有容量代价 | 有，需要一并评估 | 本轮 MTP OFF/ON 的 `max_total_num_tokens` 为 **63,744 / 43,008**，开启后减少 **20,736（32.53%）** |

因此，当前可以采用的调优判断是：**8K 作为这套单机服务的优先 Chunk 基线；MTP 作为有明显生成收益的候选配置。把两者组合用于实际业务前，需要在同一组恢复后权重、同一节点与真实请求上补齐组合验证。** 上述数字的实验条件和证据分别见第 4、5 节。

## 2. 哪些数据应作为主要依据

目录内有多轮实验，负载、权重加载状态、并发和 Chunk 值并不相同。本文采用下列优先级，避免把历史探索数字混成一组结论。

| 资料 | 实际内容 | 本文如何使用 |
|---|---|---|
| [2026-09-28 Chunk 复测](../chunk_redo_20260928/results/20260928_043035/README.md) | 节点 23，8K/1K 各 4 条；总输入 32K、输出 300、实际缓存命中 59.765625%；错开注入；采用只读恢复权重并做真实回复检查 | **判断 Chunk 的主要依据** |
| [2026-09-28 MTP 三场景](MTP_三种工作负载_45分钟_20260928.md) | 节点 23 OFF、节点 25 ON；每节点独立单机；三个场景各 45 分钟；Chunk=16K；有完整/截断请求、接受计数及输出记录 | **判断 MTP 的主要依据** |
| [2026-09-26 Chunk 首轮](ChunkSize单机实验_2026-09-26.md) | 主负载曾把 32K 误解释为公共前缀长度，造成总输入 54,614；真正 32K 总输入对照只有每档 1 条；没有真实文本语义验收 | 只作方向性历史补充，不能替代 9 月 28 日复测 |
| [2026-09-26 MTP 探索](MTP推测解码单机探索_2026-09-26.md) | 同一节点 25、Chunk=8K、每场景每模式 2 条；长输出仅 2,048，非完整 32K | 说明早期已跑通功能；不采用它的 3.3–3.6× 作为本轮通用收益 |
| [权重异常与恢复记录](模型回复异常与只读权重外挂恢复_2026-09-28.md) | 9 月 27 日发现重复字符、读取字节变化与异常浮点值；参考块逐块校验后，9 月 28 日真实中文回复恢复 | 解释为什么主要采用恢复后的实验；不能反向断言所有 9 月 26 日数据都错误 |
| [EPLB / PD 实验](../eplb_pd_agentx_20260928/README.md) | 1P1D、EPLB/冗余专家、其他负载与并发 | 用途不同，不进入本次 Chunk/MTP 的性能均值 |

本轮 MTP 两节点的修复清单文件本身不同，但目标权重的 **182 个文件头、23,093 个块 SHA-256** 比较一致；因此可以排除“清单不同就一定加载了不同目标字节”的误解。它仍不能消除节点性能差异。证据：[MTP 实验清单](../mtp_full_20260928/results/20260928_061159/experiment_manifest.json)、[权重比较](../mtp_full_20260928/results/20260928_061159/weight_comparison/comparison.json)。

## 3. 指标如何读

本文约定 **1K=1,024，8K=8,192，16K=16,384，32K=32,768 token**。`8K/1K` 在 MTP 工作负载表中表示“输入 8K、输出 1K”；只有 `ChunkSize` 一列才表示 Prefill 的分块参数。

| 指标 | 本文口径 | 优劣方向与边界 |
|---|---|---|
| Prefill 阶段用时 | 服务端 `prefill_finished_time − forward_entry_time` | 越低越好；是服务端阶段区间，不等同于纯 NPU 算子耗时 |
| TTFT | 客户端发送请求到收到首次有输出的 SSE 帧 | 越低越好，体现首 token 等待 |
| TPOT | Chunk：`(E2E − TTFT)/(输出数−1)`；MTP：窗口内 `(最后新增输出到达−首次输出到达)/(输出数−1)` | 越低越好；二者分别包含/不包含请求尾部收尾，因此不跨实验直接拼数 |
| ITL | 相邻输出事件到达客户端的时间间隔 | 看是否卡顿；Chunk 本轮每事件正好新增 1 token，MTP 每事件可能新增多个 token |
| E2E | 客户端请求开始到请求结束 | 越低越好；截断请求的 E2E 不能冒充完成目标长度的用时 |
| 45 分钟窗口输出吞吐 | 截止前收到的正式测量输出 token 总数 / 2,700 s | 越高越好；分母包含清缓存、预热、Prefill 与请求间隙，分子不含预热输出 |
| 接受长度 | 有最终累计计数的请求，`Σ实际输出 / Σverify 次数` | 含 bonus token；本配置上限约为 4，不能直接当加速倍数 |
| Draft 接受率 | `Σ接受 draft / Σ提出 draft` | 不含 bonus；按累计计数求比值，不能简单平均最近一次 Gauge |

例如“吞吐变为 **3.706×**”表示 **达到原来的 370.6%，增幅约 270.6%**；“TTFT 降低 40.89%”则以 1K 的 TTFT 为分母。本文明确区分“倍率”和“增幅”。

计时与统计定义可核查 [Chunk 分析器](../chunk_redo_20260928/analyze.py)、[MTP 分析器](../mtp_full_20260928/analyze.py)，接受计数定义见固定镜像 [tokenizer_manager.py](../mtp_full_20260928/results/20260928_061159/source_snapshot/served_image_sources/tokenizer_manager.py) 的 `_calculate_spec_decoding_metrics`。

## 4. ChunkSize：8K 为什么优于 1K

### 4.1 可比条件

| 条目 | 两档共同条件 |
|---|---|
| 节点、模型、软件 | 节点 23；GLM-5.2 W8A8；单机 TP16/EP16/DP1，16 个 NPU die；镜像 `sglang-glm52-pd:fe694986`，服务版本 `0.5.19.dev684+gfe694986a` |
| 权重加载 | 使用只读参考块校验后的恢复路径；每档生成前有中文算术与翻译检查 |
| 核心参数 | `max_running_requests=4`、`max_prefill_tokens=65536`、`mem_fraction_static=0.88`、DeepEP normal |
| 调度和图 | `enable_mixed_chunk=False`、`prefill_decode_interval=0`；CUDA/NPU Graph **实际 disabled**；MTP OFF |
| 缓存与输入 | 每组先清缓存，再预热 19,661 token 的共享前缀；每条总输入 32,768，实际命中 19,584 |
| 输出 | 每条 300 token；`temperature=0`、`ignore_eos=true`；固定随机 token ID，两档重用相同四条输入 |
| 顺序 | 先跑 8K，再切换 1K；每档四条，活跃请求上限实际为 3 |
| KV 池容量 | 两档均为 `max_total_num_tokens=63744`，本轮没有显示 1K 带来更大的可用 KV 池 |

证据：[8K 实际配置](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/before_bench/server_info.json)、[1K 实际配置](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/before_bench/server_info.json)、[数据集](../chunk_redo_20260928/results/20260928_043035/dataset.json)。启动脚本虽传入 Graph `full`，服务实际配置和日志均显示被 DeepEP normal 路径禁用，应按实际状态解释结果。

四条请求的注入规则是：A 开始；A 至少输出 20 token 后发 B；B 至少输出 20 token 后发 C；等 A 完成且 C 至少输出 20 token、仍在 Decode 时发 D。这样可以观察 A 遇 B、B 遇 C、C 遇 D 的输出停顿。规则相同，但由于两档速度不同，实际墙钟到达时刻不同。证据：[8K 调度记录](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/bench/dispatches.json)、[1K 调度记录](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/bench/dispatches.json)。

### 4.2 汇总：首 token 和 Prefill 的差距最明显

| 指标 | Chunk 8K | Chunk 1K | 8K 相对 1K |
|---|---:|---:|---|
| 服务端 Prefill 阶段均值（s） | 5.646 | 9.758 | 降低 42.14% |
| 客户端 TTFT 均值（s） | 5.976 | 10.110 | 降低 40.89% |
| 客户端 TPOT 均值（ms/token） | 492.40 | 518.35 | 降低 5.01% |
| 客户端 E2E 均值（s） | 153.205 | 165.098 | 降低 7.20% |
| 未命中输入 / Prefill 阶段时间之和（token/s） | 2335.0 | 1351.0 | 提高 72.83%；1.728× |

来源：[逐请求 CSV](../chunk_redo_20260928/results/20260928_043035/analysis/requests.csv)，本次另从每条原始 SSE 与 `result.json` 校验。均值各含四条请求；最后一行采用 **未命中 token 总数 / 各请求 Prefill 阶段用时之和**，用于对比 Prefill 阶段效率，并非服务整体吞吐。

关键数字的解释：

- **首 token 少等约 4.134 秒**：10.110 → 5.976 s。若关注“用户点发送后多久开始回复”，8K 的优势直接可见。
- **Prefill 阶段少约 4.112 秒**：9.758 → 5.646 s，说明主要差距与输入处理阶段一致。实际新计算量相同，都是 `32768 − 19584 = 13184 token/请求`。
- **E2E 只降低约 7.20%**：输出阶段仍占总时间的大部分，因此不能把 42.14% 的 Prefill 用时降低解释为整请求也降低 42.14%。
- **TPOT 平均降低约 5.01%**：这是包含长 Prefill 干扰的请求均值；它不能独立证明 8K 的纯 Decode 算子更快。

### 4.3 八条请求明细

| Chunk | 请求 | Prefill（s） | TTFT（s） | TPOT（ms/token） | E2E（s） | 全请求最长 ITL（s） |
|---|---|---:|---:|---:|---:|---:|
| 8K | A | 5.640 | 5.652 | 505.98 | 156.939 | 3.103 |
| 8K | B | 5.649 | 6.096 | 504.66 | 156.989 | 3.170 |
| 8K | C | 5.637 | 6.085 | 488.37 | 152.106 | 3.170 |
| 8K | D | 5.659 | 6.071 | 470.62 | 146.785 | 0.578 |
| 1K | A | 10.085 | 10.120 | 544.26 | 172.854 | 9.620 |
| 1K | B | 9.803 | 10.265 | 542.98 | 172.617 | 9.400 |
| 1K | C | 9.621 | 10.103 | 511.09 | 162.918 | 9.318 |
| 1K | D | 9.525 | 9.952 | 475.08 | 152.002 | 0.600 |

八条均 HTTP 200、输出达到 300、`num_retractions=0`、实际命中 59.765625%。跨档最终输出 ID 只有 B 完全一致，A/C/D 不一致。因此“输入相同、输出长度相同”成立，但“生成内容逐 token 完全相同”不成立。来源：[逐请求报告](../chunk_redo_20260928/results/20260928_043035/analysis/README.md)、[完整性审计](../chunk_redo_20260928/results/20260928_043035/analysis/integrity_audit.json)。

### 4.4 Decode 卡顿：1K 没有表现出保护输出流畅性的收益

| Chunk | 已有 Decode / 后来 Prefill | 注入前 ITL 中位数（s） | 注入窗口最大 ITL（s） | 窗口后 ITL 中位数（s） |
|---|---|---:|---:|---:|
| 8K | A 遇到 B | 0.506 | 3.100 | 0.464 |
| 8K | B 遇到 C | 0.475 | 3.103 | 0.464 |
| 8K | C 遇到 D | 0.464 | 3.170 | 0.477 |
| 1K | A 遇到 B | 0.474 | 9.620 | 0.475 |
| 1K | B 遇到 C | 0.476 | 9.400 | 0.474 |
| 1K | C 遇到 D | 0.475 | 9.318 | 0.462 |

来源：[干扰 CSV](../chunk_redo_20260928/results/20260928_043035/analysis/interference.csv)。这里的窗口从后来请求发起到其首 token 到达，包含发送、排队和交付；与窗口相交的间隔均计入，因此不等于纯服务端 Prefill 区间。三次窗口也不是三次独立实验重复。

常态输出间隔约 0.46–0.51 s，后来 Prefill 到达时，8K 最大间隔约 3.1 s，1K 约 9.3–9.6 s。**关注聊天过程中“突然不出字”的体验时，当前数据更支持 8K。** 8K 本身仍有秒级停顿，说明它也没有消除 Prefill 对 Decode 的干扰。

![Chunk 8K 与 1K 的逐输出事件间隔时间线](../chunk_redo_20260928/results/20260928_043035/analysis/itl_timeline.png)

两图使用相同纵轴。色块为后来请求发出至首 token 到达；散点横坐标是相邻输出间隔的结束时刻，跨过窗口末端的停顿点可能出现在色块右侧。

### 4.5 为什么“小块更灵活”在这里没有转化为优势

本轮实际未命中输入是 **13,184 token**。仅按分块上限估算，8K 约需 `ceil(13184/8192)=2` 块，1K 约需 `ceil(13184/1024)=13` 块；这是工作量示意，不能当作实际 forward 次数的逐步跟踪结果。

固定镜像源码和实测参数给出了更关键的条件：

1. `enable_mixed_chunk=False`，没有启用将已有 Decode 合入 Prefill batch 的分支。
2. `prefill_decode_interval=0`，没有通过该参数在 Prefill 后强制安排一段 Decode 间隔。
3. 调度器有可执行的新 Prefill batch 时优先选择它，否则才选择已有 Decode batch。

代码依据：[scheduler.py](../mtp_full_20260928/results/20260928_061159/source_snapshot/served_image_sources/scheduler.py) 中约 1225 行的 mixed 开关、1243–1263 行的间隔调度、3305–3314 行的 Prefill 优先分支、3667–3687 行的 mixed 合并分支；该快照与本轮使用相同固定镜像。实际参数见 4.1 节配置链接。

**据此可作的机制解释是：小 Chunk 增加分块轮次，却不会自动让 Decode 在块间获得执行机会；本轮更长的总 Prefill 阶段与更长停顿相吻合。** 现有计时没有把调度、kernel launch、计算与通信成本分别拆开，不能进一步把差额精确归因于某一项。换成 mixed-chunk、Graph ON、不同并发或 PD 分离后，应重新比较。

## 5. MTP：开启后的收益和代价

### 5.1 本轮到底测了什么

节点 23 固定 MTP OFF，节点 25 固定 MTP ON；两者各自运行单机 TP16/EP16/DP1，使用相同目标权重字节和固定镜像。每台客户端只有 **1 条请求在途**，完成后继续提交下一条；服务端允许的最大请求数 4 不等于实际压测并发 4。

| 场景 | 实际测量窗口开始（北京时间） | 输入 / 目标输出 | 实际命中 | Chunk | 两端 HCCL 缓冲 |
|---|---|---|---|---:|---:|
| Decode 长输出 | 2026-09-28 07:10:22 | 1,024 / 32,768 | 0 | 16,384 | 650 MB |
| 8K 输入 / 1K 输出 | 2026-09-28 07:57:38 | 8,192 / 1,024 | 0 | 16,384 | 650 MB |
| 长输入缓存场景 | 2026-09-28 09:30:28 | 32,768 / 300 | 19,584 / 32,768 = 59.765625% | 16,384 | 1,024 MB |

每场景 **2,700 秒**，两模式同时测；`temperature=0`、`ignore_eos=true`、固定随机 token 输入；Graph 两端实际均 disabled，mixed chunk 关闭。第三场景需要增大通信缓冲后才成功开始正式窗口，因而只在各场景内部做 ON/OFF 比较，不把不同场景视为仅输入长度变化的严格单变量实验。证据：[实验清单与实际时间](../mtp_full_20260928/results/20260928_061159/experiment_manifest.json)。

开启参数为归档启动命令中的：

```bash
--speculative-algorithm NEXTN \
--speculative-draft-model-quantization unquant \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

这些是本次已验证的 MTP 参数片段，需连同对应模型、外挂和完整启动条件理解。固定镜像把 `NEXTN` 归一为 `EAGLE`；3 个草稿步骤、4 个验证槽位，一轮有机会推进约 4 个输出 token。草稿层采用 `unquant`；实际配置里该量化字段被规范为 `null`。证据：[ON 启动命令](../mtp_full_20260928/results/20260928_061159/node_25/chunk_16384/launch.sh)、[ON 实际配置](../mtp_full_20260928/results/20260928_061159/final_state/node25_on/get_server_info.json)。

### 5.2 45 分钟内实际产出了多少

表内斜线均按 **OFF / ON** 排列；倍率为 **ON 吞吐 / OFF 吞吐**。

| 场景 | 窗口输出 token：OFF / ON | 窗口吞吐 token/s：OFF / ON | 吞吐倍率 | 完整请求：OFF / ON | 截断请求：OFF / ON |
|---|---:|---:|---:|---:|---:|
| 1K 输入 / 32K **目标**输出 | 5,716 / 21,184 | 2.117 / 7.846 | **3.706×** | 0 / 0 | 2 / 2 |
| 8K 输入 / 1K 输出 | 5,662 / 17,826 | 2.097 / 6.602 | **3.148×** | 5 / 17 | 1 / 1 |
| 32K 输入 / 300 输出，59.77% 命中 | 5,017 / 11,158 | 1.858 / 4.133 | **2.224×** | 16 / 37 | 1 / 1 |

来源：[decode 汇总](../mtp_full_20260928/results/20260928_061159/analysis/decode_summary.json)、[mixed 汇总](../mtp_full_20260928/results/20260928_061159/analysis/mixed_summary.json)、[prefix 汇总](../mtp_full_20260928/results/20260928_061159/analysis/prefix_summary.json)。本次从每条原始 SSE 的 `recv_perf_ns` 和累计 `completion_tokens` 按窗口截止重算，结果一致。

三个场景正式请求的非正常错误计数均为 0，但**截止截断不等于成功完成目标输出长度**。尤其 1K/32K 两端完整请求数都是 0：45 分钟内 OFF 约生成 5.7K，ON 约生成 21.2K，不能称为“完成了 32K 输出评测”。两端各有一个截止附近的极短补发请求，不能据“2 条截断”理解为完成了两次有代表性的长输出测量。

prefix 的每条测量前都要清缓存并重新预热，预热时间包含在窗口分母中。ON 完成更多请求，也付出更多次预热时间。因此这里的 2.224× 是**该压测流程下的窗口产出比**，既不是纯 Decode 加速比，也不是线上缓存常驻条件下的最大服务容量。

### 5.3 TTFT 与 TPOT：生成更快，首 token 未必更快

| 场景 | TTFT 均值 s：OFF / ON | ON TTFT 变化 | TPOT 均值 ms/token：OFF / ON | TPOT 降幅 | 延迟样本数：OFF / ON |
|---|---:|---:|---:|---:|---:|
| 1K 输入 / 32K **目标**输出 | 1.062 / 1.128 | +6.13% | 472.08 / 127.36 | 73.02% | 1 / 1 |
| 8K 输入 / 1K 输出 | 3.637 / 3.488 | -4.10% | 473.06 / 150.55 | 68.17% | 6 / 18 |
| 32K 输入 / 300 输出，59.77% 命中 | 5.396 / 6.411 | +18.79% | 476.53 / 186.45 | 60.87% | 17 / 38 |

来源同 5.2 节及 [逐请求统计](../mtp_full_20260928/results/20260928_061159/analysis/all_requests.csv)。截止前最后 3 秒才开始且未完成的极短补发请求不进入延迟均值，实际输出仍按到达时间计入吞吐。1K/32K 只有各 1 条有意义的长输出延迟样本。

可以据此判断：MTP 将后续生成的平均每 token 等待从约 **0.47 秒**压缩到约 **0.13–0.19 秒**。但它不是本轮可靠的 TTFT 优化手段：长输入缓存场景首 token 明确变慢，而 8K/1K 的全窗口 TTFT 小幅降低还受到输入样本数量不同的影响，需看下面的同输入配对。

![MTP 接受长度、窗口吞吐、TTFT 和 TPOT](../mtp_full_20260928/results/20260928_061159/analysis/summary.png)

### 5.4 同一输入、完整输出的配对结果

ON 更快，在 45 分钟内会访问更多输入；直接比较全窗口均值时，双方样本构成不同。下表仅取**相同输入 SHA、双方都完整生成目标长度**的请求，适合判断同一道输入的整请求用时。

| 场景 | 完整配对数 | TTFT 均值 s：OFF / ON | TPOT 均值 ms/token：OFF / ON | E2E 均值 s：OFF / ON | E2E 均值之比 OFF/ON |
|---|---:|---:|---:|---:|---:|
| 8K 输入 / 1K 输出 | 5 | 3.706 / 3.754 | 473.55 / 146.25 | 488.148 / 153.365 | 3.183× |
| 32K 输入 / 300 输出，59.77% 命中 | 16 | 5.395 / 6.383 | 476.68 / 187.72 | 147.923 / 62.513 | 2.366× |

数据：[配对逐请求 CSV](../mtp_full_20260928/results/20260928_061159/analysis/paired_requests.csv)、[配对汇总](../mtp_full_20260928/results/20260928_061159/analysis/paired_summary.json)。E2E 均值由 CSV 重算，完成标记与原始请求结果再次核对。

- **8K/1K**：相同输入的 TTFT 是 3.706 → 3.754 s，约 **增加 1.31%**；E2E 则从 488.148 → 153.365 s，约 **减少 68.58%**。所以更可信的说法是“首 token 基本接近，之后生成明显加速”，而不是“MTP 稳定改善了 TTFT”。
- **32K/300**：TTFT 5.395 → 6.383 s，约 **增加 18.31%**；E2E 147.923 → 62.513 s，约 **减少 57.74%**。首 token 多等约 0.99 s，仍换来整请求少约 85.41 s。
- **1K/32K**：没有完整配对，不能给出实测的完整 32K E2E 加速比。

**统计方式特别说明：**原报告的 3.234×、2.432× 是“每对请求先算 `E2E_OFF/E2E_ON`，再平均”；本表的 3.183×、2.366× 是“双方 E2E 均值之比”。两种方式不相等，均可复算；不能把它们混写成同一个指标，也不能与 45 分钟窗口吞吐倍率混用。

### 5.5 接受率：收益为何随负载不同

| 场景（均为 ON） | Verify 次数 | 接受 / 提出 draft | Draft 接受率 | 接受长度，含 bonus | 接受计数覆盖窗口输出 |
|---|---:|---:|---:|---:|---:|
| 1K 输入 / 32K **目标**输出 | 5,299 | 15,884 / 15,897 | 99.92% | 3.998 | 100.00% |
| 8K 输入 / 1K 输出 | 5,033 | 12,380 / 15,099 | 81.99% | 3.459 | 97.66% |
| 32K 输入 / 300 输出，59.77% 命中 | 4,033 | 7,148 / 12,099 | 59.08% | 2.767 | 100.00% |

接受长度分别是 `21184/5299`、`17408/5033`、`11158/4033`。8K/1K 尾部截断请求缺最终累计接受计数，因此接受统计覆盖 **17,408 / 17,826 = 97.66%** 的窗口输出；它的全部已收到输出仍计入吞吐，没有把缺失接受计数假定为 0 或 100%。证据：[三场景汇总](../mtp_full_20260928/results/20260928_061159/analysis/README.md)及各场景 `measurement/result.json`。

从原始请求终态计数重算的接受直方图如下；每格是“一次 verify 恰好接受该数量 draft”的次数。

| 场景 | 接受 0 个 | 接受 1 个 | 接受 2 个 | 接受 3 个 |
|---|---:|---:|---:|---:|
| 1K 输入 / 32K **目标**输出 | 2 | 2 | 3 | 5,292 |
| 8K 输入 / 1K 输出 | 471 | 469 | 368 | 3,725 |
| 32K 输入 / 300 输出，59.77% 命中 | 952 | 774 | 547 | 1,760 |

1K 长输出大部分 verify 都接受满 3 个 draft，所以接近每次前进 4 token；32K/300 的接受率约 59%，平均每次只前进约 2.77 token。**接受指标与本轮生成收益的变化方向一致**，但草稿、验证、同步等都要耗时，接受长度本身不是加速倍数。指标定义以 [固定镜像计数源码](../mtp_full_20260928/results/20260928_061159/source_snapshot/served_image_sources/tokenizer_manager.py)为准，本文使用最终累计计数，未把瞬时 Gauge 当全程均值。

## 6. 影响结论使用的限制

### 6.1 1K/32K 的 3.706× 不能直接用于真实对话预测

[输出模式审计](../mtp_full_20260928/results/20260928_061159/analysis/output_patterns.json)显示，双方首条长请求都重复同一组 **`[168, 111, 238]`**，序列周期为 3；OFF 这条实际收集 5,715 token，ON 收集 21,184 token。窗口总数与单条数的差别来自截止附近极短请求，以及是否落在截止前。

这意味着接近 **99.92%** 的 draft 接受率是在高度重复的输出上取得的。它能证明这条执行路径获得了加速，不能证明自然语言回答也有接近 100% 接受率。两模式还走到了不同输出长度，整窗均值不是同等最终上下文长度的完整比较。

[按输出进度分段的速度 CSV](../mtp_full_20260928/results/20260928_061159/analysis/decode_progress.csv)中，双方共有的前约 5,120 输出 token 范围，OFF 各段约 **2.11–2.12 token/s**，ON 约 **7.87–8.00 token/s**；共同进度内也能看到加速，但仍然是这条重复 token 序列。

### 6.2 输出一致性与质量尚未完成验收

本次核对的完整配对中，8K/1K 有 **4/5 对**输出 ID 不一致，32K/300 有 **16/16 对**不一致；Chunk 对照也有 A/C/D 不一致。`temperature=0` 并未让本次两配置输出逐 token 相同，实际配置中的 `enable_deterministic_inference` 也为 false。

现有材料没有定位这些差异的原因，既不能直接断言 MTP 导致质量下降，也不能宣布质量完全等价。真实中文检查证明短请求能正常回复，不等于长文本正确率、分布等价或业务效果已经通过验收。证据：[配对记录](../mtp_full_20260928/results/20260928_061159/analysis/paired_requests.csv)、[最终语义检查目录](../mtp_full_20260928/results/20260928_061159/semantic_checks_final/)。

### 6.3 容量代价需要和速度收益一起看

| 运行配置 | 可用 KV token 池上限 | 相对变化 |
|---|---:|---|
| Chunk 8K、MTP OFF，节点 23 | 63,744 | Chunk 基线 |
| Chunk 1K、MTP OFF，节点 23 | 63,744 | 本轮没有增加 KV 池上限 |
| Chunk 16K、MTP OFF，节点 23 | 63,744 | MTP 对照基线 |
| Chunk 16K、MTP ON，节点 25 | 43,008 | 减少 20,736，即 32.53% |

证据：[Chunk 8K 配置](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/before_bench/server_info.json)、[Chunk 1K 配置](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/before_bench/server_info.json)、[MTP OFF 终态配置](../mtp_full_20260928/results/20260928_061159/final_state/node23_off/get_server_info.json)、[MTP ON 终态配置](../mtp_full_20260928/results/20260928_061159/final_state/node25_on/get_server_info.json)。ON 的[固定服务日志快照](../mtp_full_20260928/results/20260928_061159/final_state/node25_on/docker_logs.stdout)还记录额外 NextN 模型加载约 **1.72 GB/rank**，单位沿用日志。

这里减少的是**该配置的 KV 池容量**，不能简化成“NPU 总显存占用下降了 32.5%”或“并发能力必然下降 32.5%”。在长上下文和多并发下，KV 紧张可能影响排队、回收与可容纳请求数；单条请求提速的证据还不足以推导整体容量收益。

### 6.4 统计与实验设计边界

- **Chunk 样本少且顺序固定**：每档仅四条，先 8K 后 1K；注入窗口互相影响，没有交叉顺序重复。适合作为当前配置选择依据，不足以估计稳定的请求 P95/P99。
- **MTP 与节点绑定**：OFF 固定 23、ON 固定 25，即使权重字节相同，仍未用交换节点或同节点交叉试验消除机器差异。
- **MTP 并发仅为 1**：它回答单条在途请求与该串行流程的产出，不回答高并发吞吐、最大承载能力或服务 SLO。
- **Graph 两边实际关闭**：当前倍数不能直接套给已经启用 Graph 的优化服务；不同基线可能改变 MTP 相对收益。
- **MTP SSE 可以一次交付多个 token**：客户端平均 TPOT 可比较，但不能把相邻 SSE 帧间隔冒充逐 token ITL。
- **随机输入与强制长度**：`ignore_eos=true`、固定随机 token 与普通对话不同；需要真实业务数据验证接受率和输出质量。

上述限制分别来自前述实际配置、调度记录、数据集、逐请求配对与统计样本数；不是额外测得的性能退化结论。

## 7. 对实际配置的建议，以及还缺哪组对照

| 使用目标 | 现有证据支持的选择 | 需要保留的条件 |
|---|---|---|
| 这套单机 32K 输入 / 300 输出服务，优先降低首 token 与插入卡顿 | **选 Chunk=8192** | 当前 Graph disabled、mixed chunk 关闭、MTP OFF 的同机对照最直接 |
| 优先降低生成时间、当前负载并发低 | **优先验证并使用 MTP ON** | 已测 Chunk=16384 下有明显收益；观察真实输入接受率、输出质量与 KV 余量 |
| 重点是长输入短回答的首 token | 先用较优 Chunk 基线；MTP 要同时看 TTFT 与 E2E | 本轮 32K/300 开启 MTP 的首 token 约慢 1 s，但整请求明显更快 |
| 计划同时采用 8K 与 MTP | 作为下一组候选组合 | 恢复后尚缺同一数据、同一节点、相同调度下的完整 2×2 对照 |
| 高并发或 PD 分离 | 重新测业务并发与部署方式 | 不直接沿用此处单条在途请求的倍率 |

建议补齐的最小组合矩阵如下，所有格子统一用恢复后权重、真实输入和相同并发策略：

| Chunk | MTP OFF | MTP ON |
|---|---|---|
| 8K | 9 月 28 日已有四请求注入基线 | **需要恢复后同条件补测**；9 月 26 日探索仅作历史参考 |
| 1K | 9 月 28 日已有四请求注入基线 | **需要恢复后同条件补测** |

16K 的现有 MTP ON/OFF 可以作为另一条参考线，但不能填入上面 8K/1K 的格子。**不要将“8K 的 Prefill 速率约为 1K 的 1.73×”与“MTP 的 3.15× 输出吞吐”相乘**；阶段、负载、并发与分母都不同。

若后续执行补测，应优先做：同节点或交换节点的 OFF/ON 交叉重复；固定一组代表性真实 prompt 并验收输出；让 32K 目标输出真正完成；逐步提高并发并同时记录 TTFT、TPOT、E2E、KV 使用和错误/回收。若改变 mixed chunk 或 Graph，应将其列为单独变量，再评估 1K 是否出现新的价值。

## 8. 本次复核内容与追溯入口

本次重读 **8 条 Chunk、83 条 MTP** 的压缩原始 SSE，并对照请求结果和保存的最终输出 ID，核对实际输出数、HTTP 状态、缓存命中、计时、完成状态。Chunk 的三次注入窗口最大 ITL 从输出到达序列复算；MTP 的窗口输出按截止时刻重新计数，延迟按同一有效样本规则复算，并检查各请求接受直方图的总数与加权和。主要指标与归档汇总一致；另复核完整配对请求的 E2E 与输出不一致数量。

| 追溯内容 | 文件入口 |
|---|---|
| Chunk 逐请求与阶段用时 | [requests.csv](../chunk_redo_20260928/results/20260928_043035/analysis/requests.csv) |
| Chunk 注入停顿 | [interference.csv](../chunk_redo_20260928/results/20260928_043035/analysis/interference.csv) |
| Chunk 原始流与真正最终输出 | [8K 请求目录](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_8192/bench/requests/)、[1K 请求目录](../chunk_redo_20260928/results/20260928_043035/node_23/chunk_1024/bench/requests/) |
| MTP 窗口统计和逐请求计数 | [all_requests.csv](../mtp_full_20260928/results/20260928_061159/analysis/all_requests.csv)、[analysis 目录](../mtp_full_20260928/results/20260928_061159/analysis/) |
| MTP 三场景原始测量 | [decode](../mtp_full_20260928/results/20260928_061159/decode/)、[mixed](../mtp_full_20260928/results/20260928_061159/mixed/)、[prefix](../mtp_full_20260928/results/20260928_061159/prefix/) |
| 配对结果与输出重复形态 | [paired_requests.csv](../mtp_full_20260928/results/20260928_061159/analysis/paired_requests.csv)、[output_patterns.json](../mtp_full_20260928/results/20260928_061159/analysis/output_patterns.json) |
| 原实验完整性审计 | [Chunk audit](../chunk_redo_20260928/results/20260928_043035/analysis/integrity_audit.json)、[MTP audit](../mtp_full_20260928/results/20260928_061159/analysis/integrity_audit.json) |

**原始数据的一个重要陷阱：**Chunk 采集器早期的 `output_ids.json` 拼接了累计 SSE 数组，长度 45,150 不是实际生成量。应使用从最后一帧恢复的 **`output_ids_final.json`（300 个 ID）**；本次核查也采用该口径。MTP 则使用 `output_ids_final_or_partial.json` 并区分窗口前后到达。说明见 [Chunk 原实验报告](../chunk_redo_20260928/results/20260928_043035/README.md)。

下面的代码可在本机只读复算本文主要均值与倍率，不会重新运行压测：

```python
from pathlib import Path
import csv
import json
from statistics import mean

root = Path('/home/chengyetai/workspace/current_exp')
chunk = root / 'chunk_redo_20260928/results/20260928_043035'
mtp = root / 'mtp_full_20260928/results/20260928_061159'

with (chunk / 'analysis/requests.csv').open() as f:
    rows = list(csv.DictReader(f))
for key in ('server_prefill_s', 'ttft_s', 'tpot_ms', 'e2e_s'):
    a = mean(float(r[key]) for r in rows if r['chunk'] == '8192')
    b = mean(float(r[key]) for r in rows if r['chunk'] == '1024')
    print(key, '8K=', a, '1K=', b, '8K 降幅=', 1 - a / b)

for scene in ('decode', 'mixed', 'prefix'):
    summary = json.loads((mtp / 'analysis' / f'{scene}_summary.json').read_text())
    off, on = summary['off'], summary['on']
    print(scene, '窗口吞吐倍率=', on['observed_tokens'] / off['observed_tokens'],
          'TPOT 降幅=', 1 - on['mean_observed_tpot_ms'] / off['mean_observed_tpot_ms'],
          'ON 接受长度=', on['weighted_accept_length'])

with (mtp / 'analysis/paired_requests.csv').open() as f:
    pairs = list(csv.DictReader(f))
for scene in ('mixed', 'prefix'):
    rows = [r for r in pairs if r['scenario'] == scene and r['both_complete'] == 'True']
    off = mean(float(r['off_e2e_s']) for r in rows)
    on = mean(float(r['on_e2e_s']) for r in rows)
    print(scene, 'E2E 均值 OFF/ON=', off, on, '均值之比=', off / on,
          '逐对倍率均值=', mean(float(r['complete_pair_e2e_speedup']) for r in rows))
```

本报告中的“服务实际配置”均指**实验归档当时**的快照；后续目录另有 PD/EPLB 工作，历史报告的“服务仍运行”不构成当前在线状态证明。
