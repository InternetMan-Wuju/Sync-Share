# Chunk Size 单机复测：四条 32K 输入错开注入，比较 8K 与 1K

实验日期：2026-09-28，中国标准时间。运行节点：23，单机 TP16；25 节点原有服务保持运行，本次数据没有使用 25 节点。**这是四请求的小规模机制验证，不是正式容量或尾延迟评测。**

## 一眼看结果

| 每组 4 请求 | Chunk 8K | Chunk 1K | 解释范围 |
|:---|---:|---:|:---|
| 服务端 Prefill 平均 | 5.646 s | 9.758 s | 从 SSE 的 `prefill_finished_time - forward_entry_time` 计算；1K 为 8K 的约 1.73 倍 |
| 客户端 TTFT 平均 | 5.976 s | 10.110 s | 从请求发送到客户端收到首个输出事件 |
| 三次后来请求注入时，已有 Decode 请求的最长 ITL | 3.100 / 3.103 / 3.170 s | 9.620 / 9.400 / 9.318 s | 原始流式事件的相邻输出时间差；每列对应 A 遇 B、B 遇 C、C 遇 D |
| 注入前对应请求的 ITL 中位数 | 0.506 / 0.475 / 0.464 s | 0.474 / 0.476 / 0.475 s | 三次注入前 Decode 约 0.5 s/输出 ID |
| 客户端 E2E 平均 | 153.205 s | 165.098 s | 仅四条，每条加入系统的时点随前一条解码变化，不应读作隔离后的单请求性能 |

![8K 与 1K 的四请求 ITL 时间线，两个子图使用同一纵轴](analysis/itl_timeline.png)

橙/绿/紫色虚线是 B/C/D 发起请求的时刻，色块延伸到后来请求的**客户端首 token**。色块包括请求传输、排队、Prefill 及首 token 交付；不是严格服务端 Prefill 时间。点的横坐标是一个 ITL 间隔的**结束**时刻，所以跨越色块末端的长停顿会显示在色块右边。8K/1K 共用纵轴，可以直接对比停顿高度。详细的 [八条逐请求数据、三次注入窗口明细](analysis/README.md)、[逐请求 CSV](analysis/requests.csv) 和 [干扰 CSV](analysis/interference.csv) 均已保存。

这组数据体现了“长上下文 Prefill 到来时，正在输出的请求会出现停顿”：相同四条输入、相同命中量和发送规则下，1K 的 Prefill 较慢，三个被注入的 Decode 请求出现了更长的单次 ITL。不能仅凭四条、先跑 8K 后跑 1K 的结果推广到所有负载，也不能把全部停顿归因于某一种核函数、IPC 或块合并通信。正常 Decode 的 ITL 并非逐 token 完全相同；A/C/D 两种配置的最终输出 ID 也不相同。

## 输入与执行顺序

- 每组先调用 `/flush_cache`（HTTP 200），再以固定的 19,661 个随机 token ID 做一次只生成 1 token 的预热；预热记录 `cached_tokens=0`。随后四条请求 A/B/C/D 每条总输入 **32,768 token ID**，前 19,661 个 ID 完全相同，后 13,107 个 ID 各不相同；两种 Chunk 重用同一份 [完整数据集](dataset.json)，种子 `2026092801`。随机 ID 是固定长度压力输入，不可按普通中文语义解读。
- 每条请求使用 `/generate` 的 `input_ids`，`max_new_tokens=300`、`temperature=0`、`ignore_eos=true`、`stream=true`。所有八条请求各有 **300 个 SSE 输出事件、300 个最终输出 ID、HTTP 200**，finish reason 为 `length`，服务器 `num_retractions=0`；每条实际 `cached_tokens=19584`，即 **59.765625%**（缓存按页对齐，与预热的 19,661 ID 不必相等）。
- A 首先发出；A 发到第 20 个输出事件且仍运行时发 B；B 的第 20 个事件后发 C；等 A 完成、C 至少 20 个事件并且仍运行时发 D。B/C/D 发出时分别与正在 Decode 的 A/B/C 相交，活跃上限为 3。原始 [8K 调度记录](node_23/chunk_8192/bench/dispatches.json) / [1K 调度记录](node_23/chunk_1024/bench/dispatches.json) 保留了每个客户端单调时钟和挂钟时间。
- 每组“30 分钟上限”是**自发 A 起的发送截止时间**，不是强制杀掉已提交的生成。8K：04:54:33 至 04:59:37，A 至 D 发出历时约 156.985 s，A 发出至全组完成约 303.792 s；1K：05:22:45 至 05:28:10，分别约 172.900 s 和 324.922 s。两组都在截止前发完四条，也都自然完成。

## 时间口径与不能直接推断的量

- `E2E = TTFT + (E2E - TTFT)`，两项都按客户端单调时钟测。`TPOT = (请求结束 - 首个输出到达) / 299`，包括客户端收到 token 的时间及最后收尾；它并非内核纯 Decode 时间。每条流式输出的全部相邻事件差构成 ITL 序列，`TPOT` 是对应整段的平均量，不是“token 数 / Decode 时间”。
- 服务端 `queue_time` 和 `prefill_finished_time - forward_entry_time` 均来自各请求最后 SSE 的 `meta_info`，由服务端挂钟产生；TTFT 来自客户端单调时钟。`TTFT - queue_time - prefill` 作为**首 token 前未归类差值**列出，不命名为 IPC。客户发请求、服务端 API 分派与调度、Prefill 至交付之间的开销都可能在其中。各阶段不应未经同步就画成精确的硬件时间账本。
- 注入窗口最大 ITL 的判定方法是“相邻输出事件跨越或落在后来请求的发送至首 token 区间”。窗口外的长 ITL 可能是跨边界的相邻间隔，因此时间线上右边相邻的长点不代表产生了另一场额外事件。原始逐 token 数组及右端时间戳见 [traces.json](analysis/traces.json)。
- 没有把 n=4 的样本 P95/P99 作为可信尾延迟结论；三次注入是相互依赖、相互影响的窗口，不是三次独立重复。不同 Chunk 的一条请求的总解码时长还受错开发请求和其他活跃请求影响，无法据此断言 Chunk 对单独运行的 Decode 有没有影响。

## 两组服务条件与有效性

23 节点先部署 8K，结束并保存快照后切换到 1K；1K 服务在实验后保持运行。两组同一模型、镜像和只读修复外挂，启动命令、模型配置摘要、权重清单 hash、各 rank 权重来源审计在各组的 `node_23/chunk_*/` 下。同一配置为 `--tp-size 16 --ep-size 16 --max-running-requests 4 --max-prefill-tokens 65536 --mem-fraction-static 0.88`，均设 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`，`enable_mixed_chunk=False`。完整启动参数见 [8K](node_23/chunk_8192/launch.sh) / [1K](node_23/chunk_1024/launch.sh)。每组生成前均实际检查“1+1”和“Hello, world.”的中文回复；原始语义检查及手工判定分别见 [8K](node_23/chunk_8192/semantic_checks/review.json) / [1K](node_23/chunk_1024/semantic_checks/review.json)。**模型文件和镜像没有被修改**。

**CUDA Graph 实际状态：关闭。** 两组启动脚本都传了公共基准的 Decode graph `full`、bs `8 16 24 32`、max bs `32`，但固定服务镜像启动时明确打印 `Cuda graph is disabled because deepep_mode=normal`，`server_args.cuda_graph_config.decode.backend=disabled`，运行日志 `npu graph: False`。所以本轮属于**两档都未真正启用 Graph**的对比；需要 Graph ON 的正式实验须先解决固定镜像的此限制，并重新检查实际日志，不能把传参当作生效。

## 原始材料、校验与复看

[8K 组完整目录](node_23/chunk_8192/) / [1K 组完整目录](node_23/chunk_1024/) 分别保存：运行 `launch.sh`、启动与推理 `server.log`、权重加载审计、生成前后 `get_server_info`、HTTP health、Prometheus metrics、容器信息、NPU 状态、语义回复和 `bench/`。`bench/` 包含 cache flush 与公共前缀预热、调度时间、每 5 秒全量 `/metrics` 抓取（每 15 秒附 NPU 采样）、四条请求各自的**原始 SSE 全帧及双时钟时间戳**、完整输入 ID、服务端 meta、最终输出 ID、请求结果与完成判定。启动前节点快照还在实验代码目录 `before/`；试验脚本及配置见上一层目录。

特别注意：采集器的 `requests/*/output_ids.json` 将每个 SSE 帧的**累计**输出数组直接拼接，因而是 45,150 个重复记录的 ID，不是“最终输出 45,150 token”。保留该原始产物以审计采集过程；**真实的 300 个最终输出 ID** 从最后一个原始 SSE 帧重建，位于每请求 `output_ids_final.json`，其摘要与帧数在 `output_integrity.json`。跨组对照：B 的最终 300 个 ID 完全一致，A/C/D 不一致，见 [逐请求结果](analysis/README.md)。每个稳定记录文件的 SHA-256 和字节数列在 [文件清单](file_inventory.sha256.tsv)；仍运行的 1K 服务正在追加的 `server.log` 除外，已经将其固定时刻的副本保存在 [运行结束快照](final_state/node23_chunk1024_server_log_snapshot.log)，并纳入文件清单。两节点的最新健康检查及 23 节点容器状态见 [服务状态快照](final_state/service_status.json)。原始八条请求的逐项校验见 [完整性审计](analysis/integrity_audit.json)。采集与后处理脚本版本写在 [实验清单](experiment_manifest.json)。
