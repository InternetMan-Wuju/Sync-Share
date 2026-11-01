# Chunk Size 与 MTP 推理实验：概念、源码线索和测量方案

> 整理日期：2026-09-24  
> 范围：先解释任务术语，再结合 `/home/chengyetai/workspace` 中现有脚本、补丁、实验记录和 SGLang / 论文资料，给出一份能据此落地的实验设计。本文没有启动服务或运行压测。

## 0. 先把任务拆成两件事

这段任务最合理的拆法是两条实验线：

1. **Chunk Size 单机对照**：比较 `--chunked-prefill-size=8192` 和 `1024`（8K 对 1K），观察单机混合服务中长 Prefill 对 Decode 的干扰，以及长上下文下 TTFT、TPOT、吞吐和显存的变化。给定的 `32K / 输出 300 / 缓存命中 60%` 看起来是这条线的长上下文工作负载。
2. **MTP / 推测解码对照**：同一硬件、模型和请求集下比较关闭与开启 MTP，记录接受长度、接受率、吞吐、TTFT、TPOT。`1K/32K` 和 `8K/1K` 很可能是“输入 token / 输出 token”两种长度组合；`32K / 输出 300 / 缓存命中 60%` 则可能是第三个长上下文缓存场景。

原任务中的斜线没有定义，第二条的长度组合和 32K 前缀不能直接合并成同一个请求：若总输入只有 1K 或 8K，里面不可能含有 32K 前缀。第 3 节列出了需要在正式跑数前固定的解释。

## 1. Chunk Size 是什么

### 1.1 一句话

模型先对输入上下文做 **Prefill（预填充）**，再逐步生成 **Decode（解码）**。`chunked-prefill-size` 是把一次很长的 Prefill 拆成多少 token 一块来计算的上限。比如输入 32K、Chunk Size 8K 时，理论上约分成 4 块；Chunk Size 1K 时，约分成 32 块（这里先假设没有前缀缓存）。它不改变请求的上下文长度，也不是 KV Cache 的分页/block 大小。

SGLang 参数说明把 `chunked_prefill_size` 描述为按指定大小分块执行 Prefill；`max_prefill_tokens` 则是一个 Prefill batch 可接收的 token 预算，是另一个限制。参数名相近，但含义不同。较大的块通常更利于 Prefill 效率，也会提高瞬时显存需求；最终效果受 SGLang 版本、调度策略、硬件后端和并发构成影响。[SGLang server arguments](https://docs.sglang.com.cn/backend/server_arguments.html)

### 1.2 为什么 Chunk Size 会影响 Prefill 和 Decode

Prefill 一次处理很多输入 token，计算密集；Decode 每个活跃请求每轮只生成少数 token，通常对每轮响应间隔更敏感。单机同时有长 Prefill 和活跃 Decode 时，大块 Prefill 会让单轮工作变长，Decode 请求可能要等这一轮结束；分块让调度器有机会在 Prefill 块之间安排 Decode，降低长 Prefill 对 Decode 的阻塞。

Sarathi / Sarathi-Serve 论文研究的正是这类 Prefill-Decode 调度：切分 Prefill，再把 Prefill 块与 Decode 批次组合，目标是在保留吞吐的同时控制 Decode 延迟。[SARATHI](https://arxiv.org/abs/2308.16369)；[Sarathi-Serve](https://arxiv.org/abs/2403.02310)

可先用下面的方向理解，但不要把它当作结果保证：

| Chunk Size | 常见收益方向 | 常见代价方向 |
|---|---|---|
| 8K | Prefill 块较大、分块轮数较少，通常更容易保持 Prefill 效率 | 单块计算和瞬时显存压力较大；Decode 等待单块结束的时间可能更长 |
| 1K | 块更短，调度器有更多机会穿插 Decode，可能改善 TPOT 尾延迟 | Prefill 分成更多轮，调度/launch 等固定开销占比可能增加，Prefill 吞吐或长请求 TTFT 可能变差 |

这里比较的是 **chunk 大小对调度粒度的影响**。SGLang 是否确实在块之间穿插 Decode，还取决于当前版本的 scheduler、mixed-chunk 配置、Prefill/Decode 是否拆分部署以及设备后端。只看 `chunked-prefill-size` 的值，不能推断实测必然出现多少次交替。当前 Ascend 专用调优页把 Chunked Prefill 与 speculative decoding 的组合标为“部分兼容”；因此建议先分别测 Chunk Size 和 MTP，不能把二者同时打开后把结果归因给其中一个。

### 1.3 不能混为一谈的三个参数

- `--chunked-prefill-size`：每个 Prefill chunk 的 token 上限。
- `--max-prefill-tokens`：Prefill batch 的 token 预算，影响一个 batch 最多收多少 Prefill 工作。
- `--context-length`：模型允许的最大上下文长度。

Chunk Size 设为 1K 不代表模型只能处理 1K 上下文；32K 上下文可以分多轮处理。反过来，Chunk Size 设为 32K 也不会自动给模型增加 KV Cache 容量。

## 2. 前缀缓存 60% 和“32K 前缀”如何理解

### 2.1 两种含义，负载完全不同

“前缀 32K / 缓存命中 60%”至少有两种解释：

- **解释 A：总输入长度为 32K，其中 60% token 命中**。则约 19.2K token 从缓存复用，约 12.8K token 需要新算 Prefill。
- **解释 B：可复用的公共前缀本身长 32K，且它占输入的 60%**。则总输入至少约 53.3K，缓存部分约 32K，未命中部分约 21.3K。

如果“60% 命中”指的是 **60% 请求命中**，而非 60% 输入 token 命中，则上面的 token 数计算不成立。正式实验要写清楚命中率的分母。

建议统一定义为 token 口径：

```text
实际 prefix hit ratio = 所有请求的 cached_tokens 总和 / 所有请求的 prompt_tokens 总和
```

并把 `prompt_tokens`、`cached_tokens`、`uncached_tokens` 的总和及均值一起保留。请求集预热后仍要用服务端返回的 `cached_tokens` 验证实际命中，而不能只把压测参数 `prefix_ratio=0.6` 当作实测命中率。

### 2.2 Chunk 轮数示例

若输入总长是 32,768 token，实际缓存命中率为 60%，则新 Prefill 工作约为 13,108 token：

- Chunk=8,192：约 `ceil(13,108 / 8,192) = 2` 块。
- Chunk=1,024：约 `ceil(13,108 / 1,024) = 13` 块。

精确数量会受 SGLang 的 page/block 对齐、缓存匹配边界、调度 token budget 和实现版本影响。这里的数字只是帮助理解工作量，不代表 scheduler 一定按这个数字产生完全独立的 forward。

工作区的 [`bench_pd_prefix.py`](../../common/bench_pd_prefix.py) 中，`--prefix-ratio` 是输入 token 比例：脚本按 `input_tokens * prefix_ratio` 构造公共前缀，并先发预热请求。比如 `--input-tokens 32768 --prefix-ratio 0.6` 的构造目标接近解释 A。该脚本并不会自动证明服务端实际命中率恰好是 60%。

## 3. MTP / 推测解码是什么

### 3.1 核心机制

普通自回归解码一次由目标模型确定一个下一个 token，再继续下一轮。推测解码让一个较便宜的草稿预测器先提出若干 token，再由目标模型并行验证这段候选；目标模型接受相符的连续前缀，遇到不符时按目标模型规则修正并继续。这样有机会用一次目标模型验证前进多个输出 token，减少目标模型逐 token 串行执行次数。

经典推测解码算法的目标是在不改变目标模型输出分布的条件下加速生成；具体能否保持该性质，要看使用的算法及采样/验证实现。[Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)

**MTP（Multi-Token Prediction）** 有两层意思，容易混淆：

1. **训练目标 / 模型结构**：训练模型预测未来多个 token，模型检查点可能包含额外预测头。
2. **推理期推测解码**：服务框架调用这些头先生成 draft tokens，再由目标模型验证。

模型“带 MTP 权重”不等于服务已经开启 MTP，也不等于当前推理后端能成功加载草稿头。DeepSeek-V3 报告说明其 MTP 训练目标也可用于推理期 speculative decoding，但论文给出的接受率和加速数字属于 DeepSeek-V3 的特定模型、硬件和工作负载，不能直接当作 GLM-5.2 / 910C 的预期成绩。[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### 3.2 工作区里的 `NEXTN`

工作区启动脚本用的是 SGLang 的 `--speculative-algorithm NEXTN`。截至 2026-09-24 可查到的 SGLang Ascend 调参文档中，`NEXTN`（也称 `EAGLE`）使用模型内置 MTP heads，不需要另一个独立 draft model；文档建议 Prefill 侧从较小步数开始（参考 1 step / 2 draft tokens），Decode 侧可从 3 steps / 4 draft tokens 开始测试。[SGLang Ascend parameter tuning](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning.mdx)

本地 [`serve_pd.sh`](../../common/serve_pd.sh) 也写有类似参数，但默认 `SPEC=off`：

- Prefill 开启时：`NEXTN`、1 step、2 draft tokens。
- Decode 开启时：`NEXTN`、3 steps、4 draft tokens。

这是 **PD 分离服务** 的配置分支。单机统一服务只有一套 scheduler/服务参数时，是否能分别为 Prefill 和 Decode 设不同推测步数，要以实际运行版本支持的参数为准；不能直接把 PD 脚本里的两组值同时套进单机命令。

### 3.3 接受长度和接受率怎么读

SGLang 当前源码将 `spec_accept_length` 定义为“每次验证平均接受的 draft tokens **加上 bonus token**”；`spec_accept_rate` 则是“接受的 draft tokens / 提出的 draft tokens”，**不包含 bonus token**。[SGLang `metrics_collector.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py)

可按以下方式报告：

```text
平均接受长度（含 bonus）= 输出 token 总数 / verify 次数
draft 接受率（不含 bonus）= 接受的 draft token 总数 / 提出的 draft token 总数
```

接受长度为 1 不表示“模型没有生成 token”；它通常表示该次验证没有接受 draft，仍由目标模型输出 bonus/纠正 token。报告时一定注明是否含 bonus。还要同时报告直方图（每个 verify step 接受 0、1、2… 个 draft 的次数），避免只有一个均值掩盖长尾。

**接受率高不保证端到端一定更快**：草稿计算、验证 forward、同步、图执行、负载大小和 NPU kernel 都有成本。因此实验必须同时看接受指标和实际 output tok/s、TPOT；不能拿接受率替代吞吐。[SGLang metrics source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py)

## 4. 对任务中几组数字的谨慎解释

建议暂时按下表理解，并在生成正式请求集之前确认：

| 原文片段 | 最可能的含义 | 需要澄清的点 |
|---|---|---|
| Chunk Size：`8K/1K` | 两个 Chunk Size 配置：8192 与 1024 token | 是否另加 `-1`（关闭 chunked prefill）作为基线；单机是 1 个 16×910C 节点还是其他卡数 |
| `32K / 输出 300 / 命中 60%` | 长上下文缓存工作负载；很可能总输入 32K、输出 300、token 级缓存命中目标 60% | “32K”是总 prompt 长度还是缓存前缀本身；60% 是 token 比例还是请求比例 |
| MTP：`1K/32K` | 很可能输入 1K、输出 32K，偏 Decode-heavy | 如果斜线代表别的量（例如并发），必须改写成具名参数 |
| MTP：`8K/1K` | 很可能输入 8K、输出 1K | 同上；若与 32K 前缀配在一起则逻辑冲突 |
| MTP：`32K / 输出 300 / 命中 60%` | 建议作为第三个、单独的长上下文缓存 case | 若总输入是 32K，不能再称“32K 缓存前缀”；需明确前缀 token 数和总输入 token 数 |

当它们被解释为 ISL/OSL（Input Sequence Length / Output Sequence Length）时，`1K/32K` 很适合观察 MTP 对长 Decode 的影响；`8K/1K` 更像短输出混合请求，Prefill/首 token 开销在整体时延中占比更高。二者都不应隐式混入 32K cache prefix case。

## 5. 建议的实验矩阵

### 5.1 Chunk Size：单机 Prefill/Decode 干扰

建议在 **单机统一服务** 上测，而不是沿用 2P2D 脚本：PD 分离把 Prefill 和 Decode 放在不同 worker 上，会改变本实验要观测的资源争用。按 workspace 现有集群，本文暂按“一台 16×910C 主机”理解单机；如果实际是单卡/其他卡数，应在报告标题和命令中明确。

固定项：模型/权重、镜像 digest、SGLang 版本、TP/DP、采样参数、KV Cache 容量、prompt token 总量、真实缓存命中率、并发/到达节奏、Decode 请求数、输出长度。Chunk Size 对照至少两组：`1024` 与 `8192`；若配置兼容且显存允许，可额外跑 `-1`（禁用 chunked prefill）作参照。先确认当前镜像对 chunk/page size 的合法性和 NPU 后端限制。

每组做三个子场景，分清性能归因：

1. **Decode-only**：建立没有新长 Prefill 到达时的 Decode TPOT / output tok/s 基线。
2. **Prefill-only**：测 32K 输入、输出 300 的 TTFT / Prefill throughput，分别记录 cache cold 与预热后的实际 cached tokens。
3. **Mixed**：先启动持续 Decode 请求，再在 Decode 运行期间发长 Prefill；对比 Decode-only 与 Mixed 的 TPOT P50/P95/P99。这个对照才能直接回答“长 Prefill 对 Decode 干扰多大”。

如果关注命中缓存的长上下文，主 case 可按“总输入 32K、60% token 命中、输出 300”构造；同时加一个 cache-cold case，才能分辨收益来自 chunking 还是 prefix cache。记录调度队列和 KV 容量，防止某个配置因 OOM、请求排队或 cache eviction 而被误判为 chunk 性能差。

### 5.2 MTP：开关对照与工作负载

先做很小的兼容性冒烟：确认模型 MTP head 被加载、服务能启动、完成短请求、返回非空 verify/accept 指标，再进入矩阵。正式对照至少包括：

| Case | 输入 | 输出 | 主要观察对象 |
|---|---:|---:|---|
| A | 1K | 32K | Decode-heavy：接受长度变化能否转化成生成吞吐、TPOT 改善 |
| B | 8K | 1K | Prefill 和短 Decode 混合；观察 TTFT 和总体收益是否被短输出稀释 |
| C（单独） | 总输入 32K 或更长，按定义缓存 60% | 300 | 长上下文 + prefix cache 对 TTFT、TPOT、接受指标和吞吐的共同影响 |

每个 case 用同一请求集分别跑 `SPEC=off` 与 `NEXTN`。保持 Chunk Size 固定（选定实验值），避免把 chunking 和 MTP 两个变化混在一个 A/B 里。SGLang 官方调参文档给出的 Prefill/Decode 参数可作为初始值，但它不证明我们本地 GLM-5.2 的 W8A8 MTP 草稿层当前可以加载；参数必须按单机服务实际支持情况先验证。

正式矩阵建议至少重复 3 轮，打乱 OFF/ON 先后顺序，保证同一组 warmup、cache 状态和采样设置。报告中同时保留每轮原始结果和中位数，不要只保留最好一次。

## 6. 指标和计算口径

| 要回答的问题 | 指标 | 推荐报告 |
|---|---|---|
| 请求首 token 等多久 | TTFT | P50 / P90 / P99；分别报 cache cold/warm |
| 生成每个 token 等多久 | TPOT、ITL | P50 / P90 / P99；混合负载重点看 Decode 请求 TPOT P95/P99 |
| 服务总共生成多快 | Output throughput | 总 completion tokens / 稳态 wall time，tok/s；同时报 requests/s |
| 输入处理多快 | Prefill throughput | 实际新算（未命中缓存）prompt tokens / wall time，另报全 prompt tok/s，避免 cache 使分母含义不清 |
| MTP 有无被接受 | Accept length、accept rate、verify count、accept histogram | 明确 bonus token 口径，并以所有请求的 token/verify 总数作加权统计 |
| 缓存是否按设计命中 | cached tokens / prompt tokens | 汇总实际服务端计数；同时报告预热请求和正式请求 |
| 长上下文的 KV 代价 | KV used/available/evictable tokens、OOM/retraction | 每轮峰值和错误数；前缀缓存不等于 KV 不占空间 |
| Chunk 是否缓解 Decode 干扰 | `TPOT_mixed / TPOT_decode-only - 1` | 用同一批 Decode 请求比较 1K/8K，并报尾延迟变化 |

SGLang 当前 `bench_serving` 文档说明它可报告 TTFT、ITL、按 `(latency - ttft)/(tokens - 1)` 计算的 TPOT，以及 SGLang 可用时的 accept length；还可输出逐请求 JSONL。[SGLang bench_serving guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx) 当前 main 分支的 `spec_accept_length` / `spec_accept_rate` 是 `mostrecent` Gauge，反映最近 batch 的统计，不是整轮累计值；仅在压测前后各 scrape 一次不能得到完整实验的接受率。优先从每请求 `meta_info` / bench JSONL 聚合，或者在服务端逐 batch 累加 accepted/proposed/verify 次数。目标镜像里实际指标类型也要核对，不能假设和当前 main 相同。

本地 [`bench_pd_prefix.py`](../../common/bench_pd_prefix.py) 已经用 `(E2E - TTFT)/(output_tokens - 1)` 计算平均 TPOT，也支持随机 token、共享前缀和预热；不过它只解析 `prompt_tokens` / `completion_tokens`，没有把 `spec_accept_length`、接受直方图和服务端 `cached_tokens` 写进结果，也没有构造“持续 Decode 期间插入一条长 Prefill”的混合到达节奏。它适合做缓存前缀的起点，但需要补充指标采集或改用当前镜像兼容的 SGLang `bench_serving`，才能完整回答本任务。

## 7. 结合当前 workspace 的实际状态

### 7.1 现有基线和能复用的部分

- `common/serve_pd.sh` 的启动命令包含 `--chunked-prefill-size`：Prefill 默认 `PF_CPS=16384`，Decode 当前也固定 `16384`；它是 **PD 分离**启动器，不是单机统一服务启动脚本。其 `SPEC` 默认关闭；开启时 Prefill 使用 1 step / 2 draft tokens，Decode 使用 3 steps / 4 draft tokens。
- `common/bench_pd_prefix.py` 可复用随机 `input_ids`、共享前缀、前缀预热和并发压测逻辑；当前默认值是 16K 输入 / 1024 输出 / 前缀比例 90%，不是本任务所需的 32K / 300 / 60%。
- `history_exp/doc/历史总结_2026-09-20_模型交接.md` 记载的旧 PD 正式基线是 16K/1024、MTP 关闭；该文档保存时记录的 chunk 值是 Prefill 4096、Decode 512。它与 2026-09-22 修改过的 `common/serve_pd.sh` 默认 16384 不同，应以具体实验启动命令和镜像日志为准。
- `current_exp/doc/` 在本次梳理时为空；`history_exp/current_exp/` 是历史资料区，不是当前 `current_exp` 的现行代码副本。

### 7.2 MTP 的本地阻塞项

[`fixed_tests/test1_evalscope_multiturn/result.md`](../../history_exp/glm52_ep64_chengyetai/fixed_tests/test1_evalscope_multiturn/result.md) 记录过一次 GLM-5.2 W8A8 / SGLang / 910C 测试：`NEXTN` 启动因 ModelSlim 不支持 MTP 草稿层的 `W13='FLOAT', W2='FLOAT'` 而失败，所以该轮结果标题和数据都明确为“无 MTP”。这不是“当前所有镜像、所有量化模型都不支持”的证明，但它意味着 **本任务不能把开启参数当作兼容性已解决**；需要先在准备使用的 pinned image + 权重上复现或排除这个问题。

旧 EP64 启动器带 `INSTRUMENT=1` 时可以挂载 [`tokenizer_manager_instrumented.py`](../../history_exp/glm52_ep64_chengyetai/patches/tokenizer_manager_instrumented.py)，其中已有 `spec_accept_rate`、`spec_accept_length`、verify count 和接受直方图计算；PR #32209 补丁副本里也明确区分了“接受 draft token（不含 bonus）”和“verify 接受长度（含 bonus）”。但当前 `common/serve_pd.sh` 没有挂载这份 tokenizer instrumentation，且旧 patch 是否与当前 `fe694986` 镜像兼容尚未确认。

### 7.3 脚本路径过时，不能把现有启动器原样视作可执行方案

本次只读检查到：

- `workspace/current_exp/` 现在只有 `doc/`；补丁实际在 `history_exp/current_exp/patches/`。
- `common/serve_pd.sh` 的默认 `PR32209_ROOT`、`EPLB_PATCH_ROOT`、`FULL_WINDOW_PATCH_ROOT` 指向 `/home/chengyetai/workspace/current_exp/patches/...`；这些目标目录当前不存在。
- `common/next_experiment.sh` 也默认使用 `/home/chengyetai/workspace/current_exp` 和 `results_final/` 等路径，而当前 `current_exp` 下没有相应补丁/结果目录。

另一个文档时效问题：通用 [SGLang server arguments](https://docs.sglang.com.cn/backend/server_arguments.html) 页面显示 2025-05-15 更新，其 speculative algorithm 列表只写 EAGLE/EAGLE3，并给出与 chunked prefill / Radix cache 的通用兼容提示；当前 Ascend 专用调优页则描述了 `NEXTN` 内置 MTP head，并将 NEXTN + prefix cache 记为兼容、NEXTN + chunked prefill 记为部分兼容。两页内容面向的版本/后端不同，不能用较旧的通用说明替代本机镜像验证。

所以开始实验前需要为本任务准备真正的单机启动命令/脚本，并确认挂载路径、模型路径、镜像 digest 和 `/help` 中参数。本文没有改动这些运行脚本，也没有对远端服务状态作判断。

## 8. 推荐执行顺序

1. **先写清 workload 定义**：总输入长度、公共缓存前缀长度、缓存命中率分母、输出长度、并发、请求到达方式、单机卡数。
2. **确认现行环境**：镜像 digest、SGLang 版本、模型 config 中 MTP 层、单机可用 KV token 数、单机统一服务启动命令。
3. **Chunk Size 冒烟**：先固定无 MTP；跑 1K 和 8K，检查请求成功率、KV 空间、实际 cached tokens、Prefill/Decode 指标是否采到。
4. **Chunk 干扰矩阵**：Decode-only、Prefill-only、Mixed 分开测；选同一长 Prefill 插入 Decode 流的区间对比 TPOT P95/P99。
5. **MTP 兼容性冒烟**：单独开 NEXTN，确认 MTP head 成功加载、输出正常、verify/accept 指标非空。若复现 FLOAT/ModelSlim 错误，先停在兼容性问题定位，不跑正式吞吐矩阵。
6. **MTP OFF/ON 正式对照**：按 `1K/32K`、`8K/1K` 和单独的长上下文缓存 case 测，固定 Chunk Size 和其余部署参数。
7. **归档原始数据**：每次保存启动日志、服务 `/metrics`、每请求 JSONL、MTP 接受直方图、实际缓存 token 统计、最终汇总和镜像/模型版本。

## 9. 参考资料

### SGLang 文档与源码

- [SGLang Ascend Parameter Tuning（当前 main 分支）](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning.mdx) — NEXTN/MTP 的关系和 Ascend 初始参数建议。文档会随 main 分支更新；本地镜像与其不是同一版本。
- [SGLang Server Arguments](https://docs.sglang.com.cn/backend/server_arguments.html) — `chunked_prefill_size` 与 `max_prefill_tokens` 参数定义。
- [SGLang Bench Serving Guide（当前 main 分支）](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx) — TTFT、ITL、TPOT、accept length 和逐请求 JSONL。
- [SGLang metrics_collector.py（当前 main 分支）](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py) — `spec_accept_length` 与 `spec_accept_rate` 的服务端定义。

### 论文

- [Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding (2023)](https://arxiv.org/abs/2211.17192) — 推测解码与保持目标分布的基本方法。
- [DeepSeek-AI. DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) — MTP 训练目标和将 MTP 用于推理期推测解码的说明；其中模型特定性能数字不可直接移植到本集群。
- [Agrawal et al. SARATHI (2023)](https://arxiv.org/abs/2308.16369) — Chunked Prefill 与 Decode-maximal batching。
- [Agrawal et al. Sarathi-Serve (2024)](https://www.usenix.org/conference/osdi24/presentation/agrawal) — 用 chunked prefill 缓和吞吐与尾延迟的折中。

## 10. 2026-09-25 实时节点与单机容量补充

### 10.1 节点状态快照

只读检查时间：**2026-09-25 11:09–11:11 CST**。

| 节点 | 观察到的状态 | 结论 |
|---|---|---|
| 23 | `atp-epcal-dense-20260924-r1-ep-tp16` 中正在跑 SGLang 和 `/study/measure.py`；16 个 die 的 HBM 约 57,192–57,499 / 65,536 MiB，AICore 约 51–84%。SGLang 命令为 TP16/EP16、Chunk=8192、`--max-prefill-tokens 16384`、context 103424、`--mem-fraction-static 0.8`，且 `--disable-radix-cache`；命令行没有 speculative/NEXTN 参数。 | **不空闲，已有活跃计算。** 当前运行的模型路径是 `/model`，仅凭路径不能确认它就是本任务的 GLM-5.2 权重。它也没有开启本任务所需的 prefix cache 或 MTP。 |
| 25 | 有一个运行 5 天的 `sglangmain_npu_cryang` 容器；容器进程列表只有 bash，die 的 AICore 为 0%，HBM 约 2.9–3.1 GiB / 64 GiB。 | NPU 快照看起来没有加载模型，但容器仍在线，不能仅凭低负载认定可覆盖使用。 |
| 31 | `sglang0520_npu_cryang` 容器进程列表主要是 bash/tmux；另有运行中的 `ack_test` 容器。NPU AICore 为 0%，HBM 约 2.9–3.1 GiB / 64 GiB。 | NPU 当前低负载，但节点和容器并非完全空闲。 |
| 32 | `sglang0520_npu_cryang` 容器进程列表只有 bash；NPU AICore 为 0%，HBM 约 2.9–3.1 GiB / 64 GiB。 | NPU 当前低负载，但容器仍在线，使用前要确认归属。 |

因此，若要立刻准备单节点，**23 不可作为空闲节点**；25 从 NPU 使用量上最像候选，但它现有容器的使用者仍需确认。此处只是状态快照，不代表资源已获分配。

### 10.2 “能装下权重”不等于“能跑目标 workload”

SGLang 当前 GLM-5.2 Ascend 部署页称，官方 **不带 MTP 的 W8A8 checkpoint** 约 774.08 GB，可部署在 16×64GB 设备上，也就是一个有 16 个 die 的 A3 节点、TP16。该页同时说明这是最低建议配置；长上下文和目标并发仍要看 KV Cache 空间。[GLM-5.2 on Ascend](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2.mdx)

这不能直接证明本地 `/mnt/raid/user/data/models/glm-qq` 在 MTP 开启时也能按同样方式运行：本地固定测试的 checkpoint/ModelSlim 组合曾因 MTP 草稿层 `W13='FLOAT', W2='FLOAT'` 加载失败（见 §7.2）。另外，本地 [`问题总结.md`](../../history_exp/current_exp/问题总结.md) 报告过一份 TP16/DP4 单机 prefill 配置在 `MEM=0.90` 下只剩约 400 MB KV 余量、16K 输入 OOM。该历史结果与官方无 MTP checkpoint、运行镜像和实际并行配置并非同一条件，不能据此断言所有 TP16 配置都必然 OOM；但它足以说明权重可以加载与 32K 输入、缓存前缀、高并发、MTP 同时可运行是不同的容量问题。

**当前判断：**一台 16-die 节点有机会装下 GLM-5.2 W8A8 权重，官方文档明确支持其指定的无 MTP checkpoint；但本任务目标若是本地 `glm-qq` + MTP + 32K 上下文 + 60% prefix cache，还没有可直接引用的单机成功记录。必须先针对准确的模型目录和镜像做一次低并发容量冒烟，记录启动后的可用 KV token 数、单请求 32K 实际 cache 命中和输出 300 是否完成，再决定正式并发。若需要 MTP，先确认草稿层加载问题已经解决。
