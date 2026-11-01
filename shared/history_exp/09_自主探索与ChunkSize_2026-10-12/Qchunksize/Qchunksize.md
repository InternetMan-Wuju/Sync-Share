# 任务 2：Chunk Size 单机实验方案

> 当前任务（2026-10-13）：用户已收窄为只补 **8K/1K、客户端并发32、64请求** 的四档ChunkSize；前缀32K/300/60%组沿用已测结果，不再重跑，W2/W3 Chunk不测。[本轮计划、实际配置和结果](retest_20261013/ChunkSize_W1并发32补测.md)。下文保留前期预实验记录。

> 2026-10-13核对更正：**W1正式要求是客户端并发32、64请求，本次自主探索未完成这一要求。** 下文并发2/4的网格是已经执行的小规模预实验，不能替代正式W1，也不能将4K/8K候选结论直接推广到C32。原因与配置偏离见[要求核对](ChunkSize要求与自主探索偏差.md)。

W1正式对照应按AboutModels的W1基线固定客户端并发32、64请求和其他配置，仅改变ChunkSize。该基线为TP16/DP2/EP1、DP attention/LM head开启、DeepEP关闭、chunk8192；本次预实验的DP1/EP16/DeepEP auto不属于同一配置。

> 2026-10-12夜间自主预实验记录：仅使用23节点，公开量化源SHA256校验的只读恢复权重、SGLang原生random-ids。容器只停止、保留；报整机总吞吐。实际小网格为chunk 1024/4096/8192/16384，8K/1K并发2/4、32K/1K并发1、32K/300/60%前缀并发1/2；这是执行记录，不是正式要求的替代。

**文件夹**：`current_exp/Qchunksize/`（脚本、数据集、结果都放这里）
**文档要求来源**：`docs/AboutModels.md`（workload 口径）、`docs/ChunkSize_ref/TODO.md`（任务与历史）
**历史实验**：`history_exp/03_ChunkSize单机_2026-09/`、`history_exp/90_文档归档/分析报告_2026-09/ChunkSize*.md`

---

## 0. 一句话

**固定几个并发档，在每个并发下改 chunk size，做成一张"不同并发该选哪个 chunk size"的表**。
（老板当场纠正过方法：**不要自己控制请求下发时机**，直接设并发 + 改 chunk size 即可。）

---

## 1. 文档明确要求

- 任务（`TODO.md`）：
  > **Chunk Size 的选取对性能的影响**@程业泰 —— 选定不同大小 chunksize 然后在对应 workload 测
  > **Chunk Size（单机）**（8k/1k，前缀 32K / 输出 300 / 缓存命中 60%）—— 对 **Prefill/Decode 干扰**和长上下文影响很大
- Workload 口径（`AboutModels.md`）：
  | # | 场景 | 输入 | 输出 | 请求数 |
  |---|---|---|---|---|
  | 1 | 8k/1k | 8192 | 1024 | 并发 × 2 |
  | 2 | 32k/1k | 32768 | 1024 | = 并发 |
  | 3 | 1k/32k | 1024 | 32768 | = 并发 |
  | 4 | 前缀 32K / 输出 300 / 缓存命中 60% | 32768（含共享前缀） | 300 | 并发 × 2 |
- 要报的指标：TTFT / TPOT / E2E（p50/p95）+ **整套服务总输出吞吐**。保留请求时间戳和服务器阶段跨度；不用 ITL，不除以 NPU。

## 2. 已执行预实验的变量（W1正式并发32未覆盖）

| 变量 | 取值（建议） | 说明 |
|---|---|---|
| `--chunked-prefill-size` | 本轮 **1024 / 4096 / 8192 / 16384** | 固定 max-prefill=32768 |
| 并发 | 8K/1K：**2 / 4**；32K/1K：**1**；60%前缀：**1 / 2** | 每个并发档都扫描所有 chunk |
| 场景 | **8K/1K、32K/1K、前缀32K/300/60%** | 1K/32K以decode为主，此轮不重复小时级长输出 |
| 其他 | 固定：并行/显存/DeepEP 等**除 chunk 外全不变** | 一次只动一个变量 |

产出：**一张表**（行 = chunk，列 = 并发；值 = 吞吐/TPOT/TTFT），并在每个并发下标出最优 chunk。

## 3. 机器与镜像

| 机器 | 现状 |
|---|---|
| **23** | 当前唯一使用的 NPU 节点；MTP诊断结束后顺序跑ChunkSize |
| 其他节点 | 本轮不占用 NPU、不停止其服务；早前的空闲记录不是当前授权 |

- 使用 **23**（单机16卡、TP16/EP16/DP1），保存与AboutModels默认DP2的差异。
- 镜像：`quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3`（KV 池最大）；**不要挂 v0.5.16 的 loader 补丁**（新镜像会 `ImportError`）。
- **不要开 MTP**（chunk 实验的主变量不是投机解码）。

## 4. 可复用资产（都在 `current_exp/` 下）

本轮实际使用 `autonomous_20261012/scripts/service.py`、`export_chunk_native.py`、`chunk_collect.py` 和 `overnight_runner.py`。输入由安装镜像的 `RandomDataset` 原生 random-ids 分支导出；共享前缀拼接是明确记录的扩展。下面旧资产只作参考，不能将旧独立 uniform IDs 混入本轮矩阵。

| 路径 | 用途 |
|---|---|
| `Q1MTPExp/scripts/quick_accept_test.py` | **并发客户端**：读带 `input_ids` 的数据集，逐请求收集 TTFT/E2E/输出 token + 服务端 meta（可加 `--flush-cache` / `--warmup-prefix`） |
| `Q1MTPExp/scripts/build_doc_datasets.py` | 按 AboutModels + "固定前缀随机数据集"口径造数据集（含共享前缀/命中率） |
| `Q1MTPExp/scripts/serve_mtp_q1.sh` | 单机起停（v0.5.19/0.5.16 可选、`MEM`/`CTX`/`MAXPREF`/`MRR`/`GRAPH`/`DEEPEP` 开关）→ **改 chunk 只需传 `CHUNK=`** |
| `Q1MTPExp/scripts/summarize_run.py` | 把一次结果 + KV 池汇总成一行 |
| `pd_colloc_20261011/analyze_q4exp.py` | 分析窗口/rank 负载（本任务用不到，留作参考） |

## 5. 必须注意的坑（我们这两天踩过的）

1. **KV池影响服务器驻留与调度，不等于客户端并发上限**：本次Chunk预实验server-info实测KV容量77824 token。输入+最大输出只能作单请求峰值驻留估算，不能据此把客户端并发改为2/4；客户端C32时，部分请求可排队。必须实际测C32并保存running/queued/KV及包含排队的TTFT，才能判断该配置是否可用。
2. **OOM 风险**：`mem-fraction-static 0.92` + 大 prefill（max-prefill 65536）会在 **warmup 阶段就 OOM** → 用 **0.90 + `--max-prefill-tokens 32768`** 更稳。
3. **chunk size 与 max-prefill**：`chunk ≤ max-prefill-tokens`；扫大 chunk 时记得同步放大 max-prefill，否则不生效。
4. **不要挂 v0.5.16 的 loader 补丁到新镜像**（`ImportError: maybe_precompile_model_kernels`）。
5. **Decode graph**：本镜像在 `deepep normal` 下会禁用 graph（`Cuda graph is disabled because deepep_mode=normal`）；报数时注明 graph 是否生效。
6. **落盘**：本轮新结果写到 **`/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012`**；服务和权重审计放 `autonomous_20261012/logs`。只生成差异页和压缩流，不复制整套721GiB权重；其他人的目录只读。
7. **两次起服务之间留 30~60 秒**（NPU/显存释放），否则可能出现服务起不来。
8. **缓存命中场景（前缀32K/300/60%）**：必须先 `flush_cache` 再**预热共享前缀**，命中率以服务端 `cached_tokens/prompt_tokens` 为准。设置共享前缀 19661/32768≈60.0006%；实际命中可能受页粒度影响，不能把设置长度直接当作实测命中率。

## 6. 已执行的小规模预实验（不代表正式任务完成）

```
场景：8K/1K（请求2C）、32K/1K（请求C）、前缀32K/300（请求2C）
并发：分别2/4、1、1/2
chunk：1024、4096、8192、16384
→ 共20单元；权重校验加载和每组用时按真实日志记录，不预先承诺一小时完成
先看：每个并发下最优 chunk 是哪个；再看 1K vs 8K chunk 的差距（历史上是 1.7×）
```
基础网格完成后复测基线与主要候选，差距很小时不凭单次结果宣布最优。

## 7. 报数要求（对齐团队规矩）

- 一次只改一个变量；**能超过 baseline 才记入榜单**；
- **必须解释"为什么"**（例：chunk 越小 → prefill 与 decode 交替越频繁 → decode 被打断更少，但 prefill 本身效率下降）；
- 报数注明：镜像 / 并行 / mem-fraction / max-prefill / **chunk** / 并发 / **实际 KV 池** / graph 是否生效。

## 本轮统一预热口径

所有基础和复测单元先用本场景前C个固定输入、并发C、输出16token预热，单独保存原始流；然后flush_cache。冷缓存场景直接测量，前缀场景再独立预热共享前缀。预热不计入请求数、吞吐墙钟或延迟统计；正式组仍逐请求检查缓存命中。该统一口径在首个性能单元开始前确定。
