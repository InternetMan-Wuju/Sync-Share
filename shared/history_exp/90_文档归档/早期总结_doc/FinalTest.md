# GLM-5.2 W8A8 最终测试方案（测试4：PD 分离 + 长上下文 + EPLB）

> 本文件是 `Pre_FinalTest.md` 的完整扩写版，把其中的数据集、三个待解决问题、实验要求和数据清单逐条展开成可执行的方案。
> 范围：**只看测试4（PD 分离）**，统一服务 EP32/64 只作为对照组/历史基线引用。

## 0. 一句话目标

在 **4 机 910C（每机 16 卡）** 上，用 **GLM-5.2 W8A8**，跑通 **2 prefill + 2 decode 的 PD 分离**，
用 **cc-traces（真实 agentic trace）** 复现参考测试4 的「64K/128K 输入 + 1K 输出 + 前缀缓存 90% + 并发 10/20/30」，
并且**开启 CUDA Graph**，同时产出 EP 维度性能、EPLB cost、Expert×Layer 热力图三套数据。

## 1. 参考基准（UCM 测试4 原文）

来源：<https://ucm.readthedocs.io/en/latest/user-guide/best-practices/GLM-5.1-A3_4Node_PD_Disaggregation.html>

| 维度 | 值 |
|---|---|
| 模型 | GLM-5.1 W8A8 |
| 拓扑 | 4 节点 A3，2 prefill + 2 decode（每节点 8×910C，双 die） |
| 输入 / 输出 | 64K / 1K、128K / 1K |
| TPOT 目标 | ≤ 30ms |
| 前缀缓存命中率 | 90% |
| KV 前缀多样性 | 取并发值的一半 |
| 并发 | 10 / 20 / 30 |
| 请求数 | 并发 × 4 |

对比口径只报「数值 + 与参考的差异」，**硬件不同（910C vs 参考 910C/910B 共封，模型 5.2 vs 5.1），绝对数字不做直接对标**。

## 2. 数据集（cc-traces）

### 2.1 数据形式

- HuggingFace：`semianalysisai/cc-traces-weka-062126-256k`
- 本地目录：`/mnt/raid/user/data/dataset/cc-traces-weka-062126-256k/`
- 主文件：`traces.jsonl`（约 568MB，393 条 session）

每条 session 是一段真实的 Claude Code / 多子代理对话 trace，结构：

```json
{
  "id": "...",
  "models": ["claude-haiku-4-5-...", "claude-opus-4-8"],
  "block_size": 64,
  "hash_id_scope": "local",
  "requests": [
    {"t": 0.0, "model": "...", "in": 640, "out": 20,
     "hash_ids": [0,1,2,...], "api_time": 1.979, "type": "s", "ttft": 1.741}
  ]
}
```

字段含义（对测试4 有用的）：

| 字段 | 含义 |
|---|---|
| `requests[]` | 该 session 内按时间排序的一串请求（多轮 / 多子代理并发） |
| `t` | 请求相对起始时间（秒），用于重放并发节奏 |
| `in` / `out` | 该请求的输入 / 输出 token 数（**真实 trace 值**） |
| `hash_ids[]` | 输入按 `block_size=64` 分块后每块的 hash id；**相同 hash_id = 相同内容块**，用来构造前缀缓存共享 |
| `ttft` / `api_time` | 原始 trace 的时延标注（仅参考，不作为我们服务指标） |

实测统计（前 300 条 session 抽样）：

| 指标 | 值 |
|---|---|
| 每 session 请求数 | 5 ~ 1622（均值 ~81） |
| 输入 token | P50 ≈ 129K，P90 ≈ 224K，max ≈ 256K（已做 256K cap） |
| 输出 token | P50 ≈ 436，mean ≈ 1069，max ≈ 40K |
| `hash_ids` 存在率 | ~95% |
| `hash_ids` 块数 | ≈ `in / 64` |

结论：**这个数据集的输入普遍在 100K~256K，天然适合 64K/128K 长上下文测试**，且 `hash_ids` 直接给出「哪些输入块被共享」，正好匹配测试4 的「前缀缓存命中率 90% + 前缀多样性 = 并发一半」。

### 2.2 怎么把 trace 变成可送进 GLM 的请求

数据里只有 token 数 + hash id，**没有真实文本**。重放时按 block 粒度用随机 token-id 填充：

1. 建一张 `hash_id -> 随机 token 块（64 个 token-id）` 的映射表（session 内共享）。
2. 对每个 request，按 `hash_ids[]` 依次取对应 token 块，拼成该请求的 `input_ids`。
   - 相同 hash_id 的块在所有请求里 token 完全一致 → 前缀缓存可命中。
3. `in` 超长时按目标档位（64K / 128K）截断到整块边界；`out` 用 `ignore_eos` 强制生成到目标长度（参考口径是固定 1K）。
4. 并发节奏：按 `t` 把同一 session 的请求按参考并发（10/20/30）分组/节流发送。

### 2.3 前缀缓存数据构造：`repeat_rate` + `prefix_num`

参考里的「90%」和「并发的一半」其实是**两个构造参数**，不是两个结果指标，所以不矛盾：

| 参数 | 含义 | 取值 |
|---|---|---|
| `repeat_rate` | 每条请求**前多少比例的 token 是公共前缀**（可命中缓存） | 0.9（≈90%） |
| `prefix_num` | 准备**多少种不同的公共前缀** | 并发 ÷ 2 |

以 `input_len=64K、并发=20` 为例：

```text
prefix_num = 20 / 2 = 10 种公共前缀
每种前缀 ≈ 64K × 90% = 59K token
尾部独有 ≈ 64K × 10% = 6.5K token

请求 A1 = [公共前缀A 59K] + [独有尾部1 6.5K]
请求 A2 = [公共前缀A 59K] + [独有尾部2 6.5K]
请求 B1 = [公共前缀B 59K] + [独有尾部3 6.5K]
请求 B2 = [公共前缀B 59K] + [独有尾部4 6.5K]
...
```

**关键步骤是预热**：正式压测前先用 `--prefix_test`（或等价动作）把这 `prefix_num` 种公共前缀**预先送进 KV cache**。
于是正式压测时，每个请求的 90% 前缀都直接命中缓存，真正需要重新 Prefill 的只有尾部约 10%。

所以：

- **「命中率 90%」＝每个请求里 90% 的输入 token 命中缓存**（不是「90% 的请求整体命中」）。
- **「多样性 = 并发一半」＝准备 `并发/2` 种不同的公共前缀**，每种前缀配约 2 个并发请求，模拟 Agent 同时发多条「前缀高度相似」的请求。

参考效果（64K、并发 20）：无缓存 TTFT≈36.8s → 90% 前缀缓存 TTFT≈7.85s（降 78.6%），吞吐 314→604 token/s，
但 TPOT 20.2→23.5ms 基本不变——**前缀缓存省的是 Prefill，不是 Decode**。

**映射到我们的数据**：从 cc-traces 的 `hash_ids` 提取「多个请求共享的前缀块」作为公共前缀、互不相同的尾部块作为独有尾部；
按 `prefix_num = 并发/2` 聚类出若干种前缀，压测前先预热。`repeat_rate`/实际命中率按真实 hash 重叠度报。

## 3. 三个待解决问题

### 3.1 问题0：bench3 是什么、要不要用

`/mnt/raid/user/data/tools/bench3/` 是 **别人（wzc）的公用基准包**，定位是：

- 用 `vllm bench serve` 对 GLM-5.2-W8A8 做**随机数据 21403/1024** 的 e2e 基准（TTFT/TPOT 分位 + 吞吐）。
- 支持纯端点驱动（`BASE_URL` 指向任意 sglang 兼容服务，含 PD 前端）。
- 关键旋钮：`MEM=0.65`、`CPS=16384`、`CGBS_LIST="1 2 4 8 16 32"`、`LL_CAP=64`、`MRR=32`。

**结论：本实验不直接用 bench3 当主工具**，原因是：
1. 它是 vllm 口径 + 随机 21403/1024，跟我们要的「真实 cc-traces + 前缀缓存 + PD + EPLB + 专家热力图」不匹配；
2. 它是别人的公用包，改动/占用有协作风险。

但 bench3 的 SOP 里有**三条对我们排查 CUDA Graph 卡死非常有用的经验**（见 3.3），值得读透并复用其中的参数纪律。

### 3.2 问题1：shared expert 的位置和数量（已基本确认）

从模型 `config.json` + `sglang/srt/models/glm4_moe_lite.py` 确认：

| 项 | 值 |
|---|---|
| `n_routed_experts` | 256（每 token 路由 top-8） |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 8 |
| `first_k_dense_replace` | 3（前 3 层是 dense MLP） |
| `num_hidden_layers` | 78（后 75 层是 sparse MoE） |

关键结论：

- shared expert **只有 1 个**，是独立的一份 MLP（`shared_experts`），**接收所有 token**，和 256 个 routed experts 是分开算的。
- `top_k = 8 routed + 1 fused shared = 9`；我们一直加 `--disable-shared-experts-fusion`，则 shared expert 单独算。
- **shared expert 不参与 EPLB 的负载均衡**（它不是「热/冷」专家，而是全量共享），所以热力图/rank 负载里要把它的热度单独标记，不能混进 256 个 routed experts 里做 CV。
- **shared expert 的 placement（已由历史热力图印证）**：过去测试结果里「前几个 rank 热度明显偏高」，正是 shared expert 集中放置在前几个 rank 导致（它接收所有 token）。
  所以画专家热力图 / 算 rank 负载时，**必须把 shared expert 单独拎出来**，否则会把「前几个 rank 偏热」误判成 routed expert 不均衡。
  代码里有 `_shared_expert_tp1` 这类开关，具体是否随 EP 切分仍可在启动日志/热力图中二次确认。

### 3.3 问题2（首要）：CUDA Graph 卡死，必须解决

**现象（已复现）**：decode 实例加 `--cuda-graph-max-bs 32` 后，日志卡在

```text
[rankN]:[W913 ... NPUGraph.cpp:223] Warning: Waiting for pending NCCL work to finish before starting graph capture.
```

然后迟迟不出现 `The server is fired up and ready to roll!`，只能 `--cuda-graph-backend-decode disabled` 才正常。

**重新评估（不能轻易下“无解”结论）**，bench3 的经验给了三条反向线索：

1. **可能是“慢”而不是“死”**：bench3 明确「图捕获 8-10 分钟属正常，容器 Up+无错≠死」。我上次只等了 ~3 分钟就回退，可能误判。
2. **显存配比太高**：bench3 用 `MEM=0.65`，并标注「deepep-normal 路径 MEM>0.70 会 OOM」。我上次 PD 用 `MEM=0.90`，graph capture 时很可能显存不足/等通信 buffer。
3. **镜像版本**：bench3 用 `quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`，我之前用 `...:cann9.0.0-a3-v0.5.16`。新镜像可能已修 graph。

**排查计划（按顺序）**：

1. 用统一服务（非 PD）先确认 graph 在 `MEM=0.65~0.70` + v0.5.18 下能正常捕获（隔离「PD 特有」还是「通用配置」问题）。
2. 再回到 PD decode，`MEM` 降到 0.65~0.70、`CGBS_LIST="1 2 4 8 16 32"`、`LL_CAP=64`，开 graph 后**等足 15 分钟**确认是慢还是死。
3. 若 PD 下仍卡死，检查 `--disaggregation-transfer-backend ascend` 与 graph capture 的 NCCL 等待是否冲突（PD 的 bootstrap/transfer 线程是否占着通信通道）。
4. 记录每一步的日志（尤其 NPUGraph / OOM / aclnn tiling 报错），**graph 必须最终开启**后才能进正式实验。

## 4. 部署拓扑

### 4.1 PD 分离 2p2d

固定 **2 prefill + 2 decode**（共 4 机）：

| 角色 | 节点 | 说明 |
|---|---|---|
| prefill | 10.120.72.23 + 10.120.72.25 | 组成一个 prefill 服务 |
| decode | 10.120.72.31 + 10.120.72.32 | 组成一个 decode 服务 |
| router + MemFabric store | 10.120.72.23 | 前端路由 + 传输配置存储 |

传输：`--disaggregation-transfer-backend ascend`，协议 `ASCEND_MF_TRANSFER_PROTOCOL`（sdma 已验证可用；device_rdma 初始化失败，需另行排查，见 §8）。

### 4.2 EP 的含义（已定：EP=16 / EP=32 两档）

我们每机 16 卡。PD 分离下「每侧 EP」就两档，**已拍板都跑**：

| 档位 | prefill | decode | KV 池 | 说明 |
|---|---|---|---|---|
| **EP=16** | 2 个独立实例（各 TP=EP=16） | 2 个独立实例（各 TP=EP=16） | 每实例 ~8 万 token | 单机版，上次已跑通，作对照 |
| **EP=32** | 1 个服务 TP=EP=32（跨 2 机） | 1 个服务 TP=EP=32（跨 2 机） | 翻倍 ~16 万 token | 跨机版，能支撑 64K/128K，**主对象** |

> 「EP=64」在 PD 分离下做不了（每侧 4 机需要 8 机，我们只有 4 机），不再作为本测试的 PD 档位；统一服务 4 机（非 PD）仅作历史对照，复用 `ep_exp2` 产物。

## 5. 实验矩阵

PD 分离 2p2d 固定，按 **EP=16 / EP=32 × EPLB 关/开** 出 4 格（其余变量固定）：

| 编号 | 每侧 EP | EPLB | 输入档 | 目的 | 优先级 |
|---|---|---|---|---|---|
| T4-1 | 32 | 关 | 64K / 128K | PD 基线：Prefill/TPOT/E2E 分解 + 前缀命中率 | 主 |
| T4-2 | 32 | 开 | 64K / 128K | EPLB cost + 重排前后负载 + 性能对比 | 主 |
| T4-3 | 16 | 关 | 64K / 128K | 单机对照基线 | 次 |
| T4-4 | 16 | 开 | 64K / 128K | 单机 EPLB cost 对照 | 次 |

每格内跑并发 10 / 20 / 30，请求数 = 并发 × 4（参考口径）。先跑 T4-1、T4-2。

> EPLB 与补丁/NZ 的组合沿用 `ep_exp2.md` 已验证的「补丁1+2 / NZ开」推荐配置；若 graph 开启后与 EPLB 冲突，再按 ep_exp2 的 5/6/7/8 号配置排查。

## 6. 数据采集清单（对应要求 0~4）

### 0. 数据集形式（必须写清楚）

- cc-traces 的字段说明、`block_size`、`hash_ids` 语义（§2.1）。
- 前缀缓存构造参数：`input_len`（64K / 128K）、`repeat_rate`（0.9）、`prefix_num`（并发/2）、预热方式（`--prefix_test` 或等价）、`out` 目标（1K，`ignore_eos`）。
- 实际本次采样：选用了哪些 session、请求数、每档输入的 token 统计（total/mean/P50/P90/P99/max）、`out` 统计、实际前缀命中率（由 `hash_ids` 重叠算得）。
- 构造脚本与 `hash_id -> token 块` 映射的 seed，保证可复现。

### 1. 运行时全部命令和参数

- 镜像 tag、CANN/driver 版本、`docker run` 完整命令、容器名。
- prefill / decode / router / mfstore 的**完整启动命令原文**（含所有 `--` 参数与 `export` 环境变量）。
- 压测客户端命令 + 参数（数据集路径、seed、并发、max_tokens、ignore_eos、prefix 构造方式）。
- 保留每个实验的**原始日志文件**（服务日志、压测日志、`metrics*.prom`、`npusmi*.txt`）。

### 2. EP=16/32 下的性能分解

每个实验要拆出（沿用 `TTFT残差分析.md` 的口径）：

```text
e2e = queue + prefill_forward + [tokenizer↔scheduler 的 IPC 残差] + decode
```

| 指标 | 来源 / 口径 |
|---|---|
| Prefill token/s | `prefill_forward` 内 prefill 吞吐；长输入场景单独报 64K/128K 的 prefill 时间 |
| TPOT token/s | `(e2e - TTFT) / (out - 1)`，或服务端 ITL 分位 |
| TTFT | P50/P90/P99 |
| queue | `queue_time_seconds` |
| prefill_forward | `per_stage_req_latency{stage=prefill_forward}` |
| 中间时长（IPC 残差） | `TTFT - queue - prefill_forward`（上次已定位是 tokenizer↔scheduler 收包/回传） |
| decode | e2e 的剩余段 |
| 吞吐 | completion token/s、total token/s、requests/s |

「EP=16」和「EP=32」各出上面这套；统一服务 EP64（非 PD）仅作历史对照。

### 3. EPLB cost（日志绝对不能删）

- `rebalance start` / `rebalance end time=X.XXXs` 的**每一次**记录，算：rebalance 次数、总耗时、均值/最大、占总运行时间比例。
- rebalance 前后 expert/rank 负载的 CV、peak/mean、max/min（验证 EPLB 是否真的均衡了）。
- 明确区分「rebalance 前 / rebalance 后」的稳态性能。

### 4. Expert × Layer 热力图（和上次一样）

- 全局：256 个 routed expert 的 `logical_count` 按 layer 求和的热力图 + 逐层热力图。
- 4 个随机时刻的 snapshot 热力图（记录每个 snapshot 的采样时刻 t）。
- 单独标出 shared expert 的热度（不混入 routed experts 做 CV）。
- 不均衡三件套：CV、peak/mean、max/min；零热度专家数、top-k 集中度。
- rank 负载：按 physical→logical 映射汇总到 rank（EPLB 开启时用 rebalance 后的映射）。

## 7. 执行步骤

1. **先解决 CUDA Graph**（§3.3 排查计划，graph 不开通不进入正式实验）。
2. **确定 shared expert placement**（§3.2，用一次短压测 + 热力图验证）。
3. **数据构造**：写 `cc_traces` 转换脚本，产出可复现的请求集（§2.2/2.3），并做一次单请求冒烟。
4. **T4-A（EP32，无 EPLB）**：起 PD 2p2d，跑 64K/128K × 并发 10/20/30，采 §6 全量数据。
5. **T4-B（EP32，EPLB）**：同上但开 EPLB，重点保留 rebalance 日志。
6. 每个实验跑完**立即做产物齐全检查**（关键文件/指标缺一就停，不盲目进下一个）。
7. 汇总成报告：性能分解表 + EPLB cost 表 + 热力图 + 与参考测试4 的对照表。

## 8. 已定结论与剩余风险

**已拍板：**

1. **EP 档位 = 16 / 32**（PD 分离下每侧单机 TP16 或跨 2 机 TP32）；EP64 不再作为 PD 档位。
2. **前缀缓存构造 = `repeat_rate=0.9` + `prefix_num=并发/2` + 预热**：每条请求前 90% 是公共前缀（预热后命中缓存），只有尾部 10% 重新 prefill；命中率按真实 hash 重叠度报。
3. **shared expert = 1 个，集中在前几个 rank**（历史热力图已印证），热力图里单独拎出来。

**剩余风险（未解决，按顺序处理）：**

1. **CUDA Graph 卡死**：首要，必须先解决（§3.3）。先隔离是 PD 特有还是通用配置（显存太高）。
2. **device_rdma 初始化失败**：长上下文 PD 传输理想上要 RDMA。当前 sdma 已验证但慢；device_rdma 报 `Ascend Transfer Engine initialization failed`，需单独排查（RoCE 设备已确认存在：`rocep162s0f0/f1`，10.120.73.x）。

## 9. 复用与产物

- 历史 EP 实验方案/产物：`doc/ep_exp2.md`、`doc/notuse_ep_exp.md`、`doc/datasets_run.md`。
- 时延残差结论：`doc/TTFT残差分析.md`（中间时长 = tokenizer↔scheduler IPC）。
- 参考原文：`doc/固定测试.md`（测试4）、`doc/固定测试转移方案.md`。
- 已跑通的 PD 脚本（单机 TP16 版，需升级到跨 2 机 TP32 版）：`glm52_ep64_chengyetai/fixed_tests/test4_pd/`。

---

## 10. 执行日志（持续更新）

### 10.1 CUDA Graph 排查（进行中）

**关键新线索（来自 bench3 `serve-workload.sh` + `patch/`）：**

1. 我之前 PD 实验的 graph 卡死，很可能是**配置错误**，不是 PD 特有：
   - 我 PD 脚本只写了 `--tp-size 16`、**没写 `--ep-size`**，于是 `ep_size=1`、`moe_a2a_backend=none`（256 experts 纯 TP 切分，没做真正的 expert parallel，也没 deepep all-to-all）。
   - 我用了 `MEM=0.90`；bench3 明确 `deepep-normal` 路径 `MEM>0.70` 会 OOM。
   - 我用了 `--cuda-graph-max-bs 32`；bench3 用的是 `--cuda-graph-bs "1 2 4 8 16 32"`。
2. bench3 的**正确基线参数**（已跑通 graph，SOP 里「图捕获 8-10 分钟属正常」）：
   - 镜像 `quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`
   - `--tp-size N --ep-size N --moe-a2a-backend deepep --deepep-mode auto`
   - `MEM=0.65`、`CPS=16384`、`--max-prefill-tokens 131072`、`--context-length 131072`
   - `--cuda-graph-bs "1 2 4 8 16 32"`、`MRR=32`、`LL_CAP=64`（`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64`）
3. bench3 还带一个 **DPFIX 热修**（`patch/deepep/deepep_patched`）：治「deepep-auto 模式下，decode graph 的 LL path 与 normal path 共享 DeepEP buffer，TP/EP 大时 CamMoeCombineNormal 死循环、AICore 100% 空转」。这跟「卡住不动」现象高度相关，若 deepep auto + graph 复现卡死就上这个补丁。
4. EPLB 双补丁（补丁1+2）bench3 也有，与我们历史用的同一套。

**下一步**：停旧 PD 服务 → 用 v0.5.18 + MEM0.65 + 显式 EP + deepep auto + `--cuda-graph-bs` 重测 PD decode 的 graph，看是否能在 8-10 分钟内捕获完成。

**实测进度（2026-09-14）：**

- 停掉了旧 PD 服务，重写 `serve_pd.sh`（显式 `--ep-size`、`--moe-a2a-backend deepep --deepep-mode auto`、`--cuda-graph-bs "1 2 4 8 16 32"`、MEM/CGBS/LL_CAP 参数化）。
- **发现 A：单机 EP16 时权重占比 ~79%**，`MEM=0.65` 直接报 `Loaded weights leave no GPU memory ... Raise --mem-fraction-static above 0.791`。
  也就是说 bench3 的 `MEM=0.65` 只在 TP32/EP32（跨 2 机）下成立；单机 16 卡装 GLM-5.2 权重就要 ~79%，KV 余量极小 → **EP16 单机对 64K/128K 基本不可行**，主对象应是 EP32（跨 2 机）。
- 改用 `MEM=0.82` 后进入 graph capture，但**仍卡在** `NPUGraph.cpp:223 Waiting for pending NCCL work to finish before starting graph capture`（这次是真正的 EP16+deepep-auto）。
- 待确认：是「慢（8-10 分钟属正常）」还是「真死」。正在等足 10-15 分钟观察是否到 `ready to roll`。

**重要修正（graph 大概率没卡死，是我误判）：**

- 这次（EP16+deepep-auto+`--cuda-graph-bs`）日志时间线：21:57:41 进 graph capture → 21:59:54 已推进到 disaggregation 初始化，**说明 graph capture 在 ~2 分钟内正常走完，没卡住**。
- 真正报错是 `SMEM smem_trans_entry.cpp:70 Initialize] store helper initialize failed: -2002` → `Ascend Transfer Engine initialization failed`，
  **根因是我把 mfstore 停了**，transfer engine 连不上 `ASCEND_MF_STORE_URL`，不是 graph 问题。
- 结论：之前「graph 卡死」极可能是我（a）等太短（bench3 说 8-10 分钟正常）＋（b）当时环境不完整/配置错（ep_size=1、MEM 0.90、`--cuda-graph-max-bs`）造成的误判。
- 下一步：**搭完整 PD 环境（mfstore+prefill+decode+router）**，用 MEM0.82（单机 EP16）跑一遍，看 decode 能否到 `ready to roll`，坐实 graph 可开。

**硬阻塞1（CUDA Graph）已解决 ✅**

- 完整环境（mfstore + prefill@23 + decode@31，EP16 单机，MEM0.82，`--cuda-graph-bs`，v0.5.18）下：
  - decode 的 graph capture 从 22:11:37 开始，22:11:49 就进入 transfer-engine 初始化，**只用了 ~12 秒，没卡死**；
  - prefill 22:12:00、decode 22:12:06 都到 `The server is fired up and ready to roll!`。
- 结论：**graph 可开，之前「卡死」是误判**（等太短 + ep_size=1/MEM0.90/`--cuda-graph-max-bs` 配置错 + mfstore 没起）。
- 正确配置要点：显式 `--ep-size`、`--moe-a2a-backend deepep --deepep-mode auto`、`--cuda-graph-bs "1 2 4 8 16 32"`、`MEM` 按权重占比给足、mfstore 先起。

**新问题：EP16+deepep-auto 下模型输出退化（token 66810 循环）**

- 端到端 smoke 通了（router health 200、能出 token），但 `/generate` 和 `/v1/chat/completions` 都输出 `//</ //</ ...`（token 66810 重复），语义退化。
- 旧配置（ep_size=1、moe none、v0.5.16）输出是正常的，所以是「真 EP + deepep」这条新路径引入的。
- 待排查变量：`--disable-shared-experts-fusion`、`--deepep-mode auto`、v0.5.18 镜像、PD+EP 交互。

**乱码根因定位（隔离实验）**

- 逐个排除：`--disable-shared-experts-fusion`（fusion on 仍乱码）、NANFIX 清 NaN（补丁已生效仍乱码）→ 都不是根因。
- **关键隔离实验**：统一服务（非 PD）+ TP=EP=16 + deepep auto + NANFIX → **输出正常**（"我是一个对生活充满热情的人…"）。
  而 PD 分离（同样 EP16+deepep）→ 乱码。
- 结论：**乱码是 PD 分离 + EP16 交互特有的**，不是 deepep 本身、不是 NaN、不是 fusion。
- 下一步：查 disaggregation 传输里 EP 相关的 KV/状态映射（bootstrap_room / rank 映射 / expert 状态）。

**下一步动作**：参考测试4 本应是「每侧 2 节点（EP32）」而非单机 EP16。先搭 **跨 2 机 EP32 的 PD**（主对象），验证乱码是否只在单机 EP16 出现、EP32 跨机是否正常；device_rdma 随后排查。

**找到关键线索（SGLang 官方 Expert Parallelism 文档）**：`--moe-a2a-backend` 在 NPU 上只支持 `deepep` / `ascend_fuseep`；**PD Disaggregation 下 prefill 必须 `--deepep-mode normal`、decode 必须 `--deepep-mode low_latency`**。我之前两边都用 `auto`，很可能就是乱码根因（auto 在 PD 下选错路径）。

已改 `serve_pd.sh`：按角色默认 prefill=normal / decode=low_latency，EP32 跨 2 机 PD 正在重启验证。

**deepep-mode 修正无效 + 更精确的定位：**

- EP32 跨 2 机 PD + deepep normal/low_latency 仍乱码；EP16/EP32、deepep auto/normal/low_latency、fusion on/off 都乱码。
- 所以乱码跟 deepep-mode、fusion 无关，是 **PD 分离 + EP（deepep 的 all-to-all）这条路径本身**的问题。
- 高度相关的上游线索：PR #28237「fused shared-expert scaling on DeepEP 路径被 routed_scaling_factor 错缩 → runaway repetition loop」，现象（token 死循环）与我们的 66810 循环**完全一致**；该 PR 是 AMD aiter 路径，Ascend deepep 可能也有同族问题，但需确认 Ascend 侧是否同样在 PD 下被触发。
- 已确认 DSA（indexer）的 PD 传输有专门支持（`enable_dsa_cache_layer_split` 是 prefill-CP 优化，非必需），MiniMax index_kv_pool 才 not-supported，与 GLM-5.2 无关。
- 下一步试：`--moe-a2a-backend ascend_fuseep`（Ascend 专用 MoE 后端）替代 deepep，看能否避开该路径。

**ascend_fuseep 也乱码 → 锁定「PD + EP（expert parallel）本身」**

- `--moe-a2a-backend ascend_fuseep` 也输出 66810 循环。
- 排除清单已完整：deepep-mode（auto/normal/low_latency）、fusion（on/off）、NaN（NANFIX）、MoE 后端（deepep/ascend_fuseep）都不是根因。
- 唯一稳定的边界是：**统一服务 + EP = 正常；PD 分离 + EP = 乱码；PD 分离 + EP1（无 expert parallel）= 正常**。
- 结论：乱码是 **PD disaggregation + expert parallel 交互**的特有 bug，高度怀疑是 GLM-5.2 的 DSA（稀疏注意力 indexer）KV 在 PD+EP 下的传输/状态问题（GLM-5.2 是 DSA 模型，DeepSeek-R1 无 indexer，官方 PD+EP 教程是 R1，未覆盖 GLM-5.2 DSA）。
- 待办：查 GLM-5.2 DSA indexer 在 PD 传输里的处理；device_rdma 仍待排查。

**重大突破（官方 GLM-5.2 PD 教程给出了正确配置）：**

官方 PD 分离教程的关键差异（我之前全做错了）：

1. **`--load-balance-method round_robin`**（我之前是 PD 默认的 `follow_bootstrap_room`）。
2. **`--enable-dp-attention`**（prefill/decode 都要）——GLM-5.2 DSA 的稀疏注意力在 PD 下需要 DP attention。
3. **prefill 不做 EP**：官方 prefill 是 `--tp-size 16 --dp-size 4 --enable-dp-lm-head --moe-dense-tp 1`，无 `--ep-size`；只有 **decode 做 EP**（`--tp-size 16 --dp-size 16 --ep-size 16`）。
4. **`--disable-shared-experts-fusion`**（官方明确关）。
5. prefill 用 `--disable-cuda-graph`，decode 用 `--cuda-graph-max-bs-decode 4`。

我之前的错误核心是：prefill/decode 都做纯 TP+EP、无 DP attention、load-balance 用了 follow_bootstrap_room。正在按官方配置重写 `serve_pd.sh`。

**官方配置实测结果（v0.5.16 + DP-attention + round_robin + decode EP16）：**

- 两个硬阻塞里 CUDA Graph 已确认解决；官方配置也能加载到 `ready to roll`（prefill 23:12:41 / decode 23:12:53）。
- 但 **smoke 仍乱码**：`output_ids=[1773, 66810, 66810, ...]`——**第一个 token 正常（prefill 算对），第二个 token 起全退化**。
- 这进一步定位到：**decode 用 prefill 传来的 KV 做第一个 decode step 时就出错**，是 GLM-5.2 的 DSA（稀疏注意力 indexer）状态在 PD 传输中没正确恢复/传递（prefill.py 里 `indexer_topk_output.finalize()`、conn.py 里 `c4_indexer_layer` 的传输路径疑似有 bug 或与 GLM-5.2 的 `full/shared` indexer 结构不匹配）。
- 结论：这是 SGLang 上游的 GLM-5.2 DSA + PD 深层次问题，非参数能绕开；需要上游 fix 或深入改 disaggregation 的 indexer 传输代码。

**找到上游已知 bug 与修复（关键）：**

- PR #22811「Fix low-probability garbled output in GLM-5 series」：NSA/DSA indexer 在 HiCache/异步加载下读到 stale/partial 的 K-scale buffer → **garbled output、重复 token、罕见字符**（Z.ai 生产环境踩坑，现象与我们的 66810 循环完全一致，修法是在 indexer kernel 前加显式同步）。
- PR #31108：GLM-5.2 DSA-indexer sidecar 是 ALL_PAGES 池，与 KV 1:1 co-load，读到 garbage indexer state → wrong top-k → garbled/runaway output。
- PR #23539：PD disaggregation 下 NSA/MTP 的 index/KV 状态没从 prefill 传到 decode → wrong results。
- 这些修复都在较新的 SGLang（7-8 月合并）。本地有 `main-cann9.0.0-a3`（2 周前）镜像，大概率已包含。下一步换 main 镜像重试官方 PD 配置。

### 10.2 换 main 镜像实测（2026-09-14 深夜）：乱码仍未解决，且定位更精确

**结论一句话**：`main-cann9.0.0-a3`（sglang `0.5.19.dev684+gfe694986a`，2 周前）**依然乱码**；`round_robin` 与 `follow_bootstrap_room` 两种 load-balance 都乱码。这已排除「镜像版本」与「load-balance 方法」两个变量，坐实是 GLM-5.2 DSA + PD + EP 的深层次路径 bug。

**本轮实测（1P1D 最小复现，prefill@23 + decode@31 + router@23，TP16/DP4，decode EP16）**：

1. 官方 PD 配置 + main 镜像：smoke 输出 `让我 //</ //</ ...`（token 66810 循环），与 v0.5.16 完全同症。
2. 换 `follow_bootstrap_room`：输出 `用户 //</ //</ ...`，依旧乱码。→ load-balance 方法不是根因。
3. 期间 router 打印关键警告：
   > `Model glm-5.2 has dp_size > 1 but load_balance_method is not 'follow_bootstrap_room' on prefill workers. This may cause rank mismatch in disaggregated mode.`
   这是 main 镜像（新版 Rust router）新增的告警，v0.5.16 没有。**rank mismatch 线索仍在，但两种 load-balance 都试过仍乱码，说明单靠它不能解。**

**关键新发现：模型身份与本地自定义镜像**

- 确认 GLM-5.2（`glm-q`）是 **DSA 模型**：`architectures=['GlmMoeDsaForCausalLM']`、`model_type=glm_moe_dsa`、`index_topk=2048`、`index_n_heads=32`、`indexer_types=[full/shared ...]`、`index_topk_freq=4`、`index_skip_topk_offset=3`。注意它继承 `DeepseekV2ForCausalLM`，DSA 稀疏注意力（indexer）代码在 `deepseek_v2.py` 的 `DeepseekV2AttentionMLA`，而 `glm4_moe.py` 的 `Glm4MoeAttention` 是普通 GQA（不涉及 indexer）。
- 发现一台本地自定义镜像 `sglang-npu-pd:v0.5.16-glm52-indexer`（**只存在于 prefill 节点 23/25，decode 节点 31/32 没有**），它 patch 了 `deepseek_v2.py` 的 DSA indexer `skip_topk` 逻辑：
  - v0.5.16 原版：`is_nextn` 层 `skip_topk=True`；且 `self.indexer` **无条件创建**（所有 DSA 层都建）。
  - glm52-indexer 版：`is_nextn` 层 `skip_topk=False`；`self.indexer` 只在 `not skip_topk` 时创建。
  - main 镜像（第三种写法）：`is_nextn` 层仍 `skip_topk=True`，但创建条件为 `if not skip_topk or is_nextn`。
  - 三个版本对**主模型非 nextn 层**等价（都只给非 skip 层建 indexer），差异只在 MTP/nextn 层；我们测试未开 MTP，故该 patch 大概率不是本次乱码根因（main 已有等价修复仍乱码）。

**修正后的根因画像**：

- 稳定边界仍是：统一服务+EP=正常；PD+EP1（无 deepep）=正常；**PD+EP16（deepep all-to-all）=乱码**。
- 症状「首 token 正常、次 token 起 66810 循环」指向 decode 侧在 EP（deepep）路径下，DSA indexer 的 topk 状态/传输在 PD disaggregation 里没被正确恢复（对应上游 PR #22811 / #31108 / #23539 所述 indexer 状态 stale/partial/未传递），但本地 main 镜像尚未包含能治本的那个修复。

**下一步（按优先级）**：

1. 深挖 `disaggregation/common/conn.py` 的 `c4_indexer_layer` 传输 + `prefill.py` 的 `indexer_topk_output.finalize()`，对照上游 PR #23539 / #31108 的 diff，定位 Ascend 侧缺的那块。
2. 把 `sglang-npu-pd:v0.5.16-glm52-indexer` 从 23 拉到 31/32 后实测（虽然预计非根因，但成本低、可闭环排除）。
3. 留意 `sglang-mooncake-npu:v0.5.16`（31/32 上），moondcake 是另一条 KV 传输后端，可能绕开 ascend 传输的 indexer bug。

**环境备注**：单机 prefill（TP16/DP4，无 EP）`--mem-fraction-static` 需 ≥ 0.86（报错 `minimum viable = 0.8595`），本轮用 0.90 稳定；服务已全部清理（4 机无 pd- 容器）。

### 10.3 上游 PR 定位（联网核实，2026-09-14）

联网确认了与「GLM-5.2 DSA indexer + PD」直接相关的上游修复，按相关度排序：

1. **PR #30839「Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay」**（2026-07-14 合并，作者 zRzRzRzRzRzRzR）——**最相关**。核心改动就是「Transfer per-request DSA top-k seeds from prefill nodes to decode nodes in PD disaggregation, with explicit invalid sentinels and retry-state cleanup」，即把 DSA 的 top-k 种子从 prefill 传到 decode。但注意它 gate 在 `speculative_eagle_topk==1 && index_share_for_mtp_iteration=true`（即 EAGLE/MTP 路径）；我们没开 EAGLE，是否命中待确认。
2. **PR #32209「Fix PD decode hang with DP attention and GLM-5.2 MTP」**（2026-07-23 起，跟进 #30839，截止本文时仍在 review/未确认合并）。修的是 **hang**（不是乱码），但根因描述与我们高度同族：PD decode + DP attention + GLM-5.2 IndexShare 时，seedless 回合下 DP rank 之间 graph 路径分歧。文中给出 operator 级 workaround：「**enabling MTP on the prefill side also avoids the hang/问题**（prefill 侧开启 MTP，让 dsa_topk_indices 在 decode 开始时始终可用）」。
3. **PR #31108「Clamp HiCache prefetch prefix to sidecar-indexer hits」**（2026-07-13）——GLM-5.2 DSA-indexer sidecar 是 ALL_PAGES 池与 KV 1:1 co-load，读垃圾 indexer state → wrong top-k → garbled/runaway；但这是 HiCache 路径，我们没开 HiCache。
4. **PR #22811「Fix low-probability garbled output in GLM-5 series」**（2026-04）——NSA index-buffer 读在 HiCache 下缺同步；同样 HiCache 相关。
5. **PR #29544「docs: PD disaggregation GLM-5.2 cookbook」** 明确：GLM-5.2 = `glm_moe_dsa`，DSA 家族（同 DeepSeek-V3.2/V4），SGLang「已支持 PD disaggregation」——说明官方认为可用，我们命中的是某个未覆盖的组合（大概率是 **DP attention + EP + CUDA graph decode** 这个组合）。

**关键判断更新**：本地 main 镜像（0.5.19.dev684，~Sep 1）已含 #30839/#31108，但仍乱码 → 要么缺 #32209（及其后继），要么 Ascend 侧没有把 #30839 的「DSA top-k seed PD 传输」完整 port 过来。**下一优先级最高动作：拉一个 8 月之后的更新镜像，或直接把 #30839+#32209 的补丁 port 到本地镜像再测。**

**可立即试的 operator workaround（来自 #32209）**：给 prefill 侧开启 MTP/EAGLE（`--speculative-algorithm EAGLE --speculative-num-steps N --speculative-eagle-topk 1`），使 dsa_topk_indices 在 decode 侧始终可用，观察乱码是否消失（即便不全解，也能进一步定位是否就是 IndexShare seed 缺失）。

### 10.4 官方教程逐字核对 + DP16/SPEC 实测（2026-09-14 凌晨）：定位到「DP 大小」死结

**逐字核对官方 GLM-5.2 PD 教程后，确认我之前转述的官方配置有两处关键点一直被漏掉/改错：**

1. **官方 prefill 和 decode 都开了 speculative（NEXTN/EAGLE）**，我们一直没开：
   - prefill：`--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`
   - decode：`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
   - 两边都 `--speculative-draft-model-quantization unquant`
2. **官方 decode 是 `--dp-size 16`**，而我们的 `serve_pd.sh` 默认 `DECODE_DP=4`，之前的"官方配置实测"其实一直用的是 DP4，从未真正用过官方 DP16。

**本轮实测结果：**

| 组合 | 结果 |
|---|---|
| main + DP4 + 无 SPEC | 乱码 `//</` 循环 |
| main + DP4 + SPEC(NEXTN) | **仍乱码**（`1 //</ ...`） |
| main + DP16 + SPEC | **OOM**（权重 99.2%，`post_load_weights` 时 60.78/61.27 GiB） |
| v0.5.16 + DP16 + SPEC | **同样 OOM**（99.2%） |

**结论：**

- **speculative 不是乱码根因**（DP4 下开关都乱码）。
- **DP 大小是核心变量**：DP16 是官方正确配置，但 `enable-dp-attention` 会把注意力权重在每个 DP rank 复制，DP16（每卡 1 份完整注意力）比 DP4（组内 TP4，每卡 1/4）每卡多约 9%（5.8 GiB），导致权重顶到 99%，**单机 16×64GB 装不下**；DP4 能装下但乱码。
- 官方教程的模型是「GLM-5.2-w8a8 **without MTP**，774.08GB」，而我们的 `/mnt/raid/user/data/models/glm-q` 是 **带 MTP（num_nextn_predict_layers=1）**、721GB 的 modelslim 量化权重。这是官方 DP16+MEM0.8 能装下、我们装不下的直接原因之一。
- 模型目录实况：`glm-q`(721G, 带 MTP) / `glm-qq`(721G, 带 MTP) / `glm5.2`(1.4T, BF16)。**没有看到官方的「无 MTP W8A8」版本**。

**当前死结：** DP4 乱码（怀疑 DP-attention 组内 TP 的 DSA indexer 状态/rank 映射在 PD 下不一致）＋ DP16 OOM（带 MTP 权重 + DP 复制注意力权重超内存）。要破局需二选一：拿到无 MTP 的 W8A8 权重跑官方 DP16，或深入修 DP4 在 PD 下的 indexer/rank 问题。

### 10.5 关键发现：glm-q 与 glm-qq 是两份不同权重，glm-qq 是新拷贝（2026-09-14 凌晨）

用户提示「glm-qq 是新拷贝的，之前的 glm-q 可能有问题」。只读对比确认：

- `config.json`、`quant_model_description.json` **完全相同**；文件数都是 182 个 safetensors；目录大小都是 721G。
- 但**权重内容大面积不同**：抽样 8 个 shard，7 个 checksum 不一样（仅 shard 00179 相同）。所以 glm-q / glm-qq 不是同一份权重的副本，而是**两份不同内容**。
- `glm-qq` 里多了 msmodelslim 的 GLM-5.2 支持补丁 `1a94484….patch`（"support glm 5.2"，yejiajun@huawei.com，2026-06）+ `GLM-5.1_best_practice.yaml`（量化配置，**明确 exclude `model.layers.78.*`（MTP 层）、`*kv_b_proj`、`*wk`、`*weights_proj`**）。glm-q 的 `.msc` 头部为不可读字节，glm-qq 的 `.msc` 是正常 modelslim cache（含 chat_template/config 等文件清单与 revision）。
- 结论：**glm-q（旧，9/3）很可能量化有问题；glm-qq（新，9/9）是正确的重拷贝**。这很可能是此前「PD+EP 乱码」一直被误判为 SGLang bug 的真正根因——权重本身有问题。

**实测（glm-qq + main 镜像 + 单机 DP4/EP16 + SPEC）：**

- **decode 用 glm-qq 正常 ready**（`Load weight end ... mem usage=51.36 GB`，graph capture 完成，`ready to roll`）——权重是好的。
- **prefill 用 glm-qq 卡死**：卡在模型构建阶段 `Using ModelSlimW8A8Int8MoE for W2/W13`（01:15:38 后 15+ 分钟无新日志，scheduler CPU 250-330% 忙转，`Load weight end` 计数=0）。prefill 与 decode 的差异在 MoE 布局：prefill 无 EP（`--moe-dense-tp 1`）、decode EP16。疑似 glm-qq 的 MoE 量化格式在 prefill 无 EP 构建路径下不兼容/死循环。

**下一步**：先解决 prefill 卡死（试去掉 `--moe-dense-tp 1` 或换 prefill 的 MoE 后端/参数），再跑完整 smoke，验证 glm-qq 是否真正解决乱码。

### 10.6 ✅ 乱码根因找到并解决：换 glm-qq 权重（2026-09-14 凌晨）

**结果：用 `glm-qq` 后，PD+EP 输出完全正常，不再是 `//</` 循环。**

成功配置（最小可跑）：
- 模型 `MODEL=/mnt/raid/user/data/models/glm-qq`（关键！）
- prefill@23：TP16 + DP4，`SPEC=off`，`PF_MOE_DENSE_TP=0`（去掉 `--moe-dense-tp 1`），deepep normal，禁 graph
- decode@31：TP16 + EP16 + DP4，`SPEC=off`，deepep low_latency，graph 开
- load_balance=round_robin，镜像 main（0.5.19.dev684）

smoke 输出（正常）：
> 用户要求用一句话介绍我自己。我需要简洁明了地概括自己的身份和功能。首先，我应该明确自己的核心定位：一个由 Z.ai 开发的 GLM 大语言模型……

**根因定性**：此前「PD + EP 乱码」不是 SGLang 的 bug，而是 **`glm-q` 权重本身有问题**（旧量化，`.msc` 坏字节、缺 `rot.weight` 等），`glm-qq` 是正确重拷贝。用 glm-qq 后 decode 侧的 DSA indexer/EP 路径一切正常。

**待澄清的两个变量**（后续按官方配置逐步回填）：
1. 这次为了绕过 prefill 卡住，用了 `SPEC=off` + 去 `moe_dense_tp`；需分别验证能不能恢复官方 `SPEC=on` 和 `--moe-dense-tp 1`（glm-qq 加载慢，prefill load_weight≈491s，之前误判成卡死，实际是慢 + 可能 moe_dense_tp/SPEC 触发了构建慢）。
2. 正式测试4 应是 2 机 decode（EP32/DP32），还需在 glm-qq 上验证 2 机形态。
