# 任务 1：MTP（推测解码）实验方案

**文件夹**：`current_exp/Q1MTPExp/`（脚本、数据集、结果都放这里）
**上一轮同类实验**：`history_exp/04_MTP推测解码_2026-09/mtp_full_20260928/`（2026-09-28，已跑完一轮，见 §2）
**最近一次快速验证**：`current_exp/mtp_quick_20261011/`（2026-10-11，单机打通了 MTP 启动）

---

## 0. 一句话

MTP 的**执行/计数**上次已经跑通；这次要解决的是**"测什么"** ——
上次用 uniform 随机 token 当输入，模型输出塌成"3 个 token 无限循环"，导致接受长度假高（2.77~4.00）。
**这次的核心任务：换一种数据集构造方式，让接受长度回到 ≈2.0（对齐线上高点 2.11），再报 MTP 开/关的收益。**

---

## 1. 文档明确要求的东西（引用来源）

### 1.1 用例（`docs/MTP.md`）

| 用例 | 输入 | 输出 | 缓存命中 | 备注 |
|---|---|---|---|---|
| 用例 1 | 前缀 **100K** | 1K | **90%** | 新增（9-28 没做） |
| 用例 2 | 前缀 **32K** | 300 | **60%** | 9-28 已做（实测命中 59.77%） |

### 1.2 固定配置与硬指标（`docs/MTP.md`）

- **固定 MTP = 3**（3 个 draft 步骤 + 1 个 bonus → 接受长度上限 4）
- **精度**：KVV 误差 ≤ 3% ⚠️（口径待确认）；**AL ≈ 2.0（±5%）**
- **延迟**：P50 TPOT < 33ms；P50 TTFT < 15s
- **目标**：上述两个场景下**最大化吞吐**
- 参考：**线上实测 AL 高点 2.11、低点接近 0**
- per-position 接受率参考值：`0.498 / 0.345 / 0.267`（另一组 `0.148 / 0.093 / 0.056` ⚠️对应场景待确认）
- 若要做 MTP≠3 的方案：在上述基准完成后，按 `0.498 / 0.345 / 0.267` 配

### 1.3 要报的指标（`docs/ChunkSize_ref/TODO.md` MTP 行）

> MTP/推测解码（1k/32k、8k/1k、前缀 32K / 输出 300 / 缓存命中 60%）→ **接受长度、对应吞吐、ttft、tpot**

### 1.4 团队标准 workload（`docs/AboutModels.md`）

请求数量 = 并发数 × 2（如并发 8 → 16 条）；`32k/1k` 与 `1k/32k` 的请求数 = 并发数。

1. `8k/1k`（输入 8192 / 输出 1024）
2. `32k/1k`（输入 32768 / 输出 1024）
3. `1k/32k`（输入 1024 / 输出 32768，小时级）
4. `前缀 32K / 输出 300 / 缓存命中 60%`

---

## 2. 上一轮（2026-09-28）做了什么、结论如何

### 2.1 数据集口径（**现成，可直接复用**）

```python
# mtp_full_20260928/bench.py::data()
prefix = [rng.randrange(1, 154880) for _ in range(19661)]   # 19661 个随机 token = 共享前缀
prefix 场景：输入 32768 = 共享前缀 19661 + 各自随机 13107
          → 命中率 = 19661 / 32768 = 59.77% ≈ 60%        ← 正是文档的用例 2
```
- `dataset.json`（60MB）里存了**完整 input_ids + seed(2026092802) + SHA**，用例数：decode 4 / mixed 64 / prefix 80。
- 预检脚本 `prefix_preflight.py` 要求实测 `cached_tokens / prompt_tokens ∈ [0.585, 0.615]`。
- 配置：单机 TP16/EP16、chunk 16384、`temperature=0`、`ignore_eos=true`、每台**最大在途 1 条**、每场景 2700s。

### 2.2 结果（MTP OFF vs ON，两机并行，45 分钟/场景）

| 场景 | OFF 吞吐 | ON 吞吐 | **AL** | 接受率 | OFF TPOT | ON TPOT |
|---|---|---|---|---|---|---|
| 1K / 32K | 2.12 tok/s | 7.85 | **3.998** | 99.9% | 472ms | 127ms |
| 8K / 1K | 2.10 | 6.60 | **3.459** | 82.0% | 473ms | 151ms |
| 32K / 300 / 命中60% | 1.86 | 4.13 | **2.767** | 59.1% | 477ms | 186ms |

TTFT：OFF 1.06~5.40s → ON 1.13~6.41s（**ON 略高**，两点：MTP 本身不省 TTFT，反而多一步草稿）。

### 2.3 为什么不可信（会上被质疑的原因）

`analysis/output_patterns.json` 的原始证据：

```
场景 decode(1K/32K) MTP=OFF：输出了 5715 个 token，但**只有 3 个不同的 token id**，
周期 = 3（168, 111, 238, 168, 111, 238, ...）—— 输出塌成无限循环
```
→ 输入是 uniform 随机 token 时，模型的输出分布退化 → MTP 的草稿几乎每轮全中 → **AL≈4 是假的**。
→ 32K/300 的 2.767 也高于线上高点 2.11，同样偏高。

**老板的结论（会议）**：随机数据集可以用，**但必须换构造方式再测**；不能拿退化结果当结论。

---

## 3. 这次的核心问题：数据集怎么构造

**退化根因 = 输入是 uniform 随机 token id**（不是"随机"这件事本身错）。三条路：

| 方案 | 做法 | 是否符合文档 | 评价 |
|---|---|---|---|
| **A（推荐）** | 保留"**固定前缀 + 随机采样后缀**"的结构与命中率口径，但 token 从**真实语料**随机采样（ShareGPT / SWE-agent 轨迹） | ✅ 仍是"固定前缀**随机**数据集"（随机采样），命中率算法不变 | 输出是真实分布 → AL 可信 |
| B | 沿用 uniform 随机 id（= 上一轮口径） | ✅ 照抄 | ❌ AL 必然假高 |
| C | 用真实 token 频率分布采样随机 id | 部分 | 未必治好退化 |

**建议**：主实验用 **A**；同时把 **B 作为对照跑一组**，用来量化"退化导致的假高"（把上次的 2.77~4.00 解释清楚）。

**可用的真实语料（只读，勿写 /mnt/raid）**

| 源 | 规模 | 可用性 |
|---|---|---|
| `/mnt/raid/user/data/dataset/ShareGPT_V3_unfiltered_cleaned_split.json` | 672MB 真实多轮对话 | ✅ 已实测可解析 |
| `/mnt/raid/user/data/dataset/SWE-agent-trajectories/*.parquet` | 12 文件 / 8 万行真实 agent 轨迹（中位 35 轮） | ✅ 需要 pyarrow |

**本机就能 tokenize**：`tokenizers` 库 + 模型目录 `tokenizer.json`（20MB，vocab 154856），已实测可用（无需起容器）。

---

## 4. 实验设计

### 4.1 平台与对照

- **单机 16 卡**（tp16 / dp2 / ep16）—— MTP 是单机特性
- **ON / OFF 各一台机并行**（和 9-28 同构，可比性最好）→ 需要 **2 台机**
- 对照：`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（9-28 用的就是这套）
- 其余照 9-28：`temperature=0`、`ignore_eos=true`、`chunked-prefill-size 16384`、`mem-fraction-static 0.90`、`context-length 180000`

### 4.2 场景矩阵

| # | 场景 | 输入 | 输出 | 命中率 | 数据构造 | 状态 |
|---|---|---|---|---|---|---|
| S1 | 8K/1K | 8192 | 1024 | 0（每次清缓存） | 真实语料随机采样 | 9-28 做过（随机 id 版） |
| S2 | 1K/32K | 1024 | 32768 | 0 | 真实语料随机采样 | 9-28 做过；**须解决退化** |
| S3 | 前缀 32K / 300 | 32768 | 300 | **60%** | 共享前缀 19661（可复用现有实现） | 9-28 做过 |
| S4 | 前缀 100K / 1K | 100K | 1024 | **90%** | 新增（context 180K 够） | **新增** |
| S5（对照） | S2 或 S3 的 **uniform 随机 id 版** | — | — | — | = 上一轮口径 | 用于量化假高 |

每组都要：**MTP OFF vs ON**。

### 4.3 两道 Gate（必须过，才跑正式）

1. **语义检查**（复用 `chat_once.py`）：真实提问 → 回复是人话，不是乱码/循环；
2. **小规模探针**（复用 `probe.py`，1K 输入 / 64 输出）：
   - 输出**无重复循环**（复用 `output_patterns.py` 的"distinct ids / 周期"统计）；
   - per-position 接受率量级接近 `0.498 / 0.345 / 0.267`；
   - 命中率落进目标区间（S3: 0.585~0.615）。

> 过不了 Gate 就不跑正式，先改数据集构造。

### 4.4 指标与口径

| 指标 | 口径 | 来源 |
|---|---|---|
| **接受长度 AL** | 该请求**实际输出 token 数 / `spec_verify_ct`**（含 bonus token） | 服务端 `meta_info` / `/metrics` 的 `spec_accept_length` |
| **Draft 接受率** | `spec_num_correct_drafts / spec_num_proposed_drafts`（不含 bonus） | `/metrics` |
| **per-position 接受率** | 逐位置累计（需要从 SSE 终态帧取 `spec_*` 计数） | 复用 9-28 的采集代码 |
| **缓存命中率** | `cached_tokens / prompt_tokens`（服务端 meta_info） | 预检 + 正式都要记 |
| **吞吐** | 窗口输出 token / 墙钟（另记"纯生成速率"= (输出-1)/(首末输出间隔)） | 9-28 有两个口径，保留 |
| **TTFT** | 客户端发出 → 首个有输出的 SSE 帧 | 同上 |
| **TPOT** | (末次新增输出 − 首次输出) / (输出 token 数 − 1) | 注意：MTP 一帧含多 token，单帧内部 ITL 拿不到 |
| **E2E** | 客户端首末 | |

---

## 5. 待确认（5 条）

1. **同意把 token 来源换成"真实语料随机采样"吗？**（保持"固定前缀 + 命中率"结构不变；这是让 AL 回到 ≈2.0 的关键。若必须严格用 uniform 随机 id，就只能报"该口径 AL 偏高、不代表业务"）
2. **机器**：给 **2 台**（ON/OFF 并行，同 9-28）还是 1 台串行（时间翻倍）？用哪几台？
3. **"前缀 100K / 命中 90%" 的口径**：100K 是**输入总长**，还是共享前缀长度？
   （9-28 的实现是"总输入 32K + 共享前缀 19661 → 59.77%"，即**命中率 = 共享前缀 / 输入总长**。）
4. **"KVV 误差 ≤ 3%"** 指什么？（文档里没定义，需找人确认）
5. **镜像**：沿用 9-28 那套（`v0.5.16` + 只读权重外挂 + loader 补丁），还是试 `v0.5.18`（PD 实验用的那个）？

---

## 6. 我要写的代码（不占机器，先做）

1. `build_real_dataset.py`：读真实语料 → tokenize → 造"固定前缀 + 随机采样后缀"数据集
   （输出 `dataset_real.json`，含 input_ids / SHA / 命中率，格式与 9-28 的 `dataset.json` 对齐）；
2. `bench.py`（新）：合并两套能力 —— 9-28 的「显式 input_ids + 逐请求 SSE/终态计数」+ 我们的「并发控制（请求数 = 并发×2）」；
3. `serve_mtp.sh`：单机起停（`MTP=on/off`、`img`、`port`、日志落盘），参考 `mtp_quick_20261011/serve_final.sh`；
4. `run_mtp.sh`：单场景编排（清缓存 → 预热前缀 → 打负载 → 抓 `/metrics` + npu-smi → 归档）；
5. 复用改造：`chat_once.py`（语义检查）、`probe.py`（小规模探针）、`output_patterns.py`（循环检测）、`prefix_preflight.py`（命中率预检）。

---

## 7. 现有可复用资产

| 路径 | 内容 |
|---|---|
| `history_exp/04_MTP推测解码_2026-09/mtp_full_20260928/` | **上一轮完整实验**：`bench.py`（数据集+采集）、`serve_verified.sh`、`prefix_preflight.py`、`probe.py`、`chat_once.py`、`output_patterns.py`、`analysis/`（各种 csv/json/图） |
| └ `results/20260928_061159/dataset.json` | 现成的固定前缀随机数据集（60MB，含 input_ids） |
| └ `results/20260928_061159/analysis/output_patterns.json` | **退化证据**（3 个 distinct token 的循环） |
| `current_exp/mtp_quick_20261011/serve_final.sh` | 最近一次单机 MTP 起服务的脚本（v0.5.16，spec 参数齐全） |
| `current_exp/docs/MTP.md` / `AboutModels.md` / `ChunkSize_ref/TODO.md` | 任务要求与 workload 口径 |
| `/data_lib/data/models/glm-q/tokenizer.json` | 本地 tokenize 用（配 `tokenizers` 库） |
| `/mnt/raid/user/data/dataset/ShareGPT_V3_...json`、`SWE-agent-trajectories/` | 真实语料源（**只读**） |

---

## 8. 风险与坑（上次踩过的）

1. **退化输出**：uniform 随机 token → 输出循环 → AL 假高。**必须检查输出**，不能只看 AL。
2. **长输出场景（1K/32K）45 分钟跑不完**：要么缩短目标输出，要么按"共同输出长度区间"比速度（9-28 的做法），不能拿整窗均值下结论。
3. **CUDA Graph 被镜像禁用**（DeepEP normal 模式）→ 不能把 Graph 参数当成生效。
4. **两机对比要交换节点交叉复测**，否则不能排除机器差异（9-28 没做，是遗留问题）。
5. **`temperature=0` 仍有逐 token 差异**（原因未定位）→ 比较 ON/OFF 时用"统计量"而不是"逐 token 相同"。
6. **两机工作目录不共享**：远端日志要在结束后同步回来。
7. 四台机是**短租**（10 月底可能被收走）→ 要抓紧排期。

---

## 9. 时间预估（拍板后）

```
准备（数据集 + bench + serve/run 脚本）        ~2 小时（不占机器）
Gate 1 + Gate 2（含起服务 ~15 分钟）           ~1 小时（2 台机）
正式 S1~S4（ON/OFF 并行，每场景 45 分钟）       ~3 小时（2 台机）
（可选）S5 对照 + MTP steps 扫描                ~2 小时
```
