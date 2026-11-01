# Q4 实验总结报告 —— 任务 3+4 联合实验

**实验时间**：2026-10-12 04:41 ~ 07:38（8 轮，共约 3 小时）
**硬件**：4 台机器 / 64 张 NPU，`quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`
**模型**：GLM-5.2（`/data_lib/data/models/glm-q`，78 层，前 3 层 dense，256 专家，top-8）
**并行**：`--tp-size 64 --dp-size 4 --ep-size 64 --enable-dp-attention`，DeepEP auto，PD 共置
**产物**：`current_exp/pd_colloc_20261011/results/`（4.1G，8 轮全量数据）+ 数据索引 `results/Q4EXP_INDEX.md`

---

## 0. 一页纸结论

1. **EPLB 的负优化在真机上完整复现**：开启 EPLB 后，rank 负载从 **1.33× 跳到 5.63~5.69×**（agentX），
   random 负载下同样从 **1.44× 跳到 6.11×**，且**每个后续周期都稳定在这个量级**。
2. **根因不是"算法写错"，而是目标不等价**：真实指标是"**跨层求和之后**的最大 rank 负载"，
   而 EPLB 是**逐层独立**求解"每层各自的最大负载"——逐层独立 + 每层都从 rank 0 开始 → 每层尖峰对齐到同一列 →
   原本靠"各层歪向不同方向"的跨层抵消被破坏。
3. **单层存在物理下限**：冗余=0 时最热专家只能独占一张卡，它一个人 = 每卡平均的 **6.90 倍**（中位，最大 8.0×），
   任何摆放算法都突破不了 → 解释了为什么各种算法都落在 5~7×。
4. **解法有效**：`--ep-num-redundant-experts 64`（每卡 4→5 槽）把负载**拉回 1.26~1.38×**（agentX）/ **1.30~1.37×**（random）。
5. **代价与副作用**：负优化真实拖慢吞吐 **4%（agentX）/ 7%（random）**；冗余专家要 **+25% 专家权重显存**（KV cache 变小）。
6. **测量本身的代价可忽略**：完全不开采集（A0）vs 开采集（A）吞吐只差 **1~2%**。
> 纯算法层面的形式化、最小反例与离线仿真见 **§5.9**；可外发的问题陈述在 `eplb_algo_sim/PROBLEM_EPLB负优化.md`。

7. **random 组是"证伪实验"但它没证伪**：prompt 是纯随机 token（无语义结构），EPLB 照样把负载搞到 6.11×
   → 说明"热点集中在少数专家"是**模型/路由自身的性质，与请求内容无关**。

---

## 1. 任务与目标

| 任务 | 要求 | 本次实验如何覆盖 |
|---|---|---|
| **任务 3**（PD 共置 + 热力 + 经典指标） | 4 机 PD 共置能跑；拿 TTFT/TPOT/E2E；拿到 token→layer→expert 热力；PD 分开统计 | A 组（EPLB 关）给出正式性能数字 + 逐周期热力；**PD 全程分开统计** |
| **任务 4**（EPLB 负优化归因 + 解法） | 找出"为什么 EPLB 反而更差"；给出可行修法 | A/B/C 三组对比 + 机制验证（源码/数据/仿真三条独立证据链） |

---

## 2. 实验设计

### 2.1 四组配置（每组 15 分钟、并发 8、0 错误）

| 组 | EPLB | 采集 | 冗余专家 | 说明 |
|---|---|---|---|---|
| **A0** | 关 | **关** | 0 | 纯性能基线（证明"采集不干扰"） |
| **A** | 关 | 开 | 0 | 热力基线；布局 = identity（槽 p 装专家 p） |
| **B** | 开 | 开 | 0 | 负优化主证据（`--eplb-rebalance-num-iterations 2000`） |
| **C** | 开 | 开 | **64** | 解法验证（每卡 4→5 槽，物理槽 256→320） |

### 2.2 两种负载

| 负载 | 输入/输出 | 说明 |
|---|---|---|
| **agentX** | 16384 / 512 | 真实业务回放（带 trace 前缀共享结构），与上一轮实验可比 |
| **random** | 8192 / 64 | 纯随机 token id（请求之间**无共享前缀**）——即 sglang bench 的 random-ids 语义 |

同组不同负载使用**固定数据集 + 固定请求顺序**，四组之间完全一致。

### 2.3 采集内容（每个 run 一个目录）

- `windows/*.pt`：热度窗口（见 §4 定义），含 prefill/decode 分开的逻辑专家计数 + 该窗口生效的映射
- `metrics/*.txt`：每 60 秒抓一次服务端 `/metrics`（TTFT/TPOT/E2E/queue/retract/kv…）
- `npu/*.txt`：每 60 秒抓一次四台机器的 `npu-smi`
- `bench/summary.json`、`bench/requests.jsonl`、`bench/sse/*.gz`：逐请求指标与原始流式事件
- `logs_node_*.log`：四个节点的服务日志（重排事件在里面）
- `orchestrator.log`：跑批过程（含每次 dump 的耗时）

---

## 3. 口径与定义（重要，容易算错）

1. **热度矩阵** `H[l][e]` = 该窗口内第 l 层、第 e 个**逻辑专家**被路由到的 **token 计数**（对"步"求和）。
2. **计数单位**：1 个 token 过一遍模型 = **75 层 × top-8 = 600 次计数**
   （例：`29491200 = 3 × 16384 × 75 × 8`，前 3 层 dense 无 MoE）。
3. **rank 负载**：`rank_load[r] = Σ_l Σ_{p ∈ rank r} H[l][pmap[l][p]]`，
   其中 `rank = 物理槽号 // (物理槽数 / ep_size)`（A/B 组每卡 4 槽，C 组每卡 5 槽）。
   指标 = `max/mean`（1.0 = 完美均衡）。
4. **周期**（= 重排间隔）：`开始 →[周期1]→ 重排1 →[周期2]→ 重排2 →…`。
   `buffer-size` 提到 2000 = 一个周期，所以 B/C 的每个 `eplb_rebalance` 窗口 = **一个完整周期**（≈2000 步）。
   A 组没有重排 → 用 60 秒切片窗口，做"周期统计"时把相邻窗口相加即可（支持"周期1、周期1+2…"任意增量）。
5. **⚠️ PD 不能按步平均**：prefill 一步 = 16384×块数 个 token；decode 一步 ≈ 几十个 token（差 3 个数量级），
   所有统计必须按 token/计数加权。
6. **⚠️ E2E breakdown 不要重复计排队**：`TTFT = first_token_time − created_time` **已经包含排队**，
   正确写法 `E2E = TTFT + decode`；若按 `queue + TTFT + ITL×(n−1)` 相加会把排队算两次
   （历史上"其他占 25%"很可能就是这么来的）。

---

## 4. 结果

### 4.1 性能（每组 15 分钟、并发 8、0 错误）

| 组 | 负载 | 请求数 | 吞吐 | TTFT p50 | TPOT p50 | E2E p50 |
|---|---|---|---|---|---|---|
| A0 | agentX | 168 | **91 tok/s** | 20.3s | 50ms | 46.0s |
| A | agentX | 160 | 90 | 20.4s | 51ms | 46.5s |
| **B** | agentX | 160 | **87** | 20.3s | 50ms | 46.0s |
| C | agentX | 160 | 88 | **17.0s** | 51ms | **43.3s** |
| A0 | random | 624 | **44** | 7.1s | 50ms | 10.2s |
| A | random | 608 | 43 | 7.8s | 50ms | 10.9s |
| **B** | random | 584 | **41** | 7.2s | 50ms | 10.3s |
| C | random | 608 | 43 | **5.7s** | 51ms | **8.9s** |

**读法**：
- 采集代价（A0→A）：**91→90（−1%）**、**44→43（−2%）** → 测量基本不干扰。
- 负优化的代价（A0→B）：**91→87（−4.4%）**、**44→41（−6.8%）** → 每步都在等最热的那张卡。
- C 组在两种负载下 TTFT/E2E 都更好（agentX：20.4s→17.0s、46.5s→43.3s；random：7.8s→5.7s、10.9s→8.9s），
与其更均衡的 rank 负载一致（注意 C 组 KV cache 更小，这个对比不是严格同条件）。

### 4.2 每周期 rank 负载 max/mean

| 组 | agentX | random |
|---|---|---|
| A | **1.27~1.38×**（14 个 60s 窗口，全稳） | **1.43~1.45×**（13 个窗口，全稳） |
| B | 1.33×（重排前）→ **5.63 / 5.68 / 5.63 / 5.69×** | 1.44×（重排前）→ **6.11 / 6.11×** |
| C | **1.26~1.38×** | **1.30~1.37×** |

（B/C 的窗口是完整周期 ≈2000 步；A 的窗口是 60 秒切片，agentX 中位 765 步、random 355 步。）

### 4.3 PD 分开（prefill / decode 各自的不均衡）

| 组 | 负载 | prefill | decode | 合计 |
|---|---|---|---|---|
| A | agentX | 1.44~1.50× | 1.60~1.61× | 1.27~1.38× |
| A | random | 1.48~1.49× | 1.62~1.63× | 1.43~1.45× |
| **B** | agentX | **6.19~6.27×** | 3.83~4.24× | **5.63~5.69×** |
| **B** | random | **6.31×** | 2.91~2.93× | **6.11×** |
| C | agentX | 1.33~1.46× | 1.55~1.74× | 1.26~1.38× |
| C | random | 1.30~1.42× | 1.36~1.72× | 1.30~1.37× |

两个可读点：
- **EPLB 之后 prefill 比 decode 更歪**（6.2× vs 4.0×）：摆放是按"两阶段合计"算的，而 **prefill 占 72~88% 的 token 量**，
  所以摆放实际是"为 prefill 优化"的。
- **合计反而比 prefill 单独好**（5.6× < 6.2×）：因为两阶段热点不同，在合计里互相抵消了一部分。
  换句话说：如果这个服务只跑 prefill，EPLB 会更糟。

### 4.4 重排事件与耗时（服务日志）

| 组 | 重排耗时 |
|---|---|
| B / agentX | 1.6s、**48.7s** |
| B / random | 1.4~1.5s、**47.3s** |
| C / agentX | 2.4s、11.5s、**47.0s** |
| C / random | 8.6s、**51.8s** |

**⚠️ 这个数字被我们的采集污染了**：`rebalance end time=` 的计时包含 `dump_record()`，
而 `dump` 内部有一次**全 64 rank 的 all_reduce**（buffer=2000 时单次要同步 ~320MB），
必须等所有 rank 到达同一调用点 → 只要有一个 rank 正在做长 prefill，全体就得等。
所以 **~48s 那一笔很可能是 dump 的等待，而不是搬权重本身**。
（A0/A 组 60 秒切片的 dump 只要 0.5~9.5s；把 buffer 从 1024 提到 2000 也放大了这个成本。）

### 4.5 采集代价明细

- A 组 agentX：13 次 dump，合计 **23.3s / 900s**，中位 0.94s、最大 9.5s。
- A 组 random：dump 明显更贵（6.7~7.4s），因为 buffer 更容易被填满（random 步数更快）。
- 净影响：吞吐 **−1~2%**，TTFT p50 +0.7s（random 7.1→7.8s）。

---

## 5. 归因：为什么 EPLB 让负载更歪

### 5.1 单层的物理下限（无解的那部分）

冗余=0 时每个逻辑专家只能占 **1 个物理槽**，于是一层里最热的那个专家**只能待在某一张卡上**。
实测：最热专家 ÷ 该层每卡平均 = **中位 6.90×，最大 8.0×**。
→ **任何摆放算法都不可能让单层的 max/mean 低于 ≈7×**。这解释了为什么所有算法（deepseek / deepseek_vec / 加蛇形 / 去蛇形）
都落在 5~7×。

### 5.2 不开 EPLB 为什么反而是均衡的（1.3~1.4×）

初始布局 = identity（槽 p 装专家 p），**各层最热的是不同的专家编号**，它们的 rank 散布在 64 个 rank 上
（实测：A 组 14 个窗口里，「含最热专家的 rank 恰为 rank0」的层占比 = **0%**，最高负载 rank 也是 0%）。
于是"第 l 层的尖峰落在 rank 37""第 m 层的尖峰落在 rank 12"——**跨层求和后互相抵消**。
这是**统计意义上的均衡**，不是每层都均衡。

### 5.3 EPLB 把抵消破坏了（5.6~6.1×）

现行算法（`--eplb-algorithm auto` 在本模型下选中 `deepseek`）：
```
对每一层 独立地：
   把 256 个专家按热度从高到低排序
   依次"发牌"：每张发给"当前最轻、且未满"的包（每包 = 一个 rank，容量 = 每卡槽数）
                （平局取编号最小的包）
   ⇒ 每层的第一个（最热）专家必然进 pack0 = rank 0
```
- **逐层独立**：每层独立求解 → 看不到"跨层的和"；
- **确定性 + 每层从空包开始**：每层的最热专家都被放到 **rank 0** → **所有层的尖峰对齐到同一列** → 抵消消失。
  实测该指标：A **0%** → B 重排后 **79%**（agentX）/ **100%**（random）→ C（冗余）**9%**。

变体 `deepseek_vec`（排序 → (槽×rank) 矩阵 → **奇数行反转** → 每列一个 rank）也一样：
反转只发生在奇数行，**第 0 行不反转** → 最热那个照样进 rank 0。

### 5.4 三级独立证据

| 证据类型 | 内容 | 结论 |
|---|---|---|
| **真源码** | 镜像里的 `eplb_algorithms/{deepseek,deepseek_vec}.py`、`expert_location.py`（sha256 与容器一致，**未打补丁**；补丁清单里没有算法文件） | 逐层独立、每层从 rank 0 开始 |
| **真实数据** | 8 轮实测的窗口数据（本报告 §4） | A 1.3~1.4× vs B 5.6~6.1×；**「每层最热专家所在的 rank 恰为 rank0」的层占比：A 0% → B 79%(agentX) / 100%(random) → C 9%** |
| **离线仿真** | 用实测 W[78,256] 直接喂 stock 算法 | `deepseek` 6.32×、`deepseek_vec` 6.46×、去蛇形 6.61×；冗余 R=64 → 1.37×（另一口径复算 1.20×） |

### 5.5 专项验证：EPLB 真的是逐层独立的吗？

**是**，两路验证：
- **代码**：所有排序/求和/取最大都发生在**专家轴**；每层贪心状态在进入该层时**重新初始化**
  （`balanced_packing` 的 `pack_weights/pack_items`、`pack_groups` 的 `node_tokens/node_groups`、
  `replicate_experts` 的 `logcnt` 按行）。`deepseek_vec` 里的 `sum(0)` 是**步（时间）轴**，不是层轴。
- **操作验证**（`eplb_algo_sim/verify_layer_independence.py`）：同一份数据三种跑法
  —— 78 层一起跑 / 每层单独跑 78 次 / 打乱层序跑 —— **5 条算法路径 × 2 组数据（真实+随机）全部逐层完全一致**。

### 5.6 专项验证：真实系统是"过期快照"循环，结论还成立吗？

真实运行是：跑一段 → 累计热度 → 重排 → **服务新流量（热度不会重复）** → 再重排…
用真实相邻窗口做**滚动评估**（`eplb_algo_sim/walkforward_stale_snapshot.py`）：

| 口径 | identity（固定布局） | EPLB | 冗余64 |
|---|---|---|---|
| 自评（用同一窗口的热度摆+评估） | — | 5.78× | 1.28× |
| **过期 1 个窗口**（用 t 的摆放服务 t+1） | 1.31× | **4.98×** | 1.32× |
| 过期 2 / 3 个窗口 | | 4.92× / 5.28× | 1.32× / 1.31× |
| 稳态口径（跨窗口累加） | 1.31× | 4.59× | **1.16×** |
| **真机实测**（同批窗口） | 1.31× | **5.60×** | （未跑） |

结论：**"过期"不会救它**——因为伤害来自"每次重排都把尖峰对齐到 rank 0"这个**结构性行为**，与具体热度无关；
而且滚动评估能**逐窗口复现真机的数字**（如某窗口：仿真 4.84 vs 真机 4.84）。

### 5.7 专项验证：prefill 和 decode 用的是**两批不同的专家**

| 比较 | 逐层 top-8 重合 |
|---|---|
| decode 前半 vs 后半（同阶段内） | 99.2% |
| prefill 窗口0 vs 窗口1（同阶段内） | 91.8% |
| **prefill vs decode（同一窗口）** | **14.7%**（**比随机基准 25% 还低**；75 层里 20 层完全不重合） |

- 且两阶段形状不同：prefill 覆盖该层一半的量需要 **20 个专家**，decode 只要 **6 个**（decode 更"抱团"）。
- 含义：PD 共置下**只有一套专家、一份摆放**，而 EPLB 按"合计热度"（prefill 占 72~88%）摆 → 实际上**偏袒 prefill**。
- 顺带：冗余副本按"合计热度"挑，结果是**覆盖 prefill 的 top-8 = 100%、decode 的 top-8 = 87%**（两批都照顾到了）；
  如果只按 prefill 挑，decode 只有 28% 被覆盖。

### 5.8 random 组：证伪实验（没证伪，反而加固）

random 负载的 prompt 是均匀随机 token、请求之间无共享前缀。如果"热点"来自数据语义，random 下应该没有热点、EPLB 不该作恶。
**实测：random 下 A=1.43~1.45×，B=6.11×** → 负优化照旧。
→ **"少数专家被集中使用"是模型/路由自身的性质，与请求内容无关**（也解释了为什么"换业务数据"也躲不开这个问题）。


### 5.9 纯算法视角：把它抽象成一个算法问题（形式化 + 最小反例 + 离线仿真）

**形式化**
```
输入:  W[L, E]        热度矩阵（L=78 层, E=256 专家），实测得到
       R = 64         rank 数
       k = E / R = 4  每个 rank 的槽数
约束:  摆放是一个双射 φ : {槽} → {专家}
       （冗余 = 0 时，每个逻辑专家恰好占 1 个槽，不可复制）
       rank r = 槽 {k·r, ..., k·r + k − 1}
目标:  minimize  max_r ( Σ_l Σ_{p ∈ rank r} W[l, φ(p)] ) / mean
现状:  对每个 l 独立求解 min-max(单层)。
       这优化的是 L 个【单层】代理目标，而真实指标是【跨层求和之后】的最大值。
```

**最小反例（4 rank / 每 rank 2 槽 / 8 专家 / 2 层）** —— 可运行：`eplb_algo_sim/demo_eplb_matrix.py`
```
层A = [100,20,10,5,1,1,1,1]  (热点 = 专家0)
层B = [1,1,1,1,5,10,20,100]  (同一组数值，热点 = 专家7)

① 不开 EPLB（槽 p 装专家 p，连续取值）
     层A 负载 = [120, 15,  2,  2]   （单层 3.45×）
     层B 负载 = [  2,  2, 15, 120]   （单层 3.45×）
     合计     = [122, 17, 17, 122] → 1.76×   ✅ 两层歪的方向相反，抵消
② 开 EPLB（逐层独立，每层最热进 rank0）
     层A 负载 = [101, 21, 11,  6]
     层B 负载 = [101, 21, 11,  6]
     合计     = [202, 42, 22, 12] → 2.91×   ❌ 歪的方向被对齐，抵消消失
```
一句话：**单层并没有变差，变差的是"合计"** —— 这正是"逐层独立求解"这个抽象本身的缺陷。

**为什么 stock 代码必然把最热专家放 rank 0**（源码位置 + 伪代码）
```python
# eplb_algorithms/deepseek.py :: balanced_packing()
for i in range(num_layers):                 # ← 每层【独立】
    pack_weights = [0] * num_packs          # ← 每层从【空包】开始
    pack_items   = [0] * num_packs
    for group in 该层按热度【降序】:
        pack = min( j for j in 未满的包, key=包当前总负载 )   # ← 平局取【最小编号】
        pack_items[pack] += 1; pack_weights[pack] += w
```
- "每层从空包开始" + "平局取最小编号" ⇒ **每层的第一个（最热）专家必然进 pack 0 = rank 0**。
- 变体 `deepseek_vec`：排序 → (槽 × rank) 矩阵 → `indices[:, :, 1::2, :].flip(-1)`（**只反奇数行**）→ 每列给一个 rank。
  第 0 行**不反转** ⇒ 最热那个照样进 rank 0。（这点曾有过争论，最后用源码 + 实跑确认。）

**离线仿真：把实测的 W[78,256] 直接喂 stock 算法**
| 算法 | 合计 max/mean |
|---|---|
| `deepseek`（auto 实际选中的那个） | 6.32× |
| `deepseek_vec`（带蛇形） | 6.46× |
| `deepseek_vec` 去掉蛇形 | 6.61× |
| 冗余 R=64（同一算法） | 1.37×（另一口径复算 1.20×；**本次真机实测 1.26~1.38×**） |
| 参照：identity 布局（不开 EPLB） | 1.50× |
→ **换算法救不了**（6.32 / 6.46 / 6.61 全在同一量级，蛇形只值 2%）；**加冗余能救**。

**为什么换算法救不了**：单层的物理下限 —— 最热专家 ÷ 每卡平均 = 中位 **6.90×**（最大 8.0×）。
冗余=0 时它只能待在一张卡上，任何摆放算法都突破不了 ≈7×。

**可运行的演示程序与素材**

| 脚本/文件 | 内容 |
|---|---|
| `eplb_algo_sim/demo_eplb_matrix.py` | 上面那个最小反例（初始连续 / 初始交错 / 蛇形 / 贪心 / 反例 五种摆法对照） |
| `eplb_algo_sim/demo_eplb_random.py` | 随机 4 层 × 32 专家，逐层分解成子矩阵并标注所属 rank（1.66× → 4.23×） |
| `eplb_algo_sim/PROBLEM_EPLB负优化.md` | **可直接外发的算法问题陈述**（形式化 + 最小反例 + 仿真表 + 想问别家 AI 的 5 个问题） |
| `eplb_algo_sim/verify_layer_independence.py` | 逐层独立性的操作性验证（见 §5.5） |
| `eplb_algo_sim/walkforward_stale_snapshot.py` | "过期快照"滚动评估（见 §5.6） |
| `pd_colloc_20261011/patches/stock_alg_ref/` | 从镜像取出的 stock 算法（**sha256 与容器一致**，仿真跑的就是它） |

> ⚠️ **证据分级**：§4 全部是**实测**；本节两张表是**仿真**（真代码 + 真实 W，但不在服务里跑）；
> 两个 `demo_*.py` 用的是**我编的小数据**，只用于把机制讲清楚，**不作为证据**。

---

## 6. 解法：冗余专家（`--ep-num-redundant-experts 64`）

### 6.1 机制与分配方式

- 物理槽 256 → **320**；`320 / 64 rank = 每卡 5 槽`（**每张卡多一个槽**，不是"每个节点多一个"）。
  约束：`(256+R) % 64 == 0` → **R 必须是 64 的倍数**（R=16 非法）。
- 这 64 个多出来的槽是**全局副本池**，**每层各自**挑"本层最热的那些专家"复制。
  实测（某层）：只有 **11 个专家**带副本，最热那个拿了 **10 份**，这 11 个专家覆盖该层 **86%** 的热度。
- 一张卡上的副本槽数不固定：约 **69% 的(层,卡)组合是 1 个副本槽、17% 是 0 个、14% 是 2 个**。
- 复制后，某个热专家的多份副本会**分散到多张卡**（例：某层最热专家有 10 份，落在 rank 40~49），
  运行期 dispatch 优先用"本卡 → 本节点"的副本 → 尖峰被摊开。

### 6.2 实测效果

| 负载 | B（无冗余） | **C（冗余 64）** | A（不开 EPLB 参照） |
|---|---|---|---|
| agentX | 5.63~5.69× | **1.26~1.38×** | 1.27~1.38× |
| random | 6.11× | **1.30~1.37×** | 1.43~1.45× |

→ **冗余 64 把负优化彻底修掉，回到甚至优于"不开 EPLB"的水平。**

### 6.3 代价

- **专家权重显存 +25%**（320/256）→ 同样 `mem-fraction-static` 下 **KV cache 变小** → 可能触发抢占
  （需看 `num_retracted_reqs`），**不要把 KV 差异误当成 EPLB 的收益**。
- 重排要搬的权重更多 → 重排耗时更长（C 组实测 2.4s/11.5s/47s vs B 组 1.6s/48.7s，样本少，不能下定论）。
- 换专家/搬运的总工作量也 +25%。

---

## 7. 结论与建议

### 7.1 对任务 3 / 任务 4 的交付

| 任务 | 交付 | 状态 |
|---|---|---|
| 任务 3 | 4 机 PD 共置跑通；TTFT/TPOT/E2E 正式数字（两种负载）；逐周期 token→layer→expert 热力；**PD 全程分开** | ✅ |
| 任务 4 | 负优化归因（三级证据）+ 冗余专家解法实测验证 + 机制专项验证（逐层独立/过期快照/PD 两批专家/random 证伪） | ✅ |

### 7.2 对 EPLB 改进的建议（可以拿出去讨论）

1. **最小改动**：在 `balanced_packing` 里给每层加**轮换起点**（第 l 层从 rank `l mod R` 开始发牌），
   破坏"每层都从 rank 0 开始"的对齐。**预期收益大、改动极小**，值得先仿真验证（尚未做）。
2. **进阶**：跨层联合摆放（把 L 行放在一起最小化 `max_r(Σ_l)`），代价是求解复杂度和权重搬运量（可能搬很多）。
3. **现实首选**：**开冗余专家**（≥64）。实测 5.6× → 1.3×，代价是显存 +25%。
4. **PD 共置专属**：既然 prefill/decode 是两批热点而只有一份摆放，可考虑**按阶段加权**（例如给 decode 的计数加权）
   或在冗余选择里对两阶段做分层配额——现在完全被 prefill 主导。

### 7.3 对我们后续实验的建议

- 需要稳态、可比的性能数字时：**用 A0（不采集）**；需要热力/机制分析时才开采集。
- B/C 组：**只在重排时落盘**（stock 行为），避免定时 dump；若必须定时 dump，把 buffer 调小（1024）或降低频率。
- 任何"细分窗口"实验都要**显式报告 dump 的耗时**（本次 A 组 13 次共 23.3s）。

---

## 8. 遗留问题与下一步

1. **~48s 的重排耗时到底是"搬权重"还是"我们 dump 的等待"？** 需要在补丁里做**分段计时**
   （dump / 计算映射 / 搬权重各多久）。这是下次实验第一优先。
2. **轮换起点**（§7.2 建议 1）与**跨层联合摆放**都没做实验；建议先用真实 W 离线仿真给出预期收益。
3. **随机/长尾负载下冗余的覆盖**：本次 random 只跑了 3 个周期（random 步数少、重排慢），样本偏少。
4. **C 组的显存/抢占影响**没有系统测量（KV cache 变小是确定的，抢占是否发生要看 `num_retracted_reqs`）。
5. `C_random` 第一次启动失败（服务 10 分钟没起来，疑似上一轮刚停、NPU 未完全释放）——**两次运行之间建议加 60s 缓冲**。

---

## 附录 A：文件与路径清单

| 路径 | 内容 |
|---|---|
| `current_exp/Q4Exp/Q4Exp.md` | 实验方案（跑之前定的） |
| `current_exp/Q4Exp/Q4Exp_总结报告.md` | **本报告** |
| `current_exp/pd_colloc_20261011/run_all_q4.sh` / `run_q4exp.sh` | 跑批脚本（序列 / 单轮） |
| `current_exp/pd_colloc_20261011/analyze_q4exp.py` | 分析脚本（性能 + 逐窗口 rank 负载，PD 分开） |
| `current_exp/pd_colloc_20261011/results/Q4EXP_INDEX.md` | **数据索引**（8 轮目录、字段说明、已知数字） |
| `current_exp/pd_colloc_20261011/results/<RUN>/` | 8 轮全量数据（4.1G） |
| `current_exp/pd_colloc_20261011/serve4_dp.sh` | 起停服务（`EPLB`/`REC`/`REDUNDANT`/`RECD_BUF`/`LOGDIR_ROOT` 开关） |
| `current_exp/pd_colloc_20261011/patches/expert_distribution_phase_window.py` | 采集补丁（按阶段分开计数 + 每窗口带映射；由 `make_patch_phase_window.py` 生成） |
| `current_exp/eplb_algo_sim/verify_layer_independence.py` | 逐层独立性验证 |
| `current_exp/eplb_algo_sim/walkforward_stale_snapshot.py` | "过期快照"滚动评估 |
| `current_exp/eplb_algo_sim/period_stability.py` | 周期稳定性统计 |
| `current_exp/eplb_algo_sim/PROBLEM_EPLB负优化.md` | 问题陈述（可外发） |
| `current_exp/eplb_algo_sim/demo_*.py` | 两个最小演示程序 |

## 附录 B：复现步骤

```bash
cd /home/chengyetai/workspace/current_exp/pd_colloc_20261011
# 1) 起服务（示例：A 组）
EPLB=0 REC=1 RECD_BUF=2000 LOGDIR_ROOT=/data_lib/cyt ./serve4_dp.sh start <run_id>
# 2) 等 health=200，然后
curl -X POST http://10.120.72.19:30000/start_expert_distribution_record
# 3) 打负载 + 定时抓指标 + 定时 dump（= run_q4exp.sh 做的事）
./run_q4exp.sh A agentx 900 8
# 4) 分析
python3 analyze_q4exp.py results/<RUN_DIR>
```

## 附录 C：坑与注意事项（下次别再踩）

1. **数据集路径**：`bench_agentx.py` 的 `--dataset` 必须给绝对路径（相对路径拼错会让 bench 秒退）。
2. **每个 dump 会额外写一个 ~479MB 的 stock 文件**（`expert_distribution_recorder_*.pt`）→
   跑批脚本里**每次 dump 后立刻删掉**，否则 15 分钟能写出十几 GB。
3. **落盘位置**：容器内 root 写、宿主用户读；`/data_lib` 我们没有 sudo，是用 `docker run --rm --net host -v /data_lib:/dl ... mkdir` 建的 `cyt` 目录（4 台都建了）。
4. **两次运行之间留 60s**（NPU/显存释放），否则可能像 `C_random` 第一次那样起不来。
5. **长跑批要放 tmux**：直接 `nohup &` 会随会话被回收（本次踩过，白跑一轮）。
6. **`(256+R) % 64 == 0`**：EP64 下 R 只能是 64 的倍数；R=16 会被断言拒绝。
7. **`buffer-size ≤ eplb-rebalance-num-iterations`**（有断言）；把 buffer 调到 2000 会让 dump 更贵（见 §4.4）。

## 附录 D：指标定义（避免"其他"又占 25%）

```
E2E(服务端) = finished_time − created_time
TTFT        = first_token_time − created_time        ← 【已包含排队】
queue       = forward_entry_time − wait_queue_entry_time
TPOT/ITL    = 相邻 chunk 的间隔（get_interval）
⇒ 正确分解： E2E = TTFT + decode(≈Σ ITL)
  错误做法： queue + TTFT + ITL×(n−1)   ← 排队被算两次
```
服务端 `/metrics` 现成可用：`queue_time_seconds`、`time_to_first_token_seconds`、
`inter_token_latency_seconds`、`e2e_request_latency_seconds`、`per_stage_req_latency_seconds{stage=…}`；
要更细的阶段树可以开 `--enable-request-time-stats-logging`（按请求打印各阶段耗时）。

## 附录 E：我们跑的 EPLB 算法 & 全部可选项

### E.1 我们实际跑的 = `deepseek`（非层级）

证据链：
```
serve4_dp.sh 未传 --eplb-algorithm            → 默认 "auto"（日志: eplb_algorithm='auto'）
models/glm-q/config.json: n_group = 1
nnodes = 4
compute_algorithm(): (num_groups=1) % (num_nodes=4) = 1 ≠ 0 → EplbAlgorithm.deepseek   ← 实际跑的
未设置 --elastic-ep-backend → "auto" 不会被改写成 elasticity_aware
   （改写发生在 server_args._handle_elastic_ep()，仅当 elastic_ep_backend 非空）
ep_dispatch_algorithm 自动选为 "static"（有 a2a 后端时；日志一致）
```

### E.2 全部可选项：6 个算法（3 实现 × 2 变体）+ `auto`

| flag 值 | 实现 | 机制 | 在我们模型上可用？ |
|---|---|---|---|
| `deepseek` | `deepseek.py` | 逐层独立 LPT 贪心（降序发牌给"当前最轻的包"，平局取最小 rank） | ✅ **实际跑的** |
| `deepseek_hierarchical` | `deepseek.py` | 先按 group 分到 node，再在 node 内复制+打包 | ❌ 会断言失败（`1 % 4 ≠ 0`） |
| `deepseek_vec` | `deepseek_vec.py` | 排序 → (槽×rank) 矩阵 → **奇数行反转** → 每列给一个 rank | ✅ 能跑（离线仿真用过） |
| `deepseek_vec_hierarchical` | `deepseek_vec.py` | 同上 + 层级 | ❌ 同上 |
| `elasticity_aware` | `elasticity_aware.py` | 弹性 EP：部分 rank 掉线时按存活 rank 重摆 | ✅ 能跑；**全 rank 健康时与 `deepseek` 结果完全一致**（内部调用同一函数） |
| `elasticity_aware_hierarchical` | `elasticity_aware.py` | 同上 + 层级 | ❌ 同上 |

`auto` 规则（`eplb_algorithms/__init__.py::compute_algorithm`）：
```
num_groups 非空 且 能被 nnodes 整除  →  deepseek_hierarchical
否则                                →  deepseek
```
本模型 `n_group = 1`、4 机 → **永远落在非层级**，hierarchical 那三个既选不到也跑不起来。

### E.3 两个正交旋钮（别和"选哪个算法"混）

| 旋钮 | 可选值 | 我们的取值 | 作用 |
|---|---|---|---|
| `--ep-dispatch-algorithm` | `static` / `dynamic` / `fake` / `lp` | **static**（自动） | 一个专家有多份副本时，token 派给**哪一份副本** |
| `--ep-num-redundant-experts` | EP64 下必须是 64 的倍数 | A/B = 0，**C = 64** | 副本**数量** |

小发现：**开 `--enable-eplb` 而未传 recorder mode 时，代码会自动设成 `stat`**
（`server_args._handle_eplb_and_dispatch`）→ "EPLB 一开就必然在采集"，这也解释了采集补丁的必要性。
