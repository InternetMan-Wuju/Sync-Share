# Online 配额与 Decode EP 代理实验：审阅报告

- run_id：`20261103_051202`
- 执行时间：2026-11-03 05:12:02–05:32（+08:00），硬截止 08:12:02
- 资源：单张 Ascend NPU，仅绑定 `/dev/davinci0`
- 设备：`Ascend910_9382`，容器内 `device_count=1`
- 源提交：`3eeb16b`
- 路径身份：**真实 NPU 上执行的 eager 局部专家链代理**
- 生产服务 graph：未验证
- 多卡 HCCL/dispatch/combine：未测

## 0. 给审阅者的结论

### 可以支持的结论

1. Online 双 rank、双专家机制 case 中，当前计数规划确实能产生局部计算空间：
   - 960/64 → 512/512；
   - 并行计算代理 0.3158 → 0.2478 ms，节省 **0.0679 ms（21.5%）**。
2. 但同一路径的 CPU planner + 本地 state copy 已经超过这项局部收益：
   - planner 0.00104 ms；
   - local copy 0.07499 ms；
   - 串行剩余 **−0.00808 ms**。
3. Decode batch1 扩大 EP 没有正计算收益；batch8 的 EP32/EP64 出现约 **0.040–0.041 ms/层**的回本预算。
4. 真实 layer49 decode dump 的契约是：每 step 128 tasks、33–44 个非零专家；它**不是**单 token × TP16 重复，更像 16 个有效 token × top8。

### 不能支持的结论

1. 不能宣布 UltraEP online 在生产服务中净收益。
2. 不能把本地 copy 0.07499 ms 当成远端 HCCL 复制时间。
3. 不能把 decode 六格称为“真实 decode trace 六格”；六格路由是固定种子人工输入。
4. 不能输出 EP64 实际 TPOT，也不能把单层代理乘 75。
5. 不能把生产 graph 与本报告 eager 链等价。

## 1. 证据等级

| 项目 | 本轮实测 | CPU 实算 | 人工输入 | 旧数据复用 | 未测 |
|---|---|---|---|---|---|
| Online 配额 | 是 | 是 | 是 | 否 | 全局交换 |
| Online 完整本地链 | 是（NPU） | 否 | 是 | 否 | 生产 graph |
| Online local copy | 是（NPU） | 否 | 真实 e43 状态 | 否 | 远端 HCCL |
| Decode 工作类型耗时 | 是（NPU） | 否 | 代表权重 | 否 | 真实 token route |
| Decode 六格路由 | 否 | 是 | 固定种子 | 否 | 真实 trace 六格 |
| 真实 dump 契约 | 否 | 是（逐 step 读取） | 否 | 是 | token top8 |

### NPU 实测内容

所有性能耗时均由 `torch.npu.Event` 在真实 NPU 上采集：

- Online e43/e40：真实 W8A8 权重；
- dynamic quant → grouped GMM1 → SwiGLU → dynamic requant → GMM2；
- 独立 buffer 的权重/scale copy；
- `npu_format_cast` 的 format prep；
- Decode 45 个去重工作类型，均使用完整 16/8/4 槽向量。

### CPU 实测内容

- 物理槽余数分配与任务守恒；
- row quota 独立整数枚举；
- 两专家 CPU planner；
- 固定种子人工路由生成；
- 六格工作量聚合。

### 人工输入部分

- Online hidden 是固定 shape 的人工激活；960/64 和 512/512 是机制输入，不是线上 trace。
- Decode top8 是固定种子人工路由；batch8 为离线重组，不是真实 token 共选。

## 2. Online 实验

### 2.1 方法和对照

两个逻辑专家：

- e43：owner rank0；
- e40：owner rank1。

每个 rank 最多一个临时槽。空 adapter 与候选保持相同双专家 residency/workspace；空 adapter 的全部 extra slot 任务严格为 0。

热点输入：

```text
原布局 A：
rank0 = 960 × e43
rank1 =  64 × e40

新布局 B：
rank0 = 512 × e43
rank1 =  64 × e40 + 448 × e43
```

### 2.2 完整本地链实测

| 指标 | A 原布局 | B 新布局 | 变化 |
|---|---:|---:|---:|
| rank0 | 0.3158 ms | 0.2277 ms | −27.9% |
| rank1 | 0.2237 ms | 0.2478 ms | +10.8% |
| `max_r(local_chain)` | **0.3158 ms** | **0.2478 ms** | **−0.0679 ms / −21.5%** |
| 顺序和 | 0.5395 ms | 0.4755 ms | 非 EP 时间，仅参考 |

单卡顺序测量，`max_r`只是两边可并行时的计算代理，不是真实跨 rank 时延。

### 2.3 Online 成本账

| 成本项 | 实测 | 是否进入真实关键路径 |
|---|---:|---|
| CPU planner | 0.00104 ms/次 | 未来可能被全局交换/重叠改变 |
| e43 独立 state copy | 0.07499 ms | 本地 device copy，不是远端 |
| 复制字节 | 37,769,216 B | 完整 w13/down/scale |
| format prep | 0.12447 ms | 是否每步支付未证明 |
| 全局计数交换 | 未测 | — |
| 远端 HCCL copy | 未测 | — |
| dispatch/combine 增量 | 未测 | — |
| 并行争用 | 未测 | — |

串行模型：

```text
Δcompute                         = 0.06795 ms
planner + local copy             = 0.00104 + 0.07499
residual before any network      = -0.00808 ms

再加 format prep:
0.06795 - 0.00104 - 0.07499 - 0.12447 = -0.13255 ms
```

因此，本轮能确定的是：

> 局部计算空间约 21.5%，但在单卡实测的串行本地准备模型中，在线收益尚未为正；远端成本和重叠方式尚未测。

### 2.4 均匀 512/512 对照

| 指标 | A 原布局 | B 新布局 |
|---|---:|---:|
| 计算代理 | 0.2424 ms | 0.2395 ms |
| planner | 1.03 µs | 1.03 µs |
| state copy | 0 | 0 |
| extra slot 任务 | 0 | 0 |

差异约 0.0029 ms，处于本轮波动范围；没有可证明收益，但也没有发生复制或额外 slot 任务。

### 2.5 Online 正确性

| 检查 | 结果 |
|---|---|
| e43/e40 offset | 全部为 0 |
| source scale / kernel scale | float32 / bfloat16 |
| dense 对 FP32 反量化参考 | NRMSE 1.2385% / 1.3189% |
| S1 grouped 对 dense | 0 / 4.63e-6 |
| fused timed chain 对 dense | 0 / 4.63e-6 |
| 9/8 分流恢复 | 逐位一致 |
| 空分支 / 单行 | 通过 |
| copy 后状态 | 四类状态逐位一致 |

## 3. Decode 六格

### 3.1 输入和映射

- 固定 layer49、256 个逻辑专家、top8、无副本。
- 固定种子：20261103。
- 每个 EP 使用同一份逻辑 token route。
- 槽数：EP16/32/64 = 16/8/4。
- 每格 8 个 step：4 个 dev + 4 个 holdout。
- batch8 是人工 batch1 样本的离线重组。

### 3.2 六格结果

| EP / batch | 活跃 rank | R | 计算代理 median | dev / holdout | 相对 EP16 | 通信/调度预算 |
|---|---|---:|---:|---:|---:|---:|
| EP16 / 1 | 6–7 | 4.0–6.0 | 0.2017 ms | 0.2017 / 0.2017 | — | — |
| EP32 / 1 | 6–8 | 4.0–8.0 | 0.2061 ms | 0.2061 / 0.2061 | −2.22% | 0 |
| EP64 / 1 | 7–8 | 8.0–16.0 | 0.2218 ms | 0.2190 / 0.2218 | −10.0% | 0 |
| EP16 / 8 | 15–16 | 1.5–2.25 | 0.2628 ms | 0.2825 / 0.2628 | — | — |
| EP32 / 8 | 26–29 | 2.5–3.0 | 0.2423 ms | 0.2423 / 0.2325 | 15.27% | **0.04013 ms/层** |
| EP64 / 8 | 38–44 | 3.0–4.0 | 0.2218 ms | 0.2218 / 0.2218 | **15.58%** | **0.04095 ms/层** |

### 3.3 解释

- batch1 只有 8 个不可分 token-expert tasks。EP16/32/64 都有约 6–8 个活跃 rank，固定 grouped-kernel 成本主导，没有正收益。
- batch8 有 64 个 tasks，EP32/64 把每 rank 工作压小后才出现约 40 µs/层空间。
- 这 40 µs 必须支付新增 dispatch/combine、HCCL、同步和争用；它不是净收益。
- 没有源 rank 映射，因此不计算理论 payload；没有网络标定，不报 EP64 TPOT。

### 3.4 计时覆盖

- 45 个工作类型全部在 NPU 上实测。
- 20 次预热、5 个正式块、每块至少 20 次调用。
- 全部正式块 ≥100 ms。
- 槽位置换误差：EP16 0.061%、EP32 0.320%、EP64 0.210%。

## 4. 真实 decode trace 检查

源文件：

`/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/logs/20261102_081322/eplb_windows/000000_1793579172.815737.pt`

逐 step 读取结果：

- 64 个 decode step；
- 每个 step 都是 128 tasks；
- 每 step 33–44 个非零专家；
- counts 并非全部 16 的倍数；
- “单 token × TP16 重复”解释被排除；
- 更像 16 个有效 token × top8。

缺口：dump 只有专家直方图，没有 token top8、router weight、token 共选、padding 和源 rank。因此不能恢复该轮的 batch1/8 逻辑输入。六格实验的真实 trace 版本仍标为**未完成**。

## 5. 修正记录

1. 物理槽余数分配改为按 physical index；old/corrected 映射的 32768 个任务都严格守恒。
2. 空 adapter 的 17th extra slot 全部强制为 0。
3. 旧 row quota 重复计入热点的问题已修正；独立整数枚举得到：
   - calibration：max 3983，配额 3367/2790；
   - holdout：max 3943，配额 3371/3015。
4. S1 grouped 与本轮 fused 计时链已通过同一输入的数值桥。
5. 计时试跑中低于 100 ms 的块全部丢弃；最终 block floor 复核通过。

## 6. 审阅检查点

请本地 AI 重点检查：

1. `online_npu.py` 中 planner 是否确实从 host counts 开始，是否遗漏 D2H/全局交换。
2. `rank1_bank` 的 extra slot 是否与空 adapter 和候选使用相同 residency/workspace。
3. 0.0679 ms 的 `max_r` 计算代理是否与 A/B 局部链测量边界一致。
4. local copy 0.07499 ms 是否应视为每步支付、一次性支付或可重叠。
5. 六格中工作类型按 counts multiset 去重是否足以代表完整 16/8/4 槽向量。
6. R 定义是否为 `max rank tasks / (total tasks / EP)`。
7. 真实 dump 的“batch16 × top8”推断是否有其他可行解释。
8. 哪些结论仍需至少两块 NPU 或生产 graph 才能闭环。

## 7. 证据索引

| 文件 | 用途 |
|---|---|
| `MANIFEST.json` | 环境、版本、输入身份、正确性、文件哈希 |
| `CPU_AUDIT.json` | 守恒、空 adapter、配额枚举、真实 dump 检查 |
| `INPUTS_PLANS.json` | Online 计划、完整 decode 16/8/4 槽向量、人工输入身份 |
| `ONLINE_RESULT.json` | Online 原始结构化结果 |
| `DECODE_RESULT.json` | 六格、45 个工作类型、槽位置换验证 |
| `TIMINGS.csv` | 305 条原始计时/planner 记录 |
| `RESULTS.md` | 精简结果报告 |
| `CORRECTIONS.md` | 本轮修正细节 |
| `REPRODUCE.md` | 最小复现命令 |
| `cpu_audit.py` / `online_npu.py` / `decode_npu.py` | 最小复现代码 |

## 8. 一句话审阅结论

本轮完成了**真实 NPU 上的 eager 局部机制验证和成本账**：online 有约 21.5% 的局部计算空间但本地串行准备已吃掉收益；decode batch8 的 EP32/64 有约 40 µs/层的通信回本预算；真实 trace 六格仍因缺少 token-level 路由而未完成。
