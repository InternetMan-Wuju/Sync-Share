# 3 小时 Online / Decode EP 结果

- 起始：2026-11-03 05:12:02 +08:00
- 截止：2026-11-03 08:12:02 +08:00
- 资源：仅一张 Ascend NPU（容器内 `device_count=1`）
- 基础提交：`3eeb16b`
- 路径身份：**eager 局部专家链代理**；未验证生产 graph/封装
- 真实跨 rank、HCCL、dispatch/combine：未测
- 数字身份：本报告性能数字全部来自本轮；旧 45–48% 和旧 0.117 ms 只作背景，不与本轮计算收益拼接。

## 1. Online：当前计数 → 复制决策 → 整数配额

方法固定为两个逻辑专家、两个虚拟 rank、每个 rank 最多一个临时槽。输入为固定机制 case，人工 hidden；不是真实 prefill trace。

### 热点 960 / 64

| 项目 | 原布局 A（extra slot 全零） | 新布局 B（复制 e43 并分流） |
|---|---:|---:|
| rank0 | 960 × e43 | 512 × e43 |
| rank1 | 64 × e40 | 64 × e40 + 448 × e43 |
| rank0 完整本地链 | 0.3158 ms | 0.2277 ms |
| rank1 完整本地链 | 0.2237 ms | 0.2478 ms |
| 并行计算代理 `max(A_r)` | **0.3158 ms** | **0.2478 ms** |

- CPU planner：从 host 已有计数开始，**1.04 µs/次**。
- 配额：迁出 448，得到 512/512。
- 计算代理节省：**0.0679 ms（21.5%）**。
- 本地独立 buffer 复制：**0.07499 ms**，37,769,216 B，权重/scale 逐位一致。
- 必要 format prep：**0.12447 ms**（e43 的 w13/down 各一次）。
- 若 planner 与本地复制都在关键路径串行执行，计算收益剩余：
  `0.06795 - 0.00104 - 0.07499 = -0.00808 ms`。
- 若每题还串行支付 format prep，则该项本身已超过计算收益：
  `0.06795 - 0.00104 - 0.07499 - 0.12447 = -0.13255 ms`。

边界：本地复制不是远端 HCCL；format prep 可能是一次性/可重叠。全局计数交换、远端复制、dispatch/combine 增量、并行争用均未测。因此本轮不能宣布 online 净收益。

### 均匀 512 / 512

| 项目 | 原布局 | 新布局 |
|---|---:|---:|
| rank0 | 512 × e43 | 512 × e43 |
| rank1 | 512 × e40 | 512 × e40，extra slot=0 |
| 并行计算代理 | 0.2424 ms | 0.2395 ms |

- planner 仍执行：**1.03 µs**。
- 复制任务：**0**；复制字节：**0**。
- extra slot 任务：**0**。
- 计算差异 `+0.0029 ms`，在约 3 µs 的计时波动内；没有可证明收益。
- 两边仍保留相同双专家 residency/workspace，未通过卸载权重制造假对照。

### Online 正确性

- e43/e40 offset 全零；source scale float32，kernel scale bfloat16。
- dense 对 FP32 反量化参考 NRMSE：1.2385% / 1.3189%。
- S1 grouped、实际 fused timed chain 对 dense：最大 NRMSE 4.63e-6。
- e43 9/8 分流恢复：逐位一致。
- 空分支、单行输入：通过。
- 独立 buffer 复制后四类状态逐位一致。

## 2. Decode：EP16/32/64 × batch1/8

输入身份：**固定种子 20261103 的人工 top8 路由**；batch8 是由 batch1 样本离线重组，不是真实 decode token 路由。EP 改变时保持同一逻辑 token 路由；映射为连续 logical expert 到 16/8/4 个槽。

| EP / batch | 输入身份 | 活跃 rank | R（全体 rank） | 计算代理 median | dev / holdout | 相对 EP16 节省 | 允许新增通信/调度/层 |
|---|---|---:|---:|---:|---:|---:|---:|
| EP16 / 1 | 人工 batch1 | 6–7 | 4.0–6.0 | 0.2017 ms | 0.2017 / 0.2017 | — | — |
| EP32 / 1 | 同一逻辑路由 | 6–8 | 4.0–8.0 | 0.2061 ms | 0.2061 / 0.2061 | **−2.22%** | **0** |
| EP64 / 1 | 同一逻辑路由 | 7–8 | 8.0–16.0 | 0.2218 ms | 0.2190 / 0.2218 | **−10.0%** | **0** |
| EP16 / 8 | 人工 batch8，离线重组 | 15–16 | 1.5–2.25 | 0.2628 ms | 0.2825 / 0.2628 | — | — |
| EP32 / 8 | 同一逻辑路由 | 26–29 | 2.5–3.0 | 0.2423 ms | 0.2423 / 0.2325 | **15.27%** | **0.04013 ms/层** |
| EP64 / 8 | 同一逻辑路由 | 38–44 | 3.0–4.0 | 0.2218 ms | 0.2218 / 0.2218 | **15.58%** | **0.04095 ms/层** |

解释：

- batch1 只有 8 个不可分 token-expert 任务；扩大 EP 仍最多有 8 个有效 rank，且 eager grouped 链受固定开销主导。EP32/64 没有正计算节省。
- batch8 正好 64 个任务，EP32/64 把每 rank 工作压到更小的工作类型后，才出现约 0.040 ms/层的计算代理空间。
- 该预算必须覆盖新增 dispatch/combine、HCCL、同步、调度与并行争用。预算只有约 40 µs/层，不能视为足够或已验证。
- 未从专家直方图猜 payload；缺少 attention-DP/TP 下源 rank 映射，理论 payload 留空。
- 不输出 EP64 实际 TPOT，不把单层乘 75。

### Decode 测量覆盖

- 去重后实测 **45 个工作类型**，保留 16/8/4 槽、零槽和 workload 向量。
- 每个工作类型 20 次预热、≥20 次/块、5 个正式块；最终所有块均 ≥100 ms。
- 槽位置换验证误差：EP16 0.061%、EP32 0.320%、EP64 0.210%，支持按 counts multiset 复用同工作类型。

### 真实 dump 核验与缺口

- 读取 `logs/20261102_081322/eplb_windows/000000_1793579172.815737.pt` 的 layer49 逐 step 数据。
- 64/64 step 都是 128 个任务；非零专家数 33–44；不是 16×重复签名。
- 最合理契约是 16 个有效 token × top8。
- dump 只有专家计数，没有 token top8、router weight、padding、源 rank；不能直接还原 batch1/8。
- 因此：**真实 decode trace 六格实验未完成**；六格是人工机制输入上的 eager 计算代理。

## 3. 失败、未测与适用边界

- 未测：全局计数交换、远端复制、dispatch/combine、HCCL、并行争用、生产 graph、真实 EP64 TPOT。
- 未测：真实 batch1/8 decode token top8 与 router weight。
- online 热点 case 的人工 hidden/计数不是真实 trace；只能回答机制和成本账。
- 本地 copy/format 不代表远端状态准备。
- 本报告不要求正结果；EP64/batch1 在当前 eager 代理中没有正计算收益。

## 4. 证据入口

- `CPU_AUDIT.json`：守恒、空 adapter、配额枚举、真实 dump 契约
- `ONLINE_RESULT.json`：online case、planner、copy、正确性
- `DECODE_RESULT.json`：六格、完整槽向量、45 个工作类型
- `TIMINGS.csv`：所有正式块、调用数和块总长
- `MANIFEST.json`：环境、版本、输入身份和文件 hash
