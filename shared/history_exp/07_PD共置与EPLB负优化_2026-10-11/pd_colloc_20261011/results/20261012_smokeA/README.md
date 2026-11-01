# 短测 A 组数据说明（EPLB 关 + 采集开）

**用途**：验证"采集链路能拿到真实数据"。配置 = A 组（EPLB 关、初始 identity 布局）。
**负载**：agentX，输入 16384 / 输出 512，并发 8，时长 180s（共 3 个窗口）。
**服务**：4 机 EP64 共置（tp64 / dp4 / ep64 / dp-attention），`--expert-distribution-recorder-mode stat`，`buffer-size 2000`。

## 文件

`windows/NNNNNN_<unixtime>.pt` —— 每个文件 = 一个"窗口"（上次 dump 到这次 dump 之间的累计计数）。

## 每个 .pt 里的字段

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `num_layers` | 78 | 总层数。**前 3 层是 dense（无 MoE），所以矩阵里那 3 行恒为 0，只有 75 层有数** |
| `num_logical_experts` | 256 | 逻辑专家数 |
| `num_physical_experts` | 256（冗余 64 时 = 320） | 物理槽数 |
| `ep_size` | 64 | rank 数；**rank = 物理槽号 // (物理槽数/ep_size)**（本组每卡 4 槽） |
| `logical_count_prefill` | `[步数, 78, 256]` int32 | prefill 阶段的**逻辑专家**计数（与映射无关） |
| `logical_count_decode` | `[步数, 78, 256]` int32 | decode 阶段同上 |
| `logical_count_other` | 同上或 None | 其它/空闲阶段 |
| `physical_to_logical_map` | `[78, 256]` int64 | 本窗口生效的映射：物理槽 → 逻辑专家。**A 组 = identity**（槽 p 装专家 p） |
| `step_timestamps_by_phase` | dict | 每个 phase 的逐步时间戳 |
| `n_steps_by_phase` | dict | 每个 phase 的步数 |
| `mode_histogram` | dict | 原始 ForwardMode 值 → 步数（诊断用） |
| `window_start_timestamp` / `window_end_timestamp` | float | 窗口起止时间 |
| `window_kind` | str | `manual_final`（手动 dump）/ `eplb_rebalance`（重排时自动 dump） |
| `dump_index` | int | 窗口序号 |

## 用法（三条）

```python
o = torch.load(path, map_location="cpu", weights_only=False)
H_pre  = o["logical_count_prefill"].float().sum(0).numpy()   # [78, 256] 热度（prefill）
H_dec  = o["logical_count_decode"].float().sum(0).numpy()    # [78, 256] 热度（decode）
pmap   = o["physical_to_logical_map"].numpy()                # [78, 256]
```

1. **热度矩阵** = 对 `步数` 维求和。
2. **rank 负载** = 把逻辑热度按 `pmap` 摊到物理槽，再按 `rank = 槽号 // 每卡槽数` 聚合：
   `rank_load[r] = Σ_l Σ_{p∈rank r} H[l][pmap[l][p]]`；指标 = `max/mean`。
3. **计数单位**：1 个 token 过一遍模型 = 75 层 × top-8 = 600 次计数
   （例：`29491200 = 3 × 16384 × 75 × 8`）。

## ⚠️ 计数单位（用之前必看）

| 阶段 | 每 token 的计数 | 换算 token 数 |
|---|---|---|
| prefill | ≈ **600**（= 75 MoE 层 × top-8，已用真实 prompt token 数验证 ✓） | 计数 / 600 |
| decode | ≈ **4800**（= 600 × 8；LL dispatch 路径口径不同，已用真实 output token 数验证） | 计数 / 4800 |

**直接用 /600 换算 decode 会把 token 数高估 8 倍**（这会连带影响 prefill:decode 的量级对比）。
对"形状/分布"类分析（哪些专家热、top-8 重合、rank 分布）无影响（只是整体缩放）。

## ⚠️ 两个坑

1. **prefill 与 decode 的量级差 3 个数量级**（prefill 一步 = 16384×块数 个 token，decode 一步 ≈ 几十个 token）
   → **绝不能按"步"平均**，必须按 token/计数加权。
2. 本组 `window_kind` 全是 `manual_final`（A 组没有 EPLB，不会有重排时自动 dump）。

## 本批数据的实测值（核对用）

| 窗口 | prefill 步数 | decode 步数 | 非空层 | rank max/mean | PD top-8 重合 |
|---|---|---|---|---|---|
| 000000 | 4 | 513 | 75/78 | 1.39× | 14.7% |
| 000001 | 2 | 801 | 75/78 | 1.31× | 14.7% |
| 000002 | 2 | 736 | 75/78 | 1.29× | 14.3% |

- 映射 = identity ✓（A 组就是初始布局）
- **PD 重合 ≈14.5% 复现了上一轮的 14.7%**（prefill 和 decode 用的是两批不同的专家，低于随机的 25%）
