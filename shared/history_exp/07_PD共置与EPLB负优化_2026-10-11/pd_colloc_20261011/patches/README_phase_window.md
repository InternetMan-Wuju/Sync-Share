# 补丁：expert_distribution_phase_window.py（基于 sglang v0.5.18 原文件）

用途：让 **4 机共置（EP64 + dp-attention）** 服务在开启 EPLB 时，能拿到
**"P/D 分开、且每个重排窗口都不丢"** 的专家计数数据。

挂载点（只读覆盖）：
```
-v .../expert_distribution_phase_window.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro
```
环境变量：`SGLANG_DUMP_EPLB_WINDOWS=1`（不设则该补丁除了 phase 路由外不改变行为）
落盘目录：`SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`（默认 `/tmp`，要挂到工作区）

## 解决的四个问题

| # | 原版（stock v0.5.18）的问题 | 本补丁 |
|---|---|---|
| 1 | **自动重排在 `output_mode="object"` 下只进内存、不落盘** → 每个重排窗口的计数全丢 | 自动 dump（object）也写文件 |
| 2 | dump 只带 `last_physical_to_logical_map`（**dump 那一刻**的布局）→ 拿它去换算早先窗口的计数是错的 | 每个窗口写**它自己当时的** `physical_to_logical_map` |
| 3 | 计数**不分 prefill/decode**（共置服务里两者混在一起） | 用**两个 buffer 分相累积**，all_reduce 后再分别落盘 |
| 4 | `stat` + `deepep auto` 直接 `raise NotImplementedError` | 走路由侧 gatherer（`on_select_experts` 在 NPU topk 路径对 normal+LL 都触发） |

## 落盘的每个文件内容

```
<窗口序号:06d>_<时间戳>.pt
  ├─ logical_count_prefill   [steps, 78, 256]   ← prefill 相，逐 step 计数
  ├─ logical_count_decode    [steps, 78, 256]   ← decode 相，逐 step 计数
  ├─ logical_count_other     [steps, 78, 256]   ← 其他 phase（IDLE 等），正常为空
  ├─ physical_to_logical_map [78, 256]          ← ★ 本窗口当时的布局
  ├─ window_kind             "eplb_rebalance"(自动) / "manual_final"(手动)
  ├─ dump_index / dump_timestamp
  └─ num_layers / num_logical_experts / num_physical_experts
```
（文件由 rank0 写，因为 all_reduce 后才是全局计数。）

## ⚠️ 三个必须记住的约束

1. **`eplb_rebalance_num_iterations` 必须 > `expert_distribution_recorder_buffer_size`**
   （源码 `eplb_manager.py:62` 断言），否则启动即挂。
2. **buffer 只保留最近 N 步**：想看完整重排周期，`buffer_size` 要 ≥ 重排周期，
   代价是内存 = `3 × buffer_size × 78 × 256 × 4B`（buffer=128 → 30MB/rank；
   buffer=1000 → 240MB/rank）。
   想让重排更容易触发就**同时调小**（例如 rebalance=201 / buffer=200）。
3. **buffer 是环形的**：dump 里固定 `buffer_size` 行，没填满的部分是**全零行**；
   环形绕回后**行的顺序不是时间序**（我们只按相求和，所以不受影响）。

## 与原文件的差异

5 处改动（`make_patch_phase_window.py` 可重新生成并逐行对照）：
import os / deepep-auto 分支 / gatherer 记 forward_mode / collect 带上它 / 重写 `_StatAccumulator`
