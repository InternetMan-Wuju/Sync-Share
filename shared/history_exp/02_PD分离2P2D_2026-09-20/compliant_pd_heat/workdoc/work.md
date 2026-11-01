# PD 分离 + EPLB / redundant-experts 实验方案（2026-09-21 修订）

## 0. 目标

1. 用真实 AgentX / cc-traces 作为参数来源，构造比上一轮更长的 PD 压测。
2. 比较三组专家布局：
   - A：EPLB off，redundant=0，trivial，不重排。
   - B：EPLB on，redundant=0，只有重排。
   - C：EPLB on，redundant=32，重排 + 冗余副本。
3. 统计 PD 分离的完整热度：
   - Prefill P0：Layer × Logical Expert、Layer × EP-Rank。
   - Prefill P1：Layer × Logical Expert、Layer × EP-Rank。
   - Decode：Layer × Logical Expert、Layer × EP-Rank。
4. 每次 EPLB rebalance 都保留“rebalance 前窗口 + rebalance 后短窗口”，
   最终也保留手动窗口，不能只看最后一个窗口。

## 1. 固定配置

### 1.1 团队统一要求

```text
chunked prefill size = 16384

cuda graph:
  --cuda-graph-backend-decode full
  --cuda-graph-max-bs-decode 32
  --cuda-graph-bs-decode 8 16 24 32

env:
  SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

EPLB rebalance 周期：
  Prefill = 100
  Decode  = 1000
```

### 1.2 部署形态

继续使用 2P2D：

```text
prefill 0 = 10.120.72.23
prefill 1 = 10.120.72.25
decode   = 10.120.72.31 + 10.120.72.32
router   = 10.120.72.23
```

### 1.3 模型与镜像

```text
model = /mnt/raid/user/data/models/glm-qq
image = sglang-glm52-pd:fe694986
```

## 2. EP / EPLB 参数矩阵

三个组，其余配置尽量固定：

| 组 | EPLB | redundant | 布局 | 目的 |
|---|---|---:|---|---|
| A | off | 0 | trivial | 基线，不重排 |
| B | on | 0 | 只有 EPLB rebalance | 重排本身收益 |
| C | on | 32 | EPLB rebalance + 冗余专家 | 比 B 多出的冗余收益 |

显式参数：

```text
A:
  --enable-eplb off
  --ep-num-redundant-experts 0

B:
  --enable-eplb on
  --ep-num-redundant-experts 0

C:
  --enable-eplb on
  --ep-num-redundant-experts 32
```

其他 EP 相关参数保持默认：

```text
--ep-dispatch-algorithm default
--init-expert-location trivial
--eplb-algorithm auto
--eplb-rebalance-num-iterations 1000
--eplb-min-rebalancing-utilization-threshold 1.0
```

## 3. Recorder 维度

每组跑两种 recorder 模式：

| 模式 | 用途 |
|---|---|
| recorder off | 拿干净性能，不被数据采集干扰 |
| recorder on | 采集完整 P0/P1/D 热度 |

因此最小实验矩阵是：

```text
A-off, A-on
B-off, B-on
C-off, C-on
```

如果时间不够，先保证：

```text
A-off, B-off, C-off -> 性能主结论
B-on, C-on          -> 热度主结论
A-on                -> 可选，作为无 EPLB/无冗余热度基线
```

## 4. Workload 与输入/输出长度

上一轮问题：

- 输入 16K 太短，导致 prefill 没有触发 EPLB rebalance。
- decode 长度也可以适当加长。

本轮建议采用“先冒烟选长度，再固定主长度”：

### 4.1 候选长度

| 档位 | 输入 | 输出 | 说明 |
|---|---:|---:|---|
| L1 | 32K | 2048 | 优先级高 |
| L2 | 48K | 2048 | 内存允许时使用 |
| L3 | 64K | 2048 | 最接近 AgentX，但需确认 prefill 不 OOM |

优先目标：

```text
主实验长度 = 内存允许下最长的输入
```

如果 64K 仍 OOM，则退到 48K；再不行退到 32K。

### 4.2 并发与请求数

沿用上一轮规则：

```text
请求数 = 并发 × 4
前缀数 = 并发 / 2
```

建议固定一个主并发：

```text
主并发 = 20
```

因为矩阵已经包含 3 组 × recorder 2 种，如果再加 3 个并发档会爆炸。

扩展档：

```text
若时间允许，再对主长度补 10 / 30
```

### 4.3 冒烟门槛

在正式矩阵前，先跑最小矩阵：

```text
每组长度的：
  - 1 条 64K/1024 或候选长度的 smoke
  - 记录 prefill/decode 健康
  - 检查 prefill 是否能出现 rebalance start
```

如果 prefill 始终不能触发 rebalance，如实记录：

```text
prefill_rebalance_triggered = false
```

不要为了让 prefill 触发而强行把并发或请求数改到不合理。

## 5. 数据采集

### 5.1 性能

每个 cell：

```text
bench.json
prefill0/1 metrics before/after
decode metrics before/after
breakdown.json
```

必须保留并输出以下传统性能指标：

| 指标 | 来源 | 定义 |
|---|---|---|
| TTFT | 客户端 + 服务端 metrics | 首 token 时延；报告 mean / p50 / p90 / p99 |
| TPOT | 客户端 + 服务端 metrics | 除首 token 外每个输出 token 平均时延；报告 mean / p50 / p90 |
| E2E | 客户端 + 服务端 metrics | 请求从发出到完整结束；报告 mean / p50 / p90 / p99 |
| E2E breakdown | prefill/decode metrics 差分 | queue + prefill_forward + 中间时长 + decode |
| prefill 速度 | 客户端 token 统计 + prefill 自身时间 | 总输入 token/s |
| decode 速度 | 客户端 token 统计 | 输出 token/s；decode 段 token/s |
| rebalance 开销 | decode 日志 | rebalance 次数、每次耗时、总耗时、峰值 |

### 5.1.1 E2E breakdown

```text
queue_time
prefill_forward
middle = TTFT - queue_time - prefill_forward
decode  = E2E - TTFT
```

使用 before/after 的 histogram 差分，不能直接读累计 after。

### 5.1.2 prefill 速度

只报一个口径：

```text
prefill 吞吐 = prompt_tokens_total / prefill_phase_time
```

其中：

```text
prefill_phase_time = max(end_time_p0, end_time_p1)
                   - min(start_time_p0, start_time_p1)
```

即直接读取 P0、P1 两个 prefill 服务的实际运行时间段。P0/P1 是并行服务，
所以 prefill wall time 用两者重叠窗口：

```text
start = min(p0_start, p1_start)
end   = max(p0_end, p1_end)
```

### 5.1.3 decode 速度

```text
decode_wall_time = max(D0_end, D1_end) - min(D0_start, D1_start)
输出 token/s = completion_tokens_total / decode_wall_time
```

同时保留 TPOT p50，避免只用一个总吞吐掩盖 decode 阶段变化。

### 5.1.4 rebalance 开销

从 decode 日志提取：

```text
rebalance start / end
```

报告：

```text
rebalance 次数
每次耗时：min / mean / max
总 rebalance 时间
首个 rebalance 时间 vs 后续增量 rebalance 时间
rebalance 期间在途请求是否出现延迟尖峰
```

### 5.2 EPLB / rebalance

```text
decode_rank0.log
decode_rank1.log
rebalance start/end
```

重点保留：

```text
rebalance 次数
每次耗时
rebalance 前后 physical_to_logical_map
```

### 5.3 PD 热力

recorder on 的 cell 必须收集：

```text
P0: Layer × Logical Expert, Layer × EP-Rank
P1: Layer × Logical Expert, Layer × EP-Rank
D:  Layer × Logical Expert, Layer × EP-Rank
```

窗口要求：

```text
每次 rebalance 保存两种窗口：

```text
pre_rebalance_window
  - rebalance 发生前，使用旧 physical_to_logical_map 的累计热度

post_rebalance_short_window
  - rebalance 完成后，使用新 physical_to_logical_map 的短时间热度
```

每个窗口内嵌当时的 physical_to_logical_map。

实验结束再保存 manual_final 窗口。

编号 0..N，无缺号、无重复。
```

## 6. 结果目录

建议统一：

```text
current_exp/results/
  <date>_pd_ep32_groups_abc_<length>/
    a_off/
    a_on/
    b_off/
    b_on/
    c_off/
    c_on/
```

每个 cell 内：

```text
bench.json
breakdown.json
metrics_*.prom
prefill0.log / prefill1.log / decode_rank0.log / decode_rank1.log
windows/
  prefill0/
  prefill1/
  decode/
heat/
  *_heat.json
  *_heat.csv
  *_heat.png
```

## 7. 执行顺序

```text
1. 审计 serve_pd.sh / 补丁脚本，确认 A/B/C 参数正确。
2. 跑长度冒烟，确定主长度。
3. 先跑 recorder on：
   A-on -> B-on -> C-on
4. 再跑 recorder off：
   A-off -> B-off -> C-off
5. 每个 cell 立即做产物齐全检查。
6. 全部完成后生成 A/B/C 对比表。
```

## 8. 结论输出

至少输出：

```text
性能对比：
  TTFT / TPOT / E2E / 输出吞吐
  prefill 吞吐
  decode 输出吞吐

四段时延：
  queue / prefill / 中间 / decode

EPLB：
  rebalance 次数与耗时
  B vs C 的冗余收益

热度：
  P0 / P1 / D 的 Layer×Expert 和 Layer×EP-Rank
  rank CV / peak-mean / max-min
```
