# E0 单机可信基线结果（2026-11-02）

状态：**PASS**。服务可启动、健康、语义正确、权重审计通过，并完成两组独立路由采集和三轮固定基线。

## 1. 环境与配置

| 项目 | 值 |
|---|---|
| 节点 | `npu1-23`，单机 16 卡 |
| 镜像 | `quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3` |
| 镜像 digest | `sha256:26a020117564da514f33c4e72353d07020bc4bfe176f4e61858b4abeac6d3e9b` |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| SGLang | 0.5.18 |
| DeepEP | 已导入 |
| 并行 | TP16 / EP16 / DP1 |
| DeepEP mode | auto |
| Decode graph | full，batch 1/2/4/8/16 |
| HBM 配置 | `mem-fraction-static=0.88` |
| KV cache | 63,232 tokens |
| Context / chunk / max prefill | 131072 / 16384 / 32768 |
| Recorder | `stat`，buffer 128，EPLB 关闭 |

运行容器使用 rootless Docker，容器网络为 `tap0`（`10.0.2.100`）。当前端口只在该容器命名空间内提供，宿主机通过 `docker exec` 访问。

## 2. 权重健康

使用 publisher 只读恢复层，manifest SHA256：

```text
7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5
```

当前目录自身包含 329 个坏块/30900 个非有限值，因此**不能**跳过恢复层直接作为正式基线。运行时审计结果：

- 16 个 scheduler 进程
- 182 个唯一权重文件
- 2,912 条文件加载记录
- 369,488 个已验证块
- 0 个错误
- 0 个 fallback 块
- 原始文件身份漂移 0

聚合读取字节约 11.26 TiB，是 16 个进程的重复读取总量，不是唯一 checkpoint 大小。

## 3. 正确性与健康

- `/health` = 200
- “1+1”输出 `2`
- “输出 1–20”输出完整且顺序正确
- 固定 `/generate` 请求 = 200
- E0 三轮基线共 18 个计时请求，全部成功

## 4. 路由采集

每组输入：4 个请求 × 1024 prompt tokens × 16 output tokens。

prefill 计数精确守恒：

```text
4 × 1024 × 75 MoE layers × top-8 = 2,457,600
```

calibration 与 holdout 均观察到 2,457,600。

decode recorder 观察到 614,400，但该路径存在平行度重复，**不作为绝对 token 任务数**；只使用其相对分布和失衡比例。

跨 75 层累计 rank max/mean：

| 阶段 | calibration | holdout |
|---|---:|---:|
| prefill | 1.2150 | 1.2483 |
| decode | 1.1741 | 1.1843 |

单层 rank max/mean 可达约 5×，说明累计均衡掩盖了逐层热点，符合历史“跨层抵消”结论。

稳定热点层：

| 阶段 | calibration 前五层 | holdout 前五层 |
|---|---|---|
| prefill | 49, 36, 29, 54, 51 | 49, 15, 54, 36, 65 |
| decode | 12, 20, 10, 74, 15 | 12, 20, 10, 74, 61 |

优先候选：

- prefill：**49, 36, 54, 29**
- decode：**12, 20, 10, 74**

## 5. 无采集基线性能

每个场景先做同 shape 未计时预热，再测 3 轮。

| 场景 | TTFT p50 | TTFT p95 | TPOT p50 | E2E p50 | 输出 tok/s |
|---|---:|---:|---:|---:|---:|
| prefill 8192→1，并发1 | 2.573 s | 2.573 s | — | 2.575 s | 0.383 |
| decode 128→256，并发1 | 0.438 s | 0.442 s | 48.0 ms | 12.678 s | 20.147 |
| mixed 1024→64，并发4 | 0.979 s | 1.271 s | 50.7 ms | 4.180 s | 59.501 |

这些是单机固定随机 token 基线，不代表生产 SLO，也不能与历史四机 EP64 数字直接比较。

## 6. 启动问题与修正

1. 首次 graph capture 因临时 Triton cache 中 launcher `.so` 丢失失败。将 Triton/Inductor cache 持久化到 XFS 后通过。
2. `mem-fraction-static=0.90` 下 8K prefill 触发 NPU OOM，只剩约 1.8GB 工作区。调整到 `0.88` 后 graph 结束仍有约 3.2–3.5GB，8K prefill 正常。
3. TP16 下 graph batch size 必须按 16 对齐，`max_bs=8` 会得到 `capture_bs=[]`；最终恢复 batch 1/2/4/8/16。

## 7. 下一步 E1

先选 **prefill layer 49** 做单层复制 A/A_adapter/B/C/D 拆解，并以 layer 36/54 作为重复验证层、低失衡层作为负对照。decode 侧先修复/确认计数重复口径，再进入 layer 12/20/10 的通信与组流水研究。

关键文件：

- `results/E0_EVIDENCE.json`
- `results/<run>_route_analysis.json`
- `results/<run>_route_analysis_holdout.json`
- `logs/<run>/bench_baseline_warm.json`
- `logs/<run>/weight_audit/*.jsonl`
- `logs/<run>/eplb_windows/00000*.pt`（原始路由张量，仅本地）
