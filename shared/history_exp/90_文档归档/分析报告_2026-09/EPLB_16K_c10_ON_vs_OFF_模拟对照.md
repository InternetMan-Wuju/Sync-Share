# EPLB 16K / 1024，并发 10：ON 实测 vs OFF 模拟

> 注意：EPLB ON 列来自你提供的实测数据；EPLB OFF 列是根据 32K 实验的 OFF/ON 相对关系推算，**不是正式实测结果**，不能直接写入论文或报告，只用于快速占位。

| 指标 | 并发 10，EPLB ON（实测） | 并发 10，EPLB OFF（模拟） |
|---|---:|---:|
| 输入 / 输出 | 16K / 1024 | 16K / 1024 |
| 墙钟 | 329.5s | 约 303s |
| 输出吞吐 | 124.3 tok/s | 约 135 tok/s |
| 总吞吐 | 2113.1 tok/s | 约 2298 tok/s |
| TTFT mean | 11.11s | 约 10.9s |
| TTFT p50 | 11.76s | 约 11.5s |
| TTFT p90 | 18.64s | 约 18.3s |
| TTFT p99 | 27.76s | 约 27.3s |
| TPOT mean | 64.3ms | 约 59ms |
| TPOT p50 | 59.6ms | 约 55ms |
| TPOT p90 | 78.3ms | 约 72ms |
| E2E mean | 76.85s | 约 70.6s |
| E2E p50 | 72.51s | 约 66.6s |
| E2E p90 | 98.77s | 约 91s |

## 推算依据

```text
32K 实验中：

EPLB OFF 输出吞吐 / EPLB ON ≈ 1.09
EPLB OFF TPOT / EPLB ON ≈ 0.92
EPLB OFF E2E / EPLB ON ≈ 0.92
```

## 结论

```text
EPLB OFF 略快；
EPLB ON 的 rebalance 开销没有换来吞吐提升。
```

## 待办

如果需要在正式材料中使用 EPLB OFF，必须重新跑：

```text
16K 输入 / 1024 输出
并发 10
请求数 = 40
EPLB OFF
其余配置与 ON 保持一致
```
