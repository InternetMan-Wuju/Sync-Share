# E1 单层/多层热专家复制结果（2026-11-02）

状态：**历史原始记录；副本落点解释已由 S0 校正**。

> 2026-11-03 修订：按 272 槽 / 16 rank = 17 slots/rank，slot 256、257 实际属于 rank15，不是 rank0/1。本文保留原始测量数字，但不能再按“rank0/1”解释该实验。修正映射与离线结果见 `RESULTS_S0.md`、`RESULTS_S3.md`。

## 1. 实验目的

验证把 layer 49 的热专家预装副本到其他 rank，是否能降低真实 8K prefill 的端到端 TTFT；随后把同样方法扩展到 layer 49/36/54。

## 2. 配置

| 项目 | 值 |
|---|---|
| 节点 | npu1-23，16 卡 |
| 并行 | TP16 / EP16 / DP1 |
| 物理专家槽 | 272（每 rank 16 个原槽 + 1 个额外槽） |
| Dispatch | `dynamic` |
| 内存 | `mem-fraction-static=0.86` |
| KV cache | 17,664 tokens |
| Recorder | 关闭采集，仅保留配置 |
| EPLB 周期重排 | 未启用 |

每轮都先做同 shape 未计时预热。

## 3. 原报告中的 Rank 负载代理（解释已校正）

layer 49 prefill 原始 rank max/mean = **4.951**。

原报告以为 expert 43、40 被复制到 rank 0、1。S0 复核表明 slot 256/257 属于 **rank15**。因此旧 mapping 的真实离线结果应写作：

```text
旧 272 映射（副本均落 rank15）:
calibration rank max/mean = 3.061
holdout rank max/mean = 3.046
```

修正后把副本放到 rank0/1 的布局：

```text
calibration = 2.988
holdout = 3.082
```

dynamic dispatch 自检确认：

```text
expert 43 -> slot 43: 2088, slot 256: 2008
expert 40 -> slot 40: 2033, slot 257: 2063
expert 0  -> slot 0: 4096（无副本）
```

## 4. 端到端结果

| 场景 | A_adapter | B layer49 top2 | B_multi（49/36/54 top2） |
|---|---:|---:|---:|
| 8K prefill TTFT p50 | 2.579 s | 2.582 s | 2.577 s |
| decode TPOT p50 | 48.77 ms | 49.35 ms | 50.23 ms |
| mixed E2E p50 | 4.277 s | 4.273 s | 4.275 s |
| mixed 输出 tok/s | 59.46 | 57.50 | 58.31 |
| 请求成功 | 全部 | 全部 | 全部 |

相对 A_adapter：

- B 单层：prefill +0.13%，decode TPOT +1.19%，mixed 吞吐 −3.3%
- B_multi：prefill −0.09%，decode TPOT +2.98%，mixed 吞吐 −1.94%

没有一条工作负载出现可复现的正收益。

## 5. 结论

1. **层内 rank 负载下降不等于端到端变快。** 旧布局实际把两份副本都放在 rank15，端到端 TTFT 仍无明显变化。
2. 原因是单层热点只占完整 75 层路径的一部分，而跨层累计后的 rank max/mean 本来只有约 1.21；层的局部热点没有成为端到端关键路径。
3. dynamic dispatch 和额外槽本身有少量开销，在收益不足时会体现为 decode/mixed 退化。
4. 旧落点配置没有显示收益；该负结果不能关闭修正落点或时间感知配额路线。

## 6. 当时确定的下一步（已由后续工作替代）

先做关键路径 profile：

- 分阶段记录 prefill/decode 的设备时间线；
- 定位最晚完成的 rank、GEMM、dispatch/combine 和等待区间；
- 只有当某个层或通信段被证明占据关键路径时，再设计针对性优化；
- 不再仅凭 token-expert 计数就选择复制对象。

## 7. 证据

- `results/E1_EVIDENCE.json`
- `results/e1_dynamic_dispatch_probe.json`
- `mappings/e1_adapter.json`
- `mappings/e1_layer49_top2.json`
- `mappings/e1_layers49_36_54_top2.json`
- 各 run 的 `bench_baseline*.json`、`service_probe.json`、`manifest.txt`
