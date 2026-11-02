# E1 单层/多层热专家复制结果（2026-11-02）

状态：**负结果，停止 C/D 在线复制方向**。

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

## 3. Rank 负载代理

layer 49 prefill 原始 rank max/mean = **4.951**。

把 expert 43、40 各复制到 rank 0、1，并按约 50/50 分流后，离线工作量代理降到：

```text
rank max/mean = 2.988
改进 = 39.6%
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

1. **层内 rank 负载下降不等于端到端变快。** layer 49 的负载代理改善 39.6%，但 TTFT 不变。
2. 原因是单层热点只占完整 75 层路径的一部分，而跨层累计后的 rank max/mean 本来只有约 1.21；层的局部热点没有成为端到端关键路径。
3. dynamic dispatch 和额外槽本身有少量开销，在收益不足时会体现为 decode/mixed 退化。
4. 按 E1 停止条件，不再为 layer 49/36/54 实现 C/D 在线规划与真实搬运。

## 6. 下一步

改为做关键路径 profile：

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
