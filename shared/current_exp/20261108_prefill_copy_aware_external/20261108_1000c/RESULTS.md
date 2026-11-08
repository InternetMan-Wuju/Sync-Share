# 单 NPU 正收益本地代理结果

## 结论

在固定输入 `4096/64`、使用真实 e43/e40 W8A8 权重和完整本地专家链的单 NPU
实验中，把热点任务按 Plan 分流到 `2080/2080` 后：

| 项目 | 实测 |
|---|---:|
| EP16 逻辑输入 | `4096/64` |
| 分流后 | `2080/2080` |
| 迁出任务 `q_global` | `2016` |
| old rank0 中位耗时 | `1.115509 ms` |
| old rank1 中位耗时 | `0.197641 ms` |
| old max 计算代理 | `1.115509 ms` |
| new rank0 中位耗时 | `0.550261 ms` |
| new rank1 中位耗时 | `0.564320 ms` |
| new max 计算代理 | `0.564320 ms` |
| 计算节省 | `0.551189 ms`（旧 max 的 `49.41%`） |
| CPU planner | `0.016281 ms` |
| 真实独立 buffer 完整 state copy | `0.126537 ms` |
| 扣 planner + local copy 后 | **`+0.408371 ms`**（旧 max 的 `36.61%`） |
| 复制字节数 | `37,810,176 B` |
| Plan 触发的 state copy 调用 | `1` |

状态：`LOCAL_POSITIVE_TEST_ONLY`。

## 正确性检查

- Plan 为 `route`，目标槽 `33`，迁出配额 `2016`。
- 目的槽预置了 e40 哨兵状态，复制后 w13、down、w13 scale、down scale
  与源 e43 逐位一致。
- 三个 weight offset 全为零。
- 每个正式测量块都大于 `100 ms`；实际调用次数和块长在
  `artifacts/positive_case.json` 中保留。
- 20 次预热后，每个配置测 5 个正式块，每块至少 `100 ms`；这里只作为块间中位数，
  不称为生产 p95。
- `transport_calls_total` 包含复制预热和正式计时循环，不等于 Plan 本身的复制次数。

## 这个正收益证明了什么

它证明了在不考虑跨卡通信时，**热点重新分配带来的本地专家计算节省，可以覆盖
CPU planner、真实完整专家运行态复制和本地顺序执行代理的成本**。这是继续做
同机双 NPU 验证的必要条件。

正负收益取决于迁移量和可摊销状态复制成本。此前较小的 `1024` token 账本里，
计算节省约 `67.95 us`，而复制和规划成本接近甚至超过它，净值约为 `-8.08 us`；
本 case 的任务量和不均衡更大，因此同一套机制出现了正收益。不能把一个 case
外推成所有负载都受益。

## 明确没有证明什么

- 两个虚拟 rank 是同一张卡顺序执行，`max(old)`/`max(new)` 只是并行计算代理。
- state copy 是同一 NPU 上真实独立 buffer 的 device copy，不是 HCCL/P2P，
  不能把 `0.126537 ms` 当作跨卡复制时延。
- 未测全局计数交换、远端复制、dispatch/combine 增量、集合通信等待和并行争用。
- 没有加载完整 GLM 模型，没有启动服务，因此不是 TTFT/吞吐/SLO 结论。
- 完整量化状态跨卡复制、格式 staging、额外槽 HBM 和生产 graph 仍待验证。

## 复用命令

在已挂载模型、旧实验脚本和本 run 目录的镜像/容器内：

```bash
python -m m0_external.positive_case
```

或：

```bash
PYTHONPATH=. python tests/npu_positive_case.py
```

原始输出：`artifacts/positive_case.stdout`；结构化结果：
`artifacts/positive_case.json`。
