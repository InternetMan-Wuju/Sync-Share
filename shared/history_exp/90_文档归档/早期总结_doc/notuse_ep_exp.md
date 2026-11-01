# GLM-5.2 × 910C：EP 测试说明

目标：用一个固定的混合请求集，测试不同 EP 和并发下的性能；找出表现最好和较差的 EP，再判断问题来自负载不均衡还是通信。

这轮不比较静态专家重排和 MoE kernel。EPLB 放到后面，先把 EP 本身测清楚。

## 1. 先做什么

先复用已经跑通 GLM-5.2 的模型路径、环境和启动方式。不要先升级框架、换 CANN 或改网络。

先保存一次环境信息，后面每次实验都沿用同一份配置：

```bash
npu-smi info
npu-smi topo -m || true
lscpu
numactl -H || true
ip -br addr
```

同时保存：

- 模型路径和 `config.json` 的 hash；
- SGLang、torch、torch-npu、CANN、driver 版本；
- 启动命令；
- EP、TP、PP、DP、world size；
- 每个 rank 对应哪台机器、哪张 NPU；
- 实际使用的 A2A 通信后端。

`EP=8` 的意思是一个 EP group 有 8 个 rank，不等于一定是单机 8 卡。单机和跨节点结果要分开画图。

## 2. 数据集：不跑全量，做一个混合请求集

不要直接跑完整数据集。先把候选请求按长度分桶，从每桶固定抽样，拼成一个 mixed benchmark。

建议最少按下面两维分桶：

```text
输入 token 长度 × 输出 token 长度
```

可以再加任务来源/类别。

每个桶记录：

| 要记录什么 | 内容 |
|---|---|
| 数据来源 | 数据集名称、版本、清洗规则 |
| 分桶规则 | token 边界、抽样种子、每桶候选数和选中数 |
| 请求数 | 每桶请求数、总请求数、占比 |
| 长度统计 | 输入、目标输出、实际输出 token 的总数、均值、P50/P95/P99、min/max |
| 推理设置 | chat template、sampling、seed、max tokens、停止条件 |
| 请求清单 | 固定顺序；保存 manifest hash |

混合集中至少要有：

- 短输入：看 EP 和通信的固定开销；
- 长输入：看 Prefill；
- 短输入、长输出：看 Decode；
- 一部分真实长度混合请求。

所有 EP、并发和重复实验必须使用同一个 mixed benchmark、同一个请求顺序和同一套采样参数。

## 3. 要测试哪些 EP 和并发

先确认机器实际支持哪些 EP；只测能正常启动且 rank 映射正确的点。

```text
EP=off（全副本）/ 2 / 8 / 16 / 32 / 64
并发=C_low / C_high
每组重复至少 3 次
```

- `EP=off`：不使用 EP，全副本服务作为基线。记录有几个副本、每副本用了几张卡、请求如何分发。
- EP 增大时可能减少副本数。要分开报：
  - 单副本性能；
  - 整个集群的总吞吐。

不要把两者混成一个“提升”。

并发建议先定两档：

```text
C_low：轻载，基本不排队
C_high：接近稳定上限，但不持续超时
```

记录客户端数、请求到达方式（closed/open loop）、QPS、超时和重试设置。

## 4. 测试分两步做

### 第一步：小规模试跑

先从每个桶抽一些请求，总计 50–200 个即可：

```text
EP=off、2、8
× C_low、C_high
```

检查：

- 服务能稳定跑完；
- 没有 OOM、通信报错或不断重启；
- 客户端能记录 TTFT、TPOT、E2E latency 和错误；
- 路由、rank 时间、NPU/HBM 指标能拿到哪些，写清楚。

这一步通过后，再跑完整 EP 扫描；不要一开始就全量提交。

### 第二步：扫描完整 EP 曲线

对所有可行 EP 和两档并发，各跑至少 3 次。每组记录：

| 类别 | 指标 |
|---|---|
| 首 token | TTFT P50/P95/P99 |
| 生成阶段 | TPOT/ITL P50/P95/P99、Decode tok/s |
| Prefill | input tok/s 或 prefill latency |
| 总体 | E2E latency P50/P95/P99、request/s、总 tok/s |
| 稳定性 | 成功率、超时率、错误率、重试数 |
| 资源 | NPU util、HBM 峰值、温度/功耗（能采则采） |

每次 run 的目录至少保存：启动命令、服务端日志、客户端原始延迟、设备监控、请求 manifest、版本信息。

推荐命名：

```text
glm52__ep<EP>__c<并发>__r<重复编号>__<时间>
```

## 5. 从 EP 曲线挑两个点深入看

对每种并发，先用一个主指标选点。建议主指标选：

```text
TPOT P95 或集群总吞吐
```

从曲线里选：

1. 一个局部最好 EP，例如可能是 EP=64；
2. 一个明显较差的 EP，优先选相邻的局部低点。

再对这两个点，用完全相同的 mixed benchmark 和并发，采更细的负载和时间数据。

## 6. 深入分析时必须看什么

### 6.1 Rank 热度

Rank 热度就是每个 EP rank 实际处理了多少 routed token。

至少采：

```text
rank_token_count[r]
rank max/mean
rank CV = std(rank token) / mean(rank token)
rank P50/P95/P99
```

画 `layer × rank` 的热力图，并分别看 Prefill 和 Decode。

### 6.2 专家热度

专家热度用来解释 rank 为什么会热：

```text
tokens[layer, expert]
expert max/mean
expert CV
热点 Top-k expert 占比
```

画 `layer × expert` 热力图。

### 6.3 时间分解

对每个 rank，尽量拿到：

- MoE compute time；
- A2A dispatch / combine time；
- synchronization / wait time；
- rank step time；
- send/recv token 或 bytes（能拿到则记录）。

然后比较：

```text
rank token load
→ rank compute / A2A / wait
→ rank step time
→ TTFT / TPOT / throughput
```

判断方式很简单：

- 某些 rank token 特别多，而且这些 rank 的 compute/step/wait 更长：负载不均衡造成 straggler；
- 各 rank token 差不多，但 A2A/wait 很高：主要是通信或同步开销；
- 两边都不明显：检查副本数、TP/DP、拓扑、客户端、KV cache、资源争用和测量误差。

如果拿不到 rank 级 compute、A2A 或 wait，就只能说“观察到负载和端到端性能的关系”，不能断言已经证明了 straggler 原因。

## 7. 汇总表

每行是同一个 `EP × 并发` 的 3 次重复汇总：

| EP | 并发 | TTFT P95 | TPOT P95 | Prefill speed | Decode speed | E2E P95 | 总吞吐 | 成功率 | rank max/mean | rank CV | A2A | wait 占比 | HBM 峰值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

同时保留每次 run 的原始数据和失败记录。汇总时给均值和标准差。

## 8. 结果怎么写

可以写的结论：

- 某 EP 在某个并发下 TTFT、TPOT 或吞吐最好；
- 某 EP 出现 rank 热度不均衡；
- 高热度 rank 同时表现出更长的 compute/step/wait，说明它是 straggler；
- EP 增大后性能先升后降，且 A2A/wait 增长，可以说明通信抵消了并行收益。

不要直接写：

- “专家热点所以端到端一定慢”；
- “rank 更均衡所以一定更快”；
- “一次跑得更快，所以 EP 更好”；
- “只测 GLM-5.2，就证明跨模型泛化”。

## 9. EPLB 放在什么时候

本轮先完成 EP 曲线和两个点的根因分析。

只有在发现稳定的 rank 失衡/straggler，且当前版本确认支持并实际触发 EPLB 时，再单独加：

```text
同一个 EP、同一 mixed benchmark、同一并发
EPLB off vs. EPLB on
```

额外记录 rebalance 是否触发、触发次数、暂停时间、显存和通信变化。

## 10. 开始前检查

- [ ] 复现当前极小数据集。
- [ ] 采集环境、拓扑、rank map 和启动命令。
- [ ] 分桶、抽样并生成 mixed benchmark；保存请求数和 token P50/P95/P99。
- [ ] 写清 EP=off 的副本数、TP 和请求分发方式。
- [ ] 确认可用 EP 点和两档并发。
- [ ] 小规模试跑 `EP=off、2、8 × C_low、C_high`。
- [ ] 确认 TTFT/TPOT/E2E、路由、rank 时间和设备指标的采集方式。
- [ ] 通过试跑后，再扫描所有可行 EP 并各重复 3 次。
