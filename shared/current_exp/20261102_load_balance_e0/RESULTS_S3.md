# S3：固定副本预算下的配额与局部回放

状态：**完成（单 NPU 局部计算）**
日期：2026-11-03

## 问题

已有 e43/e40 两个逻辑副本时，50/50 均分忽略了 owner rank2 的其他固定负载。S3 比较：

1. 完全不使用副本；
2. 修正布局上 50/50；
3. 按 rank 原始行数做 minimax；
4. 按 S2/S3 实测局部时间模型做 minimax；
5. 旧布局，把两个副本都放在 rank15 后 50/50。

副本预算固定为两个额外完整专家：e43 一份、e40 一份。

## 成本模型

使用 grouped 链实测点拟合 6 个特征：

```text
intercept
total_rows
active_experts
largest_expert_rows
sum_sqrt_rows
active_experts_below_128_rows
```

留出验证只用未参与拟合的 rank 向量：

- NRMSE：**4.60%**
- p95 相对误差：**10.31%**
- 结论：模型只用于候选配额排序；最终收益必须用 NPU 回放确认。

## 校准集结果

| 策略 | e43→副本 | e40→副本 | max rows | 实测局部 max (ms) | 相对 owner |
|---|---:|---:|---:|---:|---:|
| owner only | 0 | 0 | 10140 | 2.688 | — |
| 修正布局 50/50 | 2031 | 1989 | 6120 | 1.838 | −31.6% |
| 行数 minimax | 4062 | 3978 | 5171 | 1.707 | −36.5% |
| 时间 minimax | 3392 | 2646 | 4102 | **1.475** | **−45.1%** |
| 旧布局 rank15 50/50 | 2031 | 1989 | 6269 | 1.922 | −28.5% |

时间 minimax 时 rank0/1/2 的局部分别为 1.428/1.475/1.440 ms，已接近平衡。

## 留出集结果

| 策略 | e43→副本 | e40→副本 | max rows | 实测局部 max (ms) | 相对 owner |
|---|---:|---:|---:|---:|---:|
| owner only | 0 | 0 | 10329 | 2.756 | — |
| 修正布局 50/50 | 2033 | 1986 | 6310 | 1.894 | −31.3% |
| 行数 minimax | 4066 | 3972 | 4900 | 1.657 | −39.9% |
| 时间 minimax | 3387 | 2883 | 4059 | **1.437** | **−47.8%** |
| 旧布局 rank15 50/50 | 2033 | 1986 | 6239 | 1.865 | −32.3% |

## 证据等级

- rank0/1/2 使用各自精确的 layer 49 专家权重回放；旧布局额外加载 rank15 精确权重。
- 其他 rank 使用先前的代表性 17-group 局部回放。
- 每个计划都验证了 256 个 logical expert 的任务总数守恒。
- 局部回放工作集最大约 3.21 GB，峰值 allocated 约 4.10 GB。
- 所有时间都是顺序测得的局部链时间，不是 16 rank 同时执行的 critical path。

## 结论

在 layer 49 的单卡局部计算模型中，时间感知配额比 50/50 再改善约 13–14 个百分点，比 owner-only 改善 45–48%。这支持把“配额”作为多卡 M1 的首选候选，但**不证明端到端收益**：跨 rank dispatch、combine、HCCL、统计与规划开销尚未计入。

真正的多卡验收应使用相同副本、相同计划、固定真实输入，比较 A/A_adapter/B_plan，并报告通信等待和完整层时间。

## 产物

- `results/S3_COST_CALIBRATION.json`
- `results/S3_PLANS.json`
- `results/S3_REPLAY.json`
- `results/S3_ALLOCATION.json`
- `results/S3_ALLOCATION.csv`
- `scripts/s3_cost_calibration.py`
- `scripts/s3_plan.py`
- `scripts/s3_replay_affected.py`
- `scripts/s3_analyze.py`
