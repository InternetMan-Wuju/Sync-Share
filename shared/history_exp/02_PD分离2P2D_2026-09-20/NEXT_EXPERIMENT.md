# 下一次实验：PD EP32 的 EPLB-off 对照

## 目标

在 2026-09-14 已完成的 `EPLB=1` 基线上，只改变一个变量：关闭 EPLB。
这样可以直接回答 EPLB 对吞吐、TTFT、TPOT、E2E、rank 负载和 rebalance 开销的净影响。

## 固定条件

| 项 | 配置 |
|---|---|
| 拓扑 | 2P2D：prefill 23/25，decode 31/32，router 23 |
| 模型 | `/mnt/raid/user/data/models/glm-qq` |
| 镜像 | `sglang-glm52-pd:fe694986`，image ID `30b20a3c...` |
| decode | TP32 / EP32 / DP32，DeepEP low_latency，graph 开 |
| prefill | 每机 TP16 / DP4，DeepEP normal，graph 关 |
| workload | 16384 输入 / 1024 输出，90% 公共前缀 |
| 并发 | 10 / 20 / 30 |
| 请求数 | 并发 × 4 |
| 前缀数 | 并发 / 2 |
| 随机种子 | 1024 |
| 唯一变量 | `EPLB=0`；expert recorder 保持开启 |

基线原始数据：`results_final/c10`、`results_final/c20`、`results_final/c30`。

## 执行

```bash
cd /home/chengyetai/workspace/current_exp

# 不启动服务，只校验四机的权重、固定镜像、补丁和脚本。
bash next_experiment.sh preflight

# 后续按顺序显式执行。
bash next_experiment.sh start
bash next_experiment.sh smoke
bash next_experiment.sh matrix
bash next_experiment.sh compare
bash next_experiment.sh stop
```

默认输出目录：`results/<当天日期>_pd_ep32_eplb_off_16k/`。

## 完整性门槛

每个并发 cell 必须同时具备：

- `bench.json`，完成数为并发 × 4 且错误数为 0。
- prefill 两节点和 decode 的 before/after Prometheus 快照。
- 基于 histogram 差分的 `breakdown.json`。
- `expert_distribution.pt` 和 `expert_heat.json`。
- decode rank0/rank1 完整日志。
- `rebalance.log` 必须为空；非空说明变量隔离失败。

任一 cell 未通过，脚本立即停止，不继续下一个并发档。

`compare` 会读取 `results_final/` 的 EPLB-on 基线，生成 `eplb_ab.md` 和
`eplb_ab.json`，统一比较性能四段与 rank 不均衡。

## 已处理的复现风险

1. `main-cann9.0.0-a3` 在 2026-09-19 已漂移。四机已把旧 image ID 固定为
   `sglang-glm52-pd:fe694986`，脚本会逐机校验。
2. `serve_pd.sh` 原默认模型是坏权重 `glm-q`，现已改为 `glm-qq`。
3. 原压测默认是 64K，会触发 prefill OOM；现默认改为已验证的 16K。
4. 关闭 EPLB 后 expert recorder 仍独立开启，确保能比较 rank 负载。
5. 四段分解改用 before/after 差分，并合并两个 prefill 节点，避免累计指标污染。
