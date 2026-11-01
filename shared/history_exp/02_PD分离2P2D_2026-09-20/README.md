# current_exp —— 当前实验工作目录（测试4：PD 分离 + 前缀缓存）

本目录集中存放 **测试4** 相关的脚本、文档、数据集，以及**以后所有实验结果**。

## 目录结构

```text
current_exp/
├── 实验总结与规划.md      # 总纲：结论 + 实验规划（任务0~正式矩阵）
├── 退化实验方案.md        # 2 机退化方案（1P1D+EP16 vs 统一EP32，待决策）
├── serve_pd.sh            # PD 启动脚本（prefill/decode 单机或 2 机，glm-qq）
├── bench_pd_prefix.py     # 前缀缓存压测（随机 token + 90% 前缀 + 预热 + 并发）
├── run_test4_prefix.sh    # 三档并发 10/20/30 一键跑
├── next_experiment.sh     # 下一轮 EPLB-off 对照：预检/启动/冒烟/矩阵/停止
├── compare_eplb_ab.py     # EPLB on/off 性能与负载自动对比
├── NEXT_EXPERIMENT.md     # 下一轮实验变量、固定条件与执行顺序
├── breakdown_ttft.py      # 四段分解（queue/prefill/中间/decode）
├── collect_pd_metrics.md  # 服务端指标采集流程（四段/EPLB cost/热力图）
├── dataset/
│   └── test4_requests/    # 筛选出的请求集（requests.json + README.md）
└── results/               # ★ 以后所有实验结果都放这里
```

## 结果存放约定

- **所有实验产物统一放到 `current_exp/results/`**，每个实验一个子目录。
- 子目录命名建议：`<日期>_<配置摘要>`，例如 `0914_ep32_eplb_conc20`。
- 每个实验子目录放：
  - 压测结果（`bench_*.json`，由 `run_test4_prefix.sh --out` 产出）
  - 四段分解（`breakdown.json`）
  - EPLB cost（`rebalance_cost.json`）
  - 专家热度（`heat.json` / `expert_distribution.pt`）
  - 服务端 metrics（`metrics_prefill.prom` / `metrics_decode.prom`）
  - 服务日志（`rebalance.log`、`serve_*.log`）

## 快速流程

```bash
# 1. 起 2P2D 服务（prefill@23 + decode@31/32 + router@23）
#    bash serve_pd.sh prefill 0   （见 实验总结与规划.md §4.3 部署形态）

# 2. 压测（三档并发）
URL=http://127.0.0.1:18623 RESULTS=results/<run_dir> bash run_test4_prefix.sh

# 3. 采集 + 分析（PD 四段指标分布在 prefill/decode 两侧）
python3 ./breakdown_ttft.py \
  --prefill-before prefill_before.prom --prefill-metrics prefill_after.prom \
  --decode-before decode_before.prom --decode-metrics decode_after.prom \
  --out results/<run_dir>/breakdown.json
```

## 关键环境

- 模型：`/mnt/raid/user/data/models/glm-qq`（**必须用 glm-qq，不是 glm-q**）。
- 复现实验镜像：`sglang-glm52-pd:fe694986`（固定 image ID，避免 `main` 标签漂移）。
- 数据源（只读）：`/mnt/raid/user/data/dataset/cc-traces-weka-062126-256k/traces.jsonl`。
- 详细方案见 [实验总结与规划.md](./实验总结与规划.md)。
