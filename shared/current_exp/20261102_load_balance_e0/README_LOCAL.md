# 2026-11-02/03 GLM-5.2 Ascend 负载均衡实验

本目录包含两阶段工作：

1. 2026-11-02：单机 16×Ascend 910C 的 E0/E1/Profile/E3 服务实验。
2. 2026-11-03：遵守“只使用一张 NPU”的修订方案，完成 S0–S4 与 P2 本地验证。

模型：GLM-5.2 W8A8 `/data_lib/data/models/glm-q`。
权重必须通过 publisher 只读恢复层加载并校验 SHA256。

## 当前结论

- E0 基线 PASS；权重 182 文件、2912 条审计、0 错误。
- Prefill 由 DSA attention 主导；MoE 约 14.4%。
- Decode MoE 约 29%；已验证 `capacity=16` 小幅优于 cap64。
- 旧 E1 副本实际在 rank15，不能据此否定修正布局。
- S1 量化/副本/分流/合并正确性 PASS。
- S3 时间感知配额在 layer49 单卡局部回放中把 rank-local 最大时间降低 45–48%，但尚未过 HCCL 和完整层验证。
- S4 顺序局部拆分候选均为负收益。
- 完整总结见 `RESULTS_OVERALL.md`。

## 单卡复现

容器只绑定 `/dev/davinci0`，且必须确认：

```text
torch.npu.device_count() == 1
```

步骤：

```bash
# S0：映射、slot-rank、任务守恒
python3 scripts/s0_mapping_audit.py \
  --out results/S0_MAPPING_AUDIT.json \
  --route-dir logs/.../eplb_windows

# S1 CPU：完整量化状态审计（需要 publisher loader 环境变量）
python3 scripts/s1_expert_state.py \
  --model /data_lib/data/models/glm-q \
  --out results/S1_EXPERT_STATE.json \
  --audit-dir logs/.../weight_audit

# S1 NPU：副本、分流、A/A、top-8 合并
python3 scripts/s1_npu_numerics.py \
  --out results/S1_NUMERICS.json

# S2：局部成本曲线
python3 scripts/s2_local_cost.py \
  --out results/S2_LOCAL_COST.json \
  --csv results/S2_LOCAL_COST.csv

# S3：校准、规划、精确 rank 回放、汇总
python3 scripts/s3_cost_calibration.py --out results/S3_COST_CALIBRATION.json
python3 scripts/s3_plan.py --out results/S3_PLANS.json --results-root results
python3 scripts/s3_replay_affected.py --out results/S3_REPLAY.json
python3 scripts/s3_analyze.py \
  --plans results/S3_PLANS.json \
  --replay results/S3_REPLAY.json \
  --out results/S3_ALLOCATION.json \
  --csv results/S3_ALLOCATION.csv

# S4：两个局部候选与通道切分数值反例
python3 scripts/s4_local_candidates.py --out results/S4_LOCAL_CANDIDATES.json

# P2：shared expert、双 stream、单槽生命周期
python3 scripts/p2_local_checks.py --out results/P2_LOCAL_CHECKS.json
```

S3 规划脚本在容器内运行时可保持默认 `--results-root /exp/results`；在宿主目录运行时使用 `--results-root results`。

## 服务实验入口

```bash
bash scripts/serve_e0.sh start
bash scripts/serve_e0.sh status
bash scripts/serve_e0.sh stop
```

服务原始日志写入 `logs/<run_id>/`；小结果写 `results/`。

## 边界

- S0–S4/P2 不是完整 GLM 模型正确性验收。
- 单卡局部时间不是 EP16/EP64 的端到端时间。
- 没有测到真实 HCCL payload、跨 rank 等待或服务 SLO。
- 不要把 pointer switch、NPU→NPU 本地复制当作跨 rank 权重传输。
