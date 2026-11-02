# E0 同步清单

本目录只同步总结、关键脚本和小型结构化证据。

## 本地保留，不上传 Git

- 原始路由张量：`/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/logs/20261102_081322/eplb_windows/`，约 70MB
- 原始逐文件权重审计：`/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/logs/20261102_081322/weight_audit/`，约 836KB
- 完整实验目录：`/home/chengyetai/workspace/current_exp/20261102_load_balance_e0`

## 已同步

- `RESULTS_E0.md`：E0 结论、配置、基线、选层和限制
- `scripts/`：服务启动、健康探针、路由采集、路由分析和基线 benchmark
- `evidence/E0_EVIDENCE.json`：汇总证据
- `evidence/route_analysis_calibration.json`：校准集层/rank 分析
- `evidence/route_analysis_holdout.json`：留出集层/rank 分析
- `evidence/bench_baseline_warm.json`：三轮预热后基线原始结果
- `evidence/service_probe.json`：健康、对话和固定生成
- `evidence/env_probe.json`：运行时版本与环境
- `evidence/weight_audit_summary.json`：权重审计摘要
- `evidence/server.log`：最终稳定运行的完整服务日志

## E1 追加

已同步：

- `RESULTS_E1.md`
- `mappings/e1_adapter.json`
- `mappings/e1_layer49_top2.json`
- `mappings/e1_layers49_36_54_top2.json`
- `scripts/serve_e1.sh`
- `evidence/E1_EVIDENCE.json`
- `evidence/e1_dynamic_dispatch_probe.json`
- `evidence/e1/*_manifest.txt`
- `evidence/e1/*_bench*.json`
- `evidence/e1/*_service_probe.json`

E1 原始运行日志和路由张量仍保留在本地实验目录。

## Profile / E3 追加

已同步：

- `RESULTS_PROFILE.md`
- `evidence/profile_summary.json`
- `evidence/profile_decode_summary.json`
- `evidence/e3_ops_failure.json`
- `evidence/profile/prefill_rank0/`（op/API/step/communication 统计）
- `evidence/profile/decode_rank0/`（op/API/step/communication 统计）
- `scripts/analyze_profile.py`
- `scripts/profile_one.py`

64MB 级 kernel_details、operator_details 和 17GB 原始 trace 保留在本地或已清理，不提交 Git。

## E3 追加

已同步：

- `RESULTS_E3.md`
- `evidence/E3_EVIDENCE.json`
- `evidence/e3/cap16_concurrency16.json`
- `evidence/e3/cap32_manifest.txt`、`cap32_bench_5rep.json`、`cap32_service_probe.json`
- `evidence/e3/cap16_manifest.txt`、`cap16_bench_5rep.json`、`cap16_service_probe.json`
- `evidence/e3/alltoall_manifest.txt`、`alltoall_bench_5rep.json`、`alltoall_service_probe.json`

## S0–S4 / P2 追加（2026-11-03）

已同步：

- `RESULTS_S0.md`：272 槽映射纠错、修正映射、任务守恒
- `RESULTS_S1.md`：27 张量量化状态、副本独立性、36 个分流用例、A/A、top-8 合并
- `RESULTS_S2.md`：单卡局部链 shape 曲线、活跃专家成本、重复/轮换权重
- `RESULTS_S3.md`：成本模型、固定副本预算配额、精确 affected-rank 回放
- `RESULTS_S4.md`：热点单独 GMM、token 行分组、通道二分片数值反例
- `RESULTS_P2.md`：shared expert 成本/重叠、单槽生命周期
- `mappings_s0_corrected/*.json`
- `scripts/s0_mapping_audit.py` 至 `scripts/p2_local_checks.py`
- `scripts/build_single_npu_manifest.py`
- `evidence/s0_s4_p2/*.json`、`*.csv`
- `evidence/s0_s4_p2/SINGLE_NPU_MANIFEST.json`：镜像、模型输入、脚本和结果哈希索引

本地保留、不上传：

- 完整日志与权重读取审计：`/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/logs/`
- 原始权重、模型分片、profiler trace
- 单卡实验中的临时容器和 NPU cache

## 重要解释修正

旧 E1 的 slot 256/257 属于 rank15。`RESULTS_E1.md` 保留原始负结果，但不得继续用“rank0/1”解释该实验。修正布局与 S3 配额结果见 `RESULTS_S0.md`、`RESULTS_S3.md`。
