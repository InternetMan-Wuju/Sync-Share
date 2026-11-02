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
