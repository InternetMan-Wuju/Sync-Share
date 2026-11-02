# current_exp

当前实验状态和远端同步入口。

## 20261102_load_balance_e0

- E0：**PASS**。健康基线、权重审计、两组路由采集、三轮性能基线。
- E1：**负结果**。单层及三层 top-2 热专家预装副本未改善端到端，停止在线复制方向。
- Profile：prefill 由 DSA attention 主导（约 67%）；decode 中 MoE LL dispatch/combine 约占 15.5%。
- E3：`DEEP_USE_MODE=ops` 在当前 A3/GLM-5.2/graph 组合下触发 507011，不可用。
- E0：[`20261102_load_balance_e0/RESULTS_E0.md`](20261102_load_balance_e0/RESULTS_E0.md)
- E1：[`20261102_load_balance_e0/RESULTS_E1.md`](20261102_load_balance_e0/RESULTS_E1.md)
- Profile/E3：[`20261102_load_balance_e0/RESULTS_PROFILE.md`](20261102_load_balance_e0/RESULTS_PROFILE.md)
- 证据与清单：[`20261102_load_balance_e0/MANIFEST.md`](20261102_load_balance_e0/MANIFEST.md)
