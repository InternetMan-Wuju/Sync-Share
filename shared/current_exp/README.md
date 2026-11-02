# current_exp

当前实验状态和远端同步入口。

## 20261102_load_balance_e0

- E0：**PASS**。完成健康模型、16 卡 HCCL、两组独立路由采集和三轮基线。
- E1：**负结果**。layer 49 单层及 49/36/54 三层 top-2 热专家预装副本没有改善端到端，停止 C/D 在线复制方向。
- E0 结果：[`20261102_load_balance_e0/RESULTS_E0.md`](20261102_load_balance_e0/RESULTS_E0.md)
- E1 结果：[`20261102_load_balance_e0/RESULTS_E1.md`](20261102_load_balance_e0/RESULTS_E1.md)
- 证据与清单：[`20261102_load_balance_e0/MANIFEST.md`](20261102_load_balance_e0/MANIFEST.md)

原始路由张量、完整权重审计和运行日志保留在本地实验目录，不提交 Git。
