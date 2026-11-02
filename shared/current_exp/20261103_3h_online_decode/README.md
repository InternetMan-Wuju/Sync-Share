# 2026-11-03 3 小时 Online / Decode 补测

- run：`20261103_051202`
- 起始：2026-11-03 05:12:02 +08:00
- 硬截止：2026-11-03 08:12:02 +08:00
- 资源：单张 Ascend NPU，仅绑定 `davinci0`
- 路径身份：eager 局部专家链代理，未验证生产 graph

## 结果入口

- [给本地 AI 的审阅报告](20261103_051202/REPORT_FOR_REVIEW.md)
- [结果与边界](20261103_051202/RESULTS.md)
- [输入与完整槽向量](20261103_051202/INPUTS_PLANS.json)
- [原始计时块](20261103_051202/TIMINGS.csv)
- [环境与哈希清单](20261103_051202/MANIFEST.json)
- [最小修正](20261103_051202/CORRECTIONS.md)
- [复现说明](20261103_051202/REPRODUCE.md)

## 摘要

- Online 热点 960/64：整数配额迁出 448，本地计算代理 0.3158→0.2478 ms（−21.5%）；CPU planner 1.04 µs，本地独立状态复制 37,769,216 B/0.07499 ms，format prep 0.12447 ms。
- Online 均匀 512/512：planner 1.02 µs，不复制、extra slot 零任务，计算差异在噪声内。
- Decode batch1：EP32/64 没有正计算节省。
- Decode batch8：EP32/EP64 计算代理中位节省约 0.040–0.041 ms/层（15.3%–15.6%），这是新增通信/调度的全部回本预算。
- 真实 dump 已核验为每 step 128 tasks、33–44 个非零专家，符合 batch16×top8，并非 TP16 重复；因缺 token top8/router weight，真实 trace 六格仍未完成。
