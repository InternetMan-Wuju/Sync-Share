# 20261103 复杂 MoE 模拟

- [结果](RESULTS.md)
- [全量模拟输出](SIMULATION_RESULT.json)
- [脚本](simulate.py)
- [清单](MANIFEST.json)

范围：真实 dump 的 75 个 MoE layer、4 个 prefill step、64 个 decode step；EP16/32/64 与 EP16 online 临时副本。局部时间来自 NPU 实测曲线，网络和 dispatch 是参数扫描，不是实机标定。
