# Dolly 快速 EP 指标采集

这个脚本包先用 Dolly-15k 做快速验证，目标不是全量实验，而是确认我们能稳定拿到 EP 相关数据。

## 前置条件

1. 四机 EP64 decode 服务已经启动，并且 `/health` 返回 200。
2. 服务启动脚本已经包含：

```text
--expert-distribution-recorder-mode stat
--expert-distribution-recorder-buffer-size -1
```

如果当前服务是旧参数启动的，需要先重启服务。

## 运行

```bash
cd /home/chengyetai/workspace/glm52_ep64_chengyetai/ep_lab

LIMIT=32 \
CONCURRENCY=4 \
MAX_TOKENS=256 \
bash run_dolly_ep_cell.sh
```

## 输出

结果默认保存在：

```text
../dataset/ep_runs/dolly_ep64_<时间>/
```

主要文件：

- `dolly_bench.json`：Dolly 请求性能
- `expert_distribution.pt`：SGLang 原始专家分布
- `expert_distribution.json`：专家分布可读版本
- `metrics.prom`：Prometheus 指标
- `npusmi_after.txt`：NPU 快照
- `summary.json`：汇总结果
