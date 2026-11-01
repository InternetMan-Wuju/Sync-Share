# 单机 Chunk 8K/1K：四请求错开注入实验

正式记录和图表：[2026-09-28 实验报告](results/20260928_043035/README.md)；[逐请求及干扰明细](results/20260928_043035/analysis/README.md)；[完整文件 SHA-256 清单](results/20260928_043035/file_inventory.sha256.tsv)。所有新脚本、输入及输出都保存在此目录。实验使用 23 节点，25 节点服务维持原状，模型目录和容器镜像均未修改。

代码角色：`serve.sh` / `serve_verified.sh` 为原有只读权重外挂的服务启动脚本；`chat_once.py` 记录两种配置下的真实回复；`capture_state.py` 保存实验前后节点状态；`bench_staggered.py` 创建固定随机输入并采集原始 SSE、逐帧时钟、服务端元数据、NPU 和 `/metrics`；`finalize_outputs.py` 从 SSE 最后一帧重建真正输出 ID；`analyze.py` 生成可重算的 CSV 和逐请求报告；`plot.py` 绘制 ITL 时间线；`audit.py` 逐项核对八条请求并生成文件 SHA-256 清单。`before/` 是启动本次服务前的机器快照。

本次 8K 与 1K 的实际采集器版本均记在结果目录的 `experiment_manifest.json`；后续修改的分析脚本记录了新的摘要。当前 23 节点 **1K 服务仍运行**，重复实验前须自行明确准备独立目录和服务切换时机，避免把新数据混入已归档的 `20260928_043035`。报告中的图、CSV 属于派生数据；原始 SSE、完整 `input_ids`、每组的缓存清理与预热、逐请求结果和当时的服务日志都保存在结果目录。
