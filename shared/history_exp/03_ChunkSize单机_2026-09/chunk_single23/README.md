# 23 节点单机 Chunk Size 对照（2026-09-26）

## 目标与口径

> **需求澄清（首轮实验完成后）**：用户要求的是**总输入 32K、输出 300、token 缓存命中约 60%**。当时把“前缀 32K”理解为“公共前缀本身 32K”，因而设计了额外的 54,614 输入场景；下方原计划与实施记录按历史原貌保留，不能将 54.6K 结果当作目标 32K 的结果。首轮 `length_32768`（图中 `32K Warm`）与目标长度相符：总输入 32768，预热前缀 19661，实测命中 19584/32768 ≈ 59.77%，输出 300。两种 Chunk 各仅 1 条独立暖态请求；**缺少目标 32K 的 Prefill/Decode 干扰对照**。

- 23 节点，GLM-5.2 W8A8 `glm-qq`，固定镜像 `sglang-glm52-pd:fe694986`（ID 见 `serve.sh`），单机统一服务 TP16/EP16，MTP 关闭。只改变 `--chunked-prefill-size`：8192、1024。
- 原话「前缀 32K / 输出 300 / 缓存命中 60%」按**公共前缀本身 32768 token**解释。总输入设 54614 token，随机独有后缀 21846 token，目标 token 级命中率 32768 / 54614 ≈ 60%；长输入加输出 300 不超过 `--context-length 131072`。实际命中率须由 `/generate` 的 `meta_info.cached_tokens / prompt_tokens` 验证。预热前缀的请求不计入主实验。
- 缓存冷态对照：flush 后单次 54614/300；暖态对照：flush、32768/1 前缀预热、54614/300。
- Decode 干扰：保持 32768 前缀驻留，先跑 1024/900 的独立 Decode 基线；再发相同规模 Decode，等首 token 返回才注入 54614/300 暖态长 Prefill。对比 Decode 的 TPOT、E2E、输出 token/s，并报告长请求 TTFT、缓存命中。至少 2 轮，随机种子固定。
- 长上下文参照：总输入 16384、32768，各以其 60% token 前缀预热、输出 300。各 case 的实际命中率独立核对。目标主 case 54614 对应 32K 真实公共前缀；若容量不足，不把 32K 总输入替代成满足需求的成功结果。

## 部署

```bash
bash /home/chengyetai/workspace/current_exp/chunk_single23/serve.sh start 8192
bash /home/chengyetai/workspace/current_exp/chunk_single23/serve.sh status
bash /home/chengyetai/workspace/current_exp/chunk_single23/serve.sh stop
bash /home/chengyetai/workspace/current_exp/chunk_single23/serve.sh start 1024
```

一次只运行一个配置。服务地址为 `http://10.120.72.23:30080`。启动用只读权重挂载，日志挂到当前目录 `results/<run_id>/chunk_<size>/server.log`。`stop` 只移除这项实验创建的两个容器名。镜像、模型、设备、HCCL 缓冲区、TP/EP、禁用 CUDA graph、上下文、KV 池占比、请求上限、缓存报告和采样参数固定，方便做 Chunk 的对照。启动后需以 `/health` 200 和短请求完成验证，不能仅凭容器 Up 判断可用。

首次尝试选择 `mem-fraction-static=0.93`、`max-running-requests=4`、`HCCL_BUFFSIZE=1024`、`deepep-mode=normal`，避免早期双机试运行的 HCCL buffer 不足；TP16 权重/显存以实际启动日志为准。历史记录曾指出单机 KV 余量可能不足，54.6K 场景必须先做容量冒烟，失败时记录具体显存及可用 KV，而非直接扩大并发。

## 测量

```bash
RUN=/home/chengyetai/workspace/current_exp/chunk_single23/results/<run_id>/chunk_8192
python3 /home/chengyetai/workspace/current_exp/chunk_single23/bench.py --out-dir "$RUN" --mode smoke
python3 /home/chengyetai/workspace/current_exp/chunk_single23/bench.py --out-dir "$RUN" --mode full --reps 2
```

先完成两种 Chunk 各自 smoke，再对能通过容量冒烟的配置跑 full。`smoke.jsonl`、`full.jsonl` 保留逐请求实际输出数量、TTFT、TPOT、缓存 token、服务端 `meta_info`、SSE 事件时间以及出错信息；各自 `_summary.json` 保留分阶段统计。前缀预热、flush 操作和混合负载长请求注入时间也逐条记载。Decode 的 TPOT 口径 `(E2E−TTFT)/(实际输出 token−1)`；SSE 合批时单个事件间隔不能解释为逐 token ITL。受 `max-running-requests=4` 限制，先使用一个 Decode 背景请求和一条长 Prefill；若容量/时间允许再提高背景并发，需在报告中注明。

结果比较必须注明健康状态、完整输出是否等于 300/900、长请求与 Decode 是否真正重叠、实际缓存命中是否接近 60%，以及两种 Chunk 的 rank 数、显存、KV 池数值。不完整输出、缓存失效、OOM 和服务未就绪作为实验结果记录，不能写成成功的性能对照。

## 历史资料

- `history_exp/current_exp/问题总结.md`：旧单机 TP16/DP4 曾仅剩约 400 MB KV 余量，16K 输入 OOM（记录时的模型、镜像和拓扑需与本轮区分）。
- `history_exp/current_exp/bench_pd_prefix.py`：随机 `input_ids`、流式 TTFT/TPOT、前缀预热的实现参考。
- `history_exp/current_exp/实验总结与规划.md`：`glm-q` 权重损坏，`glm-qq` 是此前验证可用的权重。
- 本目录上层 `doc/ChunkSize与MTP推理实验说明.md`：Chunk、MTP、缓存命中和测量指标的背景说明（2026-09-24，旧路径可能过时）。

## 2026-09-26 启动修正

首次试运行 `results/20260926_143359/` 中，TP0–7 权重在约 66–75 秒加载完成，TP8–15 仍在持续读盘；等待后半组 rank 的 Gloo `monitored_barrier` 在 480 秒触发，服务于 14:44 左右退出。查看固定镜像里的 `model_runner_components/load_model_utils.py` 后确认：这里的等待阈值为模块常量 `UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480`，与全局 `--dist-timeout` 分开。第二次运行 `results/20260926_144841/` 使用 `patches/load_model_utils.py` 原文件和一行补丁（3600 秒），只读挂载回容器；同时设置 `--dist-timeout 3600`。补丁和原文件的哈希记在每轮的 `code_sha256.txt`。本修正仅延长慢盘/不均匀加载的容忍时间，若某 rank 实际失败仍需看失败堆栈，不能把超时延长当作权重或 KV 容量问题的修复。

第二轮 `results/20260926_144841/` 已确认 16 个 rank 全部加载权重，TP8–15 约 355–357 秒；初始化 KV 时日志报告 `max_total_num_tokens=98944`、`available_gpu_mem=2.65 GB`。服务预热在 `aclnnNotifyDispatch` 报 HCCL 模块申请显存失败（`Memory_Allocation_Failure(EL0004)`），`/health=503`，因此这轮没有请求性能数据。第三轮把 `mem-fraction-static` 从 0.93 调为 0.88，把 normal Deepep 模式的 `HCCL_BUFFSIZE` 从 1024 MB 调回历史记录用过的 650 MB；等待实际 KV token 池和 54.6K 请求结果验证是否同时满足通信及长上下文。由于这一轮改了显存设置，两种 Chunk 的对照必须都在同一新设置下运行。

## 固定镜像里的 Prefill/Decode 调度条件

两组启动日志的 `server_args` 都显示 `enable_mixed_chunk=False`。固定镜像 `sglang/srt/managers/scheduler.py` 的本地源码中，约 1225 行据此设置 `is_mixed_chunk`，约 3305 行选新 Prefill batch 优先、否则才选已有 Decode，约 3669 行仅当 `is_mixed_chunk=True` 才将两类请求合为一个 mixed batch。因此本轮压测观测的是默认调度条件：Chunk 从 8192 减到 1024 增加 Prefill 调度轮数，但不会自动让 Decode 在每个 Prefill 块之间获得执行机会。若要研究主动交错调度，需另做两种 Chunk 都开启 `--enable-mixed-chunk` 的对照并先验证当前 Ascend 后端是否支持，不应与本轮默认模式数值混为一组。

## 已完成的正式结果

`results/20260926_145836/` 为两种 Chunk 使用同一批确定性请求、同一配置的正式实验目录；[完整实验报告](../doc/ChunkSize单机实验_2026-09-26.md) 汇总冷/暖缓存、两轮背景 Decode 干扰、16K/32K/54.6K 长上下文、服务端 KV 指标和调度器源码条件。两组请求无错误，主场景各完成 300 输出 token 并实际命中 32768/54614；所有四条背景 Decode 请求各完成 900 token。结束时 23 节点保留 Chunk=1024 的服务，`/health=200`，状态记录见 `results/20260926_145836/end_state.txt`。变更或释放本实验服务使用本目录 `serve.sh status|stop|start`；不要用旧的双机启动脚本管理它。
