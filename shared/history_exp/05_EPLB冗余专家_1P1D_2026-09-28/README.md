# EPLB / 冗余专家：AgentX 派生请求，1P1D 对照实验

2026-09-28。最终采用 **23 单机 Prefill + 25 单机 Decode**，Router 在 23。历史最初四节点 2P+2D（23、25、19、20）准备工作记录留在 `preflight/`，**正式实验仅用 23 和 25**。2026-09-28 18:59 后的有效轮次统一为每组10分钟，历史30分钟结果只作排障参考。没有改模型或镜像；所有修复是只读外挂。目标镜像 ID：`sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d`。

| 组 | EPLB | Prefill / Decode 冗余专家 | 时长 |
|---|---|---:|---:|
| A | 关 | 0 | 10 分钟 |
| B | 开 | 0 | 10 分钟 |
| C | 开 | 16（单机 EP16 的最小正冗余；如启动 OOM 记录失败） | 10 分钟 |

所有组使用相同实际输入、输出长度、并发和采集方式。先试 **16K / 512**，如果在服务运行中 OOM，统一调低负载后重跑前面组，不能拿不同长度直接作 A/B/C 对比。Prefill TP16/DP4、DeepEP normal、命令行 chunk16384（DP attention 实际调整为 **4096**）、关闭 CUDA Graph；Decode TP16/EP16/DP4、DeepEP low_latency、Decode Graph full（maxbs32，8/16/24/32）；EPLB rebalance 周期 Prefill32/Decode1000，Prefill/Decode 专家记录窗口分别为32/128 次 forward，`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`。是否真正启用 Graph 和 EPLB，需从实际日志检查。

## 数据和复现口径

原始 `/mnt/raid/user/data/dataset/cc-traces-weka-062126-256k/traces.jsonl` **只有原始请求的 token 数、相对时间和每 64-token 内容哈希块，没有原始文字或原始 token ID**。`data/agentx_16k_512.json` 按源 trace 的内容哈希与出现顺序选取 4096 行：原输入不少于 16384、原输出不少于 512；对同一 trace 的同一个哈希块确定性生成同一段 64 个合法 token。该数据保持 trace 给出的请求分布和前缀共享形状，**不是原文或原 token 重放**。18:59 新轮负载改为 **4096 输入 / 128 输出 / 闭环并发 4**，每组运行 600 秒后停止提交并等待已提交请求全部完成；旧轮使用8192/256/1800秒，仅保留历史诊断数据。没有回放原 trace 的请求到达时间。缩短输入时截取前 N/64 块，输出上限参数固定为各组同一值。每轮另发一个中文问题保存完整答案，以确认真实回复正常。

运行入口：`scripts/run_matrix.sh`；`DURATION=600 CONCURRENCY=4 INPUT_TOKENS=4096 OUTPUT_TOKENS=128`。结果在 `results/<session>/<group>/`。`bench/requests.jsonl` 保存每个请求的来源索引、输入 SHA256、请求与完成时刻、输出数、TTFT、TPOT、E2E、状态；`bench/sse/` 保存逐事件的压缩原始 SSE；`metrics/` 周期抓取前后端指标及专家热度启动/停止/dump 操作；`node_*/` 保存服务器全量日志、运行命令、容器配置、权重校验统计及 EPLB 完整窗口。启动 OOM 的每一次尝试也保留在独立的组目录下。

历史准备阶段曾把权重副本传至 19、20 RAID（`model_readonly_replica` 为工作空间软链），镜像准备与文件校验均有 `preflight/` 证据；新方案无需这些副本，也不启用 19、20 的服务。

## 已发生的失败和调整（运行中更新）

- `results/20260928_1345_1p1d/A/`：Decode 25 原 `--mem-fraction-static=0.82`，权重已加载但 KV cache 初始化报错，要求 >0.846；该尝试无正式请求，原日志/inspect 均保留。Decode 调为 0.88 后健康，并在日志中确认 full decode NPU graph 捕获完成。
- `results/20260928_1345_1p1d/A_retry_mem088/`：中文实际问答正常；最初并发 20 × 输入 16K / 输出 512，Prefill 23 第一批请求发生 NPU OOM（申请 146 MiB 时不足），请求原始 SSE、周期 metrics、服务器日志已保留；该窗口**不是有效 30 分钟测量**。
- 新尝试 `A_8k256_pf087`：三组共同缩减至 **8192 输入 / 256 输出 / 并发 4**；Prefill 内存比例改 0.87，Decode 保持 0.88，待小规模请求检验后正式起计时。此为用户授权的因 OOM 灵活调小。不能把新负载与失败的 16K/512 直接作性能比较。
- `A_8k256_pf087`：8K/256/并发4 先完成24条，随后 Decode 25 在专家记录器的 `_InfiniteBuffer.append` 扩容时申请 314 MiB、仅约307 MiB空闲，发生 OOM，健康检查失效；已立即停止，保存原始请求、指标、两端日志，**不是有效的30分钟**。该 OOM 是记录器无限缓冲增长，不能仅归咎于请求长度。下一次两端都把 `--expert-distribution-recorder-buffer-size` 设为128、Decode 静态内存比例0.87。`stat` 模式的每次热度记录**仅保留最近128步**，不是整个重排周期的完整热度；必须据此解释 EPLB 窗口和热力图。Graph 仍保留 full/maxbs32/8,16,24,32。
- 单机 EP16、256 个路由专家时，Decode full Graph 捕获后日志仅有约 1.1–1.4 GiB 可用空间。原计划 C 冗余32 很可能 OOM；依用户允许灵活缩减冗余，本次单机 C 直接先试 **冗余1**，按真实启动及运行状态判定，不虚报为32。脚本中该组的实际名字 `C_r1`。

## 重排触发及运行中热力图快照（2026-09-28 追加）

源码 `EPLBManager.on_forward_pass_end()` 每次 forward 推进一次 generator；到达 `--eplb-rebalance-num-iterations` 才尝试 `rebalance()`。1000 次并不等于 1000 个请求，在本次 1P1D、短时限、DP4 的 Prefill 上不可直接照搬 Decode 设置。原首次开启 EPLB 的 Prefill 周期100、记录窗口128，启动即断言失败（`results/20260928_1524_1p1d/B/failed_prefill_assert.log`）；预备的周期256虽能加载，但预期每个 DP rank 的 Prefill forward 数不足以保证多次重排，已保留 `B_pref256/` 的日志并在加载期间停下。有效 B/C 采用 Prefill 周期32+窗口32、Decode 周期1000+窗口128。必须查两端真实日志中 `rebalance start/end` 与 `.pt`，若没有发生，不能声称 EPLB 生效。A 已完成的初版 Prefill 记录窗口为128，与新组不同；计划按相同32窗口补跑可比基线。

每个运行组在第 **5、10、15、20、25、29 分钟**抓取两端热度快照，文件置于 `snapshots/`：`NNm_prefill23.png/.pt` 和 `NNm_decode25.png/.pt`；`snapshots.jsonl` 记录采集时刻、来源、原始张量维度和错误。图显示 **MoE 层 × 逻辑专家 ID** 的路由 token 数（色标 `log(1+count)`）；它不是设备利用率或专家当前放置图。EPLB 开启组从重排自动导出的完整窗口 `.pt` 复制最近一份，无须暂停/重置服务统计；不同图窗口长度分别为 Prefill32、Decode1000 forward，不应拿两端单张颜色深浅直接比较。关闭组通过 stop/dump/start 记录器取得分段窗口，采集动作和其耗时另记在快照索引，不掩饰测量干扰。原始 `physical_to_logical_map` 与各步计数保存在 `.pt`，可另作放置分析。

## 2026-09-28 17:17：Decode 首次重排故障和补丁核对

`results/20260928_1524_1p1d/B_pref32/` 是故障轮，**无效，不能用于 A/B 吞吐对比**：16:58 开始运行，Decode 25 于服务器日志 08:59:19（容器 UTC）进入首次 `EPLBManager rebalance start`，其后没有 `rebalance end`。节点 25 的 `decode_first_rebalance_traceback.txt` 显示在 `expert_distribution.py:_StatAccumulator.dump` 的 `torch.distributed.all_reduce(logical_count_of_buffered_step)` 处 HCCL `Memory_Allocation_Failure`，单次请求 2148532224 字节（约 2 GiB）；节点 23 后续在同一 all_reduce 阶段也报同样的 HCCL 分配失败。故障轮 12 条完成请求及其 SSE、周期 metrics、两端完整 docker 日志、inspect、快照未产出的错误索引均保留。于 17:12 停止并清理失效服务；没有人为把卡住的请求补记成成功。

`common/patches/` 的三个补丁已在故障之前**全部**以只读挂载用于 23/25，两份工作空间文件和 common 原文件的 SHA256 一致，`B_pref32/node_25/docker_inspect.json` 记录了容器真实挂载。它们分别处理专家搬运的张量格式、HCCL P2P 成批操作和完整统计窗口；本轮首次故障是更早阶段的统计 `all_reduce` 内存申请，不能误认为“忘记应用补丁”，也不能据此声称权重搬运补丁已在本轮经历真实移动验证。

新尝试 `results/20260928_1720_hccl64/`：两端 `HCCL_BUFFSIZE` 从 1024 调至 **64**，其余 workload 仍统一 8192/256/并发4、Prefill 重排/记录32/32、Decode1000/128，C 冗余1。先观察首次真实重排是否 `start/end` 成对、热力窗口是否落盘、模型能否继续正常回复；若成功才把它当作有效 30 分钟运行。最终 A 基线将同样采用64，避免拿 HCCL 设置不同的 A 与 B/C 混比。这个参数设置属于待验证的故障规避措施，不能直接由现有日志证明一定成功。

17:42 部署校验发现 23/25 的 `/home/chengyetai/workspace` **不是同一份文件系统**。`B_hccl64` 首次在 25 仍用旧版 `start_role.sh`（进程环境 HCCL_BUFFSIZE=1024），所以在未开始请求之前保存该 Decode 的部分日志/inspect 为 `B_hccl64_decode_old1024/node_25/`、移除旧容器、rsync 同步 23 的新脚本到 25，重新在 25 启动 Decode；23 已加载的 Prefill 保留。重启后的 25 启动脚本 SHA256=`8143bc747d2d243f104b550e5d3fba3ef6cf104023106e6f9510bffb9d7f6982`，两个节点 `/proc/1/environ` 已分别读取确认 `HCCL_BUFFSIZE=64`。25 的三个补丁 SHA256 也与 `common/patches` 逐个一致。新一轮 B 的正式计时尚未开始；该启动过程不计入 30 分钟。

17:49 新增预热阶段失败：23 在实际缓冲64时 `CamMoeDispatchNormal do tiling failed`，健康持续503，未发正式请求。故障容器日志/inspect/启动命令归档至 `B_hccl64_prefill64_failedwarmup/node_23/`，因此上文“两端64”**仅描述该次已失败的启动配置**。从17:50 重启的当前 Prefill 用 `HCCL_BUFFSIZE=1024` 并将静态显存比例从0.87降至 **0.82**，为首次 EPLB 统计归并留出空间；Decode 仍用 **64**、比例0.87。启动脚本已同步到 25，双方 SHA256 均为 `8142e95f3ea7cf32b47580f371220878317aa6c6425e9aa9d7c37ce262cf7995`，从 `/proc/1/environ` 验证角色对应真实值1024/64。以上为尚待实际请求、重排和热力快照验证的新组合，若有效，A/B/C 均使用这一组合。

## 18:17 新轮：仅统计热度通过 CPU/Gloo 汇总

缓存/通信缓冲排查发现：HCCL_BUFFSIZE=64 使 23 的 DeepEP normal warmup 报 `CamMoeDispatchNormal do tiling failed`，25 的 DeepEP low_latency + full Graph 捕获报 `MoeLowLatencyDispatchV2 do tiling failed`，均不是有效负载；日志/inspect 分别在上轮结果的 `B_hccl64_prefill64_failedwarmup/`、`B_hccl64_decode64_failedinit/`。随后仅靠降低静态内存比例也不可取：23 的 0.82 被 KV cache 的明确阈值 >0.851 否决 (`B_hccl64_prefill082_kvcache/`)；这几次失败均没有正式 B 数据。

检查容器原源码 `distributed/parallel_state.py`：`get_world_group().cpu_group` 已是 `backend="gloo"` 的 CPU 通信组；专家权重更新的补丁和模型保持不动。在当前实验目录由原始 `expert_distribution_full_window_patched.py` 派生只读外挂 `expert_distribution_full_window_cpu_gloo.py`，仅把 EPLB 统计 `logical_count` 的归并转换成 CPU 张量并用现有 Gloo 组做 `all_reduce`，必要时的全局利用率广播也使用同组；输出标记 `aggregation_backend=gloo_cpu`，留存原始张量和完整窗口。专家放置算法会把逻辑计数送回对应设备。两端恢复已完成过 warmup 的 **HCCL_BUFFSIZE=1024 / 静态比例0.87**，Prefill 周期/窗口32/32、Decode1000/128 不变。此新补丁在 23/25 哈希核对后一同以只读方式挂载，不修改 `common/patches/`、模型或镜像；从日志确认每次 `EPLB logical-count stats aggregated on CPU/Gloo`、`rebalance start/end` 和窗口落盘后才能判定有效。

源码 `ExpertLocationMetadata._init_common` 要求 `(num_logical_experts + ep_num_redundant_experts) % ep_size == 0`：这里 256 逻辑专家、EP16，最小正冗余是 **16**，不是此前规划的1。C 改试 `C_r16`，可能 OOM 时必须如实报告，不能把不合法的冗余1伪装成有效实验。正式新会话目录 `results/20260928_1814_cpu_gloo/`，先 B 30 分钟，再 C，最后用同样配置补跑 A 30 分钟；各组的6时点热度图与原始 `.pt` 会在 `snapshots/`，失败轮不纳入组间统计。

## 18:51 新轮：延迟搬运权重的临时副本

`results/20260928_1814_cpu_gloo/B_cpu_gloo/` 不是有效 30 分钟实验：成功中文问答、12条8192/256请求；首次 Decode 重排真实走通 **CPU/Gloo 统计全归并**，rank0 导出一个非零128×78×256窗口 (`node_25/eplb_windows/expert_distribution_window_000000_*.pt`)，已将其复制并绘成 `snapshots/first_rebalance_failed_decode25.png/.pt/.json`。之后 Decode 在 `common/patches/expert_location_updater_patched.py` 的 `_p2p_send_tensor`，一次 `npu_format_cast` 申请194MiB 时仅余78MiB，发生 OOM，不能报告成功重排。所有请求/SSE、两端日志/inspect、指标、单张真实热图都保留，停表后清理容器。

源码显示原专家搬运补丁在 `_create_isend_ops` 为一个 MoE 层所有需要发出的专家**先**创建 ND 格式副本，而 `_execute_p2p_ops` 才将其按批发送。每层大量专家重排时这些临时张量同时占用显存。新只读外挂 `patches/eplb/expert_location_updater_lazy_send.py` 从 common 原补丁派生：先保留原权重视图，再按逻辑专家 ID 批（`SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE=1`）转格式、发送、等待、释放该批副本；CPU/Gloo 统计外挂保持。Decode 静态比例从0.87调至0.86，为至少一个194MiB临时张量腾空间；Prefill维持0.87。两端脚本/补丁哈希已在新会话预检核对。新结果目录 `results/20260928_1851_lazy/`。此修改改变重排时延，所有 A/B/C 都使用相同启动参数（关 EPLB 组没有重排动作）；需要观察实际吞吐，不能假定无性能开销。

## 18:59 用户紧急调整（实际以会话日志为准）

用户将三组本轮运行时限统一改为 **每组600秒**，允许大幅降低各项负载。新轮计划输入4096、输出上限128、并发4；第60/150/240/330/420/510/570秒采集 Prefill 与 Decode 热图并保存原始 .pt。18:51 已开始的两端加载继续，避免等待再次加载；因此 Prefill 当前周期/窗口保持32/32、Decode1000/128，以实际 `launch.sh` 和窗口文件为准。B 装载完成后正式计时，随后运行同负载的 A 基线，最后尝试 C 冗余16。快照窗口不足会单独记录，不会伪装为重排成功；先保证其他组仍能运行。先前历史组的1800秒结果与新600秒组不能直接比。
