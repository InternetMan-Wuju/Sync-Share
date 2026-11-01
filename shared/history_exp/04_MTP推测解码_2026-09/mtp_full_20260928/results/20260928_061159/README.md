# MTP ON/OFF：三种输入输出，双节点各 45 分钟

状态：**已完成**。三种工作负载两节点各完整测2700秒：1K/32K于07:10:22—07:55:22，8K/1K于07:57:38—08:42:38，32K/300/约60%缓存于09:30:28—10:15:28，均为2026-09-28北京时间。

[完整结果报告](report.md)包含接受长度、吞吐、TTFT/TPOT、成对输入对照、图和限制；另有 [doc版](../../../doc/MTP_三种工作负载_45分钟_20260928.md)。83条正式测量记录全部通过原始流、输入、输出、缓存与接受计数核验，见 `analysis/integrity_audit.json`。两端正式测量无请求错误；失败启动/切换/预热尝试均另存。23 OFF、25 ON服务仍保留，结束后两端“1+1”和中文翻译均正常，见 `semantic_checks_final/` 与 `final_state/`。

前两组使用HCCL_BUFFSIZE650MB。第三组首次前缀预热触发16K Prefill块，两端需要至少797MB通信缓冲，650MB导致CamMoeDispatchNormal报错；只在工作区两端外挂启动命令提升到1024MB并重启，通过实际回复和32K/60%缓存预检后重开第三组完整45分钟。模型和镜像未修改。旧650MB启动材料在 `failed_prefix_environment/node{23,25}_hccl650_chunk_16384/`，新材料在 `node_{23,25}/chunk_16384/`。首次切换时的异步清理400另在 `failed_scene_transition/`。

第一场景取得MTP累计计数：输出21184、验证5299、接受draft15884/15897、接受长度3.997735。两端都循环重复相同3个token ID，这是随机输入与ignore_eos场景下的退化输出，**不能推广为真实对话接受率**；完整输出ID和输出模式审计均保留。窗口输出吞吐OFF=2.117、ON=7.846 token/s。两端均没有完成目标32768输出；实际长度差异也限制了长上下文速度的直接比较。分析见 [analysis/README.md](analysis/README.md)，包含全部三种场景。

## 测量设计

| 场景目录 | 每条输入 token | 每条目标输出 token | 缓存 | 每节点实际并发与时间 |
|---|---:|---:|---|---|
| `decode` | 1024 | 32768 | 请求前清缓存 | 在途1，连续运行45分钟 |
| `mixed` | 8192 | 1024 | 每请求前清缓存 | 在途1，连续运行45分钟 |
| `prefix` | 32768 | 300 | 每请求前清缓存并预热19661个公共前缀 ID，目标token命中约60% | 在途1，连续运行45分钟 |

`32K` 指**整个输入**，不是额外加在输入之前的32K公共前缀。完整输入存于 `dataset.json`；两模式按相同输入列表、顺序提交请求，但下一条发出时间由前一条完成决定，所以墙钟到达时点不同。固定随机 token ID 不代表真实业务内容，也不能用于验收回复语义。

节点 23 为 **OFF**，节点 25 为 **ON**，两者均为独立单机 TP16/EP16。同一镜像、模型配置、相同目标权重字节；MTP ON 的启动命令使用 `NEXTN`、3 个草稿步骤、topk=1、验证槽位4，运行镜像将 NEXTN 归一为 `EAGLE`。服务端实际配置见 [OFF](preflight/node23_off_server_info.json) / [ON](preflight/node25_on_server_info.json)。两台最大服务运行数仍为4，但本次客户端只保持1条测量请求。

**严格截止**：45分钟到达时，不再提交请求；对未完成请求在截止附近按指定请求ID发出 abort，让仍打开的流接收带累计统计的最终帧；客户端到截止点结束流。保存实际生成数量、原始SSE及最终计数。该固定镜像只有最终SSE才附累计接受计数，中途 `/metrics` 值为瞬时 gauge。辅助器提前约2秒请求最终帧；边界若提交极短新请求或在截止后收到输出，原始记录仍保留，窗口吞吐和延迟只统计截止前的输出，接受统计另报覆盖比例。`1K/32K` 单条完整输出可能需要超过45分钟，截断结果按 `partial_at_45m_deadline=true` 或 helper指定ID的最终abort帧识别（原采集器对此帧的partial标记可能为false，原始字段不改）；**不能把它当成完成32768输出**。后续场景开始前通过 `/metrics` 确认没有前场景残留运行请求，否则停止进入下一场景。缓存预热、清理、采集与间隙算入45分钟窗口吞吐；预热请求不算测量样本或接受统计。第一场景边界极短补发被单独标记；后续采集器不在截止前最后2.5秒提交下一测量请求。

## 默认配置与实际生效情况

- `--chunked-prefill-size 16384`，两端实际一致。
- 请求默认的 `--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 24 32`；**实际均为 disabled**，固定镜像日志说明 `deepep_mode=normal` 自动禁用了 Graph，运行日志 `npu graph: False`。不能写成已开启Graph。
- 两端 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`、DeepEP normal、`enable_mixed_chunk=False`、`mem_fraction_static=0.88`、context131072、max_prefill65536一致；所有实际命令在各节点 `chunk_16384/launch.sh`（恢复后的1024MB）；前两组650MB原命令在 `failed_prefix_environment/node{23,25}_hccl650_chunk_16384/launch.sh`。
- 实际KV容量 OFF=63744、ON=43008 token；1024+32768小于ON容量，本轮仍监视动态使用和异常。

## 实际回复与小规模接受长度预检

两个节点均通过独立的“1+1”与“Hello, world.”中文回复检查，原始响应及人工结论见 [OFF](semantic_checks/node23_off/review.json) / [ON](semantic_checks/node25_on/review.json)。随后同一1024-token输入分别生成64个输出 token，HTTP200、全部完成；原始SSE和接受指标保留在 `probe/`，[预检判定](probe/verdict.json)通过。

预检ON：输出64、验证31次、接受草稿33/提出草稿93，接受长度 **64/31=2.064516**，接受率 **33/93=35.4839%**，直方图 `[12,8,8,3]` 总和31。OFF没有接受统计。预检只是验证字段与流程，不能当作正式结果；正式过程的接受长度可能变化。

## 只读外挂、失败记录与权重等价性

25 首次MTP ON启动在草稿加载时报 `VerifiedOpen` 缺少 `get_tensors`，完整失败材料在 `node_25/failed_first_start/`。已在**本轮工作区**复制原外挂并添加 `get_tensors()`，它逐键调用原先带SHA-256校验的 `get_tensor()`，服务镜像与模型文件均未修改。25 使用该副本重启后通过真实回复和统计预检。OFF加载过程中未调用该接口，保留原加载路径；全部外挂来源和SHA在 `experiment_manifest.json` 与各节点权重来源审计中。

两节点修复清单SHA不同，因为各节点本地坏块和外挂修复跨度不同，但 [逐块比较](weight_comparison/comparison.json)表明182个文件的头信息和 **23093个目标数据块SHA全部一致**。两份完整清单也在 `weight_comparison/`。加载器对读出的恢复块逐块验SHA，不仅凭不同清单名称判断模型字节是否相同。

## 保存与统计口径

每条测量请求保存：完整 `request.json`、原始 `sse_raw.jsonl.gz`（每帧的客户端单调时钟与挂钟）、最后的真实或部分 `output_ids_final_or_partial.json`、服务器 `meta_info_last`、请求结果、缓存清理与预热材料。每节点每场景 `metrics_raw.jsonl.gz` 保存全量Prometheus及NPU采样。服务日志、容器状态、模型与外挂摘要、启动前快照都保留。

- 接受长度：实际输出 token 数 / `spec_verify_ct`，包含bonus；多请求按累计输出 / 累计验证次数加权。
- Draft接受率：`spec_num_correct_drafts / spec_num_proposed_drafts`，不含bonus。
- TTFT：请求发出至首个有输出的客户端SSE事件。
- 观察TPOT：`(末次输出到达-首个输出到达)/(实际输出token数-1)`；MTP可一次返回多token，不将流事件间隔误叫逐tokenITL。
- 窗口吞吐：45分钟内实际输出token / 2700秒；另报首末输出之间的观察生成速率和单请求E2E输出速率。
- 小样本场景不把P95/P99当可信尾延迟结论；完整输出与截断输出分开。
