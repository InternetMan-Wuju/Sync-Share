# ChunkSize时间戳口径与E2E分解

实际v19源码副本见autonomous_20261012/source_v19/req_time_stats.py与scheduler.py，原始镜像未修改。

## 已核对的源码行为

- `set_forward_entry_time`仅在forward_entry_time=0时记录首次前向；下一次chunk不会覆盖这个首次时间。
- `set_prefill_finished_time`记录最终prefill结束，chunk中途另有last_chunked_prefill_finish_time。
- Retract重置last_*字段，常规set_retract_time不清空首次forward_entry_time；特殊reset_prefill_retry_time会重置。因此请求回退次数也保存，发生回退时不能称为一次连续纯prefill计算。
- 内部perf_counter通过本机校准转为realtime返回。客户端计时用monotonic；比较的是时间差，不把两个时钟的绝对值直接相减。

## 采集与解释

每请求保存完整meta、input/output IDs和SSE。组表包含客户端TTFT、TPOT、E2E和p50/p95，以及四个有明确边界的墙钟跨度：

1. 接收请求至首次forward：包含排队与其他进入计算前的处理，不能全部叫纯queue。
2. 首次forward至prefill完成：包含所有chunk及中间调度间隙，不能叫纯算子时间。
3. prefill完成至请求完成：包含decode与结束处理，以及decode被其他请求prefill干扰的等待。
4. 客户端E2E减服务器接收到完成跨度：保留为客户端/服务器计时残差，不猜为某一种开销。

服务端queue_time独立保存。它与第1项的边界不同，不强行相等，不把差额写成神秘的“其他”后再猜原因。第1至第3项必须望远镜相加等于服务器跨度；加上第4项等于客户端E2E（允许浮点舍入误差）。不记录纯operator加速比，不将10秒metrics采样当成完整调度trace。

2026-10-13 05:25更新：实际Chunk矩阵已完成20个基础单元与14个复测，34单元全部通过时间戳单调、阶段守恒、输入/输出、回退/缓存条件及actual chunk核验。完整数字见`ChunkSize自主实验实测.md`和`ChunkSize负载与阶段分析.md`；缺字段保留缺值，不补造数据。
