# ChunkSize低并发预实验表（W1并发32未测）

场景按输入/输出token数命名，并发为客户端同时在途的请求上限。原代号见[名称说明](../autonomous_20261012/负载与配置名称说明.md)。

<!-- chunk-requirement-coverage:start -->
**ChunkSize正式要求未完成：W1应并发32、64请求，当前只测并发2/4。**

现有34单元仅作低并发预实验；4K/8K候选结论限于已测配置和并发，不能推广到W1并发32，也不能当作榜单的单项优化结果。

客户端并发不等于服务器running batch，KV容量估算不能直接证明并发32无法运行。差异与证据见[ChunkSize要求与自主探索偏差](ChunkSize要求与自主探索偏差.md)。

<!-- chunk-requirement-coverage:end -->
23单机、公开源校验权重、原生random-ids；同场景/并发下输入固定。MTP关闭；同一矩阵服务参数除chunk外固定。缓存命中和原始日志单列。

| 场景 | 并发 | Chunk | 轮次 | 成功/请求 | 总输出tok/s | TTFT p50 s | TTFT p95 s | TPOT p50 ms | TPOT p95 ms | E2E p50 s | E2E p95 s | 命中率 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1：输入8192 → 输出1024 | 2 | 1024 | 1 | 4/4 | 32.623 | 6.931 | 8.932 | 54.593 | 56.557 | 62.767 | 62.828 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk1024_c2/result.json) |
| W1：输入8192 → 输出1024 | 4 | 1024 | 1 | 8/8 | 57.643 | 11.044 | 16.938 | 58.589 | 64.877 | 71.041 | 71.253 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk1024_c4/result.json) |
| W1：输入8192 → 输出1024 | 2 | 16384 | 1 | 4/4 | 25.072 | 28.789 | 29.645 | 51.697 | 51.804 | 81.675 | 82.641 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk16384_c2/result.json) |
| W1：输入8192 → 输出1024 | 2 | 16384 | 2 | 4/4 | 29.584 | 15.353 | 25.540 | 52.653 | 52.679 | 69.217 | 79.381 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk16384_c2_repeat2/result.json) |
| W1：输入8192 → 输出1024 | 4 | 16384 | 1 | 8/8 | 38.738 | 51.482 | 53.963 | 53.004 | 53.706 | 105.714 | 107.882 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk16384_c4/result.json) |
| W1：输入8192 → 输出1024 | 4 | 16384 | 2 | 8/8 | 43.537 | 40.103 | 54.943 | 52.745 | 55.175 | 94.060 | 108.784 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk16384_c4_repeat2/result.json) |
| W1：输入8192 → 输出1024 | 2 | 4096 | 1 | 4/4 | 34.875 | 4.912 | 5.683 | 52.581 | 53.304 | 58.716 | 58.819 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk4096_c2/result.json) |
| W1：输入8192 → 输出1024 | 2 | 4096 | 2 | 4/4 | 34.788 | 4.898 | 5.667 | 52.735 | 53.459 | 58.861 | 58.960 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk4096_c2_repeat2/result.json) |
| W1：输入8192 → 输出1024 | 4 | 4096 | 1 | 8/8 | 62.915 | 8.385 | 11.249 | 55.411 | 59.523 | 65.084 | 65.205 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk4096_c4/result.json) |
| W1：输入8192 → 输出1024 | 4 | 4096 | 2 | 8/8 | 62.722 | 8.377 | 11.247 | 55.627 | 59.722 | 65.285 | 65.319 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk4096_c4_repeat2/result.json) |
| W1：输入8192 → 输出1024 | 2 | 8192 | 1 | 4/4 | 35.127 | 5.207 | 5.241 | 51.898 | 51.908 | 58.298 | 58.343 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk8192_c2/result.json) |
| W1：输入8192 → 输出1024 | 2 | 8192 | 2 | 4/4 | 35.049 | 5.203 | 5.235 | 52.029 | 52.081 | 58.428 | 58.446 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk8192_c2_repeat2/result.json) |
| W1：输入8192 → 输出1024 | 4 | 8192 | 1 | 8/8 | 63.690 | 9.077 | 10.423 | 53.975 | 57.769 | 64.292 | 64.316 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk8192_c4/result.json) |
| W1：输入8192 → 输出1024 | 4 | 8192 | 2 | 8/8 | 63.506 | 9.069 | 10.421 | 54.164 | 57.991 | 64.483 | 64.591 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk8192_c4_repeat2/result.json) |
| W2：输入32768 → 输出1024 | 1 | 1024 | 1 | 1/1 | 14.387 | 17.582 | 17.582 | 52.386 | 52.386 | 71.173 | 71.173 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk1024_c1/result.json) |
| W2：输入32768 → 输出1024 | 1 | 16384 | 1 | 1/1 | 9.463 | 54.890 | 54.890 | 52.120 | 52.120 | 108.209 | 108.209 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk16384_c1/result.json) |
| W2：输入32768 → 输出1024 | 1 | 16384 | 2 | 1/1 | 9.163 | 58.705 | 58.705 | 51.852 | 51.852 | 111.750 | 111.750 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk16384_c1_repeat2/result.json) |
| W2：输入32768 → 输出1024 | 1 | 4096 | 1 | 1/1 | 15.826 | 11.792 | 11.792 | 51.720 | 51.720 | 64.702 | 64.702 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk4096_c1/result.json) |
| W2：输入32768 → 输出1024 | 1 | 4096 | 2 | 1/1 | 15.690 | 11.778 | 11.778 | 52.274 | 52.274 | 65.255 | 65.255 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk4096_c1_repeat2/result.json) |
| W2：输入32768 → 输出1024 | 1 | 8192 | 1 | 1/1 | 15.899 | 11.025 | 11.025 | 52.180 | 52.180 | 64.405 | 64.405 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk8192_c1/result.json) |
| W2：输入32768 → 输出1024 | 1 | 8192 | 2 | 1/1 | 15.827 | 11.022 | 11.022 | 52.469 | 52.469 | 64.699 | 64.699 | 0.0000 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w2_32k_1k_chunk8192_c1_repeat2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 1024 | 1 | 2/2 | 12.996 | 7.412 | 7.505 | 52.411 | 52.436 | 23.083 | 23.184 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk1024_c1/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 1024 | 1 | 4/4 | 19.151 | 12.182 | 16.167 | 63.986 | 75.253 | 31.323 | 32.225 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk1024_c2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 16384 | 1 | 2/2 | 12.335 | 8.724 | 9.534 | 52.155 | 52.175 | 24.319 | 25.134 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk16384_c1/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 16384 | 2 | 2/2 | 12.640 | 8.234 | 8.715 | 51.831 | 51.880 | 23.732 | 24.227 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk16384_c1_repeat2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 16384 | 1 | 4/4 | 15.013 | 24.181 | 27.049 | 52.755 | 52.807 | 39.953 | 42.838 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk16384_c2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 16384 | 2 | 4/4 | 14.268 | 26.273 | 30.158 | 52.724 | 52.747 | 42.039 | 45.919 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk16384_c2_repeat2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 4096 | 1 | 2/2 | 13.569 | 6.645 | 7.784 | 51.710 | 51.728 | 22.107 | 23.251 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk4096_c1/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 4096 | 1 | 4/4 | 21.049 | 10.162 | 14.397 | 56.841 | 68.570 | 28.500 | 30.845 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk4096_c2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 4096 | 2 | 4/4 | 21.407 | 10.061 | 13.527 | 56.967 | 66.375 | 28.021 | 29.880 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk4096_c2_repeat2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 8192 | 1 | 2/2 | 14.695 | 4.821 | 4.840 | 52.148 | 52.151 | 20.413 | 20.434 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk8192_c1/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 8192 | 2 | 2/2 | 14.631 | 4.821 | 4.841 | 52.447 | 52.476 | 20.502 | 20.532 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk8192_c1_repeat2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 8192 | 1 | 4/4 | 21.489 | 10.455 | 14.171 | 54.143 | 62.823 | 27.914 | 30.436 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk8192_c2/result.json) |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 8192 | 2 | 4/4 | 21.904 | 10.204 | 13.056 | 54.344 | 61.133 | 27.385 | 29.282 | 0.5977 | [原始结果](/home/chengyetai/workspace/current_exp/Qchunksize/results/autonomous_20261012/w4_prefix32k_300_chunk8192_c2_repeat2/result.json) |

## 各并发当前最佳选择

| 场景 | 并发 | 已完成chunk数 | 当前平均吞吐最高Chunk | 平均总输出tok/s |
|---|---|---|---|---|
| W1：输入8192 → 输出1024 | 2 | 4 | 8192 | 35.088 |
| W1：输入8192 → 输出1024 | 4 | 4 | 8192 | 63.598 |
| W2：输入32768 → 输出1024 | 1 | 4 | 8192 | 15.863 |
| W4：总输入32768 → 输出300，前缀缓存60% | 1 | 4 | 8192 | 14.663 |
| W4：总输入32768 → 输出300，前缀缓存60% | 2 | 4 | 8192 | 21.697 |

最佳仅在已完成的网格内成立。接近的候选需复测；不同输入和并发不能混算速度倍率。
E2E分解保存服务器前向前、prefill墙钟跨度、decode至结束跨度及客户端残差；这些跨度包含调度间隙，不伪称纯算子耗时。
每组metrics前后快照、按10秒采样、全部请求payload/ID/全文/SSE均已保存，可离线继续分析。
