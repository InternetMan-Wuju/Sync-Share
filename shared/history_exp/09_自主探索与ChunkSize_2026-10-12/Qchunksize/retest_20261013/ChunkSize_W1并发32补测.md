# ChunkSize W1 并发32补测

最新范围：只测8K/1K。用户已明确前缀32K/300/60%组测过，不再重复；W2/W3 Chunk也不跑。W3 MTP ON已完整补齐。

客户端并发32、正式64请求，TTFT包含排队。固定输入SHA、温度0/ignore_eos、先形状预热再清缓存。四个Chunk档位保持同一输入和其余配置。

W1尝试榜单NoEP基线：TP16/DP2/EP1、DP attention/LM head开、DeepEP关、graph开、mem0.90/context180000/MRR64。v0.5.19和修复权重是已知可运行基础，与文档默认镜像标签不同。DP2会把CLI Chunk除以2，表里核对并使用实际每DP Chunk；四档对应CLI2048/8192/16384/32768。

| 场景 | 实际Chunk | 状态 | 请求通过/要求 | 总输出 tok/s | TTFT均值 s | TPOT均值 ms | E2E均值 s | 实测缓存 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| w1_8k_1k | 8192 | 失败：RuntimeError: command rc=-15; see /home/chengyetai/workspace/current_exp/Qchunksize/retest_20261013/w1_8k_1k_chunk8192_c32_collect.log | —/64 | — | — | — | — | — |
| w1_8k_1k | 1024 | 待运行 | —/64 | — | — | — | — | — |
| w1_8k_1k | 4096 | 待运行 | —/64 | — | — | — | — | — |
| w1_8k_1k | 16384 | 待运行 | —/64 | — | — | — | — | — |

逐请求、p50/p95、E2E阶段分解、metrics和原始压缩SSE保存于各结果目录；sanity与预热分开保存且不计正式时间。排队和阶段跨度来自服务器时间戳，不将它们解释成纯算子时间。
