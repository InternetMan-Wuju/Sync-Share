# 16384基线复测波动：实际调度批次不同

输入8192 → 输出1024、并发2，第一轮总吞吐25.072494 tok/s，第二轮29.584352 tok/s。两轮使用相同输入、固定并发、预热和服务配置；均冷cache，无回退。不是一项新参数优化。

第一轮TP0日志显示两批各为`new-seq=2,new-token=16384`。第二轮第一批请求被调度成两次`new-seq=1,new-token=8192`，随后两条仍合成`new-seq=2,new-token=16384`。第一轮TTFT四条约27.93/27.94/29.64/29.65秒；第二轮约5.17/5.17/25.54/25.54秒。

这说明固定客户端并发并不保证每次scheduler组成完全相同的prefill形状。按会议要求，本实验没有人工延迟或锁step控制请求发送时机，所以动态组批属于实际服务行为，不能事后只选择较慢基线来放大收益。

最终报告同时给两轮值、均值和范围，并用实际日志说明形状差异。不将不同组批造成的prefill墙钟变化直接称为某个kernel纯性能收益；没有完整operator profile也不猜内部根因。

证据：Qchunksize/results/autonomous_20261012/w1_8k_1k_chunk16384_c2/result.json与同名_repeat2目录；logs/cyt-auto-chunk-auto-hccl650-16384/server.log及logs/cyt-auto-chunk-repeat-16384/server.log。批次分析脚本按TP0正式请求时间窗、首尾同秒候选另列，见Qchunksize/ChunkSize批次形状证据。
