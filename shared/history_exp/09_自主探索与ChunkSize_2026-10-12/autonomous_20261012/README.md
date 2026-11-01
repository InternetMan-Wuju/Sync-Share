# 23节点自主MTP与ChunkSize实验

名称看不懂时，先看[负载与配置名称说明](负载与配置名称说明.md)：测试场景写成“输入多少 → 输出多少”，服务配置写成中文方案名；长度单位均为token，原代号可在说明中查找。

2026-10-13 最新结果：完整W3的MTP开启测试已完成并验收，四类规定输入/输出长度的开启结果现已齐全，见[MTP四组完整长度实测汇总](MTP四组完整长度实测汇总.md)。四档ChunkSize另在补跑，见[实时进度与结果](w3_supplement_20261013/W3补跑结果.md)、[总控状态](w3_supplement_20261013/status.json)。本轮MTP关闭对照已由用户取消。以下“开启覆盖3/4”等结论保留为本次补跑之前的阶段记录。

本目录收集2026-10-12/13用户休息期间的自主定位与实验。**既定探索阶段已结束，但MTP四组要求未完成：规定长度ON覆盖3/4，OFF性能基线0/4。** 补充检索和6次OFF正确性诊断已完成，最后节点核对时间2026-10-13 12:44；旧总控的completed不代表四组要求已经完成。

**先看[仅自主探索的四组覆盖核对](MTP自主探索四组覆盖.md)**：输入8192 → 输出1024、输入32768 → 输出1024、输入32768 → 输出300缓存60%跑了ON，输入1024 → 输出32768没有跑，不能以输入1024 → 输出1024短诊断代替。三组标准长度ON来自DP1/normal/无Graph配置；后续DP2短诊断不代表已经补齐四组。

[最新补充检索与诊断](review_20261013/MTP一般情况检索与补充诊断.md)：公开GLM-5.2校准表报告3草稿AL2.99，正常聊天AL2.4～3.3量级相容。新补测的原始随机输入在MTP关闭后仍生成完全相同的1024-token循环；它不以MTP为必要条件。共5项原输入ON/OFF对照一致，另有相关的固定循环前缀128-token续写一致；不是6条独立提示。全模型精度和既有DP预测差异仍未解决。

[探索阶段总结](自主实验最终总结.md)：典型AL接近4的请求产生两token循环；公开源恢复权重后仍复现，关闭多流也未改变完整输出及接受计数。完整ShareGPT回复实测AL2.398～3.255，并非都要接近2。原基线16次前缀重放全部保持一致，DP间预测差异仍未定位到kernel，不能宣称通过KVV精度验收。

ChunkSize仅完成小并发预实验的20个基础单元及14个复测；**W1正式并发32、64请求未跑**。输入、计数、缓存和时间戳验收只验证这些单元自身，不能证明正式任务完成。8192在已测小并发组合均值最高，4K/8K候选结论限于这些条件。具体见[ChunkSize要求与自主探索偏差](../Qchunksize/ChunkSize要求与自主探索偏差.md)。

MTP合表86条（41条历史+45条有效新增）；自主验收52组包含sanity，因此两个数口径不同。[节点释放核对](节点释放核对.json)证实本轮自有容器均已停止、8个NPU设备均无运行进程。全部容器与原始数据保留。

## 当前状态入口

- `overnight_status.json`：MTP主要对照、ChunkSize基础网格的总控状态。
- `chunk_repeat_status.json`：基础网格后的基线/候选复测状态（启动后才有）。
- `final_mtp_isolation_status.json`：最后的NPU多流单变量和严格聊天边界配对诊断（启动后才有）。
- `finish_autonomous_status.json`：基础网格完成后的顺序衔接状态；先候选复测，再最后MTP核对，再离线报告。
- `python3 -B scripts/status_brief.py`：只读、简短状态；各结果目录的status.json包含逐请求进度。
- `logs/overnight_runner.log`：总控时间线。服务启动命令、配置、日志和权重加载审计在`logs/cyt-auto-*/`。

## MTP证据

- [自主探索全量性能表](MTP自主探索全量性能表.md)：46组有效结果（含完整W3补跑），同时列接受率、吞吐、延迟和实际输出；[CSV表格](MTP自主探索全量性能表.csv)。
- [一般情况检索与MTP OFF补充诊断](review_20261013/MTP一般情况检索与补充诊断.md)
- [实测表与同输入权重对照](MTP自主探索实测.md)
- [历史与新增实验合表](../Q1MTPExp/MTP全部实验汇总_含自主探索.md)
- [公开源权重校验与恢复](权重恢复与公开源校验.md)
- [轨迹分组、主模型prefill和续写核对](MTP轨迹分组与主模型核对.md)
- [历史完整32K长输出接近满接受的直接证据](MTP历史长输出的直接证据.md)
- [相同前缀固定DP对照](固定DP前缀核对.md)
- [指标和参考口径](MTP指标与参考口径澄清.md)
- [网络资料与适用范围](网络检索依据与适用范围.md)
- [负ID诊断的流时序排查](诊断断言的流时序排查.md)
- [输入评分编译失败与标准Torch回退](评分接口编译错误与回退.md)
- `随机输入循环的具体样例.json`：含原始响应路径、输出样例、位置率与循环形状。
- `results/`：原始请求payload、输出IDs/正文、压缩SSE、计数、指标与评分原始响应；诊断与正式workload明确分开。

## ChunkSize证据

- [固定并发方案](../Qchunksize/Qchunksize.md)
- [实测完整表](../Qchunksize/ChunkSize自主实验实测.md)
- [配置、负载与阶段证据](../Qchunksize/ChunkSize负载与阶段分析.md)
- [本轮实际配置与默认差异](../Qchunksize/RUN_CONFIG_20261013.md)
- [实际批次形状证据](../Qchunksize/ChunkSize批次形状证据.md)
- [16K基线波动与动态组批说明](Chunk基线波动的批次说明.md)
- [时间戳定义](../Qchunksize/时间戳口径与E2E分解.md)
- `../Qchunksize/datasets/native_20261012`：安装镜像的原生RandomDataset导出，种子、源码SHA、tokenizer与输入SHA全部保存。
- `../Qchunksize/results/autonomous_20261012`：基础20单元与追加repeat2的原始数据。
- `ChunkGraph对齐故障证据.json`：首次OFF-MTP/TP16 Graph小batch清单为空的源码证据；修正前无性能单元。

## CPU分析与验收

```bash
python3 -B scripts/summarize_progress.py
python3 -B scripts/analyze_mtp_completed.py
python3 -B scripts/combine_mtp_history.py
python3 -B scripts/analyze_dp_prefixes.py
python3 -B scripts/analyze_chunk_completed.py
python3 -B scripts/validate_completed.py
python3 -B scripts/plot_completed.py
python3 -B scripts/analyze_chunk_batch_logs.py
python3 -B scripts/publish_final_report.py
```

这些命令不发送推理请求。验收结果在`数据验收与时间戳核对.json`，图在`figures/`。统计报告只纳入已完成记录，失败、不完整和缺少接受计数的情况保持明确标注。

容器均保留，只停止。公共模型未改，权重通过workspace只读差异页在私有内存重建；其他人的脚本仅只读复制，没有执行。当前只使用23的NPU。
