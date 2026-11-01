# 23节点ChunkSize自主实验实际配置

**这份配置对应低并发预实验，W1要求的并发32、64请求未跑。** 原因、基线差异与结果适用范围见[要求核对](ChunkSize要求与自主探索偏差.md)。

本文件描述2026-10-12/13日自主实验，原始启动命令以`../autonomous_20261012/logs/cyt-auto-chunk-*/config.json`及`launch.sh`为准。

| 项目 | 实际值 | 与AboutModels默认的关系 |
|---|---|---|
| 节点 | npu1-23，单机16卡 | 只有23的NPU授权 |
| 镜像 | quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3，实际版本0.5.19 | 默认镜像标签v0.5.16启动及内存限制已有实测；选择可运行版本 |
| 权重 | 公共glm-q只读挂载，workspace差异页在私有内存恢复；182文件对应公开量化源SHA256 | 公共文件未改，恢复清单SHA固定 |
| 并行 | TP16 / EP16 / DP1，DP attention及DP LM head关闭 | 默认DP2/DP attention/LM head开启；不能混称同配置 |
| MTP | 关闭 | ChunkSize单变量实验，用户明确允许 |
| DeepEP | auto | 对齐默认；decode走low-latency |
| Decode Graph | 开，CLI batches=1/2/4/8/16、max=16；TP16对齐后实际捕获bs16 | 默认8/16/24/32、max32；保存实际捕获日志和运行graph=True |
| MEM / CTX / MRR | 0.90 / 131072 / 16 | 默认0.90 / 180000 / 64；缩小配置以适应资源 |
| Chunk | 16384、1024、4096、8192 | 唯一网格自变量；各服务其余参数固定 |
| Max prefill tokens | 32768 | 对齐默认，固定不动 |
| KV容量 | 77824 token | 每次启动server_info核对；不沿用旧mem0.92数字 |
| Shared experts fusion | 关闭 | 默认未显式关闭；保留已有可运行设置 |
| 混合chunk | enable_mixed_chunk=False | 没有开启另一项优化；不声称这是mixed batching对照 |
| HCCL | BUFFSIZE=650、OP_EXPANSION_MODE=AIV、dispatch capacity64 | 1536 Graph通信分配失败；统一650后成功 |
| 多流与plan stream | 都为1 | 固定；最后MTP多流诊断另起服务，不能混入本表 |
| 输入评分 | SGLANG_ENABLE_FAST_INPUT_LOGPROBS=0 | 避免已复现的融合评分编译崩溃；使用镜像现有Torch分支 |

## 输入和请求

安装镜像原生RandomDataset random-ids生成固定输入；使用input_ids API直接提交，不先decode再重分词。种子、生成源码SHA、tokenizer、输入ID与SHA均保存在`datasets/native_20261012`。

| 场景 | 输入/输出token | 并发 | 正式请求数 |
|---|---|---|---|
| W1：输入8192 → 输出1024 | 8192 / 1024 | 2、4 | 4、8 |
| W2：输入32768 → 输出1024 | 32768 / 1024 | 1 | 1 |
| W4：总输入32768 → 输出300，前缀缓存60% | 总输入32768 / 300，共享前缀19661 | 1、2 | 2、4 |

所有请求temperature=0、ignore_eos=True，固定长度；这属于性能压测，不能当正常完整业务回复的质量评测。前缀拼接是明确记录的扩展，不称为原生单独提供的random-ids缓存场景。

每单元先用前C个固定输入、并发C、输出16做形状预热，再flush_cache；前缀组另预热共享前缀。预热原始流单独保存并排除计时。冷缓存实测要求<0.5%，前缀组实测要求58.5%～61.5%；设置长度60.0006%不替代cached_tokens实测。

四档chunk×五个场景/并发组合，共20基础单元。随后复测16384基线、每场景吞吐候选及相差5%以内的次候选。新结果另存repeat2，绝不覆盖初测。

本次Chunk矩阵未跑输入1024 → 输出32768完整长输出，也未跑W1并发32；不能声称完成AboutModels正式要求。其他场景或MTP的结果不能代替这些缺项。

## 指标与数据

TTFT、TPOT、E2E及p50/p95、整套服务总输出tok/s；不除以NPU。E2E分解采用同组请求均值的服务器阶段跨度，含调度间隙，不叫纯算子时间。样本数较少，p95只是该组描述统计，不用于生产尾延迟承诺。

`results/autonomous_20261012`保存正式及预热payload、输入/输出ID、全文、压缩SSE、时间戳、server_info、metrics前后快照及10秒采样。失败/缺数不补零。完整实测、复测范围、缓存条件及回退次数分别见`ChunkSize自主实验实测`、`ChunkSize负载与阶段分析`及自主实验的数据验收JSON。
