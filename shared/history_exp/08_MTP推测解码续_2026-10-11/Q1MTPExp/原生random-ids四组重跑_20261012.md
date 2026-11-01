# 原生 random-ids 四组 MTP ON 重跑

用户要求重跑完整四组，取代先前的独立随机ID输入。只开MTP，不追加OFF或100K。

## 运行与数据

- 节点：23（10.120.72.23），单机16卡；20留给ChunkSize，不操作其他节点。
- 容器：`q1-mtp-native-full-20261012-190926`；实验结束只stop，保留容器。
- 独立runner：`tmux q1nativefull`；请求和监控不依赖当前Codex会话。
- 数据目录：`/data_lib/cyt/q1mtp/native_full_20261012_190926/`。
- 工作区入口：[实时汇总](results/native_full_20261012_190926/summary.md)、[运行状态](results/native_full_20261012_190926/experiment_status.json)。

## 四组负载

| 场景 | 输入token | 输出token | 并发 | 请求数 | 缓存 |
|---|---:|---:|---:|---:|---|
| w1 8K/1K | 8192 | 1024 | 4 | 8（并发×2） | 场景开始清空 |
| w2 32K/1K | 32768 | 1024 | 1 | 1（等于并发） | 场景开始清空 |
| w3 1K/32K | 1024 | **32768，完整输出** | 1 | 1（等于并发） | 场景开始清空 |
| w4 总32K/300 | 32768 | 300 | 1 | 2（并发×2） | 共享19661，约60% |

运行次序为w1→w2→w4→w3，先完成短负载，再跑完整长输出。没有把w3截断到2048。

前三组直接调用实际镜像内的 `RandomDataset.from_args(...).load(tokenizer)`，`dataset_name=random-ids`、`random_range_ratio=1`、`tokenize_prompt=True`。使用原生函数的随机起点和连续ID构造，没有逐token独立采样。直接发送input_ids，逐请求核验实际prompt_tokens，避免默认文本路径重新分词改变长度。

原生random-ids没有60%共享前缀选项，因此第四组做明确的共享前缀扩展：同一个原生函数生成19661个公共ID，以及两条各13107个后缀ID，再拼接成32768。来源、种子、完整IDs和SHA均保存；不能称第四组是完全未经修改的官方dataset输出。前缀预热仅生成1token，其统计和时间排除在正式负载之外；以服务端cached_tokens/prompt_tokens验证58.5%–61.5%，不只按理论长度宣布命中。

## 服务配置与文档差异

沿用最近验证能跑的配置，使输入更换成为主要变化：v0.5.19-cann9.0.0-a3、实际只读glm-q路径，TP16/EP16/DP1，DP attention/LMHead关闭，DeepEP normal，MEM0.90，chunk16384，maxprefill32768，context131072，MRR8，tokenizer-workers4，共享专家fusion关闭。MTP=NEXTN（运行时归一EAGLE）、steps3/topk1/draft4，draft-unquant，temperature0、ignore_eos=True。

相对AboutModels默认的v0.5.16、DP2+DP attention、DeepEP auto、context180000、MRR64等存在差异。Graph参数仍传入，但normal路径实际禁用；记录真实状态，不报作启用。该配置已先前跑通，并非严格榜单基线。没有MTP OFF，本轮不报告加速收益。

## 保留的数据与统计

每请求保存完整payload、输入SHA、完整output_ids/全文、终态meta、finish_reason、草稿直方图/verify/correct/proposed计数、P1/P2/P3、TTFT/TPOT/E2E、缓存计数及输出循环检查。流式事件按新增ID、收到时间及meta压缩保存，可复原所有观察到的输出；服务端周期日志另有阶段AL/草稿接受率。当前镜像在请求结束前没有返回位置计数，因此不能仅凭流式事件恢复每个阶段的精确P1/P2/P3，不把批量输出事件大小冒充接受直方图。

逐位置率采用无条件概率，跨请求先合计计数再除；AL=实际输出数/verify数。TTFT只认首个新增输出ID，TPOT用首末新增输出时间，E2E至流结束；保存队列/prefill/服务端时间戳供CPU进一步拆分。没有采集无法从MTP批量输出精确恢复的逐token ITL。

每分钟保留服务metrics、server-info和NPU状态。运行状态和请求进度独立落盘；长请求运行中每15秒更新输出量及当时P1/P2/P3（服务端若尚未提供则为空）。所有终态都检查输入/输出长度、结束原因、直方图和计数是否一致；接受长度高低及精确循环作为分析结果，不以凑到2作为通过标准。

启动预检曾因客户端把聊天模板的BatchEncoding当列表而收到HTTP400。已修正为提取input_ids后重启原容器；当时尚未发送正式负载。失败证据保存在 `attempts/preflight_01_template_error/`，与正式请求分开。

## 结果及运行过程中观察

四组已全部完成；完整长输出结果及最终核验见[结果报告](原生random-ids四组重跑结果_20261012.md)。8K/1K的AL=2.526、P1/P2/P3=0.6170/0.4928/0.4172；32K/1K的AL=2.901、0.8357/0.6289/0.4363；缓存组AL=2.804、0.7243/0.5888/0.5000，实际缓存命中59.765625%。

8K组6/8条有占输出一半以上的精确周期尾段，但循环组AL=2.427，另外两条AL=2.880，不能把循环与接受率高低画等号。两条常量输出的AL为1.238、1.585，另四条循环输出为3.807–3.938；目标模型输出重复并不保证三个草稿位置都预测一致。

所有12条正式输入均核验不存在tokenizer的added token；完整输入检查在native_input_special_token_audit.json。长输出的早期约2105token仅3种ID，主要为197/532两token周期；精确周期占99.95%，日志阶段AL=4。最终数值以完成后的summary/result为准，不把运行中阶段值当完整请求均值。

w1采集器最初汇总时遇到长度3的直方图[647,164,16]，省略的第四桶表示从未接受3个草稿。已在CPU补零后重新汇总，原meta保留，无重新发请求。w1的吞吐墙钟恢复自最早请求起点到最晚请求终点，不含线程池启动及汇总开销；该时间口径与其余组有微小差异，result.json里明确记录。durable finalizer会核验四组并修正runner的该次汇总错误状态，原runner终态另行保留。
