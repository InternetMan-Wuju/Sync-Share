# TODO

参与人员

|序号|姓名|单位||
|---|---|---|---|
|1|赵龙|类脑||
|2|游兴业|中科大||
|3|程业泰|中科大||
|4|金鑫|中科大||
|5|张孟涵|中科大||
|6|贾文龙|中科大||
|7|高浚潇|中科大||
|8|吴智承|中科大||
|9|孙申奥|类脑||
|10|唐赢坤|类脑||
|11|谢琦|类脑||
|12|吴亮|类脑|预计8\.24日后投入|

# 优化目标

[GLM5\.2 推理性能榜单](https://ncngj7vcrz76.feishu.cn/sheets/JtYhsNkyhh1q6ztvmo5czrQWnUt?from=from_copylink&sheet=4xGqe0)

可参考的文档：[GLM 5\.2 NVFP4 性能对标](https://ncngj7vcrz76.feishu.cn/docx/S94SdYslNoBapxxVE1NcwKa8nmu?from=from_copylink)，[GLM 5\.2参数调优](https://ncngj7vcrz76.feishu.cn/docx/XKCBd1JToohEbFx5WXBc4KR6nHg?from=from_copylink)

1. **Model：**GLM 5\.2\-NVFP4

2. **Workload：**固定前缀随机数据集

    1. 用例 1：前缀 100K / 输出 1K / 缓存命中 90%（输入是 90K 相同的，后 10K 为随机）

    2. 用例 2：前缀 32K / 输出 300 / 缓存命中 60%

3. **固定配置：**MTP=3

    1. Per\-position acceptance rate: 0\.498, 0\.345, 0\.267

    2. Per\-position acceptance rate: 0\.148, 0\.093, 0\.056

4. **精度要求：**KVV 误差 ≤ 3%；AL ≈ 2\.0（±5%），线上实际AL高点2\.11，低点接近0；

5. **延迟要求：**P50 TPOT\<33ms；P50 TTFT\<15s

6. **优化目标：**在上述两种场景下**最大化吞吐**

# 2026\.9\.28

## B300

1. 性能调优@赵龙

    1. 制定性能榜单表

    2. 制定四机调优的计划路线

    3. 优化 aisimulate

2. 继续实现 DCP 中对Indexer cache 的切分@金鑫

3. HiCache @高浚潇

    1. 弄清楚 SGLang 有没有考虑重新计算和传输的开销对比，如果没有，这个比例如何

    2. 探索增加 HiCache 大小反而降低性能的原因

    3. 在 G1 的情况下继续减少 KVCache，直到 Host Memory 起作用

    4. 增加 32k 60% prefix 的情况，对齐用例

4. HiSparse @张孟涵

    1. 与共置hicache测试workload对齐后继续hicache\+hisparse\+开cp测试

    2. 调研PD分离下，prefill开启cp后与decode之间kvcache传输策略和pcp部分细节

5. DP Attention @贾文龙

    1. 统计 DP Attention 额外复制的模型权重大小，以及每个 DP rank 的 KV Cache 大小，算一下比例

    2. 分析随着 concurrency 增加，KV Cache 占用怎么增长，和固定的模型权重开销做对比

6. 检查新租的 4 台 B300 机器的计算与通信是否正常@贾文龙

7. SGLang 代码理解，完善文档 **@上次分到相关任务的人**

## 910C

Workload（请求数量设置为并发数的两倍，如并发8则设置请求16条）

1. 8k/1k

2. 32k /1k长文本

3. 1k/32k长输出

4. 前缀 32K / 输出 300 / 缓存命中 60%

**默认配置与baseline**：见[性能榜单](https://ncngj7vcrz76.feishu.cn/docx/SswIdEzINoQQEBxV5gtcoNJcnke)

|**任务****（workload使用上面四种）**|**测什么****（需分析为什么该技术能带来性能提升或不能）**|
|---|---|
|**制定性能榜单**@游兴业|**pernpu性能做基准**|
|DCP、PCP的PR@谢琦|调研下DCP相关的PR看看有没有在Ascend上的实现，能不能复现|
|**DPA、DP对性能的影响 @唐赢坤**<br>|不同dpsize，DP\_LM\_HEAD等对性能影响是什么，为什么|
|**PD 分离及 P:D 配比**@sunshenao|共置、不同pd配比的**pernpu性能**如何|
|**Chunk Size的选取对性能的影响**@程业泰|选定不同大小chunksize然后在对应workload测|
|**EP对性能的影响**@吴智承|在默认配置的基础上调整EP的参数测对应workload|
|**MTP开启与不开启对性能的影响**@程业泰<br>|接受长度、对应吞吐、ttft、tpot（接受率为啥那么高）<br>|

1. 制定性能榜单、aisimulate 910c版本@游兴业

2. EPLB @程业泰@吴智承

    1. 重做ChunkSize\-MTP

    2. EP共置4机器\-PD热力

    3. 理清楚EP问题：EPLB 会有负优化

3. 单机器部署调优**@唐赢坤**

    1. 基于workload清单继续单机部署的优化**@唐赢坤**

    2. 调研cuda graph相关的两个超参数意义与实际影响，为什么 cuda graph 这么大**@唐赢坤**@游兴业

4. 整理一下 910C 上遇到的 bug 以及参数问题，汇总成文档 **@all**

5. 补齐不同PD ratio的性能测试以及寻找优化配置\(与baseline比\)@sunshenao

# 2026\.9\.21

## B300

1. 在单机共置八卡上针对四种 workload 性能调优：@赵龙@贾文龙

    1. 8k/1k，对标自部署的 default 性能

    2. Agentx，对标公开性能

    3. 前缀 32K / 输出 300 / 缓存命中 60% 

    4. 前缀 100K / 输出 1K / 缓存命中 90%（已找到）

2. DCP：看一下当前性能低是什么原因导致的，是否有优化空间@金鑫

3. HiSparse \& HiCache @张孟涵@高浚潇

    1. 对齐并修改实验配置（但是不脱离用例 1 和用例 2 的要求），在 KV Cache HBM容量受限场景下验证方法的性能

    2. 记录开启优化后的性能情况，P实例和D实例要分开统计，包括但不限于

        1. 吞吐，ttft，tpot

        2. Kv cache 利用率，kv cache 命中率，换入换出次数，显存使用的分布

        3. 实际的 running batch size

        4. 内存使用方面，可以理论计算\+实测的形式，是否符合预期

    3. 看一下 Prefill CP 是如何进行数据切分的，是否切分prefill产生的KVCache，数据流是怎样 @张孟涵

    4. （HiCache）@高浚潇

        1. 确定一下并行配置实际的布局，如 tp8，dp2，ep2

        2. 看一下同TP组卸载的数据是否相同，MLA架构，TP切分后KVCache是重复的，那TP域内每个rank上都会保留相同的KVCache，卸载的KVCache也是一样的？

4. SGLang 理解

PD分离场景下HBM的分配情况，主要是KVCache的，P和D实例上的都要看，主要是GLM和K3模型，attention层是MLA结构，可能有linear attention或者sparse attention，具体的分为以下几个case：

1. P和D实例都不开offloading的技术，HBM中KVCache是不是都是seqlen \* layers这样分配的？ P实例的前缀命中，意味着在HBM中直接命中？@赵龙

2. P实例如果开了CP，会不会切分MLA的KVCache，如果要切分，CP的执行流程是什么？如果是K3这种带linear attention的，CP不影响linear attention单元？如果是GLM5\.2这种带DSA的，prefill阶段是如何使用DSA的？CP是否支持？@赵龙

3. D实例开DCP，对于full attention来讲如何切分，linear attention如何切分，DSA layer又如何？@游兴业

4. P实例开启hicache，全量KVCache可以卸载到host memory中，那此时HBM中给KVCache的内存是怎么分配的？是buffer size \* layers吗？但是最终前缀的KVCache都要被加载到HBM中进行计算，再加上后缀产生的KVCache，那HBM中KVCache会占用seqlen \* layers这么多？ 如果是这样，能否只分配seqlen \* 2的空间，然后当前层的计算与下一层的KVCache prefetch overlap？@高浚潇

5. D实例，如果不考虑hisparse，比如K3模型，那HBM中的KVCache空间怎么分配的，是seqlen \* layers？ 是否也可以只分配seqlen \* 2，来构造overlap？ 因为K3是有三层linear attention \+ 一层full attention，那一层full attention的KVCache预取可以与三层linear attention overlap @赵龙

6. D实例，如果考虑hisparse，HBM中的KVCache分配是否为6144 \* layers？ 是否有数据的预取？ 这种场景下主要就是给GLM、DeepSeek这种模型使用了。 D实例的hisparse与DCP如何结合的？@张孟涵

7. P实例的hicache与CP如何结合的？@高浚潇

## 910C

1. 整体目标：已经选定几种workload，那就按照default的测试，或者单机的折算，来确定per NPU的吞吐，作为baseline，之后确定目标，比该baseline要提升\*\*%。要达到这个目标，应该要完成哪几件事情，测试什么优化点，把任务分解出来，交由不同的同学配合测试。@游兴业

    |    **任务**|    **测什么**|
    |---|---|
    |    **跑出 baseline，锁定目标（单机和pd分离，折算pernpu）**@游兴业|    各 workload 的 default 性能|
    |    AIconfigurator适配@游兴业|    适配自动搜索调优|
    |    **寻找单机最优配置 @唐赢坤**<br>    （8k/1k，前缀 32K / 输出 300 / 缓存命中 60%）|    从 8K/1K 入手，扫描合法并行布局、chunk size、batch、内存配置；单独对比投机开/关|
    |    **PD 分离及 P:D 配比**@sunshenao<br>    （8k/1k，前缀 32K / 输出 300 / 缓存命中 60%）|    共置、1P1D 和资源允许的其他配比（3p1d,2p2d）；统计 P/D 负载、KV 传输和排队、不同配比下能多长上下文|
    |    **HiCache 验证（单机）**<br>    @谢琦|    再在 32K、100K 场景测试开关组合，观察 KV pool、命中、换入换出和 batch，重点关注显存中的 KV Cache Pool 空间不足，开启HiCache前后的 hit rate|
    |    **TP/DP/EP 组合（共置、pd分离）**@吴智承**（8k/1k）**|    TP、EP、DP、TP\+EP、TP\+DP、TP\+DP\+EP|
    |    **Chunk Size\(单机\)（**8k/1k，前缀 32K / 输出 300 / 缓存命中 60%）@程业泰|    对 Prefill/Decode 干扰和长上下文影响很大<br>    |
    |    **MTP/推测解码**（1k/32k 、8k/1k、前缀 32K / 输出 300 / 缓存命中 60%）@程业泰<br>    |    接受长度、对应吞吐、ttft、tpot<br>    |

2. 分析现有workload下的dp=1下的kv命中率低于预期的原因,换workload重新测试不同dp下的KV命中率:

    1. 前缀 32K / 输出 300 / 缓存命中 60%

    2. 前缀 100K / 输出 1K / 缓存命中 90%@sunshenao

3. 尝试实现AIconfigurator自动调优的NPU版本@游兴业

4. 调研看下910c这边DCP（有没有PR支持）、HiCache能用吗，能用的话对显存的影响，当前的极限序列长度是多少？@谢琦

    1. 测下开关DCP/Hicache情况下性能:前缀 32K、输出 300、缓存命中率 60%。重点关注显存中的 KV Cache Pool 空间不足，开启HiCache前后的 hit rate

5. 确认EPLB问题使用的workload，继续观测是否有EPLB的优化空间，主要还是大EP@吴智承

6. EPLB开启前后的效果，要使用真实的workload @程业泰

7. 固定workload探索DP\-Attentio参数对实验性能影响 @唐赢坤

    1. 理解DPA机制，做对照实验组，利用torch profile绘制数据详情图

    2. 通过调试部署参数与形式提升DPA并发，与baeline对比，同步调研cuda graph bs的机制细节

# 2026\.9\.14

## B300

1. 在GLM 5\.2\-NVFP4上找到最优的**并行策略**以及 **chunk size **作为 **baseline**，包含以下两种情况@赵龙

    1. 八卡共置

    2. 单机 PD，P 四卡，D 四卡

2. 对于每种**优化，**弄明白**优化的相关参数含义；记录**开启后的**相关指标**；对比**前后**的**性能变化**

    1. 共置以及 PD 分离下的 HiCache @高浚潇

        1. 参照的 baseline 由于数据集特性（加上显存压力不足）未见提升（HiCache 未使用到）

        TODO：计划修改数据集生成参数，降低 KVCache 大小，确定 HiCache 有取回后再行测试 \+ 弄清楚同一个 TP 组的 GPU offload 的数据到底是否相同

    2. 单机 PD 分离下的 HiSparse @张孟涵

        1. 介绍了hisparse四个配置参数

        2. 介绍了单机PD分离\(4卡P4卡D\)下吞吐和延迟情况

        3. 提及了hybrid hisparse

    TODO:在prefill阶段,开启cp/hicache等继续测试hisparse性能,观测并记录包括实际缓存命中率,kv cache使用率和内存相关等指标。

    3. DCP的性能收益来源 @金鑫

        1. 单卡读取 KV Cache 减少后，本身能带来多少性能提升

        2. KV Cache 去重后 HBM 被释放，对于 running batch 的影响是什么；确认实际 running batch 是否真的达到设定并发

        3. 尝试修复sglang中max\_req\_input\_len引发的bug

    4. GLM上不同草稿模型的性能 @贾文龙

    5. Ai Configurator 是否还能进一步优化 @赵龙

3. 其他优化的脚本搭建（暂时跑通就行，低优先级，也可不安排）

    1. Mooncake

    2. KV Cache Router

## 910C

1. **Model：**GLM 5\.2\-W8A8

2. **Workload：固定random序列长度\+**固定前缀随机数据集

    1. 3\.5k/1\.5k解码压力

    2. 4k/1k，concurrency 1024高并发

    3. 8k/1k长上下文

    4. 32k /1k长文本

    5. 1k/32k长输出

    6. 前缀 32K / 输出 300 / 缓存命中 60%

    7. AgentX workload，真实trace

3. **固定配置：**

    1. chunksize=16384

    2. cudagraph

        1. \-\-cuda\-graph\-backend\-decode full \\

        2. \-\-cuda\-graph\-max\-bs\-decode 32 \\

        3. \-\-cuda\-graph\-bs\-decode 8 16 24 32 \\

    3. SGLANG\_ENABLE\_OVERLAP\_PLAN\_STREAM=1

        

---

1. ~~hicache、hisparse在910c上能否使用？（Hisparse不能使用）3p1d\+hisparse测试超长上下文吞吐~~@游兴业

2. 设置 `--dp-size 1`看下是否影响缓存命中率以及找路由到一个dp worker的实验@sunshenao

TODO:分析现有workload下的dp=1下的kv命中率低于预期的原因,换workload重新测试不同dp下的KV命中率

a\.前缀 32K / 输出 300 / 缓存命中 60%  b\.前缀 100K / 输出 1K / 缓存命中 90%

3. 固定workload探索DP\-Attentio参数对实验性能影响@唐赢坤

a\. 理解DPA机制，做对照实验组，利用torch profile绘制数据详情图

b\. 通过调试部署参数与形式提升DPA并发，与baeline对比，同步调研cuda graph bs的细节机制

4. 排查EP64存在的kernel错误并修复；测试TP16DP4EP64验证性能 @吴智承

5. 协同解决Kernel问题，统计PD专家分开热力，统计EP\-Rank热度，调查专家分配方案/re\-rank/replicatio@程业泰

6. 单机情况找一个最优配置测在不同workload基准性能怎么样？可以用来和pd分离后的性能对比  @游兴业

    1. 其他参数都一样，开Speculative Decoding与不开的性能区别

    2. 其他参数都一样，\+dp attn后的性能区别

7. 1p1d下的基准性能（和6进行对比，测试内容和6一样）@游兴业

8. AIconfigurator自动搜索@游兴业





# 2026\.9\.7

## 一、B300

1. k3和glm共用的，我们找到两个模型各自的最优配置，比如chunk size、调度策略等，AI configurator；@赵龙

2. k3和glm共用的，得有prefix卸载，我们需要hicache；@高浚潇

3. k3和glm共用的，4机时用mooncake；

4. KVCache感知的调度router，NVIDIA dynamo；

5. glm特殊的，因为是sparse attention，那可以直接先把hisparse搬过来；@张孟涵

6. 以及投机推理算法相关的改进，这个我找人研究一下，包括MTP和Dspark。

7. Kimi K3 剩余相关测试和 Debug@高浚潇

    - 开统一内存后的 Request 并行问题

    - 完成长请求和混合请求的剩余 Workload，以及扩充了 Ratio 后的测试

### GLM模型

- 模型：可自选近期模型（建议GLM）

- 数据集：固定前缀随机数据集

\- 用例 1：前缀 100K / 输出 1K / 缓存命中 90%

\- 用例 2：前缀 32K / 输出 300 / 缓存命中 60%

- 固定配置：MTP=3

- 精度要求：KVV 误差 ≤ 3%；AL ≈ 2\.0（±5%）；

- 延迟要求：P50 TPOT\<33ms；P50 TTFT\<15s

- 优化目标：在上述两种场景下最大化吞吐

上述方案，线上实际AL高点2\.11，低点接近0

Per\-position acceptance rate: 0\.498, 0\.345, 0\.267

Per\-position acceptance rate: 0\.148, 0\.093, 0\.056

如技术团队建议做MTP不等于3的方案；

可在上述基准测试完成后，自行调整MTP，按0\.498, 0\.345, 0\.267即可

### Kimi K3模型

- 缓存能力：支持 KV\-Cache 会话复用，连续会话及相同前缀复用场景下缓存命中率 ≥50%

- 时延能力：常规输入（输入 1K‑4K token、输出 200‑800token）、稳态并发场景下，TTFT P50≤5s、P90≤10s、P99≤15s；TPOT P50≤100ms/token、P90≤150ms/token、P99≤250ms/token；超长上下文、冷启动及资源扩缩容场景单独统计，不纳入常规时延要求

- 模型能力：对齐官方能力，支持图文视频等官方多模态能力、思考强度/模式配置及标准 Tool Calling，工具调用 JSON 解析成功率 ≥95%。



1. 内存相关 

    1. 动态内存管理 @高浚潇

        1. 继续完成不同 workload 在不同  ratio 下的性能

        2. 探究 request 排队的原因

        3. 探究 prefix cache 命中率下降的原因

        4. 1M超长序列的内存使用情况

    2. Block Attention Residuals @张孟涵

        1. 探究实验二中T=31k→T=32K时,AttnRes Bank占比下降的原因以及为什么发生在T=32K左右的原因。

        2. 研究Block Attention Residuals在Decode阶段的内存问题。

2. 性能优化相关

    1. DCP @金鑫

        1. 测试DCP在不同并发下对推理效果的影响

        2. 测试DCP在混合上下文场景下的效果

        3. DCP在长序列场景下吞吐下降，原因是什么，是否能够继续扩展到1M序列长度，再观察DCP的优势是什么

    2. 投机推理：补充长输入上下文的实验 @贾文龙

    3. chunksize，混合workload场景下chunksize的设置，是否会有动态性 @游兴业

3. 性能实验 @赵龙

    1. 性能调研与价格计算

        1. 再找一下网上公开的 Kimi K3 以及 GLM 5\.2 的 benchmark 数据

        2. 补充并整理当前部署性能与公开 benchmark 数据的对比

            1. 完善 GLM 5\.2 与 Kimi K3 的 benchmark 实验数据 

        3. 制定一下后续测试的配置与标准，包括

            1. 基于市场定价测算不同 workload 下的吞吐与成本，如：

                1. 短 / 中 / 长请求； Agent 场景

                2. Prefix Cache 命中 / 不命中

                3. 不同输入输出长度分布

                4. 实验室中转站的trace数据是否能够用上

            2. 不同 workload 下的 SLO 标准

    2. GLM 5\.2 性能实验 

        1. 检查 GLM 5\.2 高并发缓存命中异常

        2. 确认 KV Cache/权重实际显存

        3. 完成整体性能最优之后，做一下控制变量实验

## 二、910C

1. GLM性能实验 @游兴业

    1. DSA **@唐赢坤**

        1. 围绕deepep\-mode auto和low\_lantency搜索aicore timeout相关问题

        2. 规范数据集后调整\-\-chunked\-prefill\-size和\-\-max\-prefill\-tokens参数再次测试

    2. PD 分离 **@孙申奥**

        1. 测试长输入\(2k\+\)下的模型性能,并补充其他的测试指标

        2. 确认测试数据集标准以及bench和参数的设置,调研业界模型的性能

    3. 确定workload后的大EP测试 @吴智承

    4. 确定workload后的EP load balancez是否存在问题，要开启EPLB @程业泰

2. 固定序列测试（详情见：[固定测试](https://ncngj7vcrz76.feishu.cn/wiki/UZdnwBmYZid19lkp4EVcuGENnuc?from=from_copylink)）开cuda graph

    1. 测试1:使用 [EvalScope](https://link.zhihu.com/?target=https%3A//github.com/modelscope/evalscope) 构造 5 轮对话：第一轮输入 16K Token，每轮最多输出 1K Token，后续每轮新增 2K Token。总请求数和并发数均为 16。**@唐赢坤**

    2. 测试2:输出、输出、并发不同下折算每卡输出（具体见测试2[固定测试](https://ncngj7vcrz76.feishu.cn/wiki/UZdnwBmYZid19lkp4EVcuGENnuc)）**@孙申奥**

    3. 测试3:\-\-dataset\-name random \\ \-\-random\-input\-len 21403 \\ \-\-random\-output\-len 1024 \\ \-\-random\-range\-ratio 0 \\ \-\-num\-prompts 1 \\ \-\-max\-concurrency 1 \\ 输出@吴智承

    4. 测试4:UCM的测试https://ucm\.readthedocs\.io/en/latest/user\-guide/best\-practices/GLM\-5\.1\-A3\_4Node\_PD\_Disaggregation\.html@程业泰

    5. agentx数据集下不同并发、chunksize @游兴业

    **agentx数据集**为：https://huggingface\.co/datasets/semianalysisai/cc\-traces\-weka\-062126\-256k（全测太长了，可以取部分）**机器上目录：**/mnt/raid/user/data/dataset/cc\-traces\-weka\-062126\-256k

# 2026\.8\.31

## 一、B300

1. 内存相关 

    1. memory 动态调整：@赵龙@高浚潇

        1. 测试不同 workload 场景下的表现，包括短请求、长请求、长短请求混合以及超长请求

        2. 尝试实现 KV Cache 和 Mamba Slot 的动态分配机制

        3. 分析并发数在 72～80 区间时，为什么 ratio=2 的 output 吞吐会高于 ratio=2\.5，并补充更多实验进行验证

    2. Block Attention Residuals： @张孟涵

        1. 测试block attnres在不同workload下的峰值内存，记录运行时的火焰图

2. 性能优化相关

    1. Chunk size：在memory ratio 的甜点区间内，测试对应数据集下chunk大小对性能的影响 @游兴业

    2. 投机推理：测试Qwen3\.8\-27B搭配dflash2时的性能，以及投机推理各个算法适用的场景，是否能够做动态路由？@贾文龙

    3. DCP：@金鑫

        1. 测试极长序列情况下DCP对推理的影响

        2. 动态DCP调研（LoongServe）

3. 其他测试

    1. GLM\-5\.2并发性能实验 **@邓若辰**

    2. 测试CuTe DSL的作用（这个暂时不用了）

## 二、910C

1. GLM 性能实验 @游兴业

    1. PD分离部署及ratio调整性能测试 **@孙申奥**

    2. 调整EP，观察大EP下的性能 @吴智承

    3. 观察在大 EP 下的 expert 不均衡现象，以及开启EPLB后的性能 @程业泰

    4. DSA相关性能测试 **@唐赢坤**

2. 模型输出乱码问题调研与解决方案

官方镜像问题

3. 环境问题：物理链路网络反馈与跟进

# 2026\.8\.24

1. 内存相关 @赵龙@张孟涵@高浚潇

    1. KDA slot运行时的变化与使用情况（Conv和SSM）

    2. 改善显存分配对推理性能的提升有多大，看一下vLLM 的动态显存分配

    3. Attention Residuals 会不会有内存问题

2. SGLang 相关

    1. 理解一下Dynamic chunk @游兴业

    2. Kimi K3 的 Load Balance  @程业泰

    3. 开启DCP 后对性能的影响（吞吐有多少，通信影响多大）；输出token的KV Cache存放方式 @金鑫

3. 投机推理 @贾文龙

    1. DFlash2 Qwen3\-4b 额外的可学习参数是怎么学习的

    2. 测试对比原生 MTP / DSpark / DFlash2 / DFlash2的性能

4. 昇腾910C平台上环境的搭建和测试  @唐赢坤 @孙申奥

    1. 基础测试，卡与卡之间的通信延迟、带宽等

    2. PD分离部署

    3. 改变PD ratio，测试端到端吞吐

    4. 大EP的测试

5. Triton 能否感知与使用 TEME @张孟涵



# 2026\.8\.17

1. [Kimi DSPARK 调研](https://ncngj7vcrz76.feishu.cn/wiki/Vo2gwg13eimsNdkSFzScObDvnOg)**：**@贾文龙

    1. 草稿模型是多少层

    2. 使用什么并行方式

    3. 推理时的数据流

    4. 业界是怎样的部署方式



2. [**SGLang 逻辑理解**](https://ncngj7vcrz76.feishu.cn/wiki/EcPDwO8N9iUEpdkwcSRc1VA0nSc?from=from_copylink)**：**@金鑫

    1. DCP 的运行情况，推理过程中序列越来越长， KV Cache 是如何存放

    2. 多模态的部分是怎么切分的

    3. KDA Slot 的计算逻辑 



3. **实验与分析：**@赵龙@游兴业@程业泰

    [推理性能与分析](https://ncngj7vcrz76.feishu.cn/docx/MmyAd5iHSoaW9OxoaaRc0MsqnVb?from=from_copylink)

    1. 补充实验，记录推理过程中的各数据变化情况，如 KV Cache、KDA Status、Activation 等；画出性能的 Breakdown，如通信计算的占比、各层执行的时间。总之就是弄明白时间和显存都用在了哪里

        1. 不同Chunk size下的性能，能否缓解队头阻塞 [不同Chunk size下的性能测试](https://ncngj7vcrz76.feishu.cn/docx/FzF4dejgOorSE8xU0Goc5YZgnJb)

        2. 关闭 prefix cache，固定并发度，做一下不同上下文长度的实验，可以测单轮的，看看性能趋势

        3. Check 不同到达率的实验数据，画一下不同时间下的吞吐性能图

        4. Attention Residuals 会不会有内存问题，峰值内存会发生在哪里。可以使用 torch 的 mem 记录工具

        5. 可以开启 EP，看看 Kimi K3 的 Load Balance 的效果如何 [EP测试](https://ncngj7vcrz76.feishu.cn/wiki/Fum8wmQvTiSKkLkfNigcXmSynNc)

    2. 分析拐点的原因

        1. 在SWE\-agent\-trajectories里，并发度实验中的 8 和 64 产生性能拐点的原因

        2. Fused kernel 为什么会限制在并发度 64，kernel 的时间占比



4. [**其他调研**](https://ncngj7vcrz76.feishu.cn/wiki/OYWiwkOmYikrhUkVwf7cSpLtnzf?from=from_copylink)** **@张孟涵

    1. SLO在业界里是怎么制定的，标准是什么

    2. TMEM 是否编程可控

# 2026\.8\.7

1. **调研与学习 Kimi K3 模型结构，并整理出 md 文件：**

    1. Gated MLA，KDA，模型总体架构，@金鑫

        1. [模型总体架构、KDA和Gated MLA](https://ncngj7vcrz76.feishu.cn/wiki/DJGZwLwYRi6xdTkl5Gxcdl6mnYf?from=from_copylink)

    2. LatentMoE，Attention Residuals，@张孟涵

        1. [Attention Residuals](https://ncngj7vcrz76.feishu.cn/docx/Vv86ds0gDoVjUoxHKuvcJjMonde?from=from_copylink)、[Stable LatentMoE](https://ncngj7vcrz76.feishu.cn/docx/Dv17d180xo2brsxQlrlcJvT3nnc?from=from_copylink)

2. **SGLang 代码阅读，弄清参数的意义：**@贾文龙

    [SGLang 请求处理全流程](https://ncngj7vcrz76.feishu.cn/wiki/TQBDwMwkUiIKaNkhBpncoumenjg)

    先理解下列参数

    ```Bash
    tp-size 8
    dcp-size 8
    mem-fraction-static 0.85
    mamba-full-memory-ratio 5.13
    speculative-algorithm DSPARK
    speculative-dspark-block-size 7
    enable-linear-replayssm-spec
    ```

    1. 理解配置参数是如何作用到 Kimi\-K3 模型上的

    2. 结合测试过程中遇到的问题进一步阅读相关代码



3. **测试不同数据集下的性能，记录数据并讨论**** **@赵龙@游兴业@程业泰

    1. [SWE\-agent\-trajectories](https://ncngj7vcrz76.feishu.cn/docx/TBIfdiITgohICFxMsqHc6FIBnKh?from=from_copylink)

    2. [SWE\-rebench OpenHands 并发测试](https://ncngj7vcrz76.feishu.cn/wiki/Tfo8wDLBwiUEzBku5LIcRbeQn8b)

    3. [SWE\-Gym Openhands 测试](https://ncngj7vcrz76.feishu.cn/wiki/ZHxiw80q7iJwIAkPdiIcCFSSnKg)

    4. 包含两类实验

        1. 与今日汇报的实验类似，测试不同并发度的性能

        2. 固定并发度，更改总 token 数

    5. 除了常规的性能数据，还需要记录的数据：

        1. 显存组成：权重、KV Cache、KDA States 等

        2. Prefix Cache 命中情况

    6. 测试完后互相同步一下数据，然后分析一下性能瓶颈的原因

        - KV Cache 容量是否不足。

        - Router 是否产生排队。

        - 其他原因



4. **B300 机器配置调研，整理成 md 文档：****@孙申奥**

    1. 调研 B300 的机器配置，包括但不限于 GPU、CPU、内存、网络等主要硬件配置与连接模式

    

# 暂时搁置的问题

## Kimi K3 剩余相关测试和 Debug

1. 开统一内存后的 Request 并行问题

2. 完成长请求和混合请求的剩余 Workload，以及扩充了 Ratio 后的测试





