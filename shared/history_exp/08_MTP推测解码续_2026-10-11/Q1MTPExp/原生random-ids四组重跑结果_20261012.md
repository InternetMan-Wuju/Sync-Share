# SGLang原生random-ids四组MTP实验结果（2026-10-12）

四组MTP ON已全部完成，23节点单机16卡，共12条请求、42584个输出token。1K/32K完整生成32768token，用时54.3分钟；本次服务已停止，容器保留。最终长度、结束原因、接受计数、输入SHA及流式输出重建均校验通过。

## 结果

| 场景 | 并发/请求 | AL | 草稿接受率 | P1/P2/P3 | 输出tok/s | TTFT均s | TPOT均ms | E2E均s | 命中率 |
|---|---|---|---|---|---|---|---|---|---|
| w1_8k_1k | 4/8 | 2.526 | 0.5090 | 0.6170/0.4928/0.4172 | 16.617 | 12.701 | 170.914 | 187.546 | 0.0000 |
| w2_32k_1k | 1/1 | 2.901 | 0.6336 | 0.8357/0.6289/0.4363 | 6.346 | 21.232 | 136.967 | 161.350 | 0.0000 |
| w3_1k_32k | 1/1 | 3.999 | 0.9995 | 0.9996/0.9995/0.9994 | 10.061 | 3.494 | 99.288 | 3256.878 | 0.0000 |
| w4_prefix32k_300 | 1/2 | 2.804 | 0.6044 | 0.7243/0.5888/0.5000 | 6.295 | 5.618 | 140.587 | 47.653 | 0.5977 |

P1/P2/P3为无条件位置概率，先累加直方图后求比值；AL=输出token/verify。全部请求均达到目标输出，finish_reason=length，没有截短长输出。吞吐报告整机总输出tok/s，没有除以NPU；样本少，P50/P95只是这些请求的描述统计，详细值保存在result.json。

## 接受率和输出有什么关系

原生random-ids并没有自动修复高接受率：长输出32768token只有3种ID，主要是197/532交替，第一token后32767个token精确周期重复，占99.997%。原始直方图为[3,1,1,8190]、verify=8195、correct=24573、proposed=24585，AL=3.998536、草稿接受率99.9512%。这是真实采集的退化输出接受统计，不能作为正常业务MTP接受率或加速收益。

8K/1K共6/8条有占输出一半以上的精确周期尾段。其中4条AL为3.807–3.938，但另外两条仅1.238、1.585；循环组加权AL=2.427，余下两条AL=2.880。目标输出重复并不保证草稿三个位置都命中，不能把所有接受率高低都归结为循环。

32K/1K和缓存组没有检测到满足阈值的精确周期尾段，但仍出现内容改写或对话续写。检测不到精确循环也不等于语义完全正常。最终输出全文和ID都保存，后续可在CPU分析。

服务启动后的正常聊天预检：你好→“你好！很高兴见到你，有什么我可以帮忙的吗？”；1+1→“2”。正常问答可以说人话，不能由此证明所有随机输入和所有数值路径都正常。

本轮所有原生输入均无tokenizer的added token；因此“旧uniform混入角色/EOS是唯一原因”不成立。原生random-ids本身是合成ID输入，并不是有意义的真实对话数据集。本轮保持benchmark的temperature=0、ignore_eos=True；没有同输入/同平台的EOS开关对照，所以不能量化强制长度的单独贡献。

## 输入方式和服务配置

前三组调用镜像内安装的RandomDataset.from_args(...).load(tokenizer)，dataset_name=random-ids、random_range_ratio=1、tokenize_prompt=True，直接发原始ID。第四组由该函数分别生成19661个共享前缀及两条13107后缀，再拼接成总32768；这是为缓存场景增加的共享前缀扩展，并非官方函数未经修改的完整prompt。缓存实测59.765625%，处在58.5%–61.5%验收区间。种子、源文件及SHA、输入及SHA均保存。

23单机，v0.5.19-cann9.0.0-a3、实际只读glm-q路径；TP16/EP16/DP1，DP attention/LMHead关，DeepEP normal，实际Graph关，context131072、MRR8、MEM0.90、chunk16384、maxprefill32768、tokenizer-workers4，共享专家fusion关，NEXTN/EAGLE steps3/topk1/draft4、draft-unquant。

这是沿用已验证能跑的服务后更换输入的重跑。相对AboutModels默认v0.5.16、DP2+DP attention、DeepEP auto、context180000、MRR64等差异明确保留；不是严格榜单基线。请求数按文档倍率8/1/1/2，仅MTP ON，不报告MTP OFF收益。历史输入、种子、请求数量及长输出长度与本轮有差异，不能将前后吞吐或AL差值全部解释成生成器的因果贡献。

## 采集和校验

每请求保留payload、完整output_ids/全文、全部终态meta及采样参数，流式事件按增量ID/客户端时间/meta压缩保存，可重建完整输出。TTFT以首个新增ID为准，TPOT取首末新增输出时间差/(输出数−1)，E2E至流结束；另存finalization时间和服务器queue/prefill时间戳。每分钟存metrics、server-info、NPU状态。预热和聊天检查不计入正式workload。

服务在请求结束前不返回逐位置计数：阶段日志可看AL/草稿接受率，不能用流式事件大小伪造每阶段P1/P2/P3。所有终态逐位置计数完整保存。

预检第一次因聊天模板BatchEncoding处理错误收到HTTP400，正式负载未开始，修正后重启原容器；失败证据在attempts/preflight_01_template_error。w1第一次汇总因服务端省略末尾零桶而失败；在CPU补零后重算，原meta不变、没有重发请求。w1吞吐墙钟由最早请求起点到最晚请求终点恢复，不含线程池启动/最终汇总；其余组用完整客户端墙钟。runner原始终态与修正后的核验终态分别保存。

## 数据位置

完整数据：`/data_lib/cyt/q1mtp/native_full_20261012_190926/`（约47MB）。工作区入口：`/home/chengyetai/workspace/current_exp/Q1MTPExp/results/native_full_20261012_190926/`。

- `summary.md`、`experiment_status.json`、`final_validation.json`：总览及最终校验。
- `datasets/`：原生输入、生成来源及完整源码。
- `w*/result.json`：每组完整结果；`request_*.json`：每请求原始终态；`request_*.payload.json`：原始请求；`request_*.events.jsonl.gz`：增量流。
- `logs/`、`monitor/`：服务日志与按分钟采集。
- `source_sha256.json`、`container_provenance.json`、`server_info.json`、代码/文档快照：运行配置及来源。
- `native_input_special_token_audit.json`、`short_workload_output_analysis.json`：输入特殊token和输出形状复核。

本轮结论是：原生random-ids数据口径已纠正，接受计数可核验，但固定长输出的退化仍存在。不能以换成原生工具就认为业务接受率应接近2，也不能据此排除所有模型数值或实现差异。
