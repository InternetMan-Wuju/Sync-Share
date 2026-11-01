# 自主MTP探索：阶段实测结果（四组要求未完成）

场景和服务配置已用中文说明，长度单位均为token。原代号及缩写见[名称说明](负载与配置名称说明.md)。

<!-- mtp-workload-coverage:start -->
**四组覆盖：自主探索的标准长度ON覆盖3/4，OFF性能基线覆盖0/4。四组ON/OFF记录尚未齐全；配置对齐也需单独核验。**

| AboutModels要求 | 自主探索ON | OFF性能基线 | 实际并发/请求 |
|---|---|---|---|
| 输入8192 → 输出1024 | 规定长度已跑完，服务配置有偏离 | 未跑 | 4/8 |
| 输入32768 → 输出1024 | 规定长度已跑完，服务配置有偏离 | 未跑 | 1/1 |
| 输入1024 → 输出32768 | 未跑；输入1024 → 输出1024只作短诊断 | 未跑 | — |
| 输入32768 → 输出300，缓存60% | 规定长度已跑完，服务配置有偏离 | 未跑 | 1/2 |

三组标准长度测试使用0.5.19版和公开源恢复权重：数据并行1路，关闭DP attention/LM head，DeepEP常规通信，关闭计算图；不能写成AboutModels默认配置四组已完成。
输入1024 → 输出1024短诊断的输出长度仅为要求32K的1/32。后续6次OFF用于输出正确性检查，未形成四组性能基线；ChunkSize的OFF矩阵也未作为MTP开关配对。
45组有效新增记录包含重复配置和诊断变体，不能用记录数代表四组场景覆盖。完整统计与原始结果见[自主探索四组覆盖核对](MTP自主探索四组覆盖.md)。

<!-- mtp-workload-coverage:end -->
权重恢复依据公开量化源SHA256。以下仅纳入已完成请求，失败启动另列；不是同条件速度收益榜单。

| 服务/轮次 | 负载 | AL | 草稿接受率 | P1/P2/P3 | 主要计数请求 | 检出周期尾段请求 | 来源 |
|---|---|---|---|---|---|---|---|
| 旧镜像，双路分流、自动通信，静态显存87%方案 | 随机ID转文本再分词 → 输出256（实际输入长度会变） | 3.066 | 0.6946 | 0.8383/0.6826/0.5629 | 2 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_default_text_roundtrip_256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 2.798 | 0.6011 | 0.8470/0.7322/0.2240 | 2 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_exact_1k256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案 | 原生随机ID直发：输入1024 → 最多输出256，允许提前结束 | 3.657 | 0.8952 | 0.9143/0.8857/0.8857 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_exact_eos_1k256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案 | 长输出场景的短诊断：输入1024 → 输出1024 | 3.969 | 0.9884 | 0.9884/0.9884/0.9884 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_longinput_1k1024/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案 | 随机正文加聊天模板 → 最多输出256，允许提前结束 | 2.544 | 0.5146 | 0.7778/0.5029/0.2632 | 3 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_random_body_chatframed_256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案 | 问候与算术功能检查 | 2.800 | 0.6000 | 1.0000/0.6000/0.2000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/sanity_chat/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案（真实聊天测试） | 真实聊天完整回复：最多输出2048，允许提前结束 | 2.717 | 0.5712 | 0.7835/0.5527/0.3775 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650_business/sharegpt_chat_complete/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案（真实聊天测试） | 真实聊天短窗口：最多输出256，允许提前结束 | 2.616 | 0.5390 | 0.7752/0.5147/0.3269 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650_business/sharegpt_chat_eos/result.json) |
| 旧镜像，双路分流、自动通信，静态显存87%方案（真实聊天测试） | 真实聊天短窗口：强制输出256，忽略结束标记 | 2.596 | 0.5315 | 0.7643/0.5095/0.3207 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650_business/sharegpt_chat_forced/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | 随机ID转文本再分词 → 输出256（实际输入长度会变） | 2.709 | 0.5697 | 0.8466/0.6984/0.1640 | 2 | 2 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_default_text_roundtrip_256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 2.813 | 0.6044 | 0.7912/0.5659/0.4560 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_exact_1k256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | 原生随机ID直发：输入1024 → 最多输出256，允许提前结束 | 3.507 | 0.8311 | 0.9589/0.8082/0.7260 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_exact_eos_1k256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | 长输出场景的短诊断：输入1024 → 输出1024 | 3.894 | 0.9645 | 0.9696/0.9620/0.9620 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_longinput_1k1024/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | 随机正文加聊天模板 → 最多输出256，允许提前结束 | 2.660 | 0.5533 | 0.8443/0.5328/0.2828 | 3 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_random_body_chatframed_256/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | 问候与算术功能检查 | 2.800 | 0.6667 | 1.0000/0.6000/0.4000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/sanity_chat/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案（真实聊天测试） | 真实聊天完整回复：最多输出2048，允许提前结束 | 2.836 | 0.6119 | 0.8013/0.6041/0.4304 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650_business/sharegpt_chat_complete/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案（真实聊天测试） | 真实聊天短窗口：最多输出256，允许提前结束 | 2.674 | 0.5587 | 0.7937/0.5313/0.3512 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650_business/sharegpt_chat_eos/result.json) |
| 旧镜像，双路分流、自动通信，静态显存90%方案（真实聊天测试） | 真实聊天短窗口：强制输出256，忽略结束标记 | 2.639 | 0.5468 | 0.7706/0.5399/0.3299 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650_business/sharegpt_chat_forced/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 同一随机正文加聊天边界 → 最多输出256，允许提前结束 | 3.407 | 0.8025 | 0.9630/0.8889/0.5556 | 1 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/framing_pair_chat_eos256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 同一随机正文直接续写 → 最多输出256，允许提前结束 | 3.879 | 0.9545 | 0.9545/0.9545/0.9545 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/framing_pair_raw_eos256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 随机ID转文本再分词 → 输出256（实际输入长度会变） | 2.473 | 0.4928 | 0.7053/0.4589/0.3140 | 2 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_default_text_roundtrip_256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 3.066 | 0.6906 | 0.8383/0.6826/0.5509 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_exact_1k256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 原生随机ID直发：输入1024 → 最多输出256，允许提前结束 | 2.803 | 0.6011 | 0.8689/0.5738/0.3607 | 1 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_exact_eos_1k256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 长输出场景的短诊断：输入1024 → 输出1024 | 3.969 | 0.9884 | 0.9884/0.9884/0.9884 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_longinput_1k1024/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 随机正文加聊天模板 → 最多输出256，允许提前结束 | 2.851 | 0.6170 | 0.8684/0.6316/0.3509 | 3 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_random_body_chatframed_256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 | 问候与算术功能检查 | 2.800 | 0.6000 | 1.0000/0.6000/0.2000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/sanity_chat/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | 随机ID转文本再分词 → 输出256（实际输入长度会变） | 2.473 | 0.4928 | 0.7053/0.4589/0.3140 | 2 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_default_text_roundtrip_256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 3.066 | 0.6906 | 0.8383/0.6826/0.5509 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_exact_1k256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | 原生随机ID直发：输入1024 → 最多输出256，允许提前结束 | 2.803 | 0.6011 | 0.8689/0.5738/0.3607 | 1 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_exact_eos_1k256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | 长输出场景的短诊断：输入1024 → 输出1024 | 3.969 | 0.9884 | 0.9884/0.9884/0.9884 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_longinput_1k1024/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | 随机正文加聊天模板 → 最多输出256，允许提前结束 | 2.851 | 0.6170 | 0.8684/0.6316/0.3509 | 3 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_random_body_chatframed_256/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | 问候与算术功能检查 | 2.800 | 0.6000 | 1.0000/0.6000/0.2000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/sanity_chat/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分（真实聊天测试） | 真实聊天完整回复：最多输出2048，允许提前结束 | 2.593 | 0.5306 | 0.7550/0.5154/0.3213 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe_business/sharegpt_chat_complete/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分（真实聊天测试） | 真实聊天短窗口：最多输出256，允许提前结束 | 2.576 | 0.5245 | 0.7560/0.5006/0.3170 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe_business/sharegpt_chat_eos/result.json) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分（真实聊天测试） | 真实聊天短窗口：强制输出256，忽略结束标记 | 2.586 | 0.5290 | 0.7652/0.4987/0.3232 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe_business/sharegpt_chat_forced/result.json) |
| 0.5.19版，双路分流、常规通信，原始断言诊断 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 3.066 | 0.6906 | 0.8383/0.6826/0.5509 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-normal/native_exact_1k256/result.json) |
| 0.5.19版，双路分流、常规通信，原始断言诊断 | 问候与算术功能检查 | 2.800 | 0.6000 | 1.0000/0.6000/0.2000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-normal/sanity_chat/result.json) |
| 0.5.19版，双路分流，调整断言检查时机 | 随机ID转文本再分词 → 输出256（实际输入长度会变） | 2.473 | 0.4928 | 0.7053/0.4589/0.3140 | 2 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_default_text_roundtrip_256/result.json) |
| 0.5.19版，双路分流，调整断言检查时机 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 3.066 | 0.6906 | 0.8383/0.6826/0.5509 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_exact_1k256/result.json) |
| 0.5.19版，双路分流，调整断言检查时机 | 原生随机ID直发：输入1024 → 最多输出256，允许提前结束 | 2.803 | 0.6011 | 0.8689/0.5738/0.3607 | 1 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_exact_eos_1k256/result.json) |
| 0.5.19版，双路分流，调整断言检查时机 | 长输出场景的短诊断：输入1024 → 输出1024 | 3.969 | 0.9884 | 0.9884/0.9884/0.9884 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_longinput_1k1024/result.json) |
| 0.5.19版，双路分流，调整断言检查时机 | 随机正文加聊天模板 → 最多输出256，允许提前结束 | 2.851 | 0.6170 | 0.8684/0.6316/0.3509 | 3 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_random_body_chatframed_256/result.json) |
| 0.5.19版，双路分流，调整断言检查时机 | 问候与算术功能检查 | 2.800 | 0.6000 | 1.0000/0.6000/0.2000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/sanity_chat/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | 原生随机ID直发：输入1024 → 输出256（短诊断） | 2.994 | 0.6667 | 0.8363/0.6725/0.4912 | 2 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/native_exact_1k256/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | 原生随机ID直发：输入1024 → 最多输出256，允许提前结束 | 3.606 | 0.8732 | 0.8873/0.8732/0.8592 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/native_exact_eos_1k256/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | 长输出场景的短诊断：输入1024 → 输出1024 | 3.969 | 0.9884 | 0.9884/0.9884/0.9884 | 1 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/native_longinput_1k1024/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | 问候与算术功能检查 | 2.800 | 0.6000 | 1.0000/0.6000/0.2000 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/sanity_chat/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | W1：输入8192 → 输出1024 | 2.991 | 0.6631 | 0.8098/0.6480/0.5316 | 8 | 5 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/w1_8k_1k/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | W2：输入32768 → 输出1024 | 2.599 | 0.5321 | 0.7970/0.5127/0.2868 | 1 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/w2_32k_1k/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | W4：总输入32768 → 输出300，前缀缓存60% | 2.667 | 0.5541 | 0.7422/0.5600/0.3600 | 2 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/w4_prefix32k_300/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图（真实聊天测试） | 真实聊天短窗口：最多输出256，允许提前结束 | 2.609 | 0.5372 | 0.7707/0.5210/0.3197 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19_business/sharegpt_chat_eos/result.json) |
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图（真实聊天测试） | 真实聊天短窗口：强制输出256，忽略结束标记 | 2.609 | 0.5372 | 0.7707/0.5210/0.3197 | 8 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19_business/sharegpt_chat_forced/result.json) |

不完整/校验失败组不纳入上表：1组；现场路径与失败校验保存在JSON中。

## 权重对照与输入证据

| 场景 | 原始副本AL | 公开源校验版AL | 原始副本草稿率 | 校验版草稿率 | 逐请求输入SHA一致 |
|---|---|---|---|---|---|
| W1：输入8192 → 输出1024 | 2.526 | 2.991 | 0.5090 | 0.6631 | 8/8 |
| W2：输入32768 → 输出1024 | 2.901 | 2.599 | 0.6336 | 0.5321 | 1/1 |
| W4：总输入32768 → 输出300，前缀缓存60% | 2.804 | 2.667 | 0.6044 | 0.5541 | 2/2 |
| 原生随机ID直发：输入1024 → 输出256（短诊断） | 2.876 | 2.994 | 0.6217 | 0.6667 | 0/2；其余未留足证据 |

输入1024 → 输出32768旧结果与新输入1024 → 输出1024短探针长度不同，不用总AL作同条件对照。逐请求完整输出、EOS和周期形状均保存。
正式短负载对照保持镜像/TP/EP/DP/MEM/chunk/max-prefill/MRR/context/MTP参数一致。旧启动声明Graph但normal下实际graph显存为0，新启动显式关闭；两者均无图执行。草稿路径视图是加载组织差异，不是另一套草稿权重。逐请求输入SHA核验结果见JSON。
原生随机ID直发旧短探针没有保存逐请求payload或输入SHA，故该行仅列历史数字，不作为严格同输入权重因果证据。前三组正式短负载的输入SHA全部一致。
允许EOS请求有时在prefill即结束，没有verify，位置率留空；未将缺计数当成0。
草稿使用同一checkpoint第78层的7个原始分片视图，仅减少加载扫描量；原NextN加载和计算实现未修改。

## 服务配置与失败记录

| 配置 | 健康 | 请求结果 | 参数 | 日志 |
|---|---|---|---|---|
| 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 | True | 0 |  | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-publisher-v19-slim/server.log) |
| 旧镜像默认方案：双路分流、自动通信、开启计算图 | False | container exited before health | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion --graph | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-default-v16/server.log) |
| 旧镜像默认方案：关闭计算图 | False | container exited before health | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-default-v16-graph-off/server.log) |
| 旧镜像默认方案：通信缓冲增至1000MB | False | container exited before health | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion --graph --env HCCL_BUFFSIZE=1000 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-default-v16-hccl1000/server.log) |
| 旧镜像，通信容量64，静态显存90%、通信缓冲1536MB | False | container exited before health | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion --mrr 8 --dispatch-tokens 64 --graph --graph-bs 1 2 4 8 --graph-max-bs 8 --env HCCL_BUFFSIZE=1536 --env HCCL_OP_EXPANSION_MODE=AIV | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v16-auto-cap64/server.log) |
| 旧镜像，通信容量64，静态显存84%、通信缓冲1536MB | False | scheduler crashed while HTTP router remained alive | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion --mrr 8 --dispatch-tokens 64 --graph --graph-bs 1 2 4 8 --graph-max-bs 8 --env HCCL_BUFFSIZE=1536 --env HCCL_OP_EXPANSION_MODE=AIV --mem 0.84 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v16-auto-cap64-mem84/server.log) |
| 0.5.19版，双路分流、常规通信，原始断言诊断 | True | 1 | --dp 2 --dp-attention --context 180000 --mrr 8 --diagnostics | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v19-dp2-normal/server.log) |
| 旧镜像，双路分流、自动通信，静态显存87%方案 | True | 0 | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion --mrr 8 --dispatch-tokens 64 --mem 0.87 --graph --graph-bs 1 2 4 8 --graph-max-bs 8 --env HCCL_OP_EXPANSION_MODE=AIV | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v16-auto-cap64-mem87-hccl650/server.log) |
| 旧镜像，双路分流、自动通信，静态显存90%方案 | True | 0 | --image quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16 --loader-v0516 --dp 2 --dp-attention --context 180000 --mrr 64 --mem 0.90 --deepep auto --fusion --mrr 8 --dispatch-tokens 64 --mem 0.87 --graph --graph-bs 1 2 4 8 --graph-max-bs 8 --env HCCL_OP_EXPANSION_MODE=AIV --mem 0.90 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v16-auto-cap64-mem90-hccl650/server.log) |
| 0.5.19版，双路分流，调整断言检查时机 | True | 0 | --dp 2 --dp-attention --context 180000 --mrr 8 --diagnostics --ordered-diagnostic | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v19-dp2-ordered-probe/server.log) |
| 0.5.19版，双路分流，关闭调度重叠（评分回退前） | False | stopped during loading before requests: avoid proven fused input-logprob compiler crash; safe scoring restart | --dp 2 --dp-attention --context 180000 --mrr 8 --diagnostics --env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v19-dp2-no-plan/server.log) |
| 0.5.19版，双路分流，关闭调度重叠、常规评分 | True | 0 | --dp 2 --dp-attention --context 180000 --mrr 8 --diagnostics --env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v19-dp2-no-plan-lp-safe/server.log) |
| 每块16384，自动通信，初始计算图批次设置 | False | Graph capture_bs empty: TP16 alignment filters bs1/2/4/8 for one token/request | --mtp off --deepep auto --graph --mrr 16 --graph-bs 1 2 4 8 --graph-max-bs 8 --dispatch-tokens 64 --env HCCL_BUFFSIZE=1536 --env HCCL_OP_EXPANSION_MODE=AIV | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-auto-16384/server.log) |
| 每块16384，常规通信，首次备选启动 | False | stopped during loading before inference: first repair Graph batch alignment | --mtp off --deepep normal --mrr 16 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-normal-16384/server.log) |
| 每块16384，自动通信，修正计算图批次、通信缓冲1536MB | False | Aligned bs16 Graph: aclnnMoeLowLatencyDispatchV2 HCCL allocation3222274048 bytes failed at HCCL1536 | --mtp off --deepep auto --graph --mrr 16 --graph-bs 1 2 4 8 16 --graph-max-bs 16 --dispatch-tokens 64 --env HCCL_BUFFSIZE=1536 --env HCCL_OP_EXPANSION_MODE=AIV | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-auto-aligned-16384/server.log) |
| 每块16384，常规通信，第二次备选启动 | False | stopped during loading before inference: single-variable HCCL650 Graph retry | --mtp off --deepep normal --mrr 16 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-normal-aligned-16384/server.log) |
| 每块16384，自动通信，修正计算图批次、通信缓冲650MB | True | 待运行 | --mtp off --deepep auto --graph --mrr 16 --graph-bs 1 2 4 8 16 --graph-max-bs 16 --dispatch-tokens 64 --env HCCL_BUFFSIZE=650 --env HCCL_OP_EXPANSION_MODE=AIV | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-auto-hccl650-16384/server.log) |
| 每块1024，自动通信、通信缓冲650MB | True | 待运行 | --mtp off --deepep auto --graph --mrr 16 --graph-bs 1 2 4 8 16 --graph-max-bs 16 --dispatch-tokens 64 --env HCCL_BUFFSIZE=650 --env HCCL_OP_EXPANSION_MODE=AIV --chunk 1024 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-auto-1024/server.log) |
| 每块4096，自动通信、通信缓冲650MB | True | 待运行 | --mtp off --deepep auto --graph --mrr 16 --graph-bs 1 2 4 8 16 --graph-max-bs 16 --dispatch-tokens 64 --env HCCL_BUFFSIZE=650 --env HCCL_OP_EXPANSION_MODE=AIV --chunk 4096 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-auto-4096/server.log) |
| 每块8192，自动通信、通信缓冲650MB | True | 待运行 | --mtp off --deepep auto --graph --mrr 16 --graph-bs 1 2 4 8 16 --graph-max-bs 16 --dispatch-tokens 64 --env HCCL_BUFFSIZE=650 --env HCCL_OP_EXPANSION_MODE=AIV --chunk 8192 | [日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-chunk-auto-8192/server.log) |

基础阶段：planned_grid_completed。基础网格、候选复测与最终MTP单变量诊断均已完成。失败配置及尚未做的精度/参考对齐项仍按证据标注。
