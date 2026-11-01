# MTP自主探索全量性能表（含完整W3补跑）

更新时间：2026-10-13T18:47:50+08:00。共46组有效探索/标准场景结果，另列7组功能检查和1组不完整结果。

长度单位为token；总吞吐＝全部完成请求的总输出/整组墙钟；TTFT、TPOT、E2E为请求均值，TPOT只平均有定义的请求。AL与草稿接受率按有验证计数的请求加权，prefill即结束而无验证的请求不补零。允许结束的输出上限和实际输出分列。

配置中文简称对应关系如下；旧版实际自报0.5.17开发版（镜像标签0.5.16）。旧版双路使用自动通信与小批次计算图，0.5.19单/双路使用常规通信且关闭计算图。

| 表中配置简称 | 具体说明 |
|---|---|
| 0.5.19单路 | 0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图 |
| 旧版双路87% | 旧镜像，双路分流、自动通信，静态显存87%方案 |
| 旧版双路90% | 旧镜像，双路分流、自动通信，静态显存90%方案 |
| 0.5.19双路原断言 | 0.5.19版，双路分流、常规通信，原始断言诊断 |
| 0.5.19双路调整检查 | 0.5.19版，双路分流，调整断言检查时机 |
| 0.5.19双路关调度重叠 | 0.5.19版，双路分流，关闭调度重叠、常规评分 |
| 0.5.19双路再关多流 | 0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分 |
| 0.5.19单路（W3补跑） | 沿用0.5.19单路配置，完整长输出补跑 |

## 四类标准场景

| 服务配置 | 测试内容 | 输入 → 输出上限 | 实际输出（逐请求） | 并发 / 请求 | AL | 草稿接受率 | P1 / P2 / P3 | 总吞吐token/s | TTFT秒 | TPOT毫秒 | E2E秒 | 周期尾段请求数 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5.19单路 | W1：输入8192 → 输出1024 | 8192 → 1024 | 全部1024 | 4 / 8 | 2.991 | 66.31% | 80.98% / 64.80% / 53.16% | 23.494 | 9.424 | 140.120 | 152.767 | 5 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/w1_8k_1k/result.json) |
| 0.5.19单路 | W2：输入32768 → 输出1024 | 32768 → 1024 | 全部1024 | 1 / 1 | 2.599 | 53.21% | 79.70% / 51.27% / 28.68% | 5.748 | 22.469 | 152.166 | 178.134 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/w2_32k_1k/result.json) |
| 0.5.19单路（W3补跑） | W3：输入1024 → 目标输出32768 | 1024 → 32768 | 全部32768 | 1 / 1 | 3.980 | 99.34% | 99.64% / 99.32% / 99.05% | 10.009 | 0.849 | 99.886 | 3273.829 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/w3_supplement_20261013/w3_1k_32k/result.json) |
| 0.5.19单路 | W4：总输入32768 → 输出300，前缀缓存60% | 32768 → 300 | 全部300 | 1 / 2 | 2.667 | 55.41% | 74.22% / 56.00% / 36.00% | 5.946 | 5.075 | 151.738 | 50.445 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/w4_prefix32k_300/result.json) |

## 随机输入与输入格式诊断

| 服务配置 | 测试内容 | 输入 → 输出上限 | 实际输出（逐请求） | 并发 / 请求 | AL | 草稿接受率 | P1 / P2 / P3 | 总吞吐token/s | TTFT秒 | TPOT毫秒 | E2E秒 | 周期尾段请求数 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 旧版双路87% | 随机ID转文本再分词 | 1088、1106 → 256 | 全部256 | 1 / 2 | 3.066 | 69.46% | 83.83% / 68.26% / 56.29% | 34.801 | 2.413 | 19.372 | 7.353 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_default_text_roundtrip_256/result.json) |
| 旧版双路87% | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 2.798 | 60.11% | 84.70% / 73.22% / 22.40% | 32.875 | 2.425 | 21.014 | 7.784 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_exact_1k256/result.json) |
| 旧版双路87% | 随机ID直发，允许结束 | 1024 → 256 | 1、256 | 1 / 2 | 3.657 | 89.52% | 91.43% / 88.57% / 88.57% | 49.472 | 0.537 | 16.134 | 2.594 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_exact_eos_1k256/result.json) |
| 旧版双路87% | 随机ID，长输出短诊断 | 1024 → 1024 | 全部1024 | 1 / 1 | 3.969 | 98.84% | 98.84% / 98.84% / 98.84% | 64.954 | 0.503 | 14.912 | 15.758 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_longinput_1k1024/result.json) |
| 旧版双路87% | 随机正文加聊天模板 | 1030 → 256 | 256、165、14 | 1 / 3 | 2.544 | 51.46% | 77.78% / 50.29% / 26.32% | 37.286 | 0.548 | 24.523 | 3.886 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/native_random_body_chatframed_256/result.json) |
| 旧版双路90% | 随机ID转文本再分词 | 1088、1106 → 256 | 全部256 | 1 / 2 | 2.709 | 56.97% | 84.66% / 69.84% / 16.40% | 32.796 | 2.365 | 21.325 | 7.803 | 2 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_default_text_roundtrip_256/result.json) |
| 旧版双路90% | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 2.813 | 60.44% | 79.12% / 56.59% / 45.60% | 33.489 | 2.373 | 20.660 | 7.641 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_exact_1k256/result.json) |
| 旧版双路90% | 随机ID直发，允许结束 | 1024 → 256 | 1、256 | 1 / 2 | 3.507 | 83.11% | 95.89% / 80.82% / 72.60% | 48.013 | 0.557 | 16.599 | 2.674 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_exact_eos_1k256/result.json) |
| 旧版双路90% | 随机ID，长输出短诊断 | 1024 → 1024 | 全部1024 | 1 / 1 | 3.894 | 96.45% | 96.96% / 96.20% / 96.20% | 64.897 | 0.505 | 14.920 | 15.769 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_longinput_1k1024/result.json) |
| 旧版双路90% | 随机正文加聊天模板 | 1030 → 256 | 141、252、256 | 1 / 3 | 2.660 | 55.33% | 84.43% / 53.28% / 28.28% | 41.291 | 0.541 | 21.995 | 5.236 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/native_random_body_chatframed_256/result.json) |
| 0.5.19双路再关多流 | 配对随机正文，加聊天边界 | 1030 → 256 | 全部92 | 1 / 1 | 3.407 | 80.25% | 96.30% / 88.89% / 55.56% | 7.795 | 0.855 | 120.268 | 11.799 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/framing_pair_chat_eos256/result.json) |
| 0.5.19双路再关多流 | 配对随机正文，直接续写 | 1024 → 256 | 全部256 | 1 / 1 | 3.879 | 95.45% | 95.45% / 95.45% / 95.45% | 10.045 | 0.777 | 96.882 | 25.482 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/framing_pair_raw_eos256/result.json) |
| 0.5.19双路再关多流 | 随机ID转文本再分词 | 1088、1106 → 256 | 全部256 | 1 / 2 | 2.473 | 49.28% | 70.53% / 45.89% / 31.40% | 6.199 | 2.751 | 151.139 | 41.292 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_default_text_roundtrip_256/result.json) |
| 0.5.19双路再关多流 | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 3.066 | 69.06% | 83.83% / 68.26% / 55.09% | 7.579 | 2.834 | 121.323 | 33.773 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_exact_1k256/result.json) |
| 0.5.19双路再关多流 | 随机ID直发，允许结束 | 1024 → 256 | 1、171 | 1 / 2 | 2.803 | 60.11% | 86.89% / 57.38% / 36.07% | 7.073 | 0.793 | 133.673 | 12.156 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_exact_eos_1k256/result.json) |
| 0.5.19双路再关多流 | 随机ID，长输出短诊断 | 1024 → 1024 | 全部1024 | 1 / 1 | 3.969 | 98.84% | 98.84% / 98.84% / 98.84% | 10.460 | 0.837 | 94.874 | 97.893 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_longinput_1k1024/result.json) |
| 0.5.19双路再关多流 | 随机正文加聊天模板 | 1030 → 256 | 137、96、92 | 1 / 3 | 2.851 | 61.70% | 86.84% / 63.16% / 35.09% | 7.135 | 0.898 | 132.199 | 15.180 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/native_random_body_chatframed_256/result.json) |
| 0.5.19双路关调度重叠 | 随机ID转文本再分词 | 1088、1106 → 256 | 全部256 | 1 / 2 | 2.473 | 49.28% | 70.53% / 45.89% / 31.40% | 5.893 | 2.760 | 159.537 | 43.442 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_default_text_roundtrip_256/result.json) |
| 0.5.19双路关调度重叠 | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 3.066 | 69.06% | 83.83% / 68.26% / 55.09% | 7.152 | 2.880 | 129.060 | 35.791 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_exact_1k256/result.json) |
| 0.5.19双路关调度重叠 | 随机ID直发，允许结束 | 1024 → 256 | 1、171 | 1 / 2 | 2.803 | 60.11% | 86.89% / 57.38% / 36.07% | 6.675 | 0.824 | 141.842 | 12.881 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_exact_eos_1k256/result.json) |
| 0.5.19双路关调度重叠 | 随机ID，长输出短诊断 | 1024 → 1024 | 全部1024 | 1 / 1 | 3.969 | 98.84% | 98.84% / 98.84% / 98.84% | 10.065 | 0.870 | 98.596 | 101.734 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_longinput_1k1024/result.json) |
| 0.5.19双路关调度重叠 | 随机正文加聊天模板 | 1030 → 256 | 137、96、92 | 1 / 3 | 2.851 | 61.70% | 86.84% / 63.16% / 35.09% | 6.800 | 0.896 | 139.121 | 15.927 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/native_random_body_chatframed_256/result.json) |
| 0.5.19双路原断言 | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 3.066 | 69.06% | 83.83% / 68.26% / 55.09% | 7.139 | 2.798 | 129.633 | 35.855 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-normal/native_exact_1k256/result.json) |
| 0.5.19双路调整检查 | 随机ID转文本再分词 | 1088、1106 → 256 | 全部256 | 1 / 2 | 2.473 | 49.28% | 70.53% / 45.89% / 31.40% | 5.905 | 2.788 | 159.075 | 43.352 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_default_text_roundtrip_256/result.json) |
| 0.5.19双路调整检查 | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 3.066 | 69.06% | 83.83% / 68.26% / 55.09% | 7.094 | 2.835 | 130.389 | 36.085 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_exact_1k256/result.json) |
| 0.5.19双路调整检查 | 随机ID直发，允许结束 | 1024 → 256 | 1、171 | 1 / 2 | 2.803 | 60.11% | 86.89% / 57.38% / 36.07% | 6.749 | 0.817 | 140.281 | 12.741 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_exact_eos_1k256/result.json) |
| 0.5.19双路调整检查 | 随机ID，长输出短诊断 | 1024 → 1024 | 全部1024 | 1 / 1 | 3.969 | 98.84% | 98.84% / 98.84% / 98.84% | 9.600 | 0.880 | 103.397 | 106.656 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_longinput_1k1024/result.json) |
| 0.5.19双路调整检查 | 随机正文加聊天模板 | 1030 → 256 | 137、96、92 | 1 / 3 | 2.851 | 61.70% | 86.84% / 63.16% / 35.09% | 6.769 | 0.909 | 140.162 | 16.000 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/native_random_body_chatframed_256/result.json) |
| 0.5.19单路 | 随机ID直发，强制输出 | 1024 → 256 | 全部256 | 1 / 2 | 2.994 | 66.67% | 83.63% / 67.25% / 49.12% | 7.000 | 2.956 | 131.817 | 36.570 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/native_exact_1k256/result.json) |
| 0.5.19单路 | 随机ID直发，允许结束 | 1024 → 256 | 1、256 | 1 / 2 | 3.606 | 87.32% | 88.73% / 87.32% / 85.92% | 9.038 | 0.902 | 104.406 | 14.215 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/native_exact_eos_1k256/result.json) |
| 0.5.19单路 | 随机ID，长输出短诊断 | 1024 → 1024 | 全部1024 | 1 / 1 | 3.969 | 98.84% | 98.84% / 98.84% / 98.84% | 9.903 | 0.897 | 100.193 | 103.395 | 1 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/native_longinput_1k1024/result.json) |

## 真实聊天

| 服务配置 | 测试内容 | 输入 → 输出上限 | 实际输出（逐请求） | 并发 / 请求 | AL | 草稿接受率 | P1 / P2 / P3 | 总吞吐token/s | TTFT秒 | TPOT毫秒 | E2E秒 | 周期尾段请求数 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 旧版双路87% | 真实聊天完整回复 | 40、68 → 2048 | 1059、848 | 2 / 2 | 2.717 | 57.12% | 78.35% / 55.27% / 37.75% | 73.468 | 0.492 | 21.512 | 21.252 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650_business/sharegpt_chat_complete/result.json) |
| 旧版双路87% | 真实聊天短窗口，允许结束 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.616 | 53.90% | 77.52% / 51.47% / 32.69% | 56.925 | 4.446 | 52.481 | 17.829 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650_business/sharegpt_chat_eos/result.json) |
| 旧版双路87% | 真实聊天短窗口，强制输出 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.596 | 53.15% | 76.43% / 50.95% / 32.07% | 105.649 | 1.728 | 29.300 | 9.199 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650_business/sharegpt_chat_forced/result.json) |
| 旧版双路90% | 真实聊天完整回复 | 40、68 → 2048 | 894、918 | 2 / 2 | 2.836 | 61.19% | 80.13% / 60.41% / 43.04% | 76.070 | 2.571 | 21.068 | 21.606 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650_business/sharegpt_chat_complete/result.json) |
| 旧版双路90% | 真实聊天短窗口，允许结束 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.674 | 55.87% | 79.37% / 53.13% / 35.12% | 51.472 | 4.545 | 58.279 | 19.407 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650_business/sharegpt_chat_eos/result.json) |
| 旧版双路90% | 真实聊天短窗口，强制输出 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.639 | 54.68% | 77.06% / 53.99% / 32.99% | 99.778 | 1.932 | 31.160 | 9.878 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650_business/sharegpt_chat_forced/result.json) |
| 0.5.19双路关调度重叠 | 真实聊天完整回复 | 40、68 → 2048 | 1235、702 | 2 / 2 | 2.593 | 53.06% | 75.50% / 51.54% / 32.13% | 9.356 | 2.752 | 149.290 | 151.522 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe_business/sharegpt_chat_complete/result.json) |
| 0.5.19双路关调度重叠 | 真实聊天短窗口，允许结束 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.576 | 52.45% | 75.60% / 50.06% / 31.70% | 18.641 | 4.797 | 186.965 | 52.473 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe_business/sharegpt_chat_eos/result.json) |
| 0.5.19双路关调度重叠 | 真实聊天短窗口，强制输出 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.586 | 52.90% | 76.52% / 49.87% / 32.32% | 21.786 | 2.596 | 167.955 | 45.425 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe_business/sharegpt_chat_forced/result.json) |
| 0.5.19单路 | 真实聊天短窗口，允许结束 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.609 | 53.72% | 77.07% / 52.10% / 31.97% | 21.317 | 4.095 | 165.480 | 46.293 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19_business/sharegpt_chat_eos/result.json) |
| 0.5.19单路 | 真实聊天短窗口，强制输出 | 25、40、53、56、68、97、276、346 → 256 | 全部256 | 4 / 8 | 2.609 | 53.72% | 77.07% / 52.10% / 31.97% | 24.347 | 1.362 | 153.017 | 40.381 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19_business/sharegpt_chat_forced/result.json) |

完整聊天测试是在三套配置下重复同两条问题；短窗口使用同一批8条问题，不能相加当作更多独立样本。配置、输入长度、并发、输出是否自然结束和预热会影响性能，各行不能直接用作单变量收益或MTP开关加速比。

W3完整补跑的两token循环尾段占输出66.77%；输入1024/输出1024的短诊断经常接近AL4且出现循环。接受率需要和输出形状一起看。W4实测缓存命中59.77%。

## 功能检查（不计入46组探索结果）

| 服务配置 | 测试内容 | 输入 → 输出上限 | 实际输出（逐请求） | 并发 / 请求 | AL | 草稿接受率 | P1 / P2 / P3 | 总吞吐token/s | TTFT秒 | TPOT毫秒 | E2E秒 | 周期尾段请求数 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 旧版双路87% | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 60.00% | 100.00% / 60.00% / 20.00% | 4.750 | 1.334 | 37.249 | 1.472 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem87-hccl650/sanity_chat/result.json) |
| 旧版双路90% | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 66.67% | 100.00% / 60.00% / 40.00% | 4.778 | 1.328 | 36.464 | 1.463 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v16-auto-cap64-mem90-hccl650/sanity_chat/result.json) |
| 0.5.19双路再关多流 | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 60.00% | 100.00% / 60.00% / 20.00% | 2.741 | 1.606 | 281.440 | 2.551 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-multistream/sanity_chat/result.json) |
| 0.5.19双路关调度重叠 | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 60.00% | 100.00% / 60.00% / 20.00% | 2.485 | 1.802 | 290.971 | 2.815 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-no-plan-lp-safe/sanity_chat/result.json) |
| 0.5.19双路原断言 | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 60.00% | 100.00% / 60.00% / 20.00% | 2.567 | 1.707 | 285.026 | 2.724 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-normal/sanity_chat/result.json) |
| 0.5.19双路调整检查 | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 60.00% | 100.00% / 60.00% / 20.00% | 2.626 | 1.658 | 283.679 | 2.664 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-ordered-probe/sanity_chat/result.json) |
| 0.5.19单路 | 问候与算术功能检查 | 14、16 → 128 | 12、2 | 1 / 2 | 2.800 | 60.00% | 100.00% / 60.00% / 20.00% | 2.512 | 1.765 | 295.796 | 2.784 | 0 | [原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/publisher_v19/sanity_chat/result.json) |

## 不完整请求组

- 0.5.19版，双路分流、常规通信，原始断言诊断：原生随机ID直发：输入1024 → 最多输出256，允许提前结束，一条请求未收到完整流结束标记；[原始结果](/home/chengyetai/workspace/current_exp/autonomous_20261012/results/mtp-v19-dp2-normal/native_exact_eos_1k256/result.json)，不计入有效性能与接受率表。

## 未形成有效结果的MTP服务启动

| 配置尝试 | 结果 | 来源 |
|---|---|---|
| 旧镜像默认方案：双路分流、自动通信、开启计算图 | 服务达到可用状态前退出，未产生该配置的有效性能结果 | [服务日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-default-v16/server.log) |
| 旧镜像默认方案：关闭计算图 | 服务达到可用状态前退出，未产生该配置的有效性能结果 | [服务日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-default-v16-graph-off/server.log) |
| 旧镜像默认方案：通信缓冲增至1000MB | 服务达到可用状态前退出，未产生该配置的有效性能结果 | [服务日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-default-v16-hccl1000/server.log) |
| 旧镜像，通信容量64，静态显存90%、通信缓冲1536MB | 服务达到可用状态前退出，未产生该配置的有效性能结果 | [服务日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v16-auto-cap64/server.log) |
| 旧镜像，通信容量64，静态显存84%、通信缓冲1536MB | 调度器退出，HTTP入口仍在；未形成有效性能结果 | [服务日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v16-auto-cap64-mem84/server.log) |
| 0.5.19版，双路分流，关闭调度重叠（评分回退前） | 加载期间停止以避开已确认的融合评分编译错误，改用常规评分重启；没有推理请求结果 | [服务日志](/home/chengyetai/workspace/current_exp/autonomous_20261012/logs/cyt-auto-mtp-v19-dp2-no-plan/server.log) |

另外6次MTP关闭正确性诊断及16次固定前缀重放分别见[补充诊断](review_20261013/MTP一般情况检索与补充诊断.md)和[固定前缀核对](固定DP前缀核对.md)。这些正确性检查不充当MTP关闭性能基线。
