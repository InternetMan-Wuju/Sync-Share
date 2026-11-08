# 本机16设备实验接线

本轮使用npu1-23的device 0–15，TP16/EP16/DP1，单节点。原模型与SGLang安装文件保持不变；容器通过`PYTHONPATH`加载外部`sitecustomize.py`，在真实模型类`GlmMoeDsaForCausalLM → DeepseekV2MoE`的layer_id=49接入。移除环境变量`CYT_EP16_EXPERIMENT`和外挂挂载即可撤销；启动布局同时必须恢复匹配的原始配置。

运行默认为off。现有native top-k得到逻辑专家后统一转主位置`17*(e//16)+e%16`，其余层及decode同样使用这个合法映射；每卡的临时slot16在off中没有任务。副本身份改变不会影响主专家。原生dispatch、ModelSlim W8A8 runner和combine保持原调用。

三个输入为AgentX实际请求长度/块标识的确定性合成token物化，非原始用户文本；完整身份见INPUTS.json，原始块与token IDs仅本地保留。纯prefill有效8192 token，top8，总计65536分支，强制生成32token、关闭prefix缓存。

## 实验模式

- off：原计算，不做新增统计、计划或复制。
- shadow：当前各输入来源计数 → HCCL交换 → CPU计划，不改路由。
- active_cold：**隔离环境test-only强制标定模式**，每次复制完整专家状态，完成全组ready后按整数配额分流。
- active_warm：相同权重进程中可复用正确的缓存副本；输入改变需要替换时仍会复制。
- replay：捕获本次实际hidden/topk，在16卡同时复测off/prepared/warm/cold；20次预热、校准每rank块长≥100ms、5个交错块。prepared不含在线控制和复制成本，不能称online净收益。

当前planner是计数启发式：最多2个热点候选、3个目的rank、一次迁移；检查所有未调整rank对最大负载的下界。它不是已标定的生产收益门控。active必须设置test_only，否则拒绝，不能作为自动服务优化交付。

## 状态复制和正确性

完整运行态按实际张量枚举，显式EP通信组，ND打包 → 真实HCCL send/recv → 目标专家bank slot16安装 → 完成同步与全组barrier。同一实例的量化runner读取更新后的bank；不更改原始槽。验证源/目的全部状态SHA256、所有来源配额和逐分支逻辑专家/权重不变，再对同一hidden/topk的native baseline输出比较NRMSE（阈值1%）和max_abs。

prefill层只在forward边界进入控制器，decode直接原forward。服务关闭overlap、prefill/decode graph，只支持此次串行单请求eager实验。当前不承诺通信故障后恢复；任何校验失败停止该实验。

## 内存与测量限制

原生统一槽配置使75个MoE层每卡均多一个专家；不能宣称仅layer49付显存成本。KV预算固定12288token，8K请求batch1，不能据此推出原服务最大并发容量不变。

诊断profile和正确性请求不进入E2E汇总。普通E2E路径关闭事件与逐层日志；实际需要的计数D2H、CPU规划、复制与ready等待仍在请求内。layer replay使用同一真实层输入、16卡并行及完整dispatch/combine；块中调用不是独立样本，五块范围不是生产p95。

为避免开发过程中反复重载全部模型，本隔离验证容器支持控制文件显式implementation_revision：在下一串行请求边界重新加载外挂experiment.py，并在16rank核验源码摘要一致。这是实验迭代工具，不是生产热更新机制；每个实测revision与摘要会保留。

## 重现

1. 用scripts/prepare_inputs.py重建本地输入，核对INPUTS.json摘要。
2. 以scripts/launch.sh启动隔离容器；只读加载已有审计权重，不上传权重或hidden。
3. scripts/smoke.py先验证；检查全部16rank的状态、总任务与误差。
4. 验证通过后运行scripts/suite.py；结果写artifacts/及local/rank*.jsonl。
5. scripts/analyze.py将实际记录整理为TIMINGS.csv和报告数据。

CPU测试：`python3 tests/test_planner.py`。硬件结果以RESULTS.md和实际artifacts为准，此文档本身不表示验证通过。

层内计时边界是`DeepseekV2MoE.experts.forward`：覆盖routed-expert的dispatch → 完整量化MLP → combine。gate、shared expert、attention、层间通信及最后的shared/routed相加在这个边界之外。live-input replay也不重复这些外围算子，所以它证明的是routed子链收益；全模型TTFT另由完整请求给出，不能把子链收益外推为整个MoE层或75层收益。

首次P2P通信包含按peer建立通信资源的冷启动，必须与通信资源已预热但每次仍复制权重的active_cold分开。active_warm在正式请求前有单独的准备请求，其成本未摊销到暖调用；这是副本已就绪条件下的上界实验。
