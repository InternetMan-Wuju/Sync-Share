数据集：
  agentx数据集为：https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k（全测太长了，可以取部分）机器上目录：/mnt/raid/user/data/dataset/cc-traces-weka-062126-256k。也可以裁剪并且构造。

问题0：基准工具 bench3 部署在集群节点（/mnt/raid/user/data/tools/bench3/），不知道这是什么有啥用

问题1：确认 shared expert 的位置和数量，在会议文稿里有提到。

问题1：
cuda-graph开启卡死，这是首要的问题，我们必须解决，因为明确要求开启Graph进行测试。
参数尽量和其他实验要求保证一致。




我们只用看测试4，别的是其他人的。

原文的测试设计：
https://ucm.readthedocs.io/en/latest/user-guide/best-practices/GLM-5.1-A3_4Node_PD_Disaggregation.html
1. 输入/输出长度：64K/1K，128K/1K
2. TPOT延迟要求：限制为30毫秒
3. 前缀缓存命中率设置：90%
4. KV缓存前缀多样性：取并发值的一半，模拟当用户代理调用时，多个请求同时发送时共享高度相似的请求前缀。
5. 测试并发：10、20、30
6. 请求次数：并发× 4
在一个4节点的A3集群（每个节点8×910C，每个910C为双910B共封芯片），基于GLM-5.1 W8A8量化模型和PD分解架构（2个预填充+2个译码），我们在4个场景下进行了UCM前缀缓存的性能测试。
测试数据：

https://ucm.readthedocs.io/en/latest/user-guide/best-practices/GLM-5.1-A3_4Node_PD_Disaggregation.html 1. Performance Test Summary 中的表格

实验要求：2机prefill+2机decode,graph一定在解决问题后要开

要求你至少给出以下数据，就像上次实验那样：
0.数据集形式
1.运行时的全部命令和参数
2.EP=32/64下的，Prefill token/s,TPOT token/s,E2E(queue+prefill+[刚刚调查的中间时长]+decode)
3.EPLB开启下的EPLB cost，你曾经删除了日志，这次千万别这么做
4.依然是ExpertXLayer热力图，分总和热度和四个随即时刻snapshot，和上次一样。