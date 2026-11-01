# 不同Chunk size下的性能测试

## **HOL（Head\-of\-Line Blocking，队头阻塞）**测试

实验内容：

Static：切分chunk：2048、4096、16384

Dynamic



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDA5OTdhNWNiYmU4Yzc0ZDQxMGYzNDVmYjMwZWJmZTJfYTQ4YjIwY2ExNzkwODVkZjZhMzcxNTBlYmVmYTVmMGVfSUQ6NzY3NzU0NjM0MjI1NjA1MzQ3Ml8xNzkxNjY2NjgwOjE3OTE3NTMwODBfVjM)

## 本次实测结果

- Dynamic切分效果最优

- 随着chunksize增大尾延迟降低

## 发现bug：Kimi K3 \+ SGLang KV Cache 越界崩溃

> SGLang 0\.5\.16 在 Kimi K3、TP8/DCP8、DSPARK、Full KV 高压场景下存在
>  间歇性 KV slot 索引越界问题。
> 
> 

SGLang 给 KV Cache 预先划分了一块固定大小的数组。高并发时 Full\-attention KV Cache 接近用满，随后某次 CUDA kernel 收到了超出缓存池有效范围的索引，GPU 触发 assertion，导致一个 TP worker 退出，**最终整个 API 服务停止**。

发生崩溃的测试配置中，每张 GPU/TP rank 的缓存组成包括：

```Plain Text
Full-KV 容量：95,424 token slots
Full-KV 显存：2.46 GiB
KDA state pool：25.93 GiB
KDA slots：469
mamba_full_memory_ratio：5.13
```

```C
正常范围为：index >= 0 && index < 95424
```

本次日志中的核心错误是：

```Plain Text
kvcache.cuh:110
Assertion `index >= 0 && index < size_limit` failed
```

CUDA kernel 检测到越界后触发 assertion。某个 TP worker 随即退出，主进程收到：

```Plain Text
SIGQUIT received
```

TP=8 中任何一个 rank 退出后，整个分布式推理服务都无法继续，因此 OpenAI 兼容 API 会整体停止，而不是只让一个请求失败。

本次 HOL 测试负载包含：

- 96 个输入约 8K 至 10K tokens 的 short 请求。

- 32 个输入约 28K 至 40K tokens 的 long 请求。

- 客户端并发 C64。

- 每个请求固定生成 128 tokens。

大致执行过程如下：

```Plain Text
64 个请求进入客户端并发窗口
        ↓
SGLang 把长输入拆成多个 prefill chunk
        ↓
已完成 prefill 的 token 不断占用 Full-KV
        ↓
部分请求开始 decode，KV 还必须继续保留
        ↓
Full-KV 使用率达到约 99.7%
        ↓
后续请求进入 Scheduler queue
        ↓
Prefix Cache 开始大量 eviction 和 slot 复用
        ↓
**某次 KV slot 映射出现不合法索引**
        ↓
CUDA assertion，TP worker 退出，服务崩溃
```

分析：github sglang有类似相关的pr，得更新最新版本sglang测一下

