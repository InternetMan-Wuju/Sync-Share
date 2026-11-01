# 最佳radio下chunksize选取

结论：chunksize 12288/16384为最佳配置，dynamic chunksize没有使用必要

## 实验内容

mambakv比例为2/2\.5

测试chunksize为2048/4096/8192/12288/16384

在16/64/84/92/100并发下的性能

下表为 `ratio=2.5` 下并发 84/92/100 的等权平均：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjYyNzMzZmUwNTk0NTU5ZTJlM2U3ZGIxZjkwNjAxY2NfNDBlYWM4OGE4MWJhNDk3YzZjNDVmZDgzNjgwYjAzNjhfSUQ6NzY4MjY4NTYxMjE1MzA2NDQyOV8xNzkxNjY2NjkxOjE3OTE3NTMwOTFfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGI1M2Y4YzI5ZmRlYWI3YjNkZjUzNTE1ZTlhYmM1ZjlfOTRjN2Y4OGMxYjQzMDVkMDUyZjVkMjQyODUxNmViNWVfSUQ6NzY4MjY4NTYzNTg0MTU1OTczN18xNzkxNjY2NjkxOjE3OTE3NTMwOTFfVjM)

**不同chunksize的首轮TTFT**

12,288\-token 首轮 prompt 在 `chunk=2048` 下需要约 6 个 prefill chunk，在 `4096` 下约 3 个，在 `8192` 下约 2 个，而 `12288/16384` 都可以在单个 chunk 范围内覆盖首轮。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjgwYTE2NzA4ZmU0ZTMyZDQ2N2VkNjYzNTc4NDRiMDNfZTVhZGU2MWJhMmQxODA1Y2NiYzM3Y2U0MWMzYWM5MGFfSUQ6NzY4MjY4NTY4NzkxNzM4MjYxNV8xNzkxNjY2NjkxOjE3OTE3NTMwOTFfVjM)

**不同chunksize后续轮TTFT**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmM1NDBkODhjNGJlMTAyZDM0ZTkzMzQwNzU3ZDIzODNfYzY1MmI3OWY4MWViYTA2ZGY2Y2M4ODVlMGY1NjhmMWRfSUQ6NzY4MjY4NTc0ODgwNjI5MDYzM18xNzkxNjY2NjkxOjE3OTE3NTMwOTFfVjM)

高负载并发测试

下表是并发

并发100下的scheduler、cache\-pool情况

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YmZjZmJjMzQ2MjgwZDE1MjJlYjUxZDNiYzdiMmFiZjdfNDQzNTA1NjBiZmU3NjhjM2JmNzZlZDAxZDE5OGY1NmNfSUQ6NzY4MjY4NTc4Nzc1MTMzNjkwMF8xNzkxNjY2NjkxOjE3OTE3NTMwOTFfVjM)

两种 chunk 的峰值队列和 Mamba state 压力几乎相同，说明 chunk 不改变由 ratio 决定的服务容量边界。



