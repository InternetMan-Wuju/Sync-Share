# GLM-5.2 × 4 机 × 64 卡：EP64 + EPLB 实验记录

## 1. 目标

只做 **EP=64**，观察两件事：

1. 大 EP 下的 expert 负载不均衡现象；
2. 开启 EPLB（专家负载均衡）之后，rebalance 是否可用，以及整体表现如何。

## 2. 环境

- 四台 Ascend 910C 机器，每机 16 卡，共 64 卡：
  - 10.120.72.23（npu1-23，rank 0）
  - 10.120.72.25（npu1-25，rank 1）
  - 10.120.72.31（npu1-31，rank 2）
  - 10.120.72.32（npu1-32，rank 3）
- 模型：`/mnt/raid/user/data/models/glm-q`（GLM-5.2 w8a8）
- 镜像：`quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16`
- 部署：decode 模式，`TP=EP=64`，`NNODES=4`
- 服务地址：`http://10.120.72.23:30000`
- 自己的目录：`/home/chengyetai/workspace/glm52_ep64_chengyetai`

## 3. 遇到的问题和解决

### 3.1 EPLB 一开始直接崩

报错：

```text
RuntimeError: For a tensor of internal format, its storage_offset must be 0
ERR02007 DIST feature not supported
```

原因：昇腾上 MoE 权重默认是 `FRACTAL_NZ` 内部格式，EPLB 跨卡搬专家权重时直接切了一行专家权重去 `isend`，通信库不接受这种内部格式。

解决：按官方文档，启动时关掉 NZ 格式：

```bash
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
```

### 3.2 关掉 NZ 后，rebalance 会卡死

关掉 NZ 后不报错了，但 `[EPLBManager] rebalance start` 之后卡住，`rebalance end` 一直不出现，服务健康检查变成 000。

用 `py-spy` 抓堆栈，卡在：

```text
torch.distributed.isend
torch_npu/distributed/distributed_c10d.py: _batch_isend_irecv
sglang.srt.eplb.expert_location_updater: _execute_p2p_ops
```

原因：torch_npu 在 910C 上走的是“逐个发 `isend`”的 coalescing 路径，而 HCCL 的 `send` 是阻塞式的，前面的 `isend` 没发完，后面的 `irecv` 永远发不出来，于是死锁。

解决：把 torch_npu 的 `batch_isend_irecv` 强制改成底层**原子批量收发**路径：

- 补丁文件：`patches/torch_npu_distributed_c10d_patched.py`
- 挂载到容器里的：
  `/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py`

### 3.3 最优组合：两个补丁一起用，NZ 保持开启

验证结论：**两个补丁可以一起用，而且这是最干净的方案。**

- 补丁 1（`patches/expert_location_updater_patched.py`）负责在 EPLB 搬专家时，把 NZ 格式临时转成 ND，搬完再转回 NZ；
- 补丁 2（`patches/torch_npu_distributed_c10d_patched.py`）负责把 P2P 通信改成原子批量收发，避免死锁。

这样 **NZ 保持开启**，不需要再设置 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`，正常推理仍然走 NZ 快速路径。

> 之前说的“关 NZ”方案也能跑，但会让正常推理变慢，而且和旧基线（NZ 开启）不可比。现在用两个补丁 + NZ 开启，EPLB 组和无 EPLB 基线可以直接公平对比。

## 4. 当前可用配置

启动脚本（都在自己的目录里）：

- `serve_glm52_ep64_decode.sh`：单机启动命令，挂载两个补丁，NZ 保持开启；
- `start_ep64_decode.sh`：把两个补丁和启动脚本推到四台机器并拉起；
- `stop_ep64_decode.sh`：四台一起停。

服务当前状态：**已拉起，健康检查 200，EPLB 可用，NZ 开启**。

## 5. 已跑通的结果

### 5.1 EPLB 开启（NZ 开启 + 两个补丁，推荐配置）

数据目录：`dataset/ep_runs/dolly_ep64_0907_051628`

- 32 条 Dolly，并发 4，`max_tokens=256`
- 成功 32 / 32，错误 0
- 总耗时 wall：200.0s
- e2e：P50 19.80s，P90 65.08s，P99 65.68s
- tokens/s：60.2
- EPLB rebalance 完成耗时：约 48.4s（rebalance 期间在途请求会变慢）

### 5.2 EPLB 开启（NZ 关闭 + torch_npu 补丁，验证用）

数据目录：`dataset/ep_runs/dolly_ep64_0907_045935`

- 32 条 Dolly，并发 4，`max_tokens=256`
- 成功 32 / 32，错误 0
- 总耗时 wall：214.0s
- e2e：P50 22.18s，P90 63.81s，P99 63.87s
- tokens/s：55.1
- EPLB rebalance 完成耗时：约 44.9s（rebalance 期间在途请求会变慢）

### 5.3 无 EPLB 基线（NZ 开启，旧数据）

数据目录：`dataset/ep_runs/dolly_ep64_0907_015148`

- 32 条 Dolly，并发 4，`max_tokens=256`
- 成功 32 / 32，错误 0
- 总耗时 wall：128.7s
- e2e：P50 16.45s，P90 16.78s，P99 17.17s
- tokens/s：93.5

> 这组基线和推荐配置的 EPLB 组（5.1）都是 **NZ 开启**，可以直接对比。

### 5.4 大 EP 下 expert 不均衡（基线观察）

从无 EPLB 基线看，expert 被选中的次数很不均衡：

- `logical_count`：shape `[2058, 78, 256]`，即 2058 步 × 78 层 × 256 个专家
- 原始 expert 计数：全局 CV ≈ 3.59，peak/mean ≈ 58.6
- 按 rank 汇总后的负载：CV ≈ 0.846，peak/mean ≈ 5.71

这正是要展示的“大 EP 下 expert 不均衡”现象。

## 6. 下一步

1. 已确认最优配置：NZ 开启 + 两个补丁；
2. 把“无 EPLB”基线（NZ 开启）和“有 EPLB”（NZ 开启 + 两个补丁）两组对比：
   - expert/rank 负载不均衡；
   - e2e 延迟、吞吐；
   - rebalance 耗时；
3. 最后整理成最终实验报告。
