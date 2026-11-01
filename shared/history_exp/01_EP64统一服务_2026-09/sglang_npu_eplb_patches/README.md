# SGLang Ascend NPU EPLB 双补丁说明

本目录提供两个补丁，用于在 **Ascend 910C + CANN 9.0.0** 上跑 **GLM-5.2（w8a8）的 EPLB（专家负载均衡）** 时，修复两个致命问题。

> 环境：镜像 `quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16`，Ascend 910C（4 机 × 16 卡），DeepEP a2a 后端，TP=EP。

## 目录内容

```text
README.md
expert_location_updater_patched.py        # 补丁 1：SGLang 侧
torch_npu_distributed_c10d_patched.py     # 补丁 2：torch_npu 侧
```

---

## 问题 1：EPLB rebalance 直接崩溃

### 症状

服务端日志报：

```text
RuntimeError: For a tensor of internal format, it's storage_offset must be 0
[ERROR] ERR02007 DIST feature not supported
```

崩溃点在 `sglang/srt/eplb/expert_location_updater.py` 的 `_execute_p2p_ops`，即 EPLB 跨卡搬运专家权重时。

### 原因

Ascend 上 MoE 权重默认被转成 **FRACTAL_NZ**（NPU 内部格式）。EPLB 重排专家时，会从这份 NZ 权重里切出“一个专家的一行”直接 `isend`。HCCL 不接受内部格式的张量（要求 `storage_offset=0`），于是崩溃。

### 修复

**补丁 1**（`expert_location_updater_patched.py`）：

- 发送前把专家权重从 FRACTAL_NZ 转回普通 **ND** 格式；
- 接收侧临时缓冲也改成普通 **ND** 连续内存；
- 收完再拷回 NZ 权重，正常推理仍保持 NZ（不降速）。

关键改动：

```python
from sglang.srt.utils import get_bool_env_var, is_npu

def _p2p_send_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if is_npu():
        from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast
        return npu_format_cast(tensor, NPUACLFormat.ACL_FORMAT_ND)
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor
```

```python
def create_temp_buffers(sample_tensors):
    return [
        torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        for tensor in sample_tensors
    ]
```

```python
P2POp(
    op=torch.distributed.isend,
    tensor=_p2p_send_tensor(
        _get_tensor(routed_experts_weights, i, src_expert_location)
    ),
    peer=dst_rank,
)
```

---

## 问题 2：EPLB rebalance 卡死（没有 `rebalance end`）

### 症状

问题 1 修掉后（或关 NZ 后），日志里出现 `[EPLBManager] rebalance start`，但一直等不到 `rebalance end`，服务健康检查变 000，进程空转。

### 原因

torch_npu 的 `_batch_isend_irecv` 在 910C 上走的是 **coalescing-manager 路径**：把 `isend`/`irecv` 一个个发出去。而 HCCL 的 `send` 是阻塞式的——前面某个 `isend` 还没发完，后面的 `irecv` 永远发不出来，于是死锁。

### 修复

**补丁 2**（`torch_npu_distributed_c10d_patched.py`）：

把 NPU 的 P2P 改成底层 **原子批量收发** `_group.batch_isend_irecv(...)`，一次提交所有 send/recv，避免逐个阻塞。

关键改动（`_batch_isend_irecv` 的 `device.type == "npu"` 分支）：

```python
elif device.type == "npu":
    global npu_device_name
    npu_device_name = npu.get_device_name()
    if group is None:
        group = _get_default_group()
        is_multi_pg = False
    _group = group._get_backend(device)

    # Always use the atomic HCCL batch P2P path（不再走 coalescing-manager）
    op_type = []
    tensors = []
    remote_rank_list = []
    for p2p_op in p2p_op_list:
        if p2p_op.tensor.device.type != "npu":
            deviceType = p2p_op.tensor.device.type
            raise RuntimeError(...)
        op_type.append(p2p_op.op.__name__)
        tensors.append(p2p_op.tensor)
        rank_for_op = get_group_rank(group, p2p_op.peer) if is_multi_pg else p2p_op.peer
        remote_rank_list.append(rank_for_op)
    return [_group.batch_isend_irecv(op_type, tensors, remote_rank_list)]
```

---

## 使用说明

### 1. 把两个补丁文件放到每台机器

例如放到每台节点的 `/tmp/`：

```bash
scp expert_location_updater_patched.py  <node>:/tmp/expert_location_updater_patched.py
scp torch_npu_distributed_c10d_patched.py <node>:/tmp/torch_npu_distributed_c10d_patched.py
```

### 2. 启动容器时挂载覆盖原路径

在 `docker run` 里加两个只读挂载：

```text
-v /tmp/expert_location_updater_patched.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro
-v /tmp/torch_npu_distributed_c10d_patched.py:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py:ro
```

### 3. 保持 NZ 开启

**不要**设置 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`。两个补丁一起用即可，正常推理仍走 FRACTAL_NZ 快速路径。

### 4. 启动参数（EPLB 相关）

```text
--enable-eplb
--expert-distribution-recorder-mode stat
--expert-distribution-recorder-buffer-size -1
```

> 可选：`export SGLANG_LOG_EXPERT_LOCATION_METADATA=1` 会在每次 rebalance 时把 `physical_to_logical_map` 打到日志，便于验证 EPLB 布局。

---

## 效果（GLM-5.2，4 机 × 16 卡，EP64）

- 补丁后 EPLB 能正常完成 rebalance，不再崩溃/死锁；
- 首次 rebalance（从恒等布局重排）约 45.7s，后续增量 rebalance 约 1.5s；
- rank 负载不均衡从 CV ≈ 0.86（无 EPLB）降到 CV ≈ 0.54（EPLB）。

## 注意

- 两个补丁是**运行期挂载覆盖**，不改公共镜像；
- 若只想要“能跑、不计较推理性能”，也可以用 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1` 关掉 NZ 绕过问题 1，但正常推理会变慢；
- 本补丁针对 `cann9.0.0-a3-v0.5.16` 镜像内的路径，升级镜像后需确认原文件路径/逻辑是否变化。
