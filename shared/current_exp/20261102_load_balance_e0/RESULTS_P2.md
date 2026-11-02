# P2：共享专家与单槽生命周期本地检查

状态：**完成（单 NPU）**
日期：2026-11-03

## 共享专家是否存在

layer 49 的 `mlp.shared_experts` 在检查点中有完整 9 个 W8A8_DYNAMIC 张量：

```text
gate_proj / up_proj / down_proj
× weight / weight_scale / weight_offset
```

完整权重及 scale/offset 约 37.77 MB。

源码检查显示：DeepEP 路径把 shared expert 设为 `tp_rank=0, tp_size=1`；本检查没有启动完整服务，因此不声称验证了运行态分片或常驻布局。E0 探针使用 `disable_shared_experts_fusion=true`，shared expert 是独立 MLP 路径。

## 单卡成本

layer 49 shared expert 对 routed expert 的局部链：

| rows | shared (ms) | routed (ms) | shared/routed |
|---:|---:|---:|---:|
| 1 | 0.231 | 0.217 | 1.066 |
| 128 | 0.217 | 0.196 | 1.112 |
| 1024 | 0.320 | 0.323 | 0.991 |
| 4096 | 1.103 | 1.104 | 0.999 |

shared expert 与小批量 routed expert 的单次成本同量级。它可以形成真实可移动工作，但移动后会增加内存、派发和同步成本。

## 双 stream 重叠

用两个本地 stream 同时执行 shared 和单 routed 专家链：

| rows | shared | routed | 顺序和 | 双 stream | 相对顺序和 | 理想 max |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.320 | 0.323 | 0.644 | 0.848 | **−31.7%** | 0.323 |
| 4096 | 1.103 | 1.104 | 2.207 | 1.825 | **+17.3%** | 1.104 |

解释：

- 1024 行时，stream 启动/等待和资源竞争超过可隐藏工作，结果是负收益。
- 4096 行时出现部分重叠，但只回收了顺序时间约 17.3%，远低于理想 50%。
- 当前源码中 NPU prefill extend 模式可启用 shared/routed 双 stream；decode 分支不启用。
- 因此 shared work 在 prefill 中可能已被部分隐藏；若要在 decode 里移动它，必须看完整 decode 时间线。

## 单槽生命周期

用 layer 49 shared expert 和 layer 48 expert0 交替装入同一组预分配 slot：

- 旧读者完成后才覆盖：**PASS**
- 版本单调递增：**PASS**
- layer49 shared 输出逐位一致：**PASS**
- layer48 expert0 复用后输出逐位一致：**PASS**

局部搬运/切换成本：

| 操作 | 中位数 |
|---|---:|
| NPU→NPU 复制一个完整专家 | 0.117 ms |
| H2D 复制一个完整专家 | 1.806 ms |
| 仅 pointer switch（host） | 0.0003 ms |

`pointer switch` 只是引用切换，不是权重传输，不能拿来代表跨 rank 复制成本。

## 结论

- shared expert 的工作量真实存在，且在 prefill 大 batch 下已有部分可重叠空间。
- 单卡不能回答“把 shared work 移到空闲 rank 是否净收益”；需要多卡完整 layer timeline。
- 单槽覆盖和版本检查在本地通过；跨 rank 搬运、旧读者完成通知和并发正确性仍未验证。

## 产物

- `results/P2_LOCAL_CHECKS.json`
- `scripts/p2_local_checks.py`
