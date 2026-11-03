# M0 验证记录

- 执行日期：2026-11-03
- 执行边界：仅 CPU 逻辑和源码静态检查；**未启动 NPU 实验，未重启模型服务**
- 容器：`m0-inspect`，无 NPU device，`--network none`，只读源码检查
- 源码基线：实际镜像 v0.5.18-cann9.0.0-a3
- 基础 `topk.py` SHA256：`f82f25bb343892b7b8b1eb833a4e6ca376a510ac74a3a935e0a1997dbd072f75`

## 1. 执行命令

```bash
./tests/run_cpu_tests.sh
patch -p1 -d apply_check < patches/prefill_copy_aware_m0.patch
python3 tests/npu_validate_prefill_copy_aware.py
```

## 2. CPU 测试结果

| 测试 | 结果 | 覆盖 |
|---|---|---|
| invalid/padding/shared/empty conservation | PASS | 无效 ID、padding、shared 列、空输入、有效分支总数 |
| multi-source 6+4 quota=7 | PASS | 6+1；q=0/10；来源交换 |
| logical/physical remap | PASS | 只改物理位置，原逻辑 expert 与权重不丢 |
| cold vs cached | PASS | cached 不复制，cold 计 prepare 成本，收益不同 |
| unknown cost / no gain | PASS | 未知成本或低于阈值不执行 |
| slot lifecycle | PASS | reader drain、READY、epoch commit、旧 epoch 拒绝 |
| slot failure | PASS | 失败进入 FAILED，不发布部分状态 |
| routing changes plan | PASS | 448 配额由当前 960/64 输入计算，非硬编码 |
| executor consumes Plan | PASS | 实际输出物理 ID 由 Plan 驱动，448 个任务到目标槽 |
| shadow | PASS | 生成计划但不复制、不改 ID |
| default off | PASS | 不统计、不改变 ID |
| unsupported layer / no-op | PASS | 回退原 ID；多虚拟 rank 对照一致 |

汇总：

```text
CPU unit tests: 12 passed
Patch integration tests: 3 passed
py_compile: passed
patch -p1 apply check: passed
```

## 3. Patch dry-run / apply check

- 对基础 `topk.py` 执行 `diff -ruN` 生成补丁；
- 在仅含基础 `topk.py` 的临时树执行 `patch -p1`；
- 应用后两个源文件与 `patched_tree/` 逐字节一致。

## 4. NPU 验证入口

命令：

```bash
python3 tests/npu_validate_prefill_copy_aware.py
```

本轮实际输出：

```json
{
  "status": "READY_NOT_RUN",
  "reason": "--execute not supplied",
  "device_touched": false,
  "service_started": false
}
```

入口在执行时还要求：

```bash
SGLANG_PREFILL_COPY_AWARE_TEST_ONLY=1
```

并且必须显式传 `--execute`。它只做 Plan→quota→physical-ID→sentinel transport 的小型测试；即使执行，也不等于完整量化状态或 HCCL 验证。

## 5. 未执行/未通过项

- `--execute` NPU 入口：**未执行**
- 完整量化专家状态复制：**未实现**
- NZ→ND staging 和恢复：**未实现**
- 真实 HCCL/P2P 通信：**未实现**
- EP 全局计数交换：**未实现**
- 真实 dispatch/combine：**未接完整模型**
- 生产 graph 等价性：**未验证**
- 额外槽显存/HBM：**未分配、未测**
- 端到端精度、TTFT、吞吐、SLO：**未测**

## 6. 结论分级

| 项目 | 级别 |
|---|---|
| CPU Plan/配额/守恒/槽状态逻辑 | 实现并通过 CPU 测试 |
| 实际服务器源码 hook | 已定位并打补丁，补丁可应用 |
| NPU 上完整功能链 | 未执行 |
| 生产环境加速 | 未验证、不可声明 |

本轮结论只能写成：

> **M0 代码实现完成，CPU 逻辑验证通过；NPU/HCCL/生产 graph 待验证。**
