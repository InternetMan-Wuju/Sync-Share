# 补丁应用与撤销

## 目标源码根

```text
/sgl-workspace/sglang/python
```

先验证基础文件：

```bash
sha256sum /sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/moe/topk.py
# 必须等于：
# f82f25bb343892b7b8b1eb833a4e6ca376a510ac74a3a935e0a1997dbd072f75
```

## 应用前检查

```bash
cd /sgl-workspace/sglang/python
patch -p1 --dry-run < <run-dir>/patches/prefill_copy_aware_m0.patch
```

## 应用

```bash
cd /sgl-workspace/sglang/python
patch -p1 < <run-dir>/patches/prefill_copy_aware_m0.patch
python3 -m py_compile \
  sglang/srt/eplb/prefill_copy_aware.py \
  sglang/srt/hardware_backend/npu/moe/topk.py
```

默认不启用：

```bash
unset SGLANG_PREFILL_COPY_AWARE_MODE
# 等价于 off
```

## 撤销

```bash
cd /sgl-workspace/sglang/python
patch -R -p1 < <run-dir>/patches/prefill_copy_aware_m0.patch
```

若 reverse patch 留下已为空的 `prefill_copy_aware.py`，人工确认内容为空后删除该文件即可。不要删除或覆盖其他服务器文件。

## 重要限制

- 不要在本轮启动服务或执行 NPU 实验。
- `active` 模式即使设置环境变量，在没有 `register_runtime_controller` 和有效成本/transport 配置时仍会回退原路径。
- 该补丁尚未包含真实 HCCL、完整量化状态传输、NZ/ND staging、EP 计数交换和真实 graph 接线。
