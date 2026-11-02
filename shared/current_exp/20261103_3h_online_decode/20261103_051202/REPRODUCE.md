# 最小复现

环境：仅绑定 `/dev/davinci0`；镜像和 publisher 恢复层与旧实验一致。
容器挂载：

```text
/exp   -> 本 run 目录
/old   -> 20261102_load_balance_e0（只读）
/models/glm-q
/study/weights/{loader,publisher}
```

必须设置：

```bash
export PYTHONPATH=/study/weights/loader:/sgl-workspace/sglang/python
export SGLANG_READONLY_WEIGHT_MANIFEST=/study/weights/publisher/manifest.json
export SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5
export SGLANG_READONLY_WEIGHT_BASE=/models/glm-q
export SGLANG_READONLY_WEIGHT_AUDIT_DIR=/exp/weight_audit
```

执行：

```bash
python3 cpu_audit.py
python3 online_npu.py > ONLINE_RESULT.stdout 2>&1
python3 decode_npu.py > DECODE_RESULT.stdout 2>&1
python3 package_results.py
```

`online_npu.py`、`decode_npu.py` 复用旧实验的 `s1_npu_numerics.py`、
`s2_local_cost.py` 和 `sglang.srt.hardware_backend.npu.utils.npu_format_cast`。
所有本地链均为 eager 局部代理，不是已验证的生产 graph 封装。
