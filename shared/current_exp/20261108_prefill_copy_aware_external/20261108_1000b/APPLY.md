# 外挂式启动与撤销

不修改 `/sgl-workspace` 源码。

```bash
export PYTHONPATH=<run-dir>:/sgl-workspace/sglang/python:$PYTHONPATH
export SGLANG_PREFILL_COPY_AWARE_MODE=off
python -m m0_external.launch_server <原有启动参数>
```

`launch_server.py` 会先调用 `install()`，再启动原 SGLang 主模块。默认 `off` 时 mapping wrapper直接调用原函数，不做新增统计、不复制、不改物理 ID。

撤销：终止进程即可；没有原文件需要反向 patch。

`active` 仅限测试 runtime；没有显式注册 cost model、transport 和 EP commit 实现时不得用于服务。
