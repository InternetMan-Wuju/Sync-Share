# DeepEP 修复包的工作区副本

复制日期：2026-09-27 21:44，北京时间。

## 内容和来源

原目录：`/mnt/raid/user/data/tools/bench3/kfix-build/deepep_fix_pkg/deep_ep`。

已完整复制原目录的 237 个文件（8,319,830 字节，约 7.93 MiB）：Python 代码、C++ 扩展 `.so`、opapi 库、Ascend 算子二进制及随包提供的头文件。原目录没有提供完整 C++ 工程和编译仓库；此处保存的是现成的运行包。复制时以只读挂载读取原目录，没有改写原包。

- `deep_ep/`：可供挂载的完整目录。
- `package.tar`：复制时保存的完整归档。
- `manifest.json`：来源、复制时间、每个文件的大小和 SHA256。
- `copy_verified.json`：副本与原目录逐文件校验一致的记录。
- `verify_copy.py`：挂载前检查文件集合、大小和 SHA256。
- `import_check.sh`：在镜像内部确认 Python 导入路径、扩展 SHA256 和接口。

校验命令：

```bash
python3 /home/chengyetai/workspace/current_exp/chunk_manual_20260927/patches/deepep_fix_20260927/verify_copy.py
```

## 外挂方式

`serve.sh` 的 `DEEPEP_FIX=1` 开关会将工作区副本**只读**挂载到：

```text
/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep
```

镜像根目录的 `deep_ep_cpp...so` 是指向包内扩展的符号链接，因此整个包挂载同时替换了实际导入的扩展和算子包。镜像、原始模型均不改写。默认 `DEEPEP_FIX=0`，使用镜像自带的 DeepEP。

容器不存在、端口空闲时可启动：

```bash
DEEPEP_FIX=1 bash /home/chengyetai/workspace/current_exp/chunk_manual_20260927/serve.sh start 25 8192
```

## 本次对照

2026-09-27 仅在 25 启用，运行标识 `20260927_215657`；23 继续运行原服务。25 的模型仍为 `/mnt/raid/user/data/models/glm-qq`，单机 TP16/EP16、Chunk=8192、MTP OFF、DeepEP normal；其他启动参数保持一致。

本次包内扩展 SHA256：

```text
62f7915d337a4531d9a4ca7264398c68417f9c310c587af0687ac2cf0fdb0ae2
```

比较记录：[deepep_ab](../../semantic_debug_20260927/deepep_ab/)。初步源码分析：[deepep_fix_review](../../semantic_debug_20260927/deepep_fix_review/)。先检查同一中文提示的回复；性能实验仍需等语言检查通过。
