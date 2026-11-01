# Share：本地代理与远程高级模型协作仓库

这个仓库用于在本地代理和远程高级模型之间同步**代码、文档、补丁、小型结构化结果和交接说明**。
大型权重、原始日志、数据集和运行结果不要提交到 Git。

## 目录约定

- `shared/`：双方共同维护的代码、配置、文档和可复现实验脚本。
- `handoff/`：阶段交接说明、待办、问题定位和下一步计划。
- 仓库根目录的 Git 历史：双方按普通分支/提交方式同步。

## 建议工作流

本地或远程每次开始工作前：

```bash
git pull --rebase
```

完成一小块工作后：

```bash
git add <files>
git commit -m "描述本次改动"
git push
```

不要提交：

- `*.safetensors`、模型权重和其他 GB 级二进制文件
- 原始实验日志、数据集、运行输出
- SSH 私钥、token、密码、内部凭证
- 大型临时缓存和虚拟环境

大文件仍放在 workspace 或 `/data_lib`，Git 仓库只保存路径说明、索引和复现所需的小文件。

## 远程仓库

当前尚未配置 `origin`。获得远程仓库地址后执行：

```bash
git remote add origin <remote-url>
git push -u origin main
```

远程高级模型使用同一 remote 克隆后，即可通过 `pull` / `commit` / `push` 与本地协作。
