# M0 外挂验证记录

## 1. CPU

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_cpu_logic.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_external_install.py
```

- CPU 逻辑：12/12 PASS
- 外挂安装静态检查：3/3 PASS
- 覆盖 invalid/padding/shared/empty、多来源 quota、q=0/n、cold/cache、unknown/no-gain、slot lifecycle、shadow、Plan→物理 ID、unsupported/no-op。

日志：`artifacts/cpu_tests.log`。

## 2. NPU 实机

命令：

```bash
PYTHONPATH=<run-dir>:/sgl-workspace/sglang/python \
python3 -m m0_external.npu_smoke
```

实际设备：`Ascend910_9382`，device 0。
无完整模型，无服务启动。

结果：

- off 目标槽计数：0
- shadow 与 off 完全相同：true
- shadow transport：0
- active quota：448
- active 目标槽 33 计数：448
- active transport：1
- slot 校验：PASS
- top-k weights finite：true
- status：`NPU_PASS_TEST_ONLY`

输出：

- `artifacts/npu_smoke.json`
- `artifacts/npu_smoke_saved.stdout`

## 3. 未验证

完整量化状态跨卡复制、HCCL/P2P、全局计数同步、真实 DeepEP dispatch/combine、生产 graph、额外槽 HBM、完整模型精度和服务性能均未验证。

结论必须写成：**外挂代码通过 CPU 和单 NPU test-only smoke；硬件完整链路尚未通过。**


## 4. 完整本地链 NPU 验证

命令：

```bash
python3 -m m0_external.full_local_smoke
```

结果：`NPU_FULL_LOCAL_PASS_TEST_ONLY`

- top-k：top-8，512 tokens
- e0 路由：off 512；active primary 256；active destination 256
- Plan q：256
- 复制运行态：37,810,176 bytes
- w13/down/scales 与源逐位一致
- offset 全零
- e43 加权贡献与基线逐位一致，NRMSE 0
- transport 调用 1

输出：`artifacts/full_local_smoke.json`、`artifacts/full_local_smoke_v5.stdout`。

这仍是 test-only 本地路径，不是完整模型或跨卡验证。

## 5. 单 NPU 正收益本地代理

命令：

```bash
python -m m0_external.positive_case
```

固定输入：`4096/64 -> 2080/2080`，迁移配额 `q_global=2016`。
设备：`Ascend910_9382`，单张 NPU，虚拟 rank 顺序执行。

| 项目 | 结果 |
|---|---:|
| old max 计算代理 | `1.115509 ms` |
| new max 计算代理 | `0.564320 ms` |
| 计算节省 | `0.551189 ms` |
| planner | `0.016281 ms` |
| 本地完整 state copy | `0.126537 ms` |
| 扣除后净值 | **`+0.408371 ms`** |
| copy bytes | `37,810,176 B` |
| 状态 | `LOCAL_POSITIVE_TEST_ONLY` |

正确性：Plan 实际迁出 2016；w13/down/scales 与源逐位一致；offset 全零；
所有正式块 `>=100 ms`。原始输出和结构化结果分别是
`artifacts/positive_case.stdout` 与 `artifacts/positive_case.json`。

该结果证明本地计算收益足以覆盖 planner 和本地 state copy；不证明跨卡
HCCL/P2P、dispatch/combine、并行争用或生产服务收益。后续同机双 NPU 验证
不需要多机。
