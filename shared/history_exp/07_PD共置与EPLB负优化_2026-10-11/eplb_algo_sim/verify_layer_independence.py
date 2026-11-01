#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：stock 的 EPLB 算法到底是不是【逐层独立】的？

做法（操作性证明，不靠读代码猜）：
   同一份数据，三种方式各跑一遍，逐层比对摆放结果：
     A) 78 层一起跑                 （官方用法）
     B) 每次只喂 1 层，跑 78 次      （若层间有耦合，结果会变）
     C) 把 78 层打乱顺序再一起跑     （若结果依赖层序，说明有跨层状态）
   三种结果全部一致  ⇒  层与层之间没有任何耦合。

被测算法（全部是从镜像里原样取出的 stock 代码，未做修改）：
   ① deepseek                 —— 我们这次实际跑的（auto 选中）
   ② deepseek hierarchical    —— 层级版对照
   ③ deepseek_vec             —— 向量版（带蛇形）
   ④ deepseek_vec 冗余=64      —— 320 个物理槽

数据：真实 dump 的热度 (78 层 x 256 专家，按 step 累计) + 一组随机数据。
跑法: python3 verify_layer_independence.py
"""
import sys, glob, pathlib
import torch

HERE  = pathlib.Path(__file__).resolve().parent
STOCK = HERE.parent / "pd_colloc_20261011" / "patches" / "stock_alg_ref"
sys.path.insert(0, str(STOCK))
import deepseek, deepseek_vec

WIN = "/home/chengyetai/workspace/current_exp/pd_colloc_20261011/results/20261012_000935/windows/*.pt"
L, E, EP = 78, 256, 64


def load_real():
    f = sorted(glob.glob(WIN))[0]
    o = torch.load(f, map_location="cpu", weights_only=False)
    return o["logical_count_decode"].to(torch.float32), pathlib.Path(f).name


def load_random(seed=7):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(1, 400, (200, L, E), generator=g).float()
    t[:, :, ::37] += 5000                    # 塞几个热点，避免退化成均匀分布
    return t, f"random(seed={seed})"


def run_deepseek(steps, hier):
    return deepseek.rebalance_experts(
        weight=steps.sum(0), num_replicas=E,
        num_groups=(8 if hier else 1), num_nodes=(4 if hier else 1),
        num_gpus=EP, enable_hierarchical=hier)


def run_vec(steps, redundant, hier):
    return deepseek_vec.rebalance_experts(
        tokens_per_expert=steps, num_physical_experts=E + redundant,
        num_local_physical_experts=(E + redundant) // EP,
        num_groups=(8 if hier else 1), num_nodes=(4 if hier else 1),
        enable_hierarchical=hier)


CASES = [
    ("① deepseek（我们跑的）",      lambda s: run_deepseek(s, False)),
    ("② deepseek hierarchical",    lambda s: run_deepseek(s, True)),
    ("③ deepseek_vec",            lambda s: run_vec(s, 0, False)),
    ("④ deepseek_vec 冗余=64",     lambda s: run_vec(s, 64, False)),
]


def check(tag, full, single, shuffled, perm, out):
    """full/single/shuffled 都是 phy2log: [L, P]；shuffled 的第 i 行对应层 perm[i]"""
    n_single = int((full != single).any(-1).sum())
    n_shuf   = int((full != shuffled[perm.argsort()]).any(-1).sum())
    ok = (n_single == 0) and (n_shuf == 0)
    out.append((tag, n_single, n_shuf, ok))
    print(f"   {tag:26s} 单层逐跑: {'✅ 完全一致' if n_single==0 else f'❌ {n_single} 层不同'}"
          f"   打乱层序: {'✅ 完全一致' if n_shuf==0 else f'❌ {n_shuf} 层不同'}")
    return ok


def main():
    all_ok = True
    for loader, kind in [(load_real, "真实数据"), (load_random, "随机数据")]:
        steps, name = loader()
        steps = steps[:20]
        print("=" * 84)
        print(f"{kind}: {name}   steps={tuple(steps.shape)}   ({L} 层 x {E} 专家)")
        print("=" * 84)
        perm = torch.randperm(L, generator=torch.Generator().manual_seed(0))
        for tag, fn in CASES:
            full = fn(steps)[0]
            single = torch.cat([fn(steps[:, l:l+1, :])[0] for l in range(L)], 0)
            shuffled = fn(steps[:, perm, :])[0]
            all_ok &= check(tag, full, single, shuffled, perm, [])
        print()

    print("=" * 84)
    print("结论: " + ("✅ 全部一致 —— 逐层独立，层与层之间没有任何耦合"
                    if all_ok else "❌ 存在不一致，逐层独立的假设不成立"))
    print("=" * 84)


if __name__ == "__main__":
    main()
