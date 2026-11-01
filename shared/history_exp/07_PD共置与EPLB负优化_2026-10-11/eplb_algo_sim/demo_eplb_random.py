#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPLB 摆放演示：随机热度，先看"没有 EPLB"，再看 EPLB 逐层的结果。

规模（真实系统的等比缩小）:
    真实: 256 专家 / 64 rank / 每 rank 4 槽 / 78 层
    这里:  32 专家 /  8 rank / 每 rank 4 槽 /  4 层

随机数据怎么造: 每层一条长度 32 的随机序列，另外随机挑 1 个专家当"热点专家"，
让它一个人占掉本层约一半的热度（真实数据里最热专家 ≈ 每 rank 平均值的 7 倍，
这里 8 个 rank 的规模下这个上限就是 4 倍）。

每一层都会被"分解成子矩阵"：某个 rank 拿到的 4 个数 = 一个子矩阵，下方标明所属 rank。

跑法:  python3 demo_eplb_random.py
"""

import random

RANKS = 8          # rank 数
K     = 4          # 每个 rank 的槽数
E     = RANKS * K  # 专家数 = 32
L     = 4          # 层数
SEED  = 20261012
W4    = 4          # 打印宽度


# ------------------------------- 造数据 -------------------------------
def make_heat(rng):
    """一层 E 个专家的随机热度：随机挑 1 个热点专家，让它占本层约一半"""
    heat = rng.sample(range(1, 401), E)      # 互不相同的随机小热度
    e_hot = rng.randrange(E)                       # 热点专家是随机的（每层不同）
    heat[e_hot] = int(sum(heat) * rng.uniform(0.8, 1.2))
    assert len(set(heat)) == E
    return heat, e_hot


# ------------------------------ 摆 放 ------------------------------
def place_no_eplb(heat):
    """没有 EPLB：槽 p 装专家 p  →  rank r 拿连续的 K 个专家"""
    return [[r * K + s for r in range(RANKS)] for s in range(K)]


def place_snake(heat):
    """EPLB：每层自己排序(热->冷)，排成 (K x RANKS)，奇数行反转(蛇形)"""
    order = sorted(range(E), key=lambda e: heat[e], reverse=True)
    M = [[-1] * RANKS for _ in range(K)]
    for m, e in enumerate(order):
        s, r = divmod(m, RANKS)
        if s % 2 == 1:                              # 奇数行反转
            r = RANKS - 1 - r
        M[s][r] = e
    return M


# ------------------------------ 打 印 ------------------------------
def show_layer(tag, heat, e_hot, M):
    num  = [[heat[e] for e in M[s]] for s in range(K)]
    load = [sum(num[s][r] for s in range(K)) for r in range(RANKS)]
    tot  = sum(heat)

    print(f"\n--- {tag}")
    print("   本层热度   " + " ".join("%5d" % v for v in heat))
    print("   专家号     " + " ".join("%5d" % e for e in range(E)))
    print("   → 热点专家 e%-2d 热度 %d (占本层总量 %.0f%%)；每个热点专家自己 = 每 rank 平均的 %.1f 倍"
          % (e_hot, heat[e_hot], 100 * heat[e_hot] / tot, heat[e_hot] / (tot / RANKS)))
    print()
    for s in range(K):
        print("      槽%d   " % s + " ".join("[%*d]" % (W4, num[s][r]) for r in range(RANKS)))
    print("      所属 rank  " + " ".join(" %-4s" % f"r{r}" for r in range(RANKS)))
    print("      负载       " + " ".join("[%*d]" % (W4, load[r]) for r in range(RANKS))
          + "   max/mean = %.2fx" % (max(load) / (sum(load) / RANKS)))
    return load


def main():
    rng = random.Random(SEED)
    data = [make_heat(rng) for _ in range(L)]

    print("=" * 92)
    print(f"随机热度: {L} 层 x {E} 专家 (种子 {SEED})，每层的热点专家位置随机")
    print("=" * 92)

    # ---------------- PART 1: 没有 EPLB ----------------
    print("\n" + "=" * 92)
    print(f"PART 1 —— 没有 EPLB：槽 p 装专家 p，于是 rank r 拿连续的 {K} 个专家 (e{K}r .. e{K}r+{K-1})")
    print("=" * 92)
    tot0, hot_rank0 = [0] * RANKS, []
    for l, (heat, e_hot) in enumerate(data):
        ld = show_layer(f"层{l}", heat, e_hot, place_no_eplb(heat))
        hot_rank0.append(e_hot // K)
        tot0 = [a + b for a, b in zip(tot0, ld)]
    print("\n   热点专家各在哪: " + "  ".join(f"层{l}->r{r}" for l, r in enumerate(hot_rank0)))
    print("   4 层合计  " + " ".join("[%4d]" % v for v in tot0)
          + "   max/mean = %.2fx" % (max(tot0) / (sum(tot0) / RANKS)))

    # ---------------- PART 2: 有 EPLB ----------------
    print("\n" + "=" * 92)
    print("PART 2 —— 有 EPLB：每层独立排序 -> (槽 x rank) 矩阵 -> 奇数行反转 -> 每列给一个 rank")
    print("=" * 92)
    tot1, hot_rank1 = [0] * RANKS, []
    for l, (heat, e_hot) in enumerate(data):
        ld = show_layer(f"层{l}", heat, e_hot, place_snake(heat))
        hot_rank1.append(ld.index(max(ld)))
        tot1 = [a + b for a, b in zip(tot1, ld)]
    print("\n   热点专家各在哪: " + "  ".join(f"层{l}->r{r}" for l, r in enumerate(hot_rank1)))
    print("   4 层合计  " + " ".join("[%4d]" % v for v in tot1)
          + "   max/mean = %.2fx" % (max(tot1) / (sum(tot1) / RANKS)))

    # ---------------- 对比 ----------------
    print("\n" + "=" * 92)
    print("对比")
    print("=" * 92)
    f = lambda t: max(t) / (sum(t) / RANKS)
    print("   没有 EPLB   合计 = %-34s max/mean = %.2fx" % (str(tot0), f(tot0)))
    print("   有   EPLB   合计 = %-34s max/mean = %.2fx" % (str(tot1), f(tot1)))
    print(f"\n   没有 EPLB: 每层的热点专家各自落在 {sorted(set(hot_rank0))} —— 各自的尖峰落在不同的 rank 上，互相抵消")
    print(f"   有   EPLB: 每层的尖峰都被挪到了 r{hot_rank1[0]} —— 4 层的尖峰叠在同一列，抵消消失")


if __name__ == "__main__":
    main()
