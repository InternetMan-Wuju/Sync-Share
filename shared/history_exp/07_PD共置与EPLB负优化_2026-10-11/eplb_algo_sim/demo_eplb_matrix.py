#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPLB 摆放算法最小演示 —— 只有矩阵和数字。

设定（把真实系统等比缩小）:
     真实:  64 个 rank x 每 rank 4 槽 = 256 槽 , 78 层 , 256 个专家
     这里:   4 个 rank x 每 rank 2 槽 =   8 槽 ,  2 层 ,   8 个专家

W[l][e] = 第 l 层里专家 e 的热度(token 数)。每个专家只能放一个槽。
"摆放" = 决定哪个专家进哪个槽；槽 p 属于 rank p // 2  （p = 0..7）

每个摆放结果都印成同一张表:
            r0    r1    r2    r3     <- 列 = rank
   槽0:   ...   ...   ...   ...     <- 行 = 该 rank 的第几个槽
   槽1:   ...   ...   ...   ...
下面两行是每 rank 的负载(列相加)和 max/mean。

跑法:  python3 demo_eplb_matrix.py
"""

RANKS = 4                      # rank 数
K     = 2                      # 每个 rank 的槽数
E     = RANKS * K              # 专家数 = 8
W     = [
    [100, 20, 10, 5, 1, 1, 1, 1],      # 层0: 热度分布 (100,20,10,5,1,1,1,1)，热点在专家 0
    [1, 1, 1, 1, 5, 10, 20, 100],      # 层1: 同样一组数值，但热点在专家 7
]
NAMES = ["层0", "层1"]

# ------------------------- 各种摆放算法：输一行热度，返回 (槽 x rank) 的专家号矩阵 -------------------------
def layout_trivial(heat):
    """stock 的初始布局: 槽 p 装专家 p  →  rank r 拿【连续的】K 个专家 {2r, 2r+1}"""
    return [[r * K + s for r in range(RANKS)] for s in range(K)]


def layout_strided(heat):
    """对照(不是 stock): rank r 拿【交错的】K 个专家 {r, r+RANKS}"""
    return [[s * RANKS + r for r in range(RANKS)] for s in range(K)]


def layout_snake(heat):
    """排序(热->冷) -> 排成 (K x RANKS) 矩阵, 奇数行反转(蛇形) -> 每列给一个 rank"""
    order = sorted(range(E), key=lambda e: heat[e], reverse=True)   # m = 第 m 热
    M = [[-1] * RANKS for _ in range(K)]
    for m, e in enumerate(order):
        s, r = divmod(m, RANKS)          # s = 槽(行), r = 列
        if s % 2 == 1:                   # 奇数行反转
            r = RANKS - 1 - r
        M[s][r] = e
    return M


def layout_greedy(heat, hot_first=True):
    """贪心: 发牌, 每张牌给"当前最空的包"(平局取编号小的 rank)"""
    order = sorted(range(E), key=lambda e: heat[e], reverse=hot_first)
    M = [[-1] * RANKS for _ in range(K)]
    load, cnt = [0] * RANKS, [0] * RANKS
    for e in order:
        r = min([r for r in range(RANKS) if cnt[r] < K], key=lambda r: (load[r], r))
        M[cnt[r]][r] = e
        load[r] += heat[e]
        cnt[r] += 1
    return M


# ------------------------------------- 打印 -------------------------------------
def block(title, layouts, layers=W):
    print("=" * 66)
    print(title)
    print("=" * 66)
    tot = [0] * RANKS
    for l, heat in enumerate(layers):
        M = layouts[l]
        num = [[heat[e] for e in M[s]] for s in range(K)]            # 热度矩阵
        load = [sum(num[s][r] for s in range(K)) for r in range(RANKS)]
        print(f"\n{NAMES[l]}  热度 = {heat}")
        print("        " + "".join("%6s" % f"r{r}" for r in range(RANKS)))
        for s in range(K):
            print("   槽%d  " % s + "".join("%6d" % v for v in num[s]))
        print("   ---- " + "".join("%6d" % v for v in load) + "   <- 每 rank 负载"
              + "   max/mean = %.2fx" % (max(load) / (sum(load) / RANKS)))
        tot = [a + b for a, b in zip(tot, load)]
    print("\n   合计 " + "".join("%6d" % v for v in tot)
          + "   max/mean = %.2fx" % (max(tot) / (sum(tot) / RANKS)))
    print()
    return tot


if __name__ == "__main__":
    print()
    for l, heat in enumerate(W):
        print(f"{NAMES[l]} 热度 = {heat}")
    print(f"\n槽 p 属于 rank p // {K} ；每个 rank 有 {K} 个槽\n")

    r1 = block("① stock 初始布局：槽 p 装专家 p  →  rank 拿【连续】的专家",
               [layout_trivial(w) for w in W])
    r2 = block("② 对照（不是 stock）：rank 拿【交错】的专家  →  两层互为镜像, 抵消",
               [layout_strided(w) for w in W])
    r3 = block("③ 排序 + 蛇形（deepseek_vec：奇数行反转）",
               [layout_snake(w) for w in W])
    r4 = block("④ 贪心投给最空的包（deepseek —— 我们实际跑的）",
               [layout_greedy(w) for w in W])
    r5 = block("⑤ 反例：把发牌方向反过来（从冷到热）—— stock 里没有这个逻辑",
               [layout_greedy(w, hot_first=False) for w in W])

    print("=" * 66)
    print("一眼看懂：③④ 每层的最热专家都落在 r0 → 两层歪的方向一样 → 抵消消失")
    print("=" * 66)
    for name, r in [("① 初始(连续)", r1), ("② 初始(交错)", r2), ("③ 蛇形", r3),
                    ("④ 贪心", r4), ("⑤ 反例(冷->热)", r5)]:
        print("   %-14s 合计 = %-24s max/mean = %.2fx"
              % (name, str(r), max(r) / (sum(r) / RANKS)))
