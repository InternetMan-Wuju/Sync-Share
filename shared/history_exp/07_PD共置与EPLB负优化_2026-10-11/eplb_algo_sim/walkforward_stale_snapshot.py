#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态验证：EPLB 的"过期快照"效应 —— 单快照结论在循环里还成立吗？

真实系统是一个循环： 跑一段时间 -> 累计热度矩阵 -> 重排 -> 新请求(热度不重复) -> ...
所以"用窗口 t 的摆放去服务窗口 t+1 的真实热度"才是它在干的事。
本脚本用实测的 10 个真实窗口做这个滚动评估（walk-forward），并与真机实测对照。

数据: results/20261012_000935/windows/*.pt （真实 token 计数 + 当窗口生效的映射）
算法: 镜像原版 stock 算法（patches/stock_alg_ref/）
跑法: python3 walkforward_stale_snapshot.py
"""
import sys, glob, pathlib
import numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pd_colloc_20261011" / "patches" / "stock_alg_ref"))
import deepseek

WIN = sorted(glob.glob(str(HERE.parent / "pd_colloc_20261011" / "results" /
                          "20261012_000935" / "windows" / "*.pt")))
E, EP, R = 256, 64, 64
P = E + R


def load_windows():
    ws = []
    for f in WIN:
        o = torch.load(f, map_location="cpu", weights_only=False)
        h = None
        for k in ("logical_count_decode", "logical_count_prefill"):
            t = o.get(k)
            if t is None:
                continue
            s = t.float().sum(0)
            h = s if h is None else h + s
        if h is None:
            h = torch.zeros(78, E)
        ws.append(dict(name=pathlib.Path(f).name[:6], W=h.numpy(),
                       pm=o["physical_to_logical_map"].numpy(),
                       empty=float(h.sum()) == 0))
    return ws


def load_of(pm, W, PP):
    """摆放 pm 在热度 W 下的每 rank 负载"""
    k = PP // EP
    tot = np.zeros(EP)
    for l in range(W.shape[0]):
        tot += W[l][pm[l]].reshape(EP, k).sum(1)
    return tot


def place(heat, PP):
    return deepseek.rebalance_experts(torch.tensor(heat), PP, 1, 1, EP, False)[0].numpy()


def main():
    ws = load_windows()
    idx = [i for i, w in enumerate(ws) if not w["empty"]]

    print("=== ① 热度自身的时变性（相邻窗口）===")
    for i in range(1, len(ws)):
        if ws[i]["empty"] or ws[i - 1]["empty"]:
            continue
        a, b = ws[i - 1]["W"], ws[i]["W"]
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        same = np.mean([a[l].argmax() == b[l].argmax() for l in range(78)])
        print(f"  {ws[i-1]['name']}->{ws[i]['name']}: 逐格相关={c:.3f}  "
              f"每层最热专家编号不变的层={same:.0%}")

    print("\n=== ② 滚动评估：用窗口 t 的摆放，去服务窗口 t+k 的真实热度 ===")
    print(f"{'lag k':>6} {'样本':>5} {'EPLB':>10} {'冗余64':>10}")
    for k in (0, 1, 2, 3):
        a, b = [], []
        for t in idx:
            if t + k not in idx:
                continue
            Wt, We = ws[t]["W"], ws[t + k]["W"]
            p_ds, p_r64 = place(Wt, E), place(Wt, P)
            a.append(load_of(p_ds, We, E).max() / load_of(p_ds, We, E).mean())
            b.append(load_of(p_r64, We, P).max() / load_of(p_r64, We, P).mean())
        print(f"{k:>6} {len(a):>5} {np.mean(a):>10.2f} {np.mean(b):>10.2f}")

    print("\n=== ③ 稳态口径：先把多个窗口的 per-rank 负载加起来，再算 max/mean ===")
    grp = [i for i in idx if i >= 4]                       # 重排之后的稳定窗口
    acc = dict(identity=np.zeros(EP), eplb_stale=np.zeros(EP),
               red64_stale=np.zeros(EP), real=np.zeros(EP))
    for t in grp:
        We = ws[t]["W"]
        acc["identity"] += load_of(np.tile(np.arange(E), (78, 1)), We, E)
        acc["real"] += load_of(ws[t]["pm"], We, E)
        if t - 1 in idx:
            Wt = ws[t - 1]["W"]
            acc["eplb_stale"] += load_of(place(Wt, E), We, E)
            acc["red64_stale"] += load_of(place(Wt, P), We, P)
    print(f"  窗口 {[ws[i]['name'] for i in grp]} (n={len(grp)})")
    for k, v in acc.items():
        print(f"  {k:14s}: {v.max()/v.mean():.2f}x")


if __name__ == "__main__":
    main()
