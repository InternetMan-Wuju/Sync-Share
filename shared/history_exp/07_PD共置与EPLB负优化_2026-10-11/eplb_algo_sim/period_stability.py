#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""周期稳定性统计：运行 = 周期1-[重排1]-周期2(用重排1)-[重排2]-周期3...

问：每个"周期"内的【逻辑专家热度规律】是不是同一套？
     - 周期内（同一窗口前半 vs 后半）
     - 周期之间（相邻 / 跨重排 / 隔 k 个）
     - 重排到底改了多少摆放（重排1 的映射 vs 重排2 的映射）

注：我们记录的是 logical_count（逻辑专家计数），与当前 physical->logical 映射无关，
    所以跨周期比较"热度规律"是干净的（不受重排影响）。

数据: results/20261012_000935/windows/*.pt       跑法: python3 period_stability.py
"""
import sys, glob, pathlib
import numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent
WIN = sorted(glob.glob(str(HERE.parent / "pd_colloc_20261011" / "results" /
                          "20261012_000935" / "windows" / "*.pt")))
L, E = 78, 256
TOPK = 8            # 每 token 选 top-8，所以看 top-8 的重合最有意义


def norm(H):
    """按层归一化：只比较"形状"不比较绝对量"""
    s = H.sum(1, keepdims=True)
    return H / np.where(s == 0, 1, s)


def per_layer_corr(A, B):
    """逐层 Pearson，返回非空层的均值/中位/最小"""
    out = []
    for l in range(L):
        a, b = A[l], B[l]
        if a.sum() == 0 or b.sum() == 0:
            continue
        if a.std() == 0 or b.std() == 0:
            continue
        out.append(np.corrcoef(a, b)[0, 1])
    return (np.mean(out), np.median(out), np.min(out), len(out)) if out else (np.nan,) * 4


def topk_overlap(A, B, k=TOPK):
    """逐层 top-k 集合重合率（|交集| / k）"""
    out = []
    for l in range(L):
        if A[l].sum() == 0 or B[l].sum() == 0:
            continue
        ta, tb = set(np.argsort(-A[l])[:k]), set(np.argsort(-B[l])[:k])
        out.append(len(ta & tb) / k)
    return np.mean(out) if out else np.nan


def main():
    ws = []
    for f in WIN:
        o = torch.load(f, map_location="cpu", weights_only=False)
        dec = o.get("logical_count_decode")
        pre = o.get("logical_count_prefill")
        H = np.zeros((L, E))
        for t in (dec, pre):
            if t is not None:
                H += t.float().sum(0).numpy()
        steps = 0 if dec is None else dec.shape[0]
        ws.append(dict(n=pathlib.Path(f).name[:6], kind=o["window_kind"], H=H,
                       pm=o["physical_to_logical_map"].numpy(), steps=steps,
                       t0=o.get("window_start_timestamp"), t1=o.get("window_end_timestamp"),
                       empty=H.sum() == 0, dec=dec))

    live = [w for w in ws if not w["empty"]]
    print("=== 0. 周期划分（我们的 dump 就等于一个【周期】）===")
    for w in ws:
        span = "" if w["t0"] is None else f"{w['t1']-w['t0']:6.1f}s"
        print(f"  {w['n']}  kind={w['kind']:<14} 步数={w['steps']:>4}  时长={span:>7} "
              f"{'（空）' if w['empty'] else ''}")

    print("\n=== 1. 周期【内】稳定性：同一周期前半 vs 后半（细粒度基准）===")
    for w in ws:
        if w["empty"] or w["dec"] is None or w["dec"].shape[0] < 10:
            continue
        h = w["dec"].float().sum(0).numpy()
        m = w["dec"].shape[0] // 2
        A = w["dec"][:m].float().sum(0).numpy()
        B = w["dec"][m:].float().sum(0).numpy()
        c, med, mn, n = per_layer_corr(A, B)
        print(f"  {w['n']}: 逐层相关 均值={c:.3f} 中位={med:.3f} 最低={mn:.3f} "
              f"| top-8 重合={topk_overlap(A,B):.1%}")


    print("\n=== 1b. 噪声基准：同一周期内切成 4 片，片与片比较（= 纯采样噪声有多大）===")
    cs, ovs = [], []
    for w in ws:
        if w["empty"] or w["dec"] is None or w["dec"].shape[0] < 40:
            continue
        d = w["dec"]; n = d.shape[0]
        parts = [d[i * n // 4:(i + 1) * n // 4].float().sum(0).numpy() for i in range(4)]
        cc = [per_layer_corr(parts[i], parts[j])[0] for i in range(4) for j in range(i + 1, 4)]
        oo = [topk_overlap(parts[i], parts[j]) for i in range(4) for j in range(i + 1, 4)]
        cs += cc; ovs += oo
        print(f"  {w['n']}: 步数={n:>4}  片间相关={np.mean(cc):.3f}  片间 top-8 重合={np.mean(oo):.1%}")
    print(f"  总体: 片间相关 均值={np.mean(cs):.3f} 最低={np.min(cs):.3f}; top-8 重合={np.mean(ovs):.1%}")

    print("\n=== 2. 周期【之间】稳定性 ===")
    print(f"  {'A->B':>16} {'逐层相关(均)':>12} {'逐层相关(最低)':>14} {'top-8重合':>10} {'top-1不变的层':>12}")
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            if ws[i]["empty"] or ws[j]["empty"]:
                continue
            A, B = ws[i]["H"], ws[j]["H"]
            c, med, mn, n = per_layer_corr(A, B)
            same1 = np.mean([A[l].argmax() == B[l].argmax() for l in range(L)])
            tag = f"{ws[i]['n']}->{ws[j]['n']}"
            mark = ""
            if any(w["kind"] == "eplb_rebalance" for w in ws[i+1:j+1]):
                mark = "  <-跨重排"
            print(f"  {tag:>16} {c:>12.3f} {mn:>14.3f} {topk_overlap(A,B):>10.1%} {same1:>12.0%}{mark}")

    print("\n=== 3. 重排到底改了多少摆放（真机映射逐层比对）===")
    maps = [(w["n"], w["pm"], w["kind"]) for w in ws if not w["empty"]]
    for i in range(len(maps) - 1):
        a, b = maps[i][1], maps[i + 1][1]
        same = [np.array_equal(a[l], b[l]) for l in range(L)]
        # 同一逻辑专家的 rank 是否变化
        print(f"  {maps[i][0]}({maps[i][2][:9]}) -> {maps[i+1][0]}({maps[i+1][2][:9]}): "
              f"逐层完全相同 = {np.mean(same):.0%}")


if __name__ == "__main__":
    main()
