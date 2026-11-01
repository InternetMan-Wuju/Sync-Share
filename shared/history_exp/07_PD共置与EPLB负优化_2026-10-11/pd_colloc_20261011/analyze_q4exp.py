#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4Exp 分析：把一个 run 目录的窗口 + bench 汇总成表。
用法: python3 analyze_q4exp.py results/<RUN_DIR> [更多目录...]
输出：每个 run 的性能指标 + 每个窗口的 rank 负载不平衡（PD 分开）
"""
import sys, json, glob, pathlib
import numpy as np, torch

def bench(p):
    f = pathlib.Path(p) / "bench" / "summary.json"
    if not f.exists(): return None
    d = json.loads(f.read_text())
    return dict(req=d["requests_sent"], err=d["errors"], thr=d["output_tokens_per_s"],
                ttft=d["ttft_s"], tpot=d["tpot_s"], e2e=d["e2e_s"])

def window_stats(f):
    o = torch.load(f, map_location="cpu", weights_only=False)
    L, P, ep = int(o["num_layers"]), int(o["num_physical_experts"]), int(o["ep_size"])
    pm = o["physical_to_logical_map"].numpy()
    k = P // ep
    out = dict(name=pathlib.Path(f).name[:6], kind=o["window_kind"], P=P, k=k)
    for phase in ("prefill", "decode", "both"):
        h = None
        for key in ("logical_count_prefill", "logical_count_decode"):
            t = o.get(key)
            if t is None: continue
            if phase != "both" and key != f"logical_count_{phase}": continue
            s = t.float().sum(0).numpy()
            h = s if h is None else h + s
        if h is None or h.sum() == 0: out[phase] = None; continue
        rank = np.zeros(ep)
        for l in range(L):
            rank += h[l][pm[l]].reshape(ep, k).sum(1)
        out[phase] = (rank.max() / rank.mean(), float(h.sum()))
    return out

def main():
    for d in sys.argv[1:]:
        print("=" * 100); print(d); print("=" * 100)
        b = bench(d)
        if b:
            print(f"  性能: 请求 {b['req']} 错 {b['err']} | 吞吐 {b['thr']:.0f} tok/s | "
                  f"TTFT p50 {b['ttft']['p50']:.1f}s p95 {b['ttft']['p95']:.1f}s | "
                  f"TPOT p50 {b['tpot']['p50']*1000:.0f}ms p95 {b['tpot']['p95']*1000:.0f}ms | "
                  f"E2E p50 {b['e2e']['p50']:.1f}s")
        ws = sorted(glob.glob(str(pathlib.Path(d) / "windows" / "*.pt")))
        if not ws:
            print("  窗口: 0 个（本组无采集）"); continue
        print(f"  窗口: {len(ws)} 个   (P=物理槽数, k=每卡槽数)")
        print(f"    {'窗口':>8} {'kind':>14} {'P':>4} {'k':>2} | {'合计 max/mean':>13} {'prefill':>9} {'decode':>8}")
        for f in ws:
            s = window_stats(f)
            fm = lambda v: ("%.2fx" % v[0]) if v else "   -"
            print(f"    {s['name']:>8} {s['kind']:>14} {s['P']:>4} {s['k']:>2} | "
                  f"{fm(s['both']):>13} {fm(s['prefill']):>9} {fm(s['decode']):>8}")
        print()

if __name__ == "__main__":
    main()
