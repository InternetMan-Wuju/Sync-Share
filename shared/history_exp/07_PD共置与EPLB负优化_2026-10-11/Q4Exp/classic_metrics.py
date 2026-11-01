#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""经典指标表 + 分解（TTFT / TPOT / E2E breakdown），纯 CPU。

输入：<run>/bench2/requests.jsonl（含服务端时间戳）
输出：<run>/tables/classic_metrics.csv          每个请求一行
      <run>/tables/classic_metrics_summary.csv  汇总（mean/p50/p95/p99）
      <run>/figs/fig5_latency_breakdown.png     堆叠时间条 + 分布

服务端时间戳（同一个时钟）：
  request_received_ts  -> 收到请求
  forward_entry_time   -> 进入 forward 队列（**注意：这之前的等待不计入 queue_time**）
  prefill_finished_time-> prefill 前向结束
  response_sent_to_client_ts -> 首 token 发给客户端
  request_finished_ts  -> 整个请求结束
"""
import argparse, json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. results/20261012_smokeA")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    rows = [json.loads(l) for l in open(run / "bench2" / "requests.jsonl")]
    rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("meta_info"), dict)]
    out = []
    for r in rows:
        m = r["meta_info"]
        try:
            seg = dict(
                serial=r["serial"], dp_rank=m.get("dp_rank"),
                prompt_tokens=r.get("prompt_tokens"), output_tokens=r.get("output_tokens"),
                cached_tokens=r.get("cached_tokens"), retractions=m.get("num_retractions"),
                s_recv_to_forward=m["forward_entry_time"] - m["request_received_ts"],
                s_queue=m.get("queue_time", float("nan")),
                s_prefill=m["prefill_finished_time"] - m["forward_entry_time"],
                s_first_send=m["response_sent_to_client_ts"] - m["prefill_finished_time"],
                s_decode=m["request_finished_ts"] - m["response_sent_to_client_ts"],
                s_ttft=m["response_sent_to_client_ts"] - m["request_received_ts"],
                s_e2e=m["request_finished_ts"] - m["request_received_ts"],
                c_ttft=r.get("ttft_s"), c_e2e=r.get("e2e_s"), c_tpot=r.get("tpot_s"),
            )
            seg["s_other"] = seg["s_e2e"] - (seg["s_recv_to_forward"] + seg["s_prefill"]
                                             + seg["s_first_send"] + seg["s_decode"])
            # 注意：queue_time 是 recv->forward 区间【内部】的一小段（wait_queue_entry..forward_entry）
            #       所以"未上报的等待" = recv->forward - queue_time
            seg["s_unreported_wait"] = seg["s_recv_to_forward"] - seg["s_queue"]
            seg["c_overhead"] = seg["c_e2e"] - seg["s_e2e"]
            out.append(seg)
        except KeyError:
            continue
    if not out:
        raise SystemExit("没有可用请求")
    (run / "tables").mkdir(exist_ok=True); (run / "figs").mkdir(exist_ok=True)

    cols = ["serial", "dp_rank", "prompt_tokens", "output_tokens", "cached_tokens", "retractions",
            "s_recv_to_forward", "s_queue", "s_prefill", "s_first_send", "s_decode", "s_ttft",
            "s_e2e", "c_ttft", "c_tpot", "c_e2e", "s_unreported_wait", "s_other", "c_overhead"]
    with open(run / "tables" / "classic_metrics.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for s in out:
            f.write(",".join(str(round(s[c], 6)) if isinstance(s[c], float) else str(s[c]) for c in cols) + "\n")

    metrics = [
        ("TTFT 客户端 (s)", "c_ttft"), ("TTFT 服务端 (s)", "s_ttft"),
        ("TPOT 客户端 (s)", "c_tpot"),
        ("E2E 客户端 (s)", "c_e2e"), ("E2E 服务端 (s)", "s_e2e"),
        ("E2E = 以下 4 段之和 → 段1 收到→进forward (s)", "s_recv_to_forward"),
        ("E2E 段2 prefill 前向 (s)", "s_prefill"),
        ("E2E 段3 首token发送 (s)", "s_first_send"),
        ("E2E 段4 decode (s)", "s_decode"),
        ("  段1 明细: queue_time 上报 (s)", "s_queue"),
        ("  段1 明细: 未上报的等待【黑洞】(s)", "s_unreported_wait"),
        ("[残差] 服务端分解剩余 (s)", "s_other"),
        ("[残差] 客户端-服务端差 (网络+客户端) (s)", "c_overhead"),
    ]
    with open(run / "tables" / "classic_metrics_summary.csv", "w") as f:
        f.write("metric,mean,p50,p95,p99,max\n")
        print(f"\n{'指标':>32} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}")
        print("-" * 82)
        vals = {k: [s[k] for s in out] for _, k in metrics}
        for name, k in metrics:
            v = np.asarray(vals[k], dtype=float)
            row = [np.mean(v), pct(v, 50), pct(v, 95), pct(v, 99), np.max(v)]
            f.write(f"{name},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f}\n")
            print(f"{name:>32} " + " ".join(f"{x:9.4f}" for x in row))
    print(f"\n请求数 {len(out)}   DP 组分布: " + str(np.bincount([s['dp_rank'] for s in out]).tolist()))
    print(f"输出 token 合计 {sum(s['output_tokens'] for s in out)}  cached_tokens 合计 {sum(s['cached_tokens'] or 0 for s in out)}"
          f"  抢占(retraction) 合计 {sum(s['retractions'] or 0 for s in out)}")

    # ---- 图：堆叠时间条 + 分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    segs = [("s_recv_to_forward", "recv->forward (NOT in queue_time)", "#f0ad4e"),
            ("s_prefill", "prefill forward", "#5bc0de"),
            ("s_first_send", "first-token send", "#cccccc"),
            ("s_decode", "decode", "#5cb85c"),
            ("s_other", "other", "#d9534f")]
    xs = np.arange(len(out))
    bottom = np.zeros(len(out))
    for k, lab, c in segs:
        v = np.clip(np.asarray([s[k] for s in out], dtype=float), 0, None)
        ax.bar(xs, v, bottom=bottom, color=c, label=lab, width=0.85)
        bottom += v
    ax.set_xlabel("request (serial order)"); ax.set_ylabel("seconds")
    ax.set_title(f"Per-request time breakdown  {a.tag}")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    for k, lab, c in [("c_ttft", "TTFT (client)", "tab:blue"), ("c_e2e", "E2E (client)", "tab:red"),
                      ("s_recv_to_forward", "recv->forward", "tab:orange")]:
        v = np.asarray([s[k] for s in out], dtype=float)
        ax.hist(v, bins=12, alpha=.55, label=f"{lab}  p50={np.median(v):.1f}s", color=c)
    ax.set_xlabel("seconds"); ax.set_ylabel("#requests")
    ax.set_title("Distributions"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(run / "figs" / "fig5_latency_breakdown.png", dpi=110); plt.close(fig)
    print("✓ tables/classic_metrics.csv, tables/classic_metrics_summary.csv, figs/fig5_latency_breakdown.png")


if __name__ == "__main__":
    main()
