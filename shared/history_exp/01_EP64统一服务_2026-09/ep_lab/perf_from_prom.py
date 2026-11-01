#!/usr/bin/env python3
"""从 multi_bench.json + metrics_*.prom 汇总性能指标（ep_exp2.md §3.3）。

请求级（multi_bench.json）：e2e 的 mean/max/P50/P90/P99、requests/s、
  total tokens/s、completion tokens/s、ok/err。
token 级（Prometheus 直方图）：TTFT / ITL / e2e 的 P50/P90/P99。
  SGLang 暴露 sglang:time_to_first_token_seconds / inter_token_latency_seconds /
  e2e_request_latency_seconds 三个直方图，按 le 桶做线性插值求分位。
"""

import argparse
import glob
import json
import os
import re


def pct_sorted(vals, q):
    if not vals:
        return None
    s = sorted(float(v) for v in vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return round(s[idx], 4)


def e2e_from_json(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    results = d.get("results", [])
    ok = [r for r in results if not r.get("error")]
    e2e = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    usages = [r.get("usage") or {} for r in ok]
    prompt_tokens = sum(u.get("prompt_tokens") or 0 for u in usages)
    completion_tokens = sum(u.get("completion_tokens") or 0 for u in usages)
    wall = d.get("summary", {}).get("wall_s") or None
    return {
        "ok": len(ok),
        "err": len(results) - len(ok),
        "e2e_mean_s": round(sum(e2e) / len(e2e), 3) if e2e else None,
        "e2e_max_s": round(max(e2e), 3) if e2e else None,
        "e2e_p50_s": pct_sorted(e2e, 0.5),
        "e2e_p90_s": pct_sorted(e2e, 0.9),
        "e2e_p99_s": pct_sorted(e2e, 0.99),
        "requests_per_s": round(len(results) / wall, 3) if wall else None,
        "total_tokens_per_s": round((prompt_tokens + completion_tokens) / wall, 1)
        if wall
        else None,
        "completion_tokens_per_s": round(completion_tokens / wall, 1) if wall else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def histogram_quantile(buckets, total, q):
    """buckets: [(le_value_or_None, cum_count), ...] 已按 le 升序，最后是 +Inf。"""
    if not buckets or total <= 0:
        return None
    target = q * total
    prev_le, prev_c = None, 0
    for le, c in buckets:
        if c >= target:
            if le is None or prev_le is None:
                return round(le if le is not None else prev_le, 4)
            if c == prev_c:
                return round(prev_le, 4)
            frac = (target - prev_c) / (c - prev_c)
            return round(prev_le + frac * (le - prev_le), 4)
        prev_le, prev_c = le, c
    return round(prev_le, 4) if prev_le is not None else None


def parse_histogram(text, metric):
    """从 /metrics 文本解析某个 histogram 的桶和 count，返回 P50/P90/P99。"""
    base = metric.replace("sglang:", "")
    buckets = []
    total = None
    bucket_re = re.compile(rf"^{re.escape(metric)}_bucket\{{[^}}]*le=\"([^\"]+)\"[^}}]*\}}\s+([\d.eE+-]+)")
    count_re = re.compile(rf"^{re.escape(metric)}_count\{{.*?\}}\s+([\d.eE+-]+)")
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(metric):
            continue
        m = bucket_re.match(line)
        if m:
            le = None if m.group(1) == "+Inf" else float(m.group(1))
            buckets.append((le, float(m.group(2))))
            continue
        m2 = count_re.match(line)
        if m2:
            total = float(m2.group(1))
    buckets.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
    return {
        "count": int(total) if total is not None else None,
        "p50": histogram_quantile(buckets, total, 0.5) if total else None,
        "p90": histogram_quantile(buckets, total, 0.9) if total else None,
        "p99": histogram_quantile(buckets, total, 0.99) if total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    req = e2e_from_json(os.path.join(a.run_dir, "multi_bench.json"))
    if req is None:
        # 兼容旧的 dolly_bench.json 命名
        req = e2e_from_json(os.path.join(a.run_dir, "dolly_bench.json"))

    metrics_files = sorted(glob.glob(os.path.join(a.run_dir, "metrics_*.prom")))
    # 取 scheduler 节点（按文件名排序取第一个，通常是 .23）
    hist = {}
    if metrics_files:
        text = open(metrics_files[0], encoding="utf-8", errors="ignore").read()
        hist = {
            "ttft": parse_histogram(text, "sglang:time_to_first_token_seconds"),
            "itl": parse_histogram(text, "sglang:inter_token_latency_seconds"),
            "e2e": parse_histogram(text, "sglang:e2e_request_latency_seconds"),
        }

    out = {
        "request_level": req,
        "token_level_prometheus": hist,
        "metrics_files_used": [os.path.basename(p) for p in metrics_files],
    }
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[perf_from_prom] wrote -> {a.out}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
