#!/usr/bin/env python3
"""从 bench_multi 的 multi_bench.json 里统计实际请求的 token 信息。"""

import argparse
import json


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


def summarize(vals):
    if not vals:
        return None
    return {
        "total": sum(vals),
        "mean": round(sum(vals) / len(vals), 1),
        "min": min(vals),
        "max": max(vals),
        "p50": pct(vals, 0.5),
        "p90": pct(vals, 0.9),
        "p99": pct(vals, 0.99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = json.load(open(a.inp, encoding="utf-8"))
    results = d.get("results", [])
    ok = [r for r in results if not r.get("error")]
    prompt_tokens = [
        (r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok
    ]
    completion_tokens = [
        (r.get("usage") or {}).get("completion_tokens") or 0 for r in ok
    ]

    stats = {
        "completed_ok": len(ok),
        "completed_err": len(results) - len(ok),
        "prompt_tokens": summarize(prompt_tokens),
        "completion_tokens": summarize(completion_tokens),
    }

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[dataset_stats] wrote -> {a.out}")
    else:
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
