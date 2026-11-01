#!/usr/bin/env python3
"""Compare the completed PD EPLB-on baseline with an EPLB-off control run."""

import argparse
import json
from pathlib import Path


def load_cell(root, concurrency):
    cell = Path(root) / f"c{concurrency}"
    bench = json.loads((cell / "bench.json").read_text(encoding="utf-8"))["summary"]
    breakdown = json.loads((cell / "breakdown.json").read_text(encoding="utf-8"))[
        "breakdown_mean_s"
    ]
    heat = json.loads((cell / "expert_heat.json").read_text(encoding="utf-8"))[
        "imbalance"
    ]
    return {
        "output_tok_s": bench["output_tokens_per_s"],
        "total_tok_s": bench["total_tokens_per_s"],
        "ttft_s": bench["ttft_mean_ms"] / 1000,
        "tpot_ms": bench["tpot_mean_ms"],
        "e2e_s": bench["e2e_mean_ms"] / 1000,
        "queue_s": breakdown["queue"],
        "prefill_s": breakdown["prefill"],
        "middle_s": breakdown["middle(ttft-queue-prefill)"],
        "decode_s": breakdown["decode(e2e-ttft)"],
        "rank_cv": heat["cv"],
        "rank_peak_mean": heat["peak_to_mean"],
        "rank_max_min": heat["max_to_min"],
    }


def delta_pct(off, on):
    if on in (None, 0) or off is None:
        return None
    return (off / on - 1) * 100


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eplb-on", required=True)
    ap.add_argument("--eplb-off", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = {
        "output_tok_s": "输出吞吐 (tok/s)",
        "total_tok_s": "总吞吐 (tok/s)",
        "ttft_s": "TTFT mean (s)",
        "tpot_ms": "TPOT mean (ms)",
        "e2e_s": "E2E mean (s)",
        "queue_s": "queue mean (s)",
        "prefill_s": "prefill mean (s)",
        "middle_s": "中间段 mean (s)",
        "decode_s": "decode mean (s)",
        "rank_cv": "rank CV",
        "rank_peak_mean": "rank peak/mean",
        "rank_max_min": "rank max/min",
    }

    lines = [
        "# PD EP32 EPLB A/B 对比",
        "",
        "> Delta = (EPLB off / EPLB on - 1) × 100%。吞吐为正代表 off 更高；时延/不均衡为正代表 off 更差。",
        "",
    ]
    raw = {}
    for concurrency in (10, 20, 30):
        on = load_cell(args.eplb_on, concurrency)
        off = load_cell(args.eplb_off, concurrency)
        raw[str(concurrency)] = {"eplb_on": on, "eplb_off": off}
        lines.extend(
            [
                f"## 并发 {concurrency}",
                "",
                "| 指标 | EPLB on | EPLB off | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, label in labels.items():
            change = delta_pct(off[key], on[key])
            lines.append(
                f"| {label} | {fmt(on[key])} | {fmt(off[key])} | "
                f"{fmt(change, 2)}% |"
            )
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
