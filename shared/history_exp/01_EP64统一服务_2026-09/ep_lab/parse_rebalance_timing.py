#!/usr/bin/env python3
"""从服务端日志解析 rebalance 时序：每次 rebalance 的暂停时长。"""

import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    text = open(a.log, encoding="utf-8", errors="ignore").read()

    # rank0（TP0 EP0）每次 rebalance 各打一条 start / end
    starts = re.findall(r"TP0 EP0\] \[EPLBManager\] rebalance start", text)
    ends = re.findall(
        r"TP0 EP0\] \[EPLBManager\] rebalance end time=([0-9.]+)s", text
    )
    durations = [float(x) for x in ends]

    out = {
        "num_rebalances": len(starts),
        "num_end_records": len(durations),
        "durations_s": [round(x, 3) for x in durations],
        "total_s": round(sum(durations), 3) if durations else 0.0,
        "mean_s": round(sum(durations) / len(durations), 3) if durations else None,
        "max_s": round(max(durations), 3) if durations else None,
        "min_s": round(min(durations), 3) if durations else None,
    }

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[rebalance_timing] wrote {a.out}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
