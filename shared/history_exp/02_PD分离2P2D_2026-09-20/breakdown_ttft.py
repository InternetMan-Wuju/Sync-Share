#!/usr/bin/env python3
"""四段分解：queue + prefill + 中间 + decode。

从 Prometheus metrics 的 before/after 快照解析：
  queue    = sglang:queue_time_seconds
  prefill  = sglang:per_stage_req_latency_seconds{stage="prefill_forward"}
  TTFT     = sglang:time_to_first_token_seconds
  e2e      = sglang:e2e_request_latency_seconds
然后：
  中间    = TTFT - queue - prefill   （即 tokenizer<->scheduler 的 IPC，见 TTFT残差分析.md）
  decode  = e2e - TTFT

说明：mean 用 histogram 的 _sum/_count 差分计算；分位用 le 桶差分后线性插值。
不传 --before 时兼容旧行为，直接使用 --metrics 中的累计值。
"""
import argparse
import json
import re


def extract_hist(text, metric, label_filter=None):
    """聚合同一 metric 的全部 scheduler/rank 序列。"""
    sample_re = re.compile(
        rf"^{re.escape(metric)}_(sum|count|bucket)\{{(.*?)\}}\s+([\d.eE+-]+)"
    )
    total_sum = 0.0
    total_count = 0.0
    buckets = {}
    found_sum = False
    found_count = False

    for line in text.splitlines():
        m = sample_re.match(line.strip())
        if not m:
            continue
        kind, labels, raw_value = m.groups()
        if label_filter:
            k, v = label_filter
            if not re.search(rf'(?:^|,){re.escape(k)}="{re.escape(v)}"(?:,|$)', labels):
                continue
        value = float(raw_value)
        if kind == "sum":
            total_sum += value
            found_sum = True
        elif kind == "count":
            total_count += value
            found_count = True
        else:
            le_match = re.search(r'(?:^|,)le="([^"]+)"(?:,|$)', labels)
            if not le_match:
                continue
            raw_le = le_match.group(1)
            le = None if raw_le == "+Inf" else float(raw_le)
            buckets[le] = buckets.get(le, 0.0) + value

    return {
        "sum": total_sum if found_sum else None,
        "count": total_count if found_count else None,
        "buckets": buckets,
    }


def parse_hist(text, metric, label_filter=None, before_text=None):
    """解析一个 histogram，返回 {count, mean, p50, p90, p99}。
    label_filter: (key, value) 只统计该 label 的序列（如 stage=prefill_forward）。
    """
    after = extract_hist(text, metric, label_filter)
    before = extract_hist(before_text, metric, label_filter) if before_text else None

    def diff(name):
        value = after[name]
        if value is None:
            return None
        if before is None or before[name] is None:
            return value
        delta = value - before[name]
        if delta < -1e-6:
            raise ValueError(
                f"{metric}_{name} decreased between snapshots; service likely restarted"
            )
        return max(0.0, delta)

    total = diff("count")
    total_sum = diff("sum")
    mean = total_sum / total if total and total_sum is not None else None

    bucket_keys = set(after["buckets"])
    if before:
        bucket_keys.update(before["buckets"])
    buckets = []
    for le in bucket_keys:
        value = after["buckets"].get(le, 0.0)
        if before:
            value -= before["buckets"].get(le, 0.0)
        if value < -1e-6:
            raise ValueError(
                f"{metric}_bucket decreased between snapshots; service likely restarted"
            )
        buckets.append((le, max(0.0, value)))
    buckets.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))

    def quantile(q):
        if not buckets or not total:
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

    return {
        "count": int(total) if total is not None else None,
        "mean": round(mean, 4) if mean is not None else None,
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None, help="兼容模式：单份 after metrics")
    ap.add_argument("--before", default=None, help="before metrics.prom 文件路径")
    ap.add_argument("--prefill-metrics", default=None, help="prefill after metrics")
    ap.add_argument("--prefill-before", default=None, help="prefill before metrics")
    ap.add_argument("--decode-metrics", default=None, help="decode after metrics")
    ap.add_argument("--decode-before", default=None, help="decode before metrics")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.prefill_metrics or a.decode_metrics:
        if not (a.prefill_metrics and a.decode_metrics):
            ap.error("PD 模式必须同时提供 --prefill-metrics 和 --decode-metrics")
        prefill_path = a.prefill_metrics
        decode_path = a.decode_metrics
        prefill_before_path = a.prefill_before
        decode_before_path = a.decode_before
    elif a.metrics:
        prefill_path = decode_path = a.metrics
        prefill_before_path = decode_before_path = a.before
    else:
        ap.error("请提供 --metrics，或提供 prefill/decode 两侧 metrics")

    def read(path):
        return open(path, encoding="utf-8", errors="ignore").read() if path else None

    prefill_text = read(prefill_path)
    decode_text = read(decode_path)
    prefill_before_text = read(prefill_before_path)
    decode_before_text = read(decode_before_path)
    h = {
        "queue": parse_hist(
            prefill_text,
            "sglang:queue_time_seconds",
            before_text=prefill_before_text,
        ),
        "prefill": parse_hist(
            prefill_text,
            "sglang:per_stage_req_latency_seconds",
            ("stage", "prefill_forward"),
            prefill_before_text,
        ),
        "ttft": parse_hist(
            decode_text,
            "sglang:time_to_first_token_seconds",
            before_text=decode_before_text,
        ),
        "e2e": parse_hist(
            decode_text,
            "sglang:e2e_request_latency_seconds",
            before_text=decode_before_text,
        ),
    }

    def seg_mean(name):
        v = h[name].get("mean")
        return v if v is not None else 0.0

    mid_mean = seg_mean("ttft") - seg_mean("queue") - seg_mean("prefill")
    dec_mean = seg_mean("e2e") - seg_mean("ttft")

    out = {
        "metrics_files": {
            "prefill_after": prefill_path,
            "prefill_before": prefill_before_path,
            "decode_after": decode_path,
            "decode_before": decode_before_path,
        },
        "components": h,
        "breakdown_mean_s": {
            "queue": round(seg_mean("queue"), 4),
            "prefill": round(seg_mean("prefill"), 4),
            "middle(ttft-queue-prefill)": round(mid_mean, 4),
            "decode(e2e-ttft)": round(dec_mean, 4),
            "ttft": round(seg_mean("ttft"), 4),
            "e2e": round(seg_mean("e2e"), 4),
        },
    }
    # 占比（相对 e2e）
    e2e = seg_mean("e2e")
    if e2e > 0:
        out["breakdown_share_of_e2e"] = {
            "queue": round(seg_mean("queue") / e2e, 4),
            "prefill": round(seg_mean("prefill") / e2e, 4),
            "middle": round(mid_mean / e2e, 4),
            "decode": round(dec_mean / e2e, 4),
        }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[breakdown_ttft] wrote -> {a.out}")


if __name__ == "__main__":
    main()
