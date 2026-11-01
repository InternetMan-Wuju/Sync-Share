"""CPU-only output shape checks; repetition is evidence, not a quality verdict."""
from collections import Counter


def output_shape(ids, max_period=64):
    n = len(ids)
    counts = Counter(ids)
    best = None
    for period in range(1, min(max_period, n // 8) + 1):
        i = n - 1
        while i >= period and ids[i] == ids[i - period]:
            i -= 1
        start = i - period + 1
        length = n - start
        if length < max(64, period * 8):
            continue
        if best is None or length > best["tokens"]:
            best = {"period": period, "start": start, "tokens": length,
                    "fraction": length / n}
    return {
        "tokens": n,
        "distinct_ids": len(counts),
        "top5_ids": counts.most_common(5),
        "top1_fraction": counts.most_common(1)[0][1] / n if n else None,
        "tail256_distinct_ids": len(set(ids[-256:])),
        "periodic_suffix": best,
        "scope": "Exact periodic suffix >=64 tokens and >=8 repetitions; not semantic quality.",
    }
