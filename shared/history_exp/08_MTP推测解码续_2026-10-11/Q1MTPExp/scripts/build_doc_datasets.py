#!/usr/bin/env python3
"""按 AboutModels 的四组 workload + MTP.md 的"固定前缀随机数据集"口径造数据。

口径（与 9-28 那轮一致）：
  - token = 均匀随机 id（1..154879）
  - 命中率 = 共享前缀 token 数 / 输入总长（32K 组用 19661 共享前缀 → 59.77%≈60%）
  - 每条 case 独立种子固定，输出长度固定（ignore_eos）
"""
import json, pathlib, random, hashlib

HERE = pathlib.Path(__file__).resolve().parent.parent
VOCAB, SEED = 154880, 20261012

def sha(ids): return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()

SCRIPTS = [
    # name,            输入,  输出,  请求数, 共享前缀
    ("w1_8k_1k",        8192, 1024,  12, 0),
    ("w2_32k_1k",      32768, 1024,   2, 0),
    ("w3_1k_32k",       1024, 32768,  2, 0),
    ("w4_prefix32k_300",32768, 300,   4, 19661),
]

def main():
    for name, inp, out, n, pref in SCRIPTS:
        rng = random.Random(SEED + hash(name) % 10000)
        prefix = [rng.randrange(1, VOCAB) for _ in range(pref)] if pref else []
        cases = []
        for i in range(n):
            ids = prefix + [rng.randrange(1, VOCAB) for _ in range(inp - pref)]
            cases.append({"index": i, "input_ids": ids, "input_sha256": sha(ids),
                          "shared_prefix_tokens": pref})
        doc = {"name": name, "seed": SEED, "vocab_id_range": [1, VOCAB - 1],
               "input_tokens": inp, "target_output_tokens": out, "n_requests": n,
               "shared_prefix_tokens": pref,
               "expected_cache_hit_ratio": round(pref / inp, 4) if pref else 0.0,
               "shared_prefix_sha256": sha(prefix) if prefix else None,
               "cases": cases}
        p = HERE / "datasets" / f"{name}.json"
        p.write_text(json.dumps(doc))
        print(f"  {p.name}: {n} 条 x {inp} 输入 / {out} 输出, 共享前缀 {pref} "
              f"(预期命中 {doc['expected_cache_hit_ratio']:.2%}), {p.stat().st_size//1024} KB")

if __name__ == "__main__":
    main()
