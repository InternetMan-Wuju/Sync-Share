#!/usr/bin/env python3
"""按 AboutModels 四组 workload + "固定前缀随机数据集" 口径造数据集。

口径（与 2026-09-28 一致）：
  - token = 均匀随机 id (1..154879)；每条 case 独立种子固定；输出长度固定(ignore_eos)
  - 命中率 = 共享前缀 token 数 / 输入总长（32K 组用 19661 → 59.77%≈60%）
  - 一个场景一个文件；请求数 = 该场景允许的最大条数（AboutModels：8k/1k 与 prefix 为 并发×2，32k/1k 与 1k/32k 为 并发）
用法: python3 build_datasets.py
"""
import json, pathlib, random, hashlib

HERE = pathlib.Path(__file__).resolve().parent.parent
VOCAB, SEED = 154880, 20261012

def sha(ids): return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()

# name, 输入, 输出, 最大请求数, 共享前缀
SCRIPTS = [
    ("w1_8k_1k",          8192,  1024, 32, 0),
    ("w2_32k_1k",        32768,  1024,  8, 0),
    ("w3_1k_32k",         1024, 32768,  8, 0),
    ("w4_prefix32k_300", 32768,   300, 32, 19661),
]

def main():
    out = HERE / "datasets"; out.mkdir(exist_ok=True)
    for name, inp, o, n, pref in SCRIPTS:
        rng = random.Random(SEED + (abs(hash(name)) % 10000))
        prefix = [rng.randrange(1, VOCAB) for _ in range(pref)] if pref else []
        cases = []
        for i in range(n):
            ids = prefix + [rng.randrange(1, VOCAB) for _ in range(inp - pref)]
            cases.append({"index": i, "input_ids": ids, "input_sha256": sha(ids),
                          "shared_prefix_tokens": pref})
        doc = {"name": name, "seed": SEED, "vocab_id_range": [1, VOCAB - 1],
               "input_tokens": inp, "target_output_tokens": o, "n_requests": n,
               "shared_prefix_tokens": pref,
               "expected_cache_hit_ratio": round(pref / inp, 4) if pref else 0.0,
               "shared_prefix_sha256": sha(prefix) if prefix else None,
               "cases": cases}
        p = out / f"{name}.json"; p.write_text(json.dumps(doc))
        print(f"  {p.name}: {n} 条 x {inp} 输入 / {o} 输出, 共享前缀 {pref} "
              f"(预期命中 {doc['expected_cache_hit_ratio']:.2%}), {p.stat().st_size//1024} KB")

if __name__ == "__main__":
    main()
