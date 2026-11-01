#!/usr/bin/env python3
"""Export the four workloads using the SGLang installed in our service image."""
import hashlib
import inspect
import json
import random
import sys
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sglang.benchmark.datasets.random import RandomDataset
from sglang.benchmark.utils import get_tokenizer


def sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer("/models/glm-q")
    source = Path(inspect.getfile(RandomDataset))
    provenance = {
        "generator": "installed RandomDataset.from_args(...).load(tokenizer)",
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_class": type(tokenizer).__name__,
        "dataset_name": "random-ids", "tokenize_prompt": True,
        "random_range_ratio": 1.0,
    }
    (out / "installed_random.py").write_bytes(source.read_bytes())

    def generate(length, output, count, seed):
        random.seed(seed)
        np.random.seed(seed)
        args = Namespace(random_input_len=length, random_output_len=output,
                         num_prompts=count, random_range_ratio=1.0, dataset_path="",
                         tokenize_prompt=True, dataset_name="random-ids")
        rows = RandomDataset.from_args(args).load(tokenizer)
        assert len(rows) == count
        ids = [list(row.prompt) for row in rows]
        assert all(len(row) == length for row in ids)
        assert all(int(row.output_len) == output for row in rows)
        return ids

    scenarios = [
        ("w1_8k_1k", 8192, 1024, 8, 16, 2026101201),
        ("w2_32k_1k", 32768, 1024, 2, 2, 2026101202),
        ("w3_1k_32k", 1024, 32768, 1, 1, 2026101203),
        ("w4_prefix32k_300", 32768, 300, 2, 4, 2026101204),
    ]
    for name, length, output, concurrency, count, seed in scenarios:
        prefix = []
        prefix_seed = None
        if name.startswith("w4"):
            prefix_seed = 2026101205
            prefix = generate(19661, 1, 1, prefix_seed)[0]
            suffixes = generate(length - len(prefix), output, count, seed)
            inputs = [prefix + suffix for suffix in suffixes]
            assert all(row[len(prefix)] != prefix[-1] + 1 for row in inputs)
            construction = "shared native random-ids prefix + independently generated native random-ids suffixes"
        else:
            inputs = generate(length, output, count, seed)
            construction = "unmodified installed SGLang random-ids prompts"
        data = {
            **provenance, "name": name, "seed": seed, "prefix_seed": prefix_seed,
            "construction": construction, "input_tokens": length,
            "target_output_tokens": output, "concurrency": concurrency,
            "n_requests": count, "shared_prefix_tokens": len(prefix),
            "shared_prefix_sha256": sha(prefix) if prefix else None,
            "expected_cache_hit_ratio": len(prefix) / length,
            "cases": [{"index": i, "input_ids": row, "input_sha256": sha(row),
                       "shared_prefix_tokens": len(prefix)} for i, row in enumerate(inputs)],
        }
        (out / (name + ".json")).write_text(json.dumps(data, ensure_ascii=False))
        print(f"{name}: {count} requests, {length}/{output}, concurrency={concurrency}, source={provenance['source_sha256']}", flush=True)

    checks = []
    for prompt in ("你好，请简短打个招呼。", "1+1等于几？只回答结果。"):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            assert len(ids) == 1
            ids = ids[0]
        assert ids and all(isinstance(token, int) for token in ids)
        checks.append({"index": len(checks), "question": prompt, "input_ids": list(ids)})
    (out / "sanity_chat.json").write_text(json.dumps({"cases": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
