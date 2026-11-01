#!/usr/bin/env python3
"""Run on CPU inside our service container; use the installed SGLang generator."""
import hashlib
import inspect
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from sglang.benchmark.datasets.random import RandomDataset
from sglang.benchmark.utils import get_tokenizer

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tokenizer = get_tokenizer("/models/glm-q")
module = Path(inspect.getfile(RandomDataset))
seed = 20261012
for tokenize in (True, False):
    random.seed(seed)
    np.random.seed(seed)
    args = Namespace(random_input_len=1024, random_output_len=256,
                     num_prompts=2, random_range_ratio=1.0, dataset_path="",
                     tokenize_prompt=tokenize, dataset_name="random-ids")
    rows = RandomDataset.from_args(args).load(tokenizer)
    cases = []
    for i, row in enumerate(rows):
        case = {"index": i, "benchmark_prompt_len": row.prompt_len}
        key = "input_ids" if isinstance(row.prompt, list) else "text"
        case[key] = row.prompt
        if key == "input_ids":
            case["input_sha256"] = hashlib.sha256(json.dumps(row.prompt, separators=(",", ":")).encode()).hexdigest()
        else:
            case["text_sha256"] = hashlib.sha256(row.prompt.encode()).hexdigest()
        cases.append(case)
    name = "native_exact_ids" if tokenize else "native_default_text"
    data = {"name": name, "seed": seed, "input_tokens": 1024, "target_output_tokens": 256,
            "n_requests": len(cases), "generator": "installed RandomDataset.from_args(...).load(tokenizer)",
            "source_file": str(module), "source_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "tokenizer_vocab_size": tokenizer.vocab_size, "tokenizer_class": type(tokenizer).__name__,
            "tokenize_prompt": tokenize, "random_range_ratio": 1.0, "cases": cases}
    p = out / (name + ".json")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(name, "cases", len(cases), "vocab_size", tokenizer.vocab_size, "source", data["source_sha256"], flush=True)
