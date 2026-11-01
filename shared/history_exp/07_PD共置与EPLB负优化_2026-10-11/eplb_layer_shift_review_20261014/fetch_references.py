#!/usr/bin/env python3
"""Download public primary references into workspace; retain fetch provenance."""
import hashlib
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

HERE = pathlib.Path(__file__).resolve().parent
SOURCES = [
    ("deepseek_eplb_README.md", "https://raw.githubusercontent.com/deepseek-ai/EPLB/main/README.md"),
    ("deepseek_eplb_upstream.py", "https://raw.githubusercontent.com/deepseek-ai/EPLB/main/eplb.py"),
    ("deepseek_deepep_README.md", "https://raw.githubusercontent.com/deepseek-ai/DeepEP/main/README.md"),
    ("sglang_expert_parallelism.html", "https://docs.sglang.ai/advanced_features/expert_parallelism.html"),
    ("sglang_expert_parallelism_source.rst", "https://raw.githubusercontent.com/sgl-project/sglang/main/docs/advanced_features/expert_parallelism.rst"),
]


def fetch(item):
    name, url = item
    r = requests.get(url, timeout=40)
    out = dict(name=name, url=url, status=r.status_code,
               fetched_at_utc=datetime.now(timezone.utc).isoformat())
    if r.ok:
        (HERE / "sources" / name).write_bytes(r.content)
        out.update(sha256=hashlib.sha256(r.content).hexdigest(), bytes=len(r.content))
    return out


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, SOURCES))
    (HERE / "sources" / "web_manifest.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
