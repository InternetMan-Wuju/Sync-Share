#!/usr/bin/env python3
"""Small fixed E0 baseline benchmark for a single running SGLang instance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = os.environ.get("SERVICE_BASE_URL", "http://127.0.0.1:30080")


def get_text(path: str, timeout: float = 60) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, repr(e)


def generate(input_len: int, output_len: int, seed: int, timeout: float = 600) -> dict:
    ids = [1000 + ((seed * 7919 + i * 37) % 150000) for i in range(input_len)]
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
    }
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/generate",
        data=raw,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start_wall = time.time()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "status": e.code,
            "error": e.read().decode("utf-8", "replace"),
            "input_len": input_len,
            "output_len": output_len,
            "seed": seed,
        }
    except Exception as e:
        return {"ok": False, "status": 0, "error": repr(e), "input_len": input_len, "output_len": output_len, "seed": seed}
    elapsed = time.perf_counter() - start
    meta = body.get("meta_info", {})
    ttft = None
    tpot = None
    if meta.get("prefill_finished_time") is not None and meta.get("request_received_ts") is not None:
        ttft = meta["prefill_finished_time"] - meta["request_received_ts"]
    if ttft is not None and meta.get("request_finished_ts") is not None and meta.get("completion_tokens", 0) > 1:
        tpot = (meta["request_finished_ts"] - meta["prefill_finished_time"]) / (meta["completion_tokens"] - 1)
    return {
        "ok": status == 200,
        "status": status,
        "input_len": input_len,
        "output_len": output_len,
        "seed": seed,
        "input_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "client_elapsed_s": elapsed,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_latency_s": meta.get("e2e_latency"),
        "completion_tokens": meta.get("completion_tokens"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "request_received_ts": meta.get("request_received_ts"),
        "response_sent_to_client_ts": meta.get("response_sent_to_client_ts"),
        "start_wall": start_wall,
        "meta": meta,
    }


def summarize(items: list[dict]) -> dict:
    ok = [x for x in items if x.get("ok")]
    def vals(key):
        return [float(x[key]) for x in ok if x.get(key) is not None]
    out = {"request_count": len(items), "success_count": len(ok)}
    for key in ("ttft_s", "tpot_s", "e2e_latency_s", "client_elapsed_s"):
        v = vals(key)
        if v:
            out[key] = {
                "mean": statistics.fmean(v),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "min": min(v),
                "max": max(v),
            }
    if ok:
        total_tokens = sum(int(x.get("completion_tokens") or 0) for x in ok)
        starts = [x["request_received_ts"] for x in ok if x.get("request_received_ts") is not None]
        ends = [x["response_sent_to_client_ts"] for x in ok if x.get("response_sent_to_client_ts") is not None]
        if starts and ends and max(ends) > min(starts):
            out["wall_s"] = max(ends) - min(starts)
            out["output_tokens_per_s"] = total_tokens / out["wall_s"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"started_at": datetime.now(timezone.utc).isoformat(), "base_url": BASE, "scenarios": []}

    # One untimed warmup after graph capture; do not count it.
    warm = generate(128, 8, 5, timeout=300)
    result["warmup"] = warm
    if not warm.get("ok"):
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    scenarios = [
        ("prefill_8k_1", 8192, 1, 1),
        ("decode_128_256", 128, 256, 1),
        ("mixed_1024_64_c4", 1024, 64, 4),
    ]
    result["scenario_warmups"] = []
    for name, input_len, output_len, concurrency in scenarios:
        flush_status = get_text("/flush_cache")[0]
        warm_seeds = [5000 + i for i in range(concurrency)]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            warm_items = list(pool.map(lambda s: generate(input_len, min(output_len, 8), s), warm_seeds))
        result["scenario_warmups"].append({
            "name": name,
            "flush": flush_status,
            "requests": warm_items,
            "summary": summarize(warm_items),
        })
        for rep in range(args.repeats):
            result["scenarios"].append({"name": name, "rep": rep, "flush": get_text("/flush_cache")[0]})
            seeds = [1000 + rep * 100 + i for i in range(concurrency)]
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                items = list(pool.map(lambda s: generate(input_len, output_len, s), seeds))
            entry = {
                "name": name,
                "rep": rep,
                "input_len": input_len,
                "output_len": output_len,
                "concurrency": concurrency,
                "wall_client_s": time.perf_counter() - start,
                "requests": items,
                "summary": summarize(items),
            }
            result["scenarios"].append(entry)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for name in [x[0] for x in scenarios]:
        rows = [x for x in result["scenarios"] if isinstance(x, dict) and x.get("name") == name and "requests" in x]
        all_items = [r for row in rows for r in row["requests"]]
        print(name, json.dumps(summarize(all_items), ensure_ascii=False))


if __name__ == "__main__":
    main()
