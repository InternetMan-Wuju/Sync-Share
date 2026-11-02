#!/usr/bin/env python3
"""Capture a short, fixed token-expert trace from a running SGLang service."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = os.environ.get("SERVICE_BASE_URL", "http://127.0.0.1:30080")


def request(path: str, payload: dict | None = None, timeout: float = 600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                body = None
            else:
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = raw.decode("utf-8", "replace")
            return {
                "status": response.status,
                "body": body,
                "elapsed_s": time.perf_counter() - start,
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "body": exc.read().decode("utf-8", "replace"),
            "elapsed_s": time.perf_counter() - start,
        }


def make_ids(seed: int, length: int) -> list[int]:
    return [1000 + ((seed * 7919 + i * 37) % 150000) for i in range(length)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--seeds", default="11,23,37,53")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE,
        "prefix_len": args.prefix_len,
        "max_new_tokens": args.max_new_tokens,
        "requests": [],
    }

    result["flush_cache"] = request("/flush_cache")
    result["start_record"] = request("/start_expert_distribution_record")
    if result["start_record"]["status"] != 200:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    for index, seed in enumerate(seeds[: args.requests]):
        input_ids = make_ids(seed, args.prefix_len)
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
        }
        response = request("/generate", payload)
        response["index"] = index
        response["seed"] = seed
        response["input_ids_sha256"] = __import__("hashlib").sha256(
            json.dumps(input_ids, separators=(",", ":")).encode()
        ).hexdigest()
        result["requests"].append(response)

    result["stop_record"] = request("/stop_expert_distribution_record")
    result["dump_record"] = request("/dump_expert_distribution_record")
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({
        "out": str(out),
        "start": result["start_record"]["status"],
        "requests": [x["status"] for x in result["requests"]],
        "stop": result["stop_record"]["status"],
        "dump": result["dump_record"]["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
