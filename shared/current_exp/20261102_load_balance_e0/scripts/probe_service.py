#!/usr/bin/env python3
"""Collect E0 service health, identity, correctness and latency evidence."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:30080"


def get_json(path: str, timeout: float = 30) -> tuple[int, object]:
    url = BASE + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, {"error": body}
    except Exception as e:
        return 0, {"error": repr(e)}


def post_json(path: str, payload: dict, timeout: float = 180) -> tuple[int, object, float]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            elapsed = time.perf_counter() - start
            return r.status, json.loads(body) if body else None, elapsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, {"error": body}, time.perf_counter() - start
    except Exception as e:
        return 0, {"error": repr(e)}, time.perf_counter() - start


def wait_health(timeout_s: int) -> dict:
    start = time.time()
    attempts = []
    while time.time() - start < timeout_s:
        status, body = get_json("/health", timeout=5)
        attempts.append({"at": datetime.now(timezone.utc).isoformat(), "status": status, "body": body})
        if status == 200:
            return {"ok": True, "elapsed_s": time.time() - start, "attempts": attempts}
        time.sleep(10)
    return {"ok": False, "elapsed_s": time.time() - start, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--health-timeout", type=int, default=1800)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE,
        "pid": os.getpid(),
    }
    result["health"] = wait_health(args.health_timeout)
    if not result["health"]["ok"]:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    for path in ("/server_info", "/get_model_info", "/metrics"):
        status, body = get_json(path)
        result[path] = {"status": status, "body": body}

    chat_payloads = [
        {
            "name": "arithmetic",
            "messages": [{"role": "user", "content": "只回答结果：1+1等于几？"}],
        },
        {
            "name": "fixed_list",
            "messages": [{"role": "user", "content": "请只输出数字1到20，用英文逗号分隔，不要解释。"}],
        },
    ]
    result["chat"] = []
    for item in chat_payloads:
        payload = {
            "model": "glm-5.2",
            "messages": item["messages"],
            "temperature": 0,
            "max_tokens": 128,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        status, body, elapsed = post_json("/v1/chat/completions", payload)
        result["chat"].append({"name": item["name"], "status": status, "elapsed_s": elapsed, "body": body})

    fixed_payload = {
        "text": "EMPTY OUTPUT",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
            "ignore_eos": True,
        },
    }
    status, body, elapsed = post_json("/generate", fixed_payload)
    result["fixed_generate"] = {"status": status, "elapsed_s": elapsed, "body": body}
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(out), "health": result["health"]["ok"], "chat": [x["status"] for x in result["chat"]], "generate": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
