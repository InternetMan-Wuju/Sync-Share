#!/usr/bin/env python3
"""Deterministic GLM-5.2 chunk benchmark: literal 32K shared prefix, 60% token reuse."""
import argparse
import concurrent.futures
import json
import math
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

VOCAB = 154880  # Historical GLM-5.2 bench uses token IDs [1,154879].


def request(url, ids, output, timeout, role, first_token=None):
    payload = json.dumps({
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": output, "temperature": 0, "ignore_eos": True},
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(url + "/generate", payload,
                                 {"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    wall_start = time.time()
    first = None
    meta = {}
    events = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                observed = obj.get("meta_info") or {}
                meta.update(observed)
                new_ids = obj.get("output_ids") or []
                total = observed.get("completion_tokens")
                if first is None and (new_ids or (isinstance(total, int) and total > 0)):
                    first = time.perf_counter()
                    if first_token is not None:
                        first_token.set()
                if first is not None:
                    events.append({"t": round(time.perf_counter() - start, 6),
                                   "ids_in_event": len(new_ids), "completion_tokens": total})
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            detail = exc.read(1200).decode("utf-8", "replace")
        else:
            detail = ""
        if first_token is not None:
            first_token.set()
        return {"role": role, "error": repr(exc), "response": detail,
                "started_at": wall_start, "input_tokens_sent": len(ids)}
    end = time.perf_counter()
    out = meta.get("completion_tokens")
    if not isinstance(out, int):
        out = max((e["completion_tokens"] for e in events if isinstance(e["completion_tokens"], int)), default=None)
    if out is None:
        out = sum(e["ids_in_event"] for e in events)
    ttft = (first - start) if first is not None else None
    return {"role": role, "started_at": wall_start, "input_tokens_sent": len(ids),
            "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "finish_reason": meta.get("finish_reason"),
            "output_tokens": out, "requested_output_tokens": output,
            "ttft_s": ttft, "e2e_s": end - start,
            "tpot_ms": (end - first) * 1000 / (out - 1) if first is not None and out > 1 else None,
            "events": events, "meta_info": meta}


def flush(url, timeout):
    req = urllib.request.Request(url + "/flush_cache", b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "body": resp.read(300).decode("utf-8", "replace")}
    except Exception as exc:
        return {"error": repr(exc)}


def ids(rng, length):
    return [rng.randrange(1, VOCAB) for _ in range(length)]


def stats(values):
    arr = sorted(x for x in values if x is not None)
    if not arr:
        return None
    return {"mean": statistics.mean(arr), "p50": arr[len(arr) // 2],
            "p95": arr[math.ceil(len(arr) * .95) - 1], "p99": arr[math.ceil(len(arr) * .99) - 1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://10.120.72.23:30080")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--input-tokens", type=int, default=54614)
    ap.add_argument("--prefix-tokens", type=int, default=32768)
    ap.add_argument("--output-tokens", type=int, default=300)
    ap.add_argument("--decode-output-tokens", type=int, default=900)
    ap.add_argument("--decode-input-tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    if a.input_tokens <= a.prefix_tokens or a.prefix_tokens < 1:
        ap.error("input-tokens must exceed prefix-tokens")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    outfile = a.out_dir / (a.mode + ".jsonl")
    def record(stage, result):
        row = {"stage": stage, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **result}
        entries.append(row)
        with outfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{stage}] {row.get('role', 'control')} output={row.get('output_tokens')} "
              f"cached={row.get('cached_tokens')} ttft={row.get('ttft_s')} "
              f"tpot_ms={row.get('tpot_ms')} error={row.get('error')}", flush=True)
        return row
    rng = random.Random(a.seed)
    prefix = ids(rng, a.prefix_tokens)
    def long_input():
        return prefix + ids(rng, a.input_tokens - a.prefix_tokens)
    def prime(stage):
        record(stage, {"role": "flush_cache", **flush(a.url, a.timeout)})
        return record(stage, request(a.url, prefix, 1, a.timeout, "warmup"))
    print(json.dumps({"mode": a.mode, "url": a.url, "input_tokens": a.input_tokens,
                      "shared_prefix_tokens": a.prefix_tokens, "target_hit_ratio": a.prefix_tokens / a.input_tokens,
                      "output_tokens": a.output_tokens, "seed": a.seed, "reps": a.reps}, ensure_ascii=False), flush=True)
    warm = prime("smoke" if a.mode == "smoke" else "init")
    if "error" in warm:
        raise SystemExit("prefix warmup failed; see jsonl")
    if a.mode == "smoke":
        record("smoke", request(a.url, long_input(), a.output_tokens, a.timeout, "long_warm"))
    else:
        record("cold", {"role": "flush_cache", **flush(a.url, a.timeout)})
        record("cold", request(a.url, long_input(), a.output_tokens, a.timeout, "long_cold"))
        for i in range(a.reps):
            decode = ids(rng, a.decode_input_tokens)
            stage = f"baseline_{i}"
            warm = prime(stage)
            if "error" in warm:
                break
            record(stage, request(a.url, decode, a.decode_output_tokens, a.timeout, "decode"))
            stage = f"mixed_{i}"
            warm = prime(stage)
            if "error" in warm:
                break
            first = threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                future = pool.submit(request, a.url, decode, a.decode_output_tokens, a.timeout, "decode", first)
                started = first.wait(timeout=a.timeout)
                if not started:
                    record(stage, {"role": "control", "error": "decode first token timeout"})
                injected_at = time.time()
                record(stage, {"role": "control", "injected_at": injected_at})
                record(stage, request(a.url, long_input(), a.output_tokens, a.timeout, "long_warm"))
                record(stage, future.result())
        # Length sweep includes the main requested case and a shorter reference, each with 60% shared tokens.
        for length in (16384, 32768):
            stage = f"length_{length}"
            record(stage, {"role": "flush_cache", **flush(a.url, a.timeout)})
            local_rng = random.Random(a.seed + length)
            p = ids(local_rng, round(.6 * length))
            record(stage, request(a.url, p, 1, a.timeout, "warmup"))
            record(stage, request(a.url, p + ids(local_rng, length - len(p)), a.output_tokens, a.timeout, "long_warm"))
    rows = [r for r in entries if r.get("role") in ("decode", "long_warm", "long_cold")]
    summary = {"config": vars(a) | {"out_dir": str(a.out_dir)},
               "stages": {stage: {
                   "ttft_s": stats([r.get("ttft_s") for r in rows if r["stage"] == stage]),
                   "tpot_ms": stats([r.get("tpot_ms") for r in rows if r["stage"] == stage]),
                   "cache_hit_ratio": (sum(r["cached_tokens"] for r in rows if r["stage"] == stage and isinstance(r.get("cached_tokens"), int)) /
                                       sum(r["prompt_tokens"] for r in rows if r["stage"] == stage and isinstance(r.get("cached_tokens"), int) and isinstance(r.get("prompt_tokens"), int))
                                       if any(r["stage"] == stage and isinstance(r.get("cached_tokens"), int) and isinstance(r.get("prompt_tokens"), int) for r in rows) else None),
                   "errors": [r.get("error") for r in rows if r["stage"] == stage and r.get("error")],
               } for stage in sorted(set(r["stage"] for r in rows))}}
    target = a.out_dir / (a.mode + "_summary.json")
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary=" + str(target), flush=True)


if __name__ == "__main__":
    main()
