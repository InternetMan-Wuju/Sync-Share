#!/usr/bin/env python3
"""Full-length MTP ON collector with durable per-request data and progress."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mtp_output_audit import output_shape


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    temp.replace(path)


def rates(meta):
    h = meta.get("spec_correct_drafts_histogram") or meta.get("spec_accept_histogram")
    if not h or not sum(h):
        return None
    return [sum(h[i:]) / sum(h) for i in (1, 2, 3)]


def histogram(meta):
    # The server emits only up to the largest observed number of accepted drafts.
    h = meta.get("spec_correct_drafts_histogram") or meta.get("spec_accept_histogram") or []
    if not 1 <= len(h) <= 4:
        return None
    return list(h) + [0] * (4-len(h))


def capture(url, payload, target, case, progress=None, timeout=1800):
    target = Path(target)
    t0 = time.monotonic()
    started_unix = time.time()
    ids, text, meta = [], "", {}
    first = last_token = end = None
    error, done = None, False
    event_count, next_progress = 0, 0
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url + "/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    write_json(target.with_suffix(".payload.json"), payload)
    stream_path = target.with_suffix(".events.jsonl.gz")
    try:
        with gzip.open(stream_path, "wt", encoding="utf-8", compresslevel=1) as journal:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    now = time.monotonic()
                    body = line[5:].strip()
                    if body == b"[DONE]":
                        done = True
                        end = now
                        journal.write(json.dumps({"elapsed_s": now-t0, "done": True}) + "\n")
                        continue
                    obj = json.loads(body)
                    if "error" in obj:
                        raise RuntimeError(str(obj["error"]))
                    new_ids = obj.get("output_ids")
                    old_count = len(ids)
                    delta = []
                    if new_ids is not None:
                        if len(new_ids) < old_count or new_ids[:old_count] != ids:
                            raise RuntimeError("stream output_ids are not cumulative")
                        delta = new_ids[old_count:]
                        ids = new_ids
                    if obj.get("text") is not None:
                        text = obj["text"]
                    if obj.get("meta_info"):
                        meta = obj["meta_info"]
                    if delta:
                        if first is None:
                            first = now
                        last_token = now
                    end = now
                    event_count += 1
                    journal.write(json.dumps({"elapsed_s": now-t0, "delta_ids": delta,
                                              "output_count": len(ids), "meta_info": meta},
                                             ensure_ascii=False, separators=(",", ":")) + "\n")
                    if progress and now >= next_progress:
                        progress(case["index"], {"state": "running", "output_tokens": len(ids),
                                                 "target_output_tokens": payload["sampling_params"]["max_new_tokens"],
                                                 "elapsed_s": round(now-t0, 1), "last_event_unix": time.time(),
                                                 "AL": meta.get("spec_accept_length"), "P1_P2_P3": rates(meta)})
                        next_progress = now + 15
                        journal.flush()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, "read"):
            error += ": " + exc.read().decode(errors="replace")[:16384]
        end = time.monotonic()
    e2e = end - t0 if end is not None else None
    ttft = first - t0 if first is not None else None
    decode = last_token - first if first is not None and last_token is not None else None
    record = {
        "index": case["index"], "error": error, "stream_done": done,
        "started_unix": started_unix, "ended_unix": time.time(),
        "input_sha256": case.get("input_sha256"), "input_tokens_sent": len(payload["input_ids"]),
        "sampling_params": payload["sampling_params"], "completion_tokens": len(ids),
        "ttft_s": ttft, "e2e_s": e2e, "decode_s": decode,
        "tpot_ms": decode*1000/(len(ids)-1) if decode is not None and len(ids)>1 else None,
        "client_finalization_s": end-last_token if end is not None and last_token is not None else None,
        "output_ids": ids, "text": text, "meta": meta, "pos_rates": rates(meta),
        "output_shape": output_shape(ids), "event_count": event_count,
        "events_file": str(stream_path), "payload_file": str(target.with_suffix(".payload.json")),
    }
    write_json(target, record)
    if progress:
        progress(case["index"], {"state": "finished" if not error else "failed",
                                 "output_tokens": len(ids), "elapsed_s": round(e2e, 1),
                                 "target_output_tokens": payload["sampling_params"]["max_new_tokens"],
                                 "last_event_unix": time.time(), "AL": meta.get("spec_accept_length"),
                                 "P1_P2_P3": rates(meta), "error": error})
    print(f"request #{case['index']}: output={len(ids)} AL={meta.get('spec_accept_length')} "
          f"Per-position acceptance rate: {rates(meta)} error={error}", flush=True)
    return record


def validate(record, length, output, prefix=False):
    meta = record["meta"]
    h = histogram(meta) or []
    verify = meta.get("spec_verify_ct")
    checks = {
        "no_error": record["error"] is None,
        "stream_done": record["stream_done"],
        "input_length": meta.get("prompt_tokens") == length,
        "output_length": record["completion_tokens"] == output == meta.get("completion_tokens"),
        "finish_length": (meta.get("finish_reason") or {}).get("type") == "length",
        "histogram_length": len(h) == 4,
        "histogram_verify": bool(h) and sum(h) == verify,
        "histogram_correct": bool(h) and sum(i*v for i,v in enumerate(h)) == meta.get("spec_num_correct_drafts"),
        "proposed": verify is not None and meta.get("spec_num_proposed_drafts") == 3*verify,
        "per_position_available": record["pos_rates"] is not None,
    }
    if prefix:
        checks["cache_hit_60pct"] = .585 <= meta.get("cached_tokens", 0)/length <= .615
    return {"index": record["index"], "checks": checks,
            "failures": [name for name, ok in checks.items() if not ok]}


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high]-values[low])*(position-low)


def aggregate(records, wall):
    valid = [r for r in records if not r["error"] and r["meta"].get("spec_verify_ct")]
    if not valid:
        return {"requests_ok": 0, "wall_s": wall}
    verify = sum(r["meta"]["spec_verify_ct"] for r in valid)
    correct = sum(r["meta"]["spec_num_correct_drafts"] for r in valid)
    proposed = sum(r["meta"]["spec_num_proposed_drafts"] for r in valid)
    count = sum(r["completion_tokens"] for r in valid)
    histograms = [histogram(r["meta"]) for r in valid]
    if any(h is None for h in histograms):
        raise ValueError("missing or unsupported acceptance histogram")
    hist = [sum(h[i] for h in histograms) for i in range(4)]
    prompt = sum(r["meta"]["prompt_tokens"] for r in valid)
    cached = sum(r["meta"].get("cached_tokens", 0) for r in valid)
    result = {"requests_ok": len(valid), "wall_s": wall, "completion_tokens": count,
              "output_tokens_per_s": count/wall, "verify_ct": verify,
              "correct_drafts": correct, "proposed_drafts": proposed,
              "histogram": hist, "AL_weighted": count/verify,
              "accept_rate_weighted": correct/proposed,
              "pos_rates_weighted": [sum(hist[i:])/sum(hist) for i in (1,2,3)],
              "cached_tokens": cached, "prompt_tokens": prompt,
              "cache_hit_ratio": cached/prompt,
              "periodic_suffix_requests": sum(r["output_shape"]["periodic_suffix"] is not None for r in valid)}
    for key in ("ttft_s", "tpot_ms", "e2e_s", "decode_s", "client_finalization_s"):
        values = [r[key] for r in valid if r[key] is not None]
        result[key+"_mean"] = statistics.mean(values) if values else None
        result[key+"_p50"] = percentile(values, .5)
        result[key+"_p95"] = percentile(values, .95)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30080")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ds = json.loads(Path(args.dataset).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = {"scenario": ds["name"], "state": "preparing", "requests": {}, "updated_unix": time.time()}
    lock = threading.Lock()

    def progress(index, info):
        with lock:
            state["state"] = "running"
            state["requests"][str(index)] = info
            state["updated_unix"] = time.time()
            write_json(out / "progress.json", state)

    write_json(out / "progress.json", state)
    request = urllib.request.Request(args.url+"/flush_cache", data=b"{}",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        (out / "flush_cache.json").write_bytes(response.read())
    prefix = ds.get("shared_prefix_tokens", 0)
    if prefix:
        payload = {"input_ids": ds["cases"][0]["input_ids"][:prefix], "stream": True,
                   "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True}}
        warmup = capture(args.url, payload, out/"prefix_warmup.json", {"index": -1})
        if warmup["error"] or warmup["completion_tokens"] != 1:
            raise RuntimeError("shared prefix warmup failed")
    started = time.monotonic()

    def run(case):
        ids = case["input_ids"]
        checksum = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if len(ids) != ds["input_tokens"] or checksum != case["input_sha256"]:
            raise RuntimeError("input length/SHA mismatch")
        payload = {"input_ids": ids, "stream": True,
                   "sampling_params": {"max_new_tokens": ds["target_output_tokens"],
                                       "temperature": 0, "ignore_eos": True}}
        progress(case["index"], {"state": "submitted", "output_tokens": 0,
                                 "target_output_tokens": ds["target_output_tokens"],
                                 "last_event_unix": time.time()})
        return capture(args.url, payload, out/f"request_{case['index']:03d}.json", case, progress)

    with ThreadPoolExecutor(max_workers=ds["concurrency"]) as executor:
        records = list(executor.map(run, ds["cases"]))
    wall = time.monotonic()-started
    write_json(out / "completed_requests.json", {"wall_s": wall, "requests": records})
    checks = [validate(r, ds["input_tokens"], ds["target_output_tokens"], bool(prefix)) for r in records]
    failed = [c for c in checks if c["failures"]]
    result = {"dataset": args.dataset, "scenario": ds["name"], "concurrency": ds["concurrency"],
              "target_output_tokens": ds["target_output_tokens"], "requests_expected": ds["n_requests"],
              "aggregate": aggregate(records, wall), "requests": records,
              "validation": checks, "valid": not failed}
    write_json(out / "result.json", result)
    state.update(state="completed" if not failed else "validation_failed", updated_unix=time.time())
    write_json(out / "progress.json", state)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print("validation_failures", failed, flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
