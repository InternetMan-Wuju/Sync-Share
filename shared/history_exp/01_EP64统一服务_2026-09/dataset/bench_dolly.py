#!/usr/bin/env python3
"""把 Dolly-15k JSONL 直接发给 GLM-5.2 服务。"""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


def one_request(url, prompt, max_tokens, timeout):
    body = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    rec = {"error": None, "status": None, "e2e_s": None, "usage": None}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rec["status"] = resp.status
            rec["e2e_s"] = round(time.time() - t0, 3)
            rec["usage"] = data.get("usage")
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            rec["finish_reason"] = choice.get("finish_reason")
            rec["content_chars"] = len(msg.get("content") or "")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}:{str(e)[:200]}"
        rec["e2e_s"] = round(time.time() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--jsonl", default="databricks-dolly-15k.jsonl")
    ap.add_argument("--out", default="dolly_bench.json")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    jobs = []
    with open(a.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instruction = row.get("instruction") or ""
            context = row.get("context") or ""
            prompt = instruction
            if context:
                prompt = f"{instruction}\n\nContext:\n{context}"
            jobs.append({"id": i, "prompt": prompt, "category": row.get("category")})

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futures = {
            ex.submit(one_request, a.url, job["prompt"], a.max_tokens, a.timeout): job
            for job in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            job = futures[fut]
            rec = fut.result()
            rec["id"] = job["id"]
            rec["category"] = job["category"]
            results.append(rec)
            print(f"[{len(results)}/{len(jobs)}] id={job['id']} "
                  f"status={rec.get('status') or 'ERR'} "
                  f"e2e={rec.get('e2e_s')} error={rec.get('error')}", flush=True)

    wall = round(time.time() - t0, 3)
    ok = [r for r in results if not r.get("error")]
    e2e = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    usages = [r.get("usage") or {} for r in ok]
    prompt_tokens = sum(u.get("prompt_tokens") or 0 for u in usages)
    completion_tokens = sum(u.get("completion_tokens") or 0 for u in usages)

    summary = {
        "url": a.url,
        "dataset": "databricks/databricks-dolly-15k",
        "limit": len(jobs),
        "concurrency": a.concurrency,
        "max_tokens": a.max_tokens,
        "wall_s": wall,
        "ok": len(ok),
        "err": len(results) - len(ok),
        "e2e_p50_s": percentile(e2e, 0.5),
        "e2e_p90_s": percentile(e2e, 0.9),
        "e2e_p99_s": percentile(e2e, 0.99),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "requests_per_s": round(len(jobs) / wall, 3) if wall else None,
        "tokens_per_s": round((prompt_tokens + completion_tokens) / wall, 1) if wall else None,
    }

    output = {"summary": summary, "results": results}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[bench] saved -> {a.out}")


if __name__ == "__main__":
    main()
