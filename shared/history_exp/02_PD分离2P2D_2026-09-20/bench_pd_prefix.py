#!/usr/bin/env python3
"""测试4 前缀缓存压测（对齐官方 UCM 构造）：
随机 token + 90% 公共前缀 + 多前缀(prefix_num=并发/2) + 前缀预热 + 并发。

客户端只测 TTFT / TPOT / E2E / token 吞吐；四段分解(queue+prefill+中间+decode)、
EPLB cost、Expert×Layer 热力图由服务端 metrics / 日志另行采集。
"""
import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request

VOCAB = 154880


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[idx]


def sse_iter(resp):
    buf = b""
    for chunk in resp:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                return
            yield data


def one_request(url, input_ids, max_new_tokens, timeout):
    body = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    req = urllib.request.Request(
        f"{url}/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.time()
    first_token_t = None
    last_token_t = None
    n_out = 0
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for data in sse_iter(resp):
            now = time.time()
            if first_token_t is None:
                first_token_t = now
            last_token_t = now
            try:
                obj = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            n_out += len(obj.get("output_ids") or [])
            meta = obj.get("meta_info") or {}
            if meta.get("prompt_tokens") is not None:
                prompt_tokens = meta.get("prompt_tokens")
            if meta.get("completion_tokens") is not None:
                completion_tokens = meta.get("completion_tokens")
    t_end = time.time()
    ttft = (first_token_t - t_start) if first_token_t else None
    out = completion_tokens if completion_tokens is not None else n_out
    # TPOT：/generate 攒批回传，SSE 事件间隔不可靠，用 (e2e - ttft)/(out-1) 口径。
    tpot = None
    if ttft is not None and out > 1:
        tpot = (t_end - first_token_t) / (out - 1)
    return {
        "e2e": t_end - t_start,
        "ttft": ttft,
        "tpot": tpot,
        "out_tokens": out,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else len(input_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18623")
    ap.add_argument("--input-tokens", type=int, default=16384)
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=0)  # 默认 = 并发*4
    ap.add_argument("--prefix-ratio", type=float, default=0.9)
    ap.add_argument("--prefix-num", type=int, default=0)  # 默认 = 并发/2
    ap.add_argument("--no-warmup", action="store_true", help="跳过前缀预热（对照用）")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.requests <= 0:
        args.requests = args.concurrency * 4
    if args.prefix_num <= 0:
        args.prefix_num = args.concurrency // 2

    rng = random.Random(args.seed)
    prefix_len = int(args.input_tokens * args.prefix_ratio)
    suffix_len = max(1, args.input_tokens - prefix_len)

    # 生成 prefix_num 种互不相同的公共前缀
    prefixes = [
        [rng.randint(1, VOCAB - 1) for _ in range(prefix_len)]
        for _ in range(args.prefix_num)
    ]

    # 前缀预热：每种前缀发一个"只含前缀"的请求，让前缀 KV 进 radix cache。
    warmup_s = 0.0
    if not args.no_warmup:
        print(f"[warmup] 预热 {args.prefix_num} 个前缀（每个 {prefix_len} token）...", flush=True)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.prefix_num) as ex:
            futs = [ex.submit(one_request, args.url, p, 1, args.timeout) for p in prefixes]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[warmup] prefix {i} error={e}", flush=True)
        warmup_s = time.time() - t0
        print(f"[warmup] done in {warmup_s:.2f}s", flush=True)

    # 正式请求：轮询选前缀 + 独有 suffix
    inputs = []
    for i in range(args.requests):
        p = prefixes[i % args.prefix_num]
        suffix = [rng.randint(1, VOCAB - 1) for _ in range(suffix_len)]
        inputs.append(p + suffix)

    print(
        f"[bench] url={args.url} input={args.input_tokens} "
        f"(prefix={prefix_len}/suffix={suffix_len}) output={args.output_tokens} "
        f"concurrency={args.concurrency} requests={args.requests} "
        f"prefix_num={args.prefix_num} warmup={'off' if args.no_warmup else 'on'}",
        flush=True,
    )

    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(one_request, args.url, ids, args.output_tokens, args.timeout)
            for ids in inputs
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:
                r = {"error": str(e)}
            results.append(r)
            if "error" in r:
                print(f"[{i}/{args.requests}] error={r.get('error')}", flush=True)
            else:
                ttft_s = f"{r['ttft']:.3f}" if r.get('ttft') is not None else "None"
                tpot_ms = f"{r['tpot']*1000:.2f}" if r.get('tpot') is not None else "None"
                print(
                    f"[{i}/{args.requests}] e2e={r['e2e']:.3f}s "
                    f"ttft={ttft_s}s tpot={tpot_ms}ms "
                    f"out={r['out_tokens']}",
                    flush=True,
                )
    wall = time.time() - t0

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    e2e = [r["e2e"] for r in ok]
    ttft = [r["ttft"] for r in ok if r.get("ttft") is not None]
    tpot = [r["tpot"] for r in ok if r.get("tpot") is not None]
    out_tokens = sum(r.get("out_tokens") or 0 for r in ok)
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok)
    total_tokens = prompt_tokens + out_tokens

    summary = {
        "url": args.url,
        "input_tokens": args.input_tokens,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "prefix_ratio": args.prefix_ratio,
        "prefix_num": args.prefix_num,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "warmup": not args.no_warmup,
        "warmup_s": round(warmup_s, 3),
        "completed": len(ok),
        "errors": len(errs),
        "wall_s": round(wall, 3),
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": out_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_s": round(out_tokens / wall, 3) if wall else None,
        "total_tokens_per_s": round(total_tokens / wall, 3) if wall else None,
        "ttft_mean_ms": round(statistics.mean(ttft) * 1000, 2) if ttft else None,
        "ttft_p50_ms": round(pct(ttft, 0.5) * 1000, 2) if ttft else None,
        "ttft_p90_ms": round(pct(ttft, 0.9) * 1000, 2) if ttft else None,
        "ttft_p99_ms": round(pct(ttft, 0.99) * 1000, 2) if ttft else None,
        "tpot_mean_ms": round(statistics.mean(tpot) * 1000, 2) if tpot else None,
        "tpot_p50_ms": round(pct(tpot, 0.5) * 1000, 2) if tpot else None,
        "tpot_p90_ms": round(pct(tpot, 0.9) * 1000, 2) if tpot else None,
        "e2e_mean_ms": round(statistics.mean(e2e) * 1000, 2) if e2e else None,
        "e2e_p50_ms": round(pct(e2e, 0.5) * 1000, 2) if e2e else None,
        "e2e_p90_ms": round(pct(e2e, 0.9) * 1000, 2) if e2e else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"[bench] wrote -> {args.out}")


if __name__ == "__main__":
    main()
