#!/usr/bin/env python3
"""Chunk Size 实验客户端：并发打 N 条请求，逐请求收 TTFT/E2E/输出token + 服务端 meta。

产出 p50/p95 的 TTFT/TPOT/E2E 与吞吐（含 per-NPU）。
用法:
  python3 chunk_client.py --url http://127.0.0.1:30080 --dataset datasets/w1_8k_1k.json \
      --out results/<run>/x.json --concurrency 4 --limit 8 [--flush-cache] [--warmup-prefix K]
"""
import argparse, json, pathlib, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

lock = threading.Lock()

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    if len(xs) == 1: return xs[0]
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def one(args, case):
    payload = {"input_ids": case["input_ids"],
               "sampling_params": {"max_new_tokens": args.max_tokens,
                                   "temperature": 0, "ignore_eos": True},
               "stream": True}
    req = urllib.request.Request(args.url + "/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = last = None; ntok = 0; meta = {}; err = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip(); now = time.monotonic()
                if body == b"[DONE]":
                    continue
                if first is None:
                    first = now
                last = now
                obj = json.loads(body)
                out = obj.get("output_ids") or []
                if len(out) > ntok: ntok = len(out)
                m = obj.get("meta_info") or {}
                if m: meta = m
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    ttft = None if first is None else first - t0
    e2e = None if last is None else last - t0
    ct = meta.get("completion_tokens") or ntok
    tpot = None
    if ttft is not None and e2e is not None and ct and ct > 1:
        tpot = (e2e - ttft) / (ct - 1)
    rec = {"index": case["index"], "error": err, "ttft_s": ttft, "e2e_s": e2e,
           "completion_tokens": ct, "tokens_seen": ntok,
           "prompt_tokens": meta.get("prompt_tokens"),
           "cached_tokens": meta.get("cached_tokens"), "tpot_s": tpot}
    with lock:
        print(f"  #{rec['index']:>2} out={ct:>5} ttft={ttft and round(ttft,2)}s "
              f"e2e={e2e and round(e2e,1)}s tpot={tpot and round(tpot*1000)}ms "
              f"cached={meta.get('cached_tokens')}/{meta.get('prompt_tokens')} {('ERR:'+err) if err else ''}")
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只取数据集前 N 条（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=0); ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--flush-cache", action="store_true")
    ap.add_argument("--warmup-prefix", type=int, default=0)
    ap.add_argument("--npu", type=int, default=16)
    a = ap.parse_args()
    ds = json.loads(pathlib.Path(a.dataset).read_text())
    cases = ds["cases"][:a.limit] if a.limit else ds["cases"]
    if not a.max_tokens: a.max_tokens = ds["target_output_tokens"]
    print(f"数据集 {pathlib.Path(a.dataset).name}: 用 {len(cases)} 条 x {ds['input_tokens']} 输入 / "
          f"{a.max_tokens} 输出, 并发={a.concurrency}")
    if a.flush_cache:
        try:
            rq = urllib.request.Request(a.url + "/flush_cache", data=b"{}",
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=60) as r: print("flush_cache:", r.status)
        except Exception as e: print("flush_cache 失败:", e)
    if a.warmup_prefix > 0:
        w = {"input_ids": cases[0]["input_ids"][:a.warmup_prefix],
             "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True}, "stream": True}
        try:
            rq = urllib.request.Request(a.url + "/generate", data=json.dumps(w).encode(),
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=900) as r:
                for _ in r: pass
            print(f"warmup 前缀 {a.warmup_prefix} token 完成")
        except Exception as e: print("warmup 失败:", e)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        recs = list(pool.map(lambda c: one(a, c), cases))
    wall = time.monotonic() - t0
    ok = [r for r in recs if not r["error"] and r["completion_tokens"]]
    agg = {}
    if ok:
        ct = sum(r["completion_tokens"] for r in ok)
        cached = [(r["cached_tokens"] or 0) / max(1, r["prompt_tokens"] or 1) for r in ok]
        agg = {
            "requests": len(cases), "requests_ok": len(ok), "wall_s": round(wall, 2),
            "output_tokens": ct, "output_tps": round(ct / wall, 3),
            "output_tps_per_npu": round(ct / wall / a.npu, 4),
            "ttft_p50_s": round(pct([r["ttft_s"] for r in ok], .5), 3),
            "ttft_p95_s": round(pct([r["ttft_s"] for r in ok], .95), 3),
            "tpot_p50_ms": round(1000 * pct([r["tpot_s"] for r in ok], .5), 2) if any(r["tpot_s"] for r in ok) else None,
            "tpot_p95_ms": round(1000 * pct([r["tpot_s"] for r in ok], .95), 2) if any(r["tpot_s"] for r in ok) else None,
            "e2e_p50_s": round(pct([r["e2e_s"] for r in ok], .5), 2),
            "e2e_p95_s": round(pct([r["e2e_s"] for r in ok], .95), 2),
            "ttft_mean_s": round(sum(r["ttft_s"] for r in ok) / len(ok), 3),
            "cache_hit_mean": round(sum(cached) / len(cached), 4),
        }
    out = {"dataset": str(a.dataset), "name": ds.get("name"), "url": a.url, "limit": len(cases),
           "max_tokens": a.max_tokens, "concurrency": a.concurrency, "npu": a.npu,
           "aggregate": agg, "requests": recs}
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n== 汇总 ==")
    for k, v in agg.items(): print(f"  {k}: {v}")
    print(f"  原始结果: {p}")

if __name__ == "__main__":
    main()
