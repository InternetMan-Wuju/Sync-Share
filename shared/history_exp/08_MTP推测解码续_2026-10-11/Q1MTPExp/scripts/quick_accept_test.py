#!/usr/bin/env python3
"""快速 MTP 接受率对比测试。

对一个数据集发 N 条请求（可并发），收集：
  - 服务端 spec 统计：spec_verify_ct / spec_num_correct_drafts / spec_num_proposed_drafts
                      -> 加权接受长度 AL、加权 draft 接受率
  - TTFT / E2E / 输出 token 数
  - 输出退化指标：不同 token 数、最小重复周期（判断是否"3 个 token 无限循环"）
用法: python3 quick_accept_test.py --url http://127.0.0.1:30080 --dataset datasets/random_1k256.json \
        --out results/<run>/random.json [--concurrency 4]
"""
import argparse, json, pathlib, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from mtp_output_audit import output_shape

lock = threading.Lock()

def period_of(ids, maxp=8):
    for p in range(1, maxp + 1):
        if len(ids) > p * 3 and all(ids[i] == ids[i - p] for i in range(p, len(ids))):
            return p
    return None

def one(args, case):
    prompt = {"text": case["text"]} if "text" in case else {"input_ids": case["input_ids"]}
    payload = {**prompt,
               "sampling_params": {"max_new_tokens": args.max_tokens,
                                   "temperature": 0, "ignore_eos": True},
               "stream": True}
    req = urllib.request.Request(args.url + "/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = last = None; ids = []; text = ""; meta = {}
    err = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip(); now = time.monotonic()
                if first is None: first = now
                last = now
                if body == b"[DONE]":
                    continue
                obj = json.loads(body)
                out = obj.get("output_ids") or []
                if len(out) >= len(ids): ids = out
                if obj.get("text"): text = obj["text"]
                m = obj.get("meta_info") or {}
                if m: meta = m
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    rec = {"index": case["index"], "error": err,
           "ttft_s": None if first is None else first - t0,
           "e2e_s": None if last is None else last - t0,
           "completion_tokens": len(ids), "distinct_ids": len(set(ids)),
           "period": period_of(ids), "text_head": text[:120].replace("\n", " "),
           "meta": meta,
           "sampling_params": payload["sampling_params"],
           "input_kind": next(iter(prompt)),
           "output_ids": ids, "text": text,
           "output_shape": output_shape(ids)}
    hist = (rec["meta"].get("spec_correct_drafts_histogram")
            or rec["meta"].get("spec_accept_histogram"))
    if hist:
        N = sum(hist)
        rec["pos_rates"] = [sum(hist[i:]) / N for i in (1, 2, 3)] if N else None
    else:
        rec["pos_rates"] = None
    with lock:
        print(f"  #{rec['index']:>2} tok={rec['completion_tokens']:>4} distinct={rec['distinct_ids']:>4} "
              f"period={rec['period']} AL={rec['meta'].get('spec_accept_length')} "
              f"rate={rec['meta'].get('spec_accept_rate')} ttft={rec['ttft_s'] and round(rec['ttft_s'],2)}s "
              f"e2e={rec['e2e_s'] and round(rec['e2e_s'],1)}s")
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=0); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--flush-cache", action="store_true")
    ap.add_argument("--warmup-prefix", type=int, default=0)
    a = ap.parse_args()
    ds = json.loads(pathlib.Path(a.dataset).read_text())
    if not a.max_tokens: a.max_tokens = ds["target_output_tokens"]
    print(f"数据集 {pathlib.Path(a.dataset).name}: {len(ds['cases'])} 条 x {ds['input_tokens']} 输入 / "
          f"{a.max_tokens} 输出, 并发={a.concurrency}")
    if a.flush_cache:
        try:
            req = urllib.request.Request(a.url + "/flush_cache", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                print("flush_cache:", r.status)
        except Exception as e:
            print("flush_cache 失败:", e)
    if a.warmup_prefix > 0:
        w = {"input_ids": ds["cases"][0]["input_ids"][:a.warmup_prefix],
             "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
             "stream": True}
        try:
            req = urllib.request.Request(a.url + "/generate", data=json.dumps(w).encode(),
                                         headers={"Content-Type": "application/json"})
            n = 0
            with urllib.request.urlopen(req, timeout=600) as r:
                for line in r:
                    if line.startswith(b"data:"): n += 1
            print(f"warmup 前缀 {a.warmup_prefix} token 完成(事件 {n})")
        except Exception as e:
            print("warmup 失败:", e)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        recs = list(pool.map(lambda c: one(a, c), ds["cases"]))
    wall = time.monotonic() - t0
    ok = [r for r in recs if not r["error"] and r["meta"].get("spec_verify_ct")]
    agg = {}
    if ok:
        sv = sum(r["meta"]["spec_verify_ct"] for r in ok)
        cd = sum(r["meta"]["spec_num_correct_drafts"] for r in ok)
        pd_ = sum(r["meta"]["spec_num_proposed_drafts"] for r in ok)
        ct = sum(r["completion_tokens"] for r in ok)
        agg = {"requests_ok": len(ok), "wall_s": round(wall, 1),
               "completion_tokens": ct, "verify_ct": sv,
               "AL_weighted": round(ct / sv, 3),
               "accept_rate_weighted": round(cd / pd_, 4) if pd_ else None,
               "ttft_mean_s": round(sum(r["ttft_s"] for r in ok) / len(ok), 2),
               "e2e_mean_s": round(sum(r["e2e_s"] for r in ok) / len(ok), 1),
               "degenerate_requests": sum(1 for r in ok if (r["period"] or 99) <= 4),
               "periodic_suffix_requests": sum(r["output_shape"]["periodic_suffix"] is not None for r in ok),
               "distinct_ids_mean": round(sum(r["distinct_ids"] for r in ok) / len(ok), 1)}
    if ok:
        den = 0; num = [0, 0, 0]
        for r in ok:
            h = r["meta"].get("spec_correct_drafts_histogram")
            if not h: continue
            den += sum(h)
            for i in (1, 2, 3):
                num[i - 1] += sum(h[i:])
        if den:
            agg["pos_rates_weighted"] = [round(num[i] / den, 4) for i in range(3)]
    out = {"dataset": str(a.dataset), "url": a.url, "max_tokens": a.max_tokens,
           "concurrency": a.concurrency, "aggregate": agg, "requests": recs}
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n== 汇总 ==")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"  原始结果: {p}")

if __name__ == "__main__":
    main()
