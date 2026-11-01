#!/usr/bin/env python3
"""多 JSONL 接续压测：分桶/抽样 + 时间预算 + 断点续跑。

把多个 JSONL 按给定顺序合并成一个任务流，逐条发给 GLM-5.2(SGLang) 服务。

常用组合：
  1) 时间预算：--duration 3600（跑满 1 小时，到点不再提交新请求，在途请求跑完）
  2) 分桶抽样：--sample 2000 --stratify category --seed 0
  3) 断点续跑：上次用 --progress 记录了完成情况，本次加 --resume 跳过已完成 key
  4) 预览：--dry-run（只打印抽样/任务流，不发请求）

说明：--jsonl 可传多个文件，按命令行顺序接续；先到 --duration 就停，
不关心单个文件要跑多久。剩余部分下次 --resume 接着跑即可。
"""

import argparse
import concurrent.futures
import json
import os
import random
import time
import urllib.request


def percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


def build_prompt(row):
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    prompt = instruction
    if context:
        prompt = f"{instruction}\n\nContext:\n{context}"
    return prompt


def load_jobs(files, per_file_limit=0):
    jobs = []
    for path in files:
        dataset = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if per_file_limit and line_no >= per_file_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                jobs.append(
                    {
                        "key": f"{dataset}#{line_no}",
                        "dataset": dataset,
                        "file": path,
                        "line_no": line_no,
                        "category": row.get("category"),
                        "prompt": build_prompt(row),
                    }
                )
    return jobs


def _bucket_key(job, field):
    v = job.get(field)
    if v is None:
        return "__none__"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def sample_jobs(jobs, sample, seed, stratify):
    """抽样：stratify 为空则均匀随机，否则按 field 分桶比例抽样。"""
    if sample <= 0 or sample >= len(jobs):
        return jobs

    rng = random.Random(seed)
    if not stratify:
        idxs = list(range(len(jobs)))
        rng.shuffle(idxs)
        selected = [jobs[i] for i in idxs[:sample]]
    else:
        buckets = {}
        for j in jobs:
            buckets.setdefault(_bucket_key(j, stratify), []).append(j)

        total = len(jobs)
        names = list(buckets.keys())
        counts = {}
        assigned = 0
        for name in names:
            n = max(1, int(round(sample * len(buckets[name]) / total)))
            counts[name] = min(n, len(buckets[name]))
            assigned += counts[name]

        if assigned < sample:
            for name in sorted(
                names, key=lambda n: len(buckets[n]) - counts[n], reverse=True
            ):
                add = min(len(buckets[name]) - counts[name], sample - assigned)
                counts[name] += add
                assigned += add
                if assigned >= sample:
                    break

        while assigned > sample:
            name = max(names, key=lambda n: counts[n])
            counts[name] -= 1
            assigned -= 1

        selected = []
        for name in names:
            idxs = list(range(len(buckets[name])))
            rng.shuffle(idxs)
            selected.extend(buckets[name][i] for i in idxs[: counts[name]])

    order = {id(j): i for i, j in enumerate(jobs)}
    selected.sort(key=lambda j: order[id(j)])
    return selected


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


def read_progress_keys(path):
    keys = set()
    if not path or not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line).get("key"))
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out", default="multi_bench.json")
    ap.add_argument("--progress", default="multi_bench.progress.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="总任务数上限，0 表示不限")
    ap.add_argument("--per-file-limit", type=int, default=0, help="每个文件的行数上限，0 表示不限")
    ap.add_argument("--duration", type=int, default=0, help="wall-clock 秒数，0 表示不限")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--sample", type=int, default=0, help="抽样条数，0 表示不抽样")
    ap.add_argument("--stratify", default="", help="分桶字段，如 category 或 dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过 --progress 中已完成 key")
    ap.add_argument("--dry-run", action="store_true", help="只预览任务流，不发请求")
    a = ap.parse_args()

    jobs = load_jobs(a.jsonl, a.per_file_limit)
    total_loaded = len(jobs)

    if a.resume:
        done = read_progress_keys(a.progress)
        before = len(jobs)
        jobs = [j for j in jobs if j["key"] not in done]
        print(f"[resume] skip {before - len(jobs)} 个已完成 key", flush=True)

    jobs = sample_jobs(jobs, a.sample, a.seed, a.stratify)
    if a.limit:
        jobs = jobs[: a.limit]

    if a.dry_run:
        from collections import Counter

        by_file = Counter(j["dataset"] for j in jobs)
        by_bucket = Counter(_bucket_key(j, a.stratify or "category") for j in jobs)
        print(f"[dry-run] files={len(a.jsonl)} loaded={total_loaded} selected={len(jobs)}")
        print("按文件:", dict(by_file))
        print(f"按分桶({a.stratify or 'category'}):", dict(by_bucket))
        print("顺序前 10 条:")
        for j in jobs[:10]:
            print(f"  {j['key']}  bucket={j.get(a.stratify or 'category')}")
        return

    deadline = time.time() + a.duration if a.duration > 0 else None
    t0 = time.time()
    results = []
    submitted = 0
    stopped_by_deadline = False

    with open(a.progress, "a", encoding="utf-8") as pf:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futures = {}
            job_iter = iter(jobs)
            exhausted = False

            while True:
                # 到点后不再提交新任务，但已提交的任务要跑完
                while len(futures) < a.concurrency and not exhausted:
                    if deadline is not None and time.time() >= deadline:
                        stopped_by_deadline = True
                        break
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    fut = ex.submit(
                        one_request, a.url, job["prompt"], a.max_tokens, a.timeout
                    )
                    futures[fut] = job
                    submitted += 1

                if not futures:
                    break

                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    job = futures.pop(fut)
                    rec = fut.result()
                    rec["key"] = job["key"]
                    rec["dataset"] = job["dataset"]
                    rec["line_no"] = job["line_no"]
                    rec["category"] = job["category"]
                    results.append(rec)
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pf.flush()
                    print(
                        f"[{len(results)}/{submitted}] {job['key']} "
                        f"status={rec.get('status') or 'ERR'} "
                        f"e2e={rec.get('e2e_s')} error={rec.get('error')}",
                        flush=True,
                    )

    wall = round(time.time() - t0, 3)
    ok = [r for r in results if not r.get("error")]
    e2e = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    usages = [r.get("usage") or {} for r in ok]
    prompt_tokens = sum(u.get("prompt_tokens") or 0 for u in usages)
    completion_tokens = sum(u.get("completion_tokens") or 0 for u in usages)

    from collections import Counter

    by_file = Counter(r["dataset"] for r in results)
    summary = {
        "url": a.url,
        "files": [os.path.basename(p) for p in a.jsonl],
        "loaded": total_loaded,
        "selected": len(jobs),
        "submitted": submitted,
        "completed": len(results),
        "sample": a.sample,
        "stratify": a.stratify or None,
        "seed": a.seed,
        "duration_budget_s": a.duration,
        "stopped_by_deadline": stopped_by_deadline,
        "wall_s": wall,
        "concurrency": a.concurrency,
        "max_tokens": a.max_tokens,
        "ok": len(ok),
        "err": len(results) - len(ok),
        "e2e_p50_s": percentile(e2e, 0.5),
        "e2e_p90_s": percentile(e2e, 0.9),
        "e2e_p99_s": percentile(e2e, 0.99),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "requests_per_s": round(len(results) / wall, 3) if wall else None,
        "tokens_per_s": round((prompt_tokens + completion_tokens) / wall, 1)
        if wall
        else None,
        "by_file": dict(by_file),
    }

    output = {"summary": summary, "results": results}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[bench_multi] saved -> {a.out}")
    print(f"[bench_multi] progress -> {a.progress}")


if __name__ == "__main__":
    main()
