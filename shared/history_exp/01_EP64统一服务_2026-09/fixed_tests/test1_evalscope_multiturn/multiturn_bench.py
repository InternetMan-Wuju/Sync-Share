#!/usr/bin/env python3
"""复刻测试1：多轮对话 + 长上下文 + 前缀缓存（SGLang radix cache 默认开）。"""
import json, time, urllib.request, concurrent.futures, random

URL = "http://127.0.0.1:30000"
JSONL = "/home/chengyetai/workspace/glm52_ep64_chengyetai/dataset/databricks-dolly-15k.jsonl"
N_CONV = 16
N_TURNS = 5
FIRST_TURN_TOKENS = 16384  # 近似
SUBSEQUENT_TOKENS = 2048
MAX_TOKENS = 1024

random.seed(1024)

# 从 dolly 抽句子，拼成约 first_turn 长度的初始 prompt
lines = []
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        s = (r.get("instruction") or "") + "\n" + (r.get("context") or "")
        lines.append(s.strip())
random.shuffle(lines)

def build_prompt(target_chars):
    parts, total = [], 0
    for s in lines:
        parts.append(s)
        total += len(s)
        if total >= target_chars:
            break
    return "\n\n".join(parts)

init_prompt = build_prompt(FIRST_TURN_TOKENS * 4)
new_chunk = build_prompt(SUBSEQUENT_TOKENS * 4)

def one_turn(messages, turn):
    body = {
        "model": "glm-5.2",
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
        "max_tokens": MAX_TOKENS,
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall = time.time() - t0
    usage = data.get("usage") or {}
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    return {
        "turn": turn, "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content_chars": len(content),
    }, content

def run_conversation(i):
    messages = [{"role": "user", "content": init_prompt}]
    results = []
    for t in range(1, N_TURNS + 1):
        rec, content = one_turn(messages, t)
        results.append(rec)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"{new_chunk}\n\n[turn {t} follow-up]"})
    return i, results

t0 = time.time()
all_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONV) as ex:
    for fut in concurrent.futures.as_completed([ex.submit(run_conversation, i) for i in range(N_CONV)]):
        i, results = fut.result()
        for r in results:
            r["conv"] = i
            all_results.append(r)

wall = time.time() - t0
out = {
    "n_conversations": N_CONV, "n_turns": N_TURNS, "wall_s": round(wall, 3),
    "results": all_results,
}
with open("multiturn_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# 统计
e2e = [r["wall_s"] for r in all_results]
e2e.sort()
def pct(a, q):
    return a[int(q * (len(a) - 1))]
print(json.dumps({
    "total_requests": len(all_results),
    "wall_s": round(wall, 3),
    "e2e_p50": round(pct(e2e, .5), 3),
    "e2e_p90": round(pct(e2e, .9), 3),
    "e2e_p99": round(pct(e2e, .99), 3),
    "prompt_tokens_sum": sum(r["prompt_tokens"] or 0 for r in all_results),
    "completion_tokens_sum": sum(r["completion_tokens"] or 0 for r in all_results),
}, ensure_ascii=False, indent=2))
