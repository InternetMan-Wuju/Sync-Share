#!/usr/bin/env python3
import json, time, urllib.request

URL = "http://127.0.0.1:30000"
JSONL = "/home/chengyetai/workspace/glm52_ep64_chengyetai/dataset/databricks-dolly-15k.jsonl"
TARGET_TOKENS = 21403  # 目标输入 token（近似）
MAX_TOKENS = 1024

parts = []
total_chars = 0
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        inst = row.get("instruction") or ""
        ctx = row.get("context") or ""
        p = f"{inst}\n\nContext:\n{ctx}" if ctx else inst
        parts.append(p)
        total_chars += len(p)
        # 英文约 4 字符/token，凑到目标字符数
        if total_chars >= TARGET_TOKENS * 4:
            break

prompt = "\n\n".join(parts)
body = {
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": prompt}],
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
choice = (data.get("choices") or [{}])[0]
msg = choice.get("message") or {}
print(json.dumps({
    "wall_s": round(wall, 3),
    "prompt_chars": total_chars,
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "finish_reason": choice.get("finish_reason"),
    "content_chars": len(msg.get("content") or ""),
}, ensure_ascii=False, indent=2))
