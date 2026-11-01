#!/usr/bin/env python3
"""Send several prompts to a node and flag degenerate/garbage output."""
import json, re, sys, urllib.request
PROMPTS = [
    "1加1等于几？只回复一句话。",
    "Hello, world. 翻译成中文。",
    "请用一句话介绍你自己。",
    "水的化学式是什么？",
]
def ask(url, prompt, max_tokens=64):
    body = json.dumps({"model":"glm-5.2","messages":[{"role":"user","content":prompt}],
                       "temperature":0,"max_tokens":max_tokens,"stream":False,
                       "chat_template_kwargs":{"enable_thinking":False}}).encode()
    req = urllib.request.Request(url.rstrip('/')+"/v1/chat/completions", data=body,
                                 headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        obj = json.loads(r.read())
    return obj["choices"][0]["message"].get("content",""), obj["choices"][0].get("finish_reason")
def degenerate(text):
    t = text.strip()
    if not t: return "EMPTY"
    if re.search(r"(.)\1{5,}", t, re.S): return "REPEATED_CHAR"
    if len(set(t)) <= 3 and len(t) > 8: return "LOW_DIVERSITY"
    if t.count(t[:6]) >= 3 and len(t) >= 18: return "REPEATED_SUBSTR"
    return None
node = sys.argv[1]
url = f"http://10.120.72.{node}:30080"
print(f"===== node {node} =====")
bad = 0
for p in PROMPTS:
    try:
        text, fr = ask(url, p)
    except Exception as e:
        print(f"  [ERROR] {p!r}: {e}"); bad += 1; continue
    flag = degenerate(text)
    if flag: bad += 1
    shown = text.strip().replace("\n"," ")[:90]
    print(f"  {'BAD(' + flag + ')' if flag else 'ok  '} | {p[:16]:16s} -> {shown}")
print(f"  => {len(PROMPTS)-bad}/{len(PROMPTS)} ok")
