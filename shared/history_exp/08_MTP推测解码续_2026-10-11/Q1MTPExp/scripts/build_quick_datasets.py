#!/usr/bin/env python3
"""快速对比用的小数据集：random-ids vs 真实文本，长度完全相同。

- random: sglang random-ids 语义 = 均匀随机 token id（1..154879），请求之间独立
- real  : ShareGPT 真实多轮对话文本，用模型自带 tokenizer.json 编码
两者都是 8 条请求 x 1024 输入 / 256 输出，便于直接对比接受率。
"""
import json, pathlib, random, re, hashlib

HERE = pathlib.Path(__file__).resolve().parent.parent
TOK = "/data_lib/data/models/glm-q/tokenizer.json"
SHAREGPT = "/mnt/raid/user/data/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
VOCAB, N_REQ, IN_LEN, OUT_LEN, SEED = 154880, 8, 1024, 256, 20261012

def sha(ids): return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()

def build_random(rng):
    out = []
    for i in range(N_REQ):
        ids = [rng.randrange(1, VOCAB) for _ in range(IN_LEN)]
        out.append({"index": i, "input_ids": ids, "input_sha256": sha(ids)})
    return out

def build_real():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    # 只读前 6MB，粗暴切出完整 conversation 对象
    with open(SHAREGPT, encoding="utf-8", errors="replace") as f:
        head = f.read(6_000_000)
    parts = re.findall(r'\{\s*"id":\s*"[^"]+",\s*"conversations":\s*\[.*?\]\s*\}', head, re.S)
    cases, used = [], 0
    for p in parts:
        try: d = json.loads(p)
        except Exception: continue
        text = "\n\n".join(c["value"] for c in d["conversations"])
        ids = tok.encode(text).ids
        if len(ids) < IN_LEN: continue
        ids = ids[:IN_LEN]
        cases.append({"index": used, "source_id": d["id"], "input_ids": ids,
                      "input_sha256": sha(ids), "text_head": text[:160]})
        used += 1
        if used >= N_REQ: break
    assert used == N_REQ, f"只有 {used} 条可用真实样本"
    return cases

def main():
    rng = random.Random(SEED)
    for name, cases in (("random_1k256", build_random(rng)), ("real_1k256", build_real())):
        doc = {"name": name, "seed": SEED, "vocab_id_range": [1, VOCAB - 1],
               "input_tokens": IN_LEN, "target_output_tokens": OUT_LEN,
               "n_requests": N_REQ, "cases": cases}
        p = HERE / "datasets" / f"{name}.json"
        p.write_text(json.dumps(doc))
        print(f"  {p}  ({p.stat().st_size//1024} KB, {len(cases)} 条)")

if __name__ == "__main__":
    main()
