#!/usr/bin/env python3
"""把一个 probe 结果 + KV 池大小 汇总成一行字段"""
import json, sys
d = json.load(open(sys.argv[1])); kv = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
a = d.get("aggregate") or {}
ok = [r for r in d["requests"] if not r["error"] and r["meta"].get("spec_verify_ct")]
tok = sum(r["completion_tokens"] for r in ok)
wall = a.get("wall_s", 1) or 1
tpot = [(r["e2e_s"]-r["ttft_s"])/(r["completion_tokens"]-1) for r in ok if r["completion_tokens"] > 1]
cached = [(r["meta"].get("cached_tokens") or 0)/max(1, r["meta"].get("prompt_tokens") or 1) for r in ok]
pr = a.get("pos_rates_weighted")
pos = "/".join(f"{x:.3f}" for x in pr) if pr else "-"
print(len(ok), tok, round(tok/wall, 1), round(tok/wall/16, 3),
      round(sum(tpot)/len(tpot)*1000) if tpot else "-",
      round(a.get("ttft_mean_s", 0), 1), a.get("AL_weighted", "-"),
      a.get("accept_rate_weighted", "-"),
      f"{sum(cached)/len(cached):.1%}" if cached else "-", pos)
