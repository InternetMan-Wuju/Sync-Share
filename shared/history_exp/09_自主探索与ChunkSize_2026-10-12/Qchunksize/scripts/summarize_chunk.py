#!/usr/bin/env python3
"""把一次 chunk_client 结果汇总成一行 markdown 表格行。
用法: python3 summarize_chunk.py <result.json> <chunk> <mem> <maxpref> <graph_effective> [kv_tokens]
"""
import json, sys
d = json.load(open(sys.argv[1])); a = d.get("aggregate") or {}
chunk = sys.argv[2]; mem = sys.argv[3]; mp = sys.argv[4]; graph = sys.argv[5]
kv = sys.argv[6] if len(sys.argv) > 6 else "-"
name = d.get("name", "?")
print(f"| {name} | {d['concurrency']} | {d['limit']} | {chunk} | {mem} | {mp} | {graph} | {kv} | "
      f"{a.get('output_tps_per_npu','-')} | {a.get('ttft_p50_s','-')} | {a.get('ttft_p95_s','-')} | "
      f"{a.get('tpot_p50_ms','-')} | {a.get('tpot_p95_ms','-')} | {a.get('e2e_p50_s','-')} | "
      f"{a.get('cache_hit_mean','-')} | {a.get('requests_ok','-')}/{a.get('requests','-')} | "
      f"{a.get('wall_s','-')} |")
