# 测试4 数据集筛选结果说明

## 产物

- `requests.json`：筛选出的请求集（约 1.2MB）。

## 数据来源

- `/mnt/raid/user/data/dataset/cc-traces-weka-062126-256k/traces.jsonl`
  （SemiAnalysis Claude Code 多轮 agentic trace，393 会话 / 30141 top-level 请求）。

## 筛选规则

- 只用 top-level `type in ('s','n')` 的 main 请求。
- 每段取 **8 个连续 main 请求**：前 7 个用于预热公共前缀，第 8 个（锚点）是正式测试请求。
- 锚点 `in ∈ [49152, 65536]`（64K 输入档）。
- 锚点前缀命中率（相对前 7 个请求的 hash_ids 并集）**≥ 0.85**。
- 优先选锚点 `out` 最接近 1024 的段（压测时 max_tokens 统一 1024）。

## 三档请求集

| 并发 | 前缀多样性 | 段数 | 请求总数 | 前缀命中率 | 锚点 out |
|---|---|---|---|---|---|
| 10 | 5 | 5 | 40 | 0.890~0.958 | 1012~1041 |
| 20 | 10 | 10 | 80 | 0.886~0.965 | 968~1078 |
| 30 | 15 | 15 | 120 | 0.900~0.968 | 915~1095 |

整体锚点：`in` mean≈58804（64K 档）、`out` mean≈880（贴近 1024）、前缀命中 mean≈0.962。

## requests.json 结构

```text
{
  "source": 数据源路径,
  "rules":  筛选规则,
  "stats":  候选/锚点统计,
  "levels": {
    "10": {"concurrency":10, "prefix_diversity":5, "total_requests":40,
           "segments":[{"trace_id","anchor_in","anchor_out","prefix_hit","requests":[8个请求]}]},
    "20": ...,
    "30": ...
  }
}
```

每个 `requests` 元素保留原 trace 的字段：`t` / `model` / `in` / `out` / `hash_ids` / `type`。

## 压测时的用法（约定）

- 数据集只有 token 计数和 `hash_ids`（每 64 token 一块的内容 hash），**没有原始文本**。
- 压测时用随机 token 构造输入：按每个请求的 `in` 生成随机 token id，`max_tokens=1024`。
- 前缀缓存：同一 segment 内先顺序发前 7 个请求（预热前缀），再发第 8 个锚点请求，
  靠相同 `hash_ids` 前缀天然命中缓存（≈90%）。
- 前缀多样性 = 段数（不同 trace 的 segment 前缀不同）。
