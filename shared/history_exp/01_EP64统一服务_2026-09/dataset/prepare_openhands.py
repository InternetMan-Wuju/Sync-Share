#!/usr/bin/env python3
"""把 OpenHands 轨迹 Parquet 转成我们自己的 JSON。

运行在 SGLang 容器内，优先使用 pyarrow。
"""

import argparse
import json


def parse_tool_arguments(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def clean_turns(turns):
    cleaned = []
    for t in turns:
        role = t.get("role")
        content = t.get("content") or ""

        if role in ("system", "user"):
            cleaned.append({"role": role, "content": content})
            continue

        if role == "assistant":
            item = {"role": "assistant", "content": content}
            tool_calls = t.get("tool_calls")
            if tool_calls:
                normalized = []
                for call in tool_calls:
                    fn = call.get("function") or {}
                    normalized.append(
                        {
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": fn.get("name"),
                                "arguments": parse_tool_arguments(fn.get("arguments")),
                            },
                        }
                    )
                item["tool_calls"] = normalized
            cleaned.append(item)
            continue

        if role == "tool":
            item = {"role": "tool", "content": content}
            if t.get("tool_call_id") is not None:
                item["tool_call_id"] = t["tool_call_id"]
            if t.get("name") is not None:
                item["name"] = t["name"]
            cleaned.append(item)
            continue

    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/trajectories.parquet")
    ap.add_argument("--out", default="/tmp/openhands_sessions.json")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-turns", type=int, default=0)
    a = ap.parse_args()

    errors = []

    try:
        import itertools
        from datasets import load_dataset

        ds = load_dataset(
            "parquet",
            data_files=a.parquet,
            split="train",
            streaming=True,
        )
        rows = list(itertools.islice(ds, a.limit))
    except Exception as e:
        errors.append(f"datasets: {e}")
        rows = None

    if rows is None:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(
                a.parquet,
                columns=["trajectory_id", "instance_id", "trajectory"],
            )
            rows = table.slice(0, a.limit).to_pylist()
        except Exception as e:
            errors.append(f"pyarrow: {e}")
            rows = None

    if rows is None:
        try:
            import pandas as pd

            df = pd.read_parquet(a.parquet)
            rows = df.head(a.limit).to_dict(orient="records")
        except Exception as e:
            errors.append(f"pandas: {e}")
            raise RuntimeError(
                "无法读取 Parquet。\n" + "\n".join(errors)
            )

    sessions = []
    for i, row in enumerate(rows):
        turns = clean_turns(row.get("trajectory") or [])
        if a.max_turns > 0:
            turns = turns[:a.max_turns]
        sessions.append(
            {
                "id": row.get("trajectory_id") or row.get("instance_id") or str(i),
                "turns": turns,
            }
        )

    output = {
        "meta": {
            "source": "nebius/SWE-rebench-openhands-trajectories",
            "limit": a.limit,
            "max_turns": a.max_turns,
            "count": len(sessions),
        },
        "sessions": sessions,
    }

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"[prepare] wrote {len(sessions)} sessions -> {a.out}")


if __name__ == "__main__":
    main()
