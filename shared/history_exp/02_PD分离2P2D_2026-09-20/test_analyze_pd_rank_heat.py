#!/usr/bin/env python3
"""Small deterministic test for full-window EP-rank aggregation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def save_window(path, values, mapping, index, kind, timestamp):
    torch.save(
        {
            "rank": 0,
            "logical_count": torch.tensor([values], dtype=torch.int64),
            "physical_to_logical_map": torch.tensor(mapping, dtype=torch.int64),
            "full_window_dump_index": index,
            "dump_timestamp": timestamp,
            "window_kind": kind,
        },
        path,
    )


def main():
    script = Path(__file__).with_name("analyze_pd_rank_heat.py")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "window_0.pt"
        final = root / "window_1.pt"
        out = root / "out"

        save_window(
            first,
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            0,
            "eplb_rebalance",
            1000.0,
        )
        save_window(
            final,
            [[10, 20, 30, 40], [50, 60, 70, 80]],
            [[2, 3, 0, 1], [1, 0, 3, 2]],
            1,
            "manual_final",
            1001.0,
        )

        subprocess.run(
            [
                sys.executable,
                str(script),
                "--pt",
                str(first),
                "--pt",
                str(final),
                "--ep-size",
                "2",
                "--role",
                "test",
                "--out-dir",
                str(out),
            ],
            check=True,
        )
        result = json.loads((out / "test_heat.json").read_text(encoding="utf-8"))

        assert result["coverage_validated"] is True
        assert result["window_index_range"] == [0, 1]
        assert result["layer_expert_tokens"] == [
            [11, 22, 33, 44],
            [55, 66, 77, 88],
        ]
        assert result["layer_rank_tokens"] == [[73, 37], [121, 165]]
        assert result["rank_totals"] == [194, 202]
        assert result["total_expert_token_assignments"] == 396

    print("full-window aggregation test passed")


if __name__ == "__main__":
    main()
