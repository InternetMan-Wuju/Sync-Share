#!/usr/bin/env python3
"""从服务端日志解析 EPLB 每次 rebalance 后的 physical_to_logical_map，
并重建最终完整布局（未 rebalance 的 dense 层用恒等映射补齐）。

日志里的 "rebalance layout after" 只打印被 rebalance 的 MoE 层（本模型是
layer 3..77），dense 层 0..2 始终保持恒等映射。这里先把恒等布局建好，
再按时间顺序用每个 "after" 块覆盖对应层，得到最终完整布局。
"""

import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    text = open(a.log, encoding="utf-8", errors="ignore").read()

    blocks = re.findall(
        r"rebalance layout after:\n"
        r"physical_to_logical_map num_layers=(\d+) "
        r"num_physical_experts=(\d+) ep_size=(\d+)\n"
        r"((?:layer=\d+: .*\n?)+)",
        text,
    )
    if not blocks:
        raise SystemExit("没有找到 rebalance layout after 的 physical_to_logical_map 日志块")

    # 从第一个块拿到总层数和物理专家数
    num_layers = int(blocks[0][0])
    num_physical = int(blocks[0][1])
    ep_size = int(blocks[0][2])

    # 初始恒等布局：physical slot p -> logical expert p
    full_map = [[p for p in range(num_physical)] for _ in range(num_layers)]

    updates = []
    for idx, (_, _, _, layer_lines) in enumerate(blocks):
        updated_layers = {}
        for line in layer_lines.splitlines():
            m = re.match(r"layer=(\d+): (.*)", line)
            if not m:
                continue
            layer_id = int(m.group(1))
            # 只取 [...] 括号里的数字，避免把 ep0/ep1 里的序号也算进去
            nums = []
            for chunk in re.findall(r"\[([^\]]*)\]", m.group(2)):
                nums += [int(x) for x in re.findall(r"-?\d+", chunk)]
            if len(nums) != num_physical:
                raise SystemExit(
                    f"块 {idx} layer={layer_id} 解析出 {len(nums)} 个专家，预期 {num_physical}"
                )
            updated_layers[layer_id] = nums

        for layer_id, nums in updated_layers.items():
            full_map[layer_id] = nums
        updates.append(
            {
                "index": idx,
                "layers_updated": sorted(updated_layers.keys()),
                "num_layers_updated": len(updated_layers),
            }
        )

    out = {
        "num_maps": len(blocks),
        "num_layers": num_layers,
        "num_physical_experts": num_physical,
        "ep_size": ep_size,
        "updates": updates,
        "physical_to_logical_map": full_map,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(
        f"[parse_layout] wrote {a.out}  num_maps={len(blocks)} "
        f"layers={num_layers} physical={num_physical}"
    )


if __name__ == "__main__":
    main()
