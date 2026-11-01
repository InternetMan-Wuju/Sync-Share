#!/usr/bin/env python3
"""把 Dolly 性能、专家分布、四机 metrics、四机 npu-smi 汇总成 summary.json。"""

import argparse
import glob
import json
import os
import re


INTERESTING_KEYWORDS = (
    "eplb",
    "expert",
    "moe",
    "time_to_first_token",
    "time_per_output_token",
    "num_running_reqs",
    "cache_hit_rate",
    "dispatch",
    "combine",
    "wait",
)


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_text(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def ip_from_filename(path, prefix, suffix):
    name = os.path.basename(path)
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return name


def parse_metrics(text):
    """解析单个 /metrics 文本，返回 rank / stage / 指标名统计。"""
    moe_ranks = set()
    stages = set()
    metric_names = set()
    interesting_names = set()
    data_lines = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = re.search(r'moe_ep_rank="(\d+)"', line)
        if m:
            moe_ranks.add(int(m.group(1)))
        m = re.search(r'stage="([^"]+)"', line)
        if m:
            stages.add(m.group(1))

        name = None
        m = re.match(r"(sglang:[A-Za-z0-9_:]+)\{", line)
        if m:
            name = m.group(1)
        elif line.startswith("# HELP ") or line.startswith("# TYPE "):
            m = re.search(r"(sglang:[A-Za-z0-9_:]+)", line)
            if m:
                name = m.group(1)

        if name:
            metric_names.add(name)
            low = name.lower()
            if any(k in low for k in INTERESTING_KEYWORDS):
                interesting_names.add(name)

        if not line.startswith("#"):
            data_lines += 1

    return {
        "data_lines": data_lines,
        "moe_ep_rank_min": min(moe_ranks) if moe_ranks else None,
        "moe_ep_rank_max": max(moe_ranks) if moe_ranks else None,
        "moe_ep_rank_count": len(moe_ranks),
        "stages": sorted(stages),
        "metric_names": sorted(metric_names),
        "interesting_metric_names": sorted(interesting_names),
    }


def parse_npu_smi(text):
    """解析 npu-smi 文本，返回 chip 级 AICore / HBM / Power 统计。"""
    chip_re = re.compile(
        r"^\|\s*(\d+)\s+(\d+)\s+\|\s*[0-9A-Fa-f:.]+?\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
    )
    npu_re = re.compile(
        r"^\|\s*(\d+)\s+Ascend\d+\s+\|\s*OK\s+\|\s*([\d.]+|-)\s+([\d.-]+)"
    )
    aicores = []
    hbm_used = []
    hbm_total = []
    powers = []

    for raw in text.splitlines():
        line = raw.strip()
        m = chip_re.match(line)
        if m:
            aicores.append(int(m.group(3)))
            hbm_used.append(int(m.group(6)))
            hbm_total.append(int(m.group(7)))
            continue
        m2 = npu_re.match(line)
        if m2:
            p = m2.group(2)
            if p not in ("", "-"):
                try:
                    powers.append(float(p))
                except ValueError:
                    pass

    def avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "chips": len(aicores),
        "aicore_pct": {
            "min": min(aicores) if aicores else None,
            "avg": avg(aicores),
            "max": max(aicores) if aicores else None,
        },
        "hbm_used_mb": {
            "min": min(hbm_used) if hbm_used else None,
            "avg": avg(hbm_used),
            "max": max(hbm_used) if hbm_used else None,
        },
        "hbm_total_mb": max(hbm_total) if hbm_total else None,
        "power_w_avg": avg(powers),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()

    dolly_json = read_json(os.path.join(a.run_dir, "dolly_bench.json"))
    dolly = dolly_json.get("summary") if isinstance(dolly_json, dict) else None
    expert = read_json(os.path.join(a.run_dir, "expert_distribution.json"))

    metrics_files = sorted(glob.glob(os.path.join(a.run_dir, "metrics_*.prom")))
    metrics_by_ip = {}
    all_moe_ranks = set()
    for path in metrics_files:
        ip = ip_from_filename(path, "metrics_", ".prom")
        text = read_text(path)
        if text is None:
            continue
        info = parse_metrics(text)
        info["path"] = os.path.basename(path)
        metrics_by_ip[ip] = info
        for line in text.splitlines():
            m = re.search(r'moe_ep_rank="(\d+)"', line)
            if m:
                all_moe_ranks.add(int(m.group(1)))

    npu_files = sorted(glob.glob(os.path.join(a.run_dir, "npusmi_*.txt")))
    npu_by_ip = {}
    for path in npu_files:
        ip = ip_from_filename(path, "npusmi_", ".txt")
        text = read_text(path)
        if text is None:
            continue
        info = parse_npu_smi(text)
        info["path"] = os.path.basename(path)
        npu_by_ip[ip] = info

    summary = {
        "run_dir": a.run_dir,
        "dolly": dolly,
        "expert_distribution": expert,
        "metrics": {
            "files": metrics_by_ip,
            "moe_ep_rank_overall": {
                "min": min(all_moe_ranks) if all_moe_ranks else None,
                "max": max(all_moe_ranks) if all_moe_ranks else None,
                "count": len(all_moe_ranks),
            },
        },
        "npu_smi": {
            "files": npu_by_ip,
            "total_chips": sum(f.get("chips", 0) for f in npu_by_ip.values()),
            "note": "npu-smi 采集于 benchmark 结束后，AICore 为瞬时值、空闲时为 0；HBM 为常驻占用。",
        },
    }

    out_path = os.path.join(a.run_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[summary] wrote -> {out_path}")
    print(
        json.dumps(
            {
                "metrics_files": len(metrics_by_ip),
                "moe_ep_rank_overall": summary["metrics"]["moe_ep_rank_overall"],
                "npu_chips": summary["npu_smi"]["total_chips"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
