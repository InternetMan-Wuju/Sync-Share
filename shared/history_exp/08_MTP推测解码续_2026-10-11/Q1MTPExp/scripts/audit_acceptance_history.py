#!/usr/bin/env python3
"""Audit saved MTP evidence on CPU. Never sends a model request."""
import hashlib
import json
from pathlib import Path

from mtp_output_audit import output_shape

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parents[1]
OUT = ROOT / "audit_20261012"
HISTORY = WORKSPACE / "history_exp/04_MTP推测解码_2026-09"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(rows):
    good = [r for r in rows if not r["error"] and r["meta"].get("spec_verify_ct")]
    verify = sum(r["meta"]["spec_verify_ct"] for r in good)
    correct = sum(r["meta"].get("spec_num_correct_drafts", 0) for r in good)
    proposed = sum(r["meta"].get("spec_num_proposed_drafts", 0) for r in good)
    tokens = sum(r["completion_tokens"] for r in good)
    hist_rows = [r for r in good if r["meta"].get("spec_correct_drafts_histogram")]
    histogram = [sum(r["meta"]["spec_correct_drafts_histogram"][i] for r in hist_rows)
                 for i in range(4)]
    den = sum(histogram)
    return {"requests": len(rows), "accepted_stat_requests": len(good),
            "output_tokens": tokens, "verify_ct": verify,
            "correct_drafts": correct, "proposed_drafts": proposed,
            "AL": tokens / verify if verify else None,
            "draft_accept_rate": correct / proposed if proposed else None,
            "histogram": histogram, "histogram_verify_coverage": den / verify if verify else None,
            "P1_P2_P3": [sum(histogram[i:]) / den for i in (1, 2, 3)] if den else None,
            "output_minus_verify_and_correct": tokens - verify - correct}


def validate(row):
    m = row["meta"]
    h = m.get("spec_correct_drafts_histogram") or m.get("spec_accept_histogram")
    checks = {"request_error_is_none": not row["error"]}
    if m.get("spec_verify_ct"):
        v = m["spec_verify_ct"]
        checks["histogram_count_equals_verify"] = bool(h) and sum(h) == v
        checks["weighted_histogram_equals_correct"] = bool(h) and sum(i*x for i, x in enumerate(h)) == m.get("spec_num_correct_drafts")
        checks["proposed_equals_three_times_verify"] = m.get("spec_num_proposed_drafts") == 3*v
        checks["server_completion_equals_saved_count"] = m.get("completion_tokens") == row["completion_tokens"]
    return checks


def current():
    paths = sorted((ROOT / "results/posrate").glob("w*.json"))
    paths.append(ROOT / "results/scenarios/w4_prefix32k_300_mtpon_174458.json")
    result = []
    for path in paths:
        d = json.loads(path.read_text())
        dataset_path = Path(d["dataset"])
        ds = json.loads(dataset_path.read_text())
        cases = {c["index"]: c for c in ds["cases"]}
        rows = d["requests"]
        checks = []
        for row in rows:
            c = cases[row["index"]]
            row_checks = validate(row)
            sha = hashlib.sha256(json.dumps(c["input_ids"], separators=(",", ":")).encode()).hexdigest()
            row_checks["input_sha_matches_dataset"] = sha == c["input_sha256"]
            row_checks["prompt_length_matches_dataset"] = row["meta"].get("prompt_tokens") == len(c["input_ids"])
            row_checks["output_length_matches_actual_run_limit"] = row["completion_tokens"] == d["max_tokens"]
            checks.append({"index": row["index"], "checks": row_checks})
        tpot = [(r["e2e_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1)*1000 for r in rows]
        low = [r for r in rows if r["distinct_ids"] <= 16]
        higher = [r for r in rows if r["distinct_ids"] > 16]
        result.append({"result_file": str(path), "result_sha256": digest(path),
            "dataset_file": str(dataset_path), "dataset_sha256": digest(dataset_path),
            "configured_output": ds["target_output_tokens"], "actual_output_limit": d["max_tokens"],
            "concurrency": d["concurrency"], "sampling_params_from_client_source": {"temperature": 0, "ignore_eos": True},
            "aggregate": rates(rows), "validation": checks,
            "output_tok_s": sum(r["completion_tokens"] for r in rows)/d["aggregate"]["wall_s"],
            "mean_tpot_ms": sum(tpot)/len(tpot), "mean_ttft_s": d["aggregate"]["ttft_mean_s"],
            "mean_e2e_s": d["aggregate"]["e2e_mean_s"],
            "low_diversity_le16_diagnostic": rates(low), "other_requests_diagnostic": rates(higher),
            "diagnostic_limit": "Excluding <=16 distinct IDs is only sensitivity analysis; other outputs can also repeat. No full output IDs or finish_reason retained in these original files.",
            "requests": rows})
    return result


def history_full():
    root = HISTORY / "mtp_full_20260928/results/20260928_061159"
    rows = []
    for scene in ("decode", "mixed", "prefix"):
        for mode in ("off", "on"):
            group = []
            for path in sorted((root / scene / mode).glob("request_*/measurement/result.json")):
                d = json.loads(path.read_text())
                ids_path = path.parent / "output_ids_final_or_partial.json"
                ids = json.loads(ids_path.read_text())
                row = {"result_file": str(path), "index": d["case_index"], "error": d["error"],
                       "completion_tokens": d["actual_output"], "meta": d["meta_info_last"],
                       "output_shape": output_shape(ids), "output_ids_sha256": digest(ids_path)}
                row["validation"] = validate(row)
                group.append(row)
            periodic = [r for r in group if (r["output_shape"].get("periodic_suffix") or {}).get("fraction", 0) >= .5]
            other = [r for r in group if r not in periodic]
            rows.append({"scene": scene, "mode": mode, "aggregate": rates(group),
                         "majority_periodic_suffix": rates(periodic), "other": rates(other),
                         "requests": group})
    return rows


def history_preview():
    root = HISTORY / "mtp_preview25/results/20260926_181500"
    result = []
    for mode in ("off", "on"):
        groups = {}
        for line in (root / mode / "requests.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("role") != "measurement":
                continue
            scene = d["scenario"].rsplit("_", 1)[0]
            groups.setdefault(scene, []).append({"error": None, "meta": d.get("meta_info") or {},
                                                 "completion_tokens": d["output_tokens"]})
        result.extend({"mode": mode, "scene": s, "aggregate": rates(rs),
                       "file": str(root / mode / "requests.jsonl")} for s, rs in groups.items())
    return result


def history_native():
    root = WORKSPACE / "history_exp/06_四节点服务与权重核查_2026-10-11/logs/20261011_015839/node_23"
    result = []
    for path in sorted(root.glob("bench_*.json")):
        d = json.loads(path.read_text())
        fields = ("dataset_name", "random_input_len", "random_output_len", "random_range_ratio",
                  "completed", "total_input_tokens", "total_output_tokens", "accept_length")
        result.append({"file": str(path), **{k: d.get(k) for k in fields},
                       "limit": "accept_length may be service cumulative avg_spec_accept_length, not workload counter deltas; variable lengths with range_ratio=0."})
    return result


def main():
    OUT.mkdir(exist_ok=True)
    data = {"current": current(), "history_20260928": history_full(),
            "history_20260926": history_preview(), "native_20261011": history_native(),
            "scope": "CPU re-analysis only; no model requests and no service started."}
    path = OUT / "acceptance_evidence.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    for s in data["current"]:
        print(Path(s["result_file"]).name, "AL", round(s["aggregate"]["AL"], 4),
              "P", [round(x, 4) for x in s["aggregate"]["P1_P2_P3"]],
              "checks_failed", sum(not v for r in s["validation"] for v in r["checks"].values()),
              "other_AL", round(s["other_requests_diagnostic"]["AL"], 4))
    for s in data["history_20260928"]:
        print("history", s["scene"], s["mode"], "majority_periodic", s["majority_periodic_suffix"]["requests"],
              "/", s["aggregate"]["requests"], "AL", s["aggregate"]["AL"])
    print("saved", path)


if __name__ == "__main__":
    main()
