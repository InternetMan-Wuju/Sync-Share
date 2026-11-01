#!/usr/bin/env python3
"""Read-only CPU investigation of saved MTP outputs and checkpoint defects.

No model requests, container changes, or checkpoint writes. Writes only its
own evidence directory. The live full run is sampled, not awaited or altered.
"""
import datetime
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from mtp_output_audit import output_shape

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parents[1]
HISTORY = WORKSPACE / "history_exp/04_MTP推测解码_2026-09/mtp_full_20260928/results/20260928_061159"
WEIGHT_AUDIT = WORKSPACE / "history_exp/06_四节点服务与权重核查_2026-10-11/weight_audit"
MODEL = Path("/data_lib/data/models/glm-q")
OUT = ROOT / "causal_review_20261012"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    good = [r for r in rows if not r["error"] and r["verify"]]
    verify = sum(r["verify"] for r in good)
    correct = sum(r["correct"] for r in good)
    proposed = sum(r["proposed"] for r in good)
    tokens = sum(r["output_tokens"] for r in good)
    hist = [sum(r["histogram"][i] for r in good) for i in range(4)]
    return {
        "records": len(rows), "accepted_stat_records": len(good),
        "output_tokens": tokens, "verify": verify, "correct": correct,
        "proposed": proposed, "histogram": hist,
        "AL": tokens / verify if verify else None,
        "draft_acceptance": correct / proposed if proposed else None,
        "P1_P2_P3": [sum(hist[i:]) / verify for i in (1, 2, 3)] if verify else None,
        "boundary_token_delta": tokens - verify - correct,
    }


def row_record(path, raw, eos, ids=None):
    meta = raw.get("meta", raw.get("meta_info_last", {}))
    count = raw.get("completion_tokens", raw.get("actual_output"))
    h = meta.get("spec_correct_drafts_histogram")
    if h is None:
        h = meta.get("spec_accept_histogram", [])
    raw_h = list(h or [])
    # SGLang uses a variable length bincount: unobserved trailing bins vanish.
    hist = (raw_h + [0] * 4)[:4]
    verify = meta.get("spec_verify_ct", 0)
    correct = meta.get("spec_num_correct_drafts", 0)
    proposed = meta.get("spec_num_proposed_drafts", 0)
    record = {
        "source": str(path), "source_sha256": sha(path),
        "index": raw.get("index", raw.get("case_index")), "error": raw.get("error"),
        "output_tokens": count, "verify": verify, "correct": correct,
        "proposed": proposed, "histogram_raw": raw_h, "histogram": hist,
        "AL": count / verify if verify else None,
        "draft_acceptance": correct / proposed if proposed else None,
        "finish_reason": meta.get("finish_reason"),
        "count_checks": {
            "histogram_length_le4": len(raw_h) <= 4,
            "histogram_sum_is_verify": sum(hist) == verify,
            "histogram_weighted_sum_is_correct": sum(i * n for i, n in enumerate(hist)) == correct,
            "proposed_is_3verify": proposed == 3 * verify,
            "saved_count_is_server_count": count == meta.get("completion_tokens"),
        } if verify else {},
    }
    if ids is None:
        ids = raw.get("output_ids")
    if ids is not None:
        record["shape"] = output_shape(ids)
        positions = [i for i, token in enumerate(ids) if token in eos]
        record["eos_count"] = len(positions)
        record["first_eos_position_zero_based"] = positions[0] if positions else None
        record["tokens_after_first_eos"] = len(ids) - positions[0] - 1 if positions else 0
        record["tokens_before_first_eos_including_eos"] = positions[0] + 1 if positions else len(ids)
        record["output_ids_sha256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        record["output_ids_count_matches_record"] = len(ids) == count
    else:
        record["saved_distinct_ids"] = raw.get("distinct_ids")
        record["shape_limit"] = "Original record did not retain full IDs."
    record["text_head"] = raw.get("text", raw.get("text_head", ""))[:200]
    return record


def group(label, rows):
    periodic = [r for r in rows if (r.get("shape", {}).get("periodic_suffix") or {}).get("fraction", 0) >= 0.5]
    other = [r for r in rows if r not in periodic]
    return {
        "label": label, "aggregate": summarize(rows),
        "majority_periodic_suffix": summarize(periodic), "remaining": summarize(other),
        "full_ids_records": sum("shape" in r for r in rows),
        "first_eos_records": sum(r.get("first_eos_position_zero_based") is not None for r in rows),
        "tokens_after_first_eos": sum(r.get("tokens_after_first_eos", 0) for r in rows),
        "rows": rows,
    }


def output_evidence(eos):
    groups = []
    for scene in ("decode", "mixed", "prefix"):
        for mode in ("off", "on"):
            rows = []
            for path in sorted((HISTORY / scene / mode).glob("request_*/measurement/result.json")):
                ids_path = path.parent / "output_ids_final_or_partial.json"
                rows.append(row_record(path, read(path), eos, read(ids_path)))
            groups.append(group("history_0928_" + scene + "_" + mode, rows))
    paths = sorted((ROOT / "results/posrate").glob("w*.json"))
    paths += [ROOT / "results/scenarios/w4_prefix32k_300_mtpon_174458.json"]
    for path in paths:
        groups.append(group("legacy_current_" + path.stem,
                            [row_record(path, r, eos) for r in read(path)["requests"]]))
    base = ROOT / "results/native_randomids_20261012_1844"
    for kind in ("native_exact_ids", "uniform_legacy", "native_default_text"):
        path = base / (kind + ".json")
        groups.append(group("short_" + kind, [row_record(path, r, eos) for r in read(path)["requests"]]))
    latest = read(ROOT / "results/latest_native_full.json")
    base = Path(latest["out"])
    for scene in ("w1_8k_1k", "w2_32k_1k", "w4_prefix32k_300", "w3_1k_32k"):
        rows = []
        for path in sorted((base / scene).glob("request_*.json")):
            if path.name.endswith(".payload.json"):
                continue
            raw = read(path)
            if "output_ids" in raw:
                rows.append(row_record(path, raw, eos))
        if rows:
            groups.append(group("native_full_" + scene, rows))
    return groups, read(base / "experiment_status.json")


def historical_pairs():
    import csv
    path = HISTORY / "analysis/paired_requests.csv"
    rows = list(csv.DictReader(path.open()))
    result = {"source": str(path), "source_sha256": sha(path), "scenes": {}}
    for scene in ("mixed", "prefix"):
        selected = [r for r in rows if r["scenario"] == scene and r["both_complete"] == "True"]
        changed = [r for r in selected if r["first_output_id_mismatch_index"]]
        # Recheck the saved output arrays, independently of the existing CSV.
        actual = []
        for r in selected:
            idx = int(r["index"])
            arrays = [read(HISTORY / scene / mode / f"request_{idx:04d}" / "measurement/output_ids_final_or_partial.json") for mode in ("off", "on")]
            mismatch = next((i for i, (a, b) in enumerate(zip(*arrays)) if a != b), None)
            expected = int(r["first_output_id_mismatch_index"]) if r["first_output_id_mismatch_index"] else None
            if mismatch != expected:
                raise ValueError("Paired output discrepancy: " + str(r))
            actual.append({"index": idx, "first_mismatch_zero_based": mismatch,
                           "off_head": arrays[0][:12], "on_head": arrays[1][:12]})
        result["scenes"][scene] = {"complete_pairs": len(selected), "different": len(changed), "pairs": actual}
    return result


def weights():
    """Read three 32MiB blocks only, not the 721GiB checkpoint."""
    result = {"model": str(MODEL), "bytes_read": 0, "blocks": [], "nonfinite": 0,
              "scope": "Current disk bytes compared with Oct11 snapshot; not a dump of already loaded NPU tensors."}
    for number in (1, 47, 89):
        name = f"quant_model_weights-{number:05d}-of-00181.safetensors"
        manifest = WEIGHT_AUDIT / "snapshot_datalib" / (name + ".json")
        d = read(manifest)
        reference = read(WEIGHT_AUDIT / "audit_seed" / (name + ".json"))
        previous = {b["offset"]: b["sha256"] for b in reference["blocks"]}
        # header_bytes in these manifests ALREADY includes the uint64 length.
        header = d["header_bytes"]
        with (MODEL / name).open("rb") as stream:
            actual_header_bytes = struct.unpack("<Q", stream.read(8))[0]
            actual_header = json.loads(stream.read(actual_header_bytes))
            if actual_header_bytes + 8 != header or actual_header != d["tensors"]:
                raise ValueError("Current checkpoint header differs from saved tensor map: " + name)
            for block in d["blocks"]:
                if previous.get(block["offset"]) == block["sha256"]:
                    continue
                lo, hi = block["offset"], block["offset"] + block["length"]
                floats = [(key, t) for key, t in d["tensors"].items()
                          if key != "__metadata__" and t["dtype"] in ("BF16", "F32")
                          and max(lo, t["data_offsets"][0] + header) < min(hi, t["data_offsets"][1] + header)]
                if not floats:
                    continue
                stream.seek(lo)
                data = stream.read(block["length"])
                result["bytes_read"] += len(data)
                digest = hashlib.sha256(data).hexdigest()
                item = {"file": name, "offset": lo, "length": len(data), "sha256": digest,
                        "matches_oct11_snapshot": digest == block["sha256"],
                        "reference_sha256": previous.get(lo), "tensor_header_matches_saved_map": True,
                        "manifest": str(manifest),
                        "manifest_sha256": sha(manifest), "anomalous_tensors": []}
                for key, tensor in floats:
                    start, end = tensor["data_offsets"]
                    left, right = max(lo, start + header) - lo, min(hi, end + header) - lo
                    dtype = tensor["dtype"]
                    width = 2 if dtype == "BF16" else 4
                    vals = np.frombuffer(data, dtype="<u2" if dtype == "BF16" else "<f4", offset=left, count=(right - left) // width)
                    if dtype == "BF16":
                        vals = (vals.astype(np.uint32) << 16).view(np.float32)
                    finite = np.isfinite(vals)
                    bad = int(np.count_nonzero(~finite))
                    extreme = int(np.count_nonzero(np.abs(vals[finite]) > 1e10))
                    if bad or extreme:
                        item["anomalous_tensors"].append({"name": key, "dtype": dtype, "shape": tensor["shape"],
                                                          "values_read": len(vals), "nonfinite": bad,
                                                          "finite_abs_gt_1e10": extreme,
                                                          "tensor_value_start": (left + lo - header - start) // width})
                        result["nonfinite"] += bad
                result["blocks"].append(item)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    config = read(MODEL / "config.json")
    eos = set(config["eos_token_id"])
    tokenizer = read(MODEL / "tokenizer.json")
    token_names = {t["id"]: t["content"] for t in tokenizer["added_tokens"]}
    groups, status = output_evidence(eos)
    data = {
        "observed_at_Asia_Shanghai": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "scope": "CPU-only analysis; no model requests or service mutation.",
        "eos": {str(k): token_names[k] for k in sorted(eos)},
        "config_sha256": sha(MODEL / "config.json"), "groups": groups,
        "historical_pairs": historical_pairs(), "weight_probe": weights(),
        "live_experiment_status_observed": status,
        "reference_rates": {"high": [0.498, 0.345, 0.267], "low": [0.148, 0.093, 0.056],
                            "high_AL_if_unconditional": 1 + 0.498 + 0.345 + 0.267,
                            "high_AL_if_conditional": 1 + 0.498 + 0.498 * 0.345 + 0.498 * 0.345 * 0.267,
                            "low_AL_if_unconditional": 1 + 0.148 + 0.093 + 0.056},
    }
    target = OUT / "root_cause_evidence.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    for g in groups:
        a = g["aggregate"]
        print(g["label"], "n", a["records"], "AL", a["AL"], "periodic", g["majority_periodic_suffix"]["records"],
              "EOS", g["first_eos_records"], "after_EOS", g["tokens_after_first_eos"])
    print("weight_nonfinite", data["weight_probe"]["nonfinite"], "bytes_read", data["weight_probe"]["bytes_read"])
    print("saved", target)


if __name__ == "__main__":
    main()
