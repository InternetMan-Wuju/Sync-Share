#!/usr/bin/env python3
"""S1 CPU audit: complete audited quant state for selected layer-49 experts.

Run with the readonly publisher loader on PYTHONPATH.  This script intentionally
loads through safetensors.safe_open so every reconstructed block is checked
against the publisher manifest before it is interpreted as model state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_LAYER = 49
DEFAULT_EXPERTS = (40, 43, 60)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("weight", "weight_scale", "weight_offset")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return sha256_bytes(data)


def tensor_stats(tensor: torch.Tensor) -> dict:
    t = tensor.detach().cpu().contiguous()
    result = {
        "finite_count": int(t.numel()),
        "nonfinite_count": 0,
        "zero_count": int((t == 0).sum().item()),
    }
    if t.dtype.is_floating_point:
        x = t.to(torch.float64).numpy()
        result.update(
            min=float(x.min()) if x.size else None,
            max=float(x.max()) if x.size else None,
            mean=float(x.mean()) if x.size else None,
            nonfinite_count=int((~torch.isfinite(t)).sum().item()),
        )
    else:
        result.update(
            min=int(t.min().item()) if t.numel() else None,
            max=int(t.max().item()) if t.numel() else None,
        )
    return result


def copy_policy_test(tensor: torch.Tensor) -> dict:
    """Test PyTorch clone versus view storage semantics on a CPU tensor."""
    original_hash = tensor_sha256(tensor)
    original_value = tensor.view(-1)[0].item()
    clone = tensor.clone()
    alias = tensor.view_as(tensor)
    clone_ptr, alias_ptr, source_ptr = clone.data_ptr(), alias.data_ptr(), tensor.data_ptr()
    clone.view(-1)[0] = original_value + 1
    clone_isolated = tensor_sha256(tensor) == original_hash
    alias.view(-1)[0] = original_value + 2
    alias_visible = tensor.view(-1)[0].item() == original_value + 2 and tensor_sha256(tensor) != original_hash
    alias.view(-1)[0] = original_value
    restored = tensor_sha256(tensor) == original_hash
    return {
        "source_data_ptr": source_ptr,
        "clone_data_ptr": clone_ptr,
        "alias_data_ptr": alias_ptr,
        "clone_has_distinct_storage": clone_ptr != source_ptr,
        "alias_shares_storage": alias_ptr == source_ptr,
        "clone_mutation_isolated": clone_isolated,
        "alias_mutation_visible": alias_visible,
        "source_restored_exactly": restored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data_lib/data/models/glm-q")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--experts", default=",".join(map(str, DEFAULT_EXPERTS)))
    args = parser.parse_args()

    model = Path(args.model).resolve()
    out = Path(args.out).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    experts = tuple(int(x) for x in args.experts.split(","))

    manifest_path = Path(os.environ["SGLANG_READONLY_WEIGHT_MANIFEST"]).resolve()
    manifest_sha = sha256_bytes(manifest_path.read_bytes())
    expected_manifest_sha = os.environ["SGLANG_READONLY_WEIGHT_MANIFEST_SHA256"]
    if manifest_sha != expected_manifest_sha:
        raise RuntimeError(f"manifest SHA mismatch: {manifest_sha}")

    manifest = json.loads(manifest_path.read_text())
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    quant_desc = json.loads((model / "quant_model_description.json").read_text())

    expected_keys = {
        f"model.layers.{MODEL_LAYER}.mlp.experts.{expert}.{proj}.{suffix}"
        for expert in experts
        for proj in PROJECTIONS
        for suffix in SUFFIXES
    }
    actual_keys = {
        key
        for key in weight_map
        if any(
            key.startswith(f"model.layers.{MODEL_LAYER}.mlp.experts.{expert}.")
            for expert in experts
        )
    }
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(f"incomplete expert state: missing={missing} unexpected={unexpected}")

    by_file: dict[str, list[str]] = {}
    for key in sorted(expected_keys):
        by_file.setdefault(weight_map[key], []).append(key)

    tensors: dict[tuple[int, str], torch.Tensor] = {}
    entries = []
    for filename, keys in sorted(by_file.items()):
        file_meta = manifest["files"][filename]
        with safe_open(model / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                expert = int(key.split(".experts.", 1)[1].split(".", 1)[0])
                tensor = handle.get_tensor(key)
                tensors[(expert, key.rsplit(".", 2)[-2] + "." + key.rsplit(".", 1)[-1])] = tensor
                file_tensor_meta = file_meta["tensors"][key]
                entries.append(
                    {
                        "key": key,
                        "layer": MODEL_LAYER,
                        "expert": expert,
                        "tensor": key.rsplit(".", 2)[-2] + "." + key.rsplit(".", 1)[-1],
                        "source_shard": filename,
                        "source_data_offsets": file_tensor_meta["data_offsets"],
                        "quantization": quant_desc.get(key),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                        "stride": list(tensor.stride()),
                        "contiguous": bool(tensor.is_contiguous()),
                        "storage_offset": int(tensor.storage_offset()),
                        "storage_bytes": int(tensor.numel() * tensor.element_size()),
                        "sha256": tensor_sha256(tensor),
                        "stats": tensor_stats(tensor),
                    }
                )

    # Each logical expert tensor appears in exactly one checkpoint key.  Extra
    # runtime replicas are separately allocated copies, not additional source
    # checkpoint tensors.
    source_copy_counts = {key: 0 for key in expected_keys}
    for tensor_key in weight_map:
        if tensor_key in source_copy_counts:
            source_copy_counts[tensor_key] += 1

    per_expert = {}
    for expert in experts:
        expert_entries = [e for e in entries if e["expert"] == expert]
        weight_entry = next(e for e in expert_entries if e["tensor"] == "gate_proj.weight")
        per_expert[str(expert)] = {
            "tensor_count": len(expert_entries),
            "expected_tensor_count": len(PROJECTIONS) * len(SUFFIXES),
            "storage_bytes": sum(e["storage_bytes"] for e in expert_entries),
            "all_tensors_contiguous": all(e["contiguous"] for e in expert_entries),
            "all_source_copy_counts_one": all(source_copy_counts[e["key"]] == 1 for e in expert_entries),
            "all_offsets_zero": all(
                e["stats"]["zero_count"] == e["stats"]["finite_count"]
                for e in expert_entries
                if e["tensor"].endswith("weight_offset")
            ),
            "gate_weight_sha256": weight_entry["sha256"],
            "copy_policy": copy_policy_test(tensors[(expert, "gate_proj.weight")]),
        }

    result = {
        "stage": "S1",
        "kind": "cpu_audited_expert_state",
        "status": "PASS" if not missing and not unexpected else "FAIL",
        "scope": {
            "layer": MODEL_LAYER,
            "experts": list(experts),
            "logical_source_copies_per_tensor": 1,
            "runtime_replica_storage": "not present in checkpoint; NPU copy test is separate",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "model": str(model),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_expected_sha256": expected_manifest_sha,
            "reference_id": manifest.get("reference_id"),
            "weight_loader_environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTHONPATH",
                    "SGLANG_READONLY_WEIGHT_MANIFEST",
                    "SGLANG_READONLY_WEIGHT_BASE",
                    "SGLANG_READONLY_WEIGHT_AUDIT_DIR",
                )
            },
        },
        "completeness": {
            "expected_tensor_count": len(expected_keys),
            "found_tensor_count": len(actual_keys),
            "missing": missing,
            "unexpected": unexpected,
            "quantization_types": sorted({quant_desc[key] for key in expected_keys}),
        },
        "source_audit": {
            "audit_dir": str(audit_dir),
            "audit_files": sorted(str(p) for p in audit_dir.glob("weight_source_*.jsonl")),
        },
        "experts": per_expert,
        "tensors": sorted(entries, key=lambda e: (e["expert"], e["tensor"])),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "status": result["status"],
        "experts": list(experts),
        "tensor_count": len(entries),
        "storage_bytes": sum(e["storage_bytes"] for e in entries),
        "all_offsets_zero": all(v["all_offsets_zero"] for v in per_expert.values()),
        "copy_policy": {k: v["copy_policy"] for k, v in per_expert.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
