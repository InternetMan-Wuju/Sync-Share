#!/usr/bin/env python3
"""Build a compact hash index for the single-NPU result package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    artifacts = []
    patterns = [
        "RESULTS_S0.md", "RESULTS_S1.md", "RESULTS_S2.md", "RESULTS_S3.md",
        "RESULTS_S4.md", "RESULTS_P2.md", "RESULTS_OVERALL.md",
    ]
    patterns += [f"results/{name}" for name in (
        "S0_MAPPING_AUDIT.json", "S0_EVIDENCE.json", "s0_layer49_expert_loads.csv",
        "S1_EXPERT_STATE.json", "S1_NUMERICS.json", "S2_LOCAL_COST.json",
        "S2_LOCAL_COST.csv", "S3_COST_CALIBRATION.json", "S3_PLANS.json",
        "S3_REPLAY.json", "S3_ALLOCATION.json", "S3_ALLOCATION.csv",
        "S4_LOCAL_CANDIDATES.json", "P2_LOCAL_CHECKS.json",
    )]
    patterns += [f"scripts/{path.name}" for path in sorted((root / "scripts").glob("s[0-4]_*.py"))]
    patterns += ["scripts/p2_local_checks.py", "scripts/build_single_npu_manifest.py"]
    for relative in patterns:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    model = Path("/data_lib/data/models/glm-q")
    model_files = ["config.json", "quant_model_description.json", "quant_model_weights.safetensors.index.json"]
    model_hashes = {name: sha256(model / name) for name in model_files}
    result = {
        "stage": "single_npu_evidence_index",
        "status": "COMPLETE",
        "scope": "S0-S4 and P2 on one Ascend NPU; no full-model or multi-rank performance claim",
        "image": {"id": args.image_id, "digest": args.image_digest},
        "device": {"count": 1, "name": "Ascend910_9382"},
        "software": {"torch": "2.10.0+cpu", "torch_npu": "2.10.0", "CANN": "9.0.0"},
        "weight_manifest_sha256": args.manifest_sha256,
        "model_file_sha256": model_hashes,
        "artifacts": artifacts,
        "local_raw_only": [
            "logs/s1_state_*/weight_audit/",
            "all container logs and profiler traces",
            "model weights and tensor shards",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(out), "artifact_count": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
