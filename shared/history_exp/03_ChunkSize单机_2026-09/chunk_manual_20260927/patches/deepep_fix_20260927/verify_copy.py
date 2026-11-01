#!/usr/bin/env python3
"""Verify the exact copied DeepEP package before external mounting."""
import hashlib, json
from pathlib import Path
base = Path(__file__).resolve().parent
manifest = json.loads((base / "manifest.json").read_text())
root = base / "deep_ep"
actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
expected = set(manifest["files"])
assert actual == expected, f"file set mismatch: missing={expected-actual}, extra={actual-expected}"
for name, info in manifest["files"].items():
    path = root / name
    assert path.stat().st_size == info["bytes"], f"size mismatch: {name}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], f"SHA256 mismatch: {name}"
print(json.dumps({"verified": True, "file_count": len(actual), "package_path": str(root)}, ensure_ascii=False))
