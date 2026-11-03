#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SGLANG_PREFILL_COPY_AWARE_MODE=off
python3 tests/test_cpu_logic.py
python3 tests/test_patch_integration.py
python3 -m py_compile patched_tree/sglang/srt/eplb/prefill_copy_aware.py patched_tree/sglang/srt/hardware_backend/npu/moe/topk.py
python3 tests/npu_validate_prefill_copy_aware.py
