#!/usr/bin/env python3
"""Emit container runtime identity relevant to E0/U2."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone


def module_info(name: str) -> dict:
    try:
        mod = __import__(name)
        return {
            "ok": True,
            "version": getattr(mod, "__version__", None),
            "file": getattr(mod, "__file__", None),
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def main() -> None:
    try:
        import torch_npu  # noqa: F401
        import torch
        npu_available = torch.npu.is_available()
        npu_count = torch.npu.device_count()
    except Exception as e:
        npu_available = False
        npu_count = 0
        npu_error = repr(e)
    else:
        npu_error = None

    result = {
        "at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_HOME_PATH", "GLOO_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
                "HCCL_BUFFSIZE", "HCCL_OP_EXPANSION_MODE", "PYTHONPATH",
                "SGLANG_READONLY_WEIGHT_MANIFEST",
                "SGLANG_READONLY_WEIGHT_MANIFEST_SHA256",
                "SGLANG_READONLY_WEIGHT_BASE",
                "SGLANG_READONLY_WEIGHT_AUDIT_DIR",
            )
        },
        "modules": {name: module_info(name) for name in ("torch", "torch_npu", "sglang", "deep_ep", "safetensors", "transformers")},
        "npu": {"available": npu_available, "count": npu_count, "error": npu_error},
        "devices": sorted(os.listdir("/dev")) if os.path.isdir("/dev") else [],
        "network_interfaces": sorted(os.listdir("/sys/class/net")) if os.path.isdir("/sys/class/net") else [],
    }
    spec = importlib.util.find_spec("sglang.srt.layers.moe.token_dispatcher.deepep")
    result["deepep_module_spec"] = None if spec is None else {"origin": spec.origin, "search_locations": spec.submodule_search_locations}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
