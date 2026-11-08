#!/usr/bin/env python3
"""External launcher: patch runtime symbols, then execute SGLang server module."""
from __future__ import annotations
import runpy
from m0_external.install import install
install()
runpy.run_module("sglang.launch_server", run_name="__main__")
