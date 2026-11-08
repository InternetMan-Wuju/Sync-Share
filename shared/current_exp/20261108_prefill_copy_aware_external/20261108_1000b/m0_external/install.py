"""Runtime monkey-patches; no source files are modified.

The plugin wraps two existing symbols:
- sglang.srt.hardware_backend.npu.moe.topk.fused_topk_npu
- topk_ids_logical_to_physical aliases already imported by callers

The fused wrapper only carries ``layer_id`` in a ContextVar. The mapping
wrapper is the actual insertion point: it receives logical top-k IDs before the
standard logical->physical transform.
"""
from __future__ import annotations
import contextvars
import logging
from typing import Optional, Tuple
from . import controller

logger = logging.getLogger(__name__)
_LAYER_ID = contextvars.ContextVar("m0_prefill_copy_aware_layer_id", default=None)
_INSTALLED = False
_ORIGINAL_FUSED = None
_ORIGINAL_MAPS: list[Tuple[object, str, object]] = []


def _mapping_wrapper(topk_ids, expert_location_dispatch_info, log2phy_prob=None):
    original = _ORIGINAL_MAPS[0][2] if _ORIGINAL_MAPS else None
    if original is None:
        raise RuntimeError("external plugin original mapping missing")
    # Default-off must take the original path with no added counting.
    if controller.configured_mode() is controller.Mode.OFF:
        return original(topk_ids, expert_location_dispatch_info, log2phy_prob) if log2phy_prob is not None else original(topk_ids, expert_location_dispatch_info)
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch
    if not get_is_extend_in_batch():
        return original(topk_ids, expert_location_dispatch_info, log2phy_prob) if log2phy_prob is not None else original(topk_ids, expert_location_dispatch_info)
    layer_id = _LAYER_ID.get()
    if layer_id is None:
        return original(topk_ids, expert_location_dispatch_info, log2phy_prob) if log2phy_prob is not None else original(topk_ids, expert_location_dispatch_info)
    try:
        remapped, applied = controller.maybe_apply_prefill_copy_aware(
            topk_ids=topk_ids,
            topk_weights=None,
            expert_location_dispatch_info=expert_location_dispatch_info,
            layer_id=layer_id,
            num_token_non_padded=None,
            num_fused_shared_experts=0,
        )
        if applied:
            return remapped
        return original(topk_ids, expert_location_dispatch_info, log2phy_prob) if log2phy_prob is not None else original(topk_ids, expert_location_dispatch_info)
    except Exception:
        if controller.configured_mode() is controller.Mode.ACTIVE:
            raise
        logger.exception("copy-aware shadow failed; falling back to original mapping")
        return original(topk_ids, expert_location_dispatch_info, log2phy_prob) if log2phy_prob is not None else original(topk_ids, expert_location_dispatch_info)


def _fused_wrapper(*args, **kwargs):
    layer_id = kwargs.get("layer_id")
    token = _LAYER_ID.set(layer_id)
    try:
        return _ORIGINAL_FUSED(*args, **kwargs)
    finally:
        _LAYER_ID.reset(token)


def install():
    global _INSTALLED, _ORIGINAL_FUSED
    if _INSTALLED:
        return
    import sglang.srt.hardware_backend.npu.moe.topk as npu_topk
    import sglang.srt.layers.moe.topk as core_topk
    original_map = npu_topk.topk_ids_logical_to_physical
    _ORIGINAL_FUSED = npu_topk.fused_topk_npu
    _ORIGINAL_MAPS.extend([
        (npu_topk, "topk_ids_logical_to_physical", original_map),
        (core_topk, "topk_ids_logical_to_physical", core_topk.topk_ids_logical_to_physical),
    ])
    npu_topk.topk_ids_logical_to_physical = _mapping_wrapper
    core_topk.topk_ids_logical_to_physical = _mapping_wrapper
    npu_topk.fused_topk_npu = _fused_wrapper
    _INSTALLED = True


def uninstall():
    global _INSTALLED
    if not _INSTALLED:
        return
    for module, name, original in _ORIGINAL_MAPS:
        setattr(module, name, original)
    import sglang.srt.hardware_backend.npu.moe.topk as npu_topk
    npu_topk.fused_topk_npu = _ORIGINAL_FUSED
    _ORIGINAL_MAPS.clear()
    _INSTALLED = False
