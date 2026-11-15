#!/usr/bin/env python3
"""Pack one layer's expert state into a SlotPool slot and install it elsewhere.

The pool intentionally stores raw bytes. This module provides the narrow bridge
between a layer's state tensors and that raw-byte slot. It does not make the
model dispatch read from the pool; that is a separate integration step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

DEFAULT_STATE_NAMES = (
    'w13_weight',
    'w2_weight',
    'w13_weight_scale',
    'w2_weight_scale',
    'w13_weight_offset',
    'w2_weight_offset',
    'w13_scale_bias',
    'w2_scale_bias',
    'w13_weight_bias',
    'w2_weight_bias',
)


@dataclass(frozen=True)
class StateField:
    name: str
    shape: tuple[int, ...]
    dtype: Any
    offset: int
    nbytes: int


@dataclass(frozen=True)
class StateSpec:
    fields: tuple[StateField, ...]
    total_bytes: int

    @classmethod
    def from_layer(cls, layer: Any, names: Iterable[str] = DEFAULT_STATE_NAMES) -> 'StateSpec':
        fields = []
        offset = 0
        for name in names:
            tensor = getattr(layer, name, None)
            if tensor is None:
                continue
            if tensor.ndim < 1 or tensor.shape[0] < 1:
                raise ValueError(f'{name} must have a slot axis')
            nbytes = tensor[0].numel() * tensor[0].element_size()
            fields.append(StateField(name, tuple(tensor.shape[1:]), tensor.dtype, offset, nbytes))
            offset += nbytes
        if not fields:
            raise ValueError('no expert state fields found')
        return cls(tuple(fields), offset)

    def assert_compatible(self, other: 'StateSpec') -> None:
        if self.total_bytes != other.total_bytes:
            raise ValueError(f'slot byte mismatch: {self.total_bytes} vs {other.total_bytes}')
        if [(f.name, f.shape, f.dtype, f.nbytes) for f in self.fields] != [
            (f.name, f.shape, f.dtype, f.nbytes) for f in other.fields
        ]:
            raise ValueError('expert state layouts are not compatible')


def _to_nd(tensor: Any) -> Any:
    """Convert an internal-format expert view to a contiguous base-format view."""
    try:
        import torch_npu  # type: ignore
        return torch_npu.npu_format_cast(tensor, 2).contiguous()
    except ImportError:
        return tensor.contiguous()


def pack_expert(layer: Any, expert_index: int, spec: StateSpec, slot_view: Any) -> None:
    """Copy one expert's state into a writable pool slot view."""
    if len(slot_view) < spec.total_bytes:
        raise ValueError('slot view is smaller than state layout')
    for field in spec.fields:
        source = getattr(layer, field.name)[expert_index]
        nd = _to_nd(source)
        raw = nd.view(__import__('torch').uint8).reshape(-1)
        if raw.numel() != field.nbytes:
            raise ValueError(f'{field.name}: packed size mismatch')
        slot_view[field.offset:field.offset + field.nbytes].copy_(raw)


def install_expert(layer: Any, target_index: int, spec: StateSpec, slot_view: Any) -> None:
    """Install a packed pool slot into a target layer/slot."""
    import torch
    if len(slot_view) < spec.total_bytes:
        raise ValueError('slot view is smaller than state layout')
    for field in spec.fields:
        raw = slot_view[field.offset:field.offset + field.nbytes]
        value = raw.view(field.dtype).reshape(field.shape)
        getattr(layer, field.name)[target_index].copy_(value)


def pack_and_mark_ready(pool: Any, lease: Any, layer: Any, expert_index: int, spec: StateSpec) -> None:
    slot = pool.slot_buffer(lease, allow_reserved=True)
    pack_expert(layer, expert_index, spec, slot)
    try:
        import torch_npu  # type: ignore
        torch_npu.npu.synchronize()
    except ImportError:
        pass
    pool.mark_ready(lease)


def install_and_release(pool: Any, lease: Any, layer: Any, target_index: int, spec: StateSpec) -> None:
    with pool.use(lease) as slot:
        install_expert(layer, target_index, spec, slot)
        try:
            import torch_npu  # type: ignore
            torch_npu.npu.synchronize()
        except ImportError:
            pass
    pool.release(lease)
