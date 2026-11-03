"""Default-off prefill copy-aware routing controller.

M0 scope: CPU logic, plan generation, quota remapping and slot lifecycle.
Hardware transport and cost calibration are explicit dependencies and are not
implemented by this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional, Protocol, Sequence

import torch

DEFAULT_LAYER_ID = 49


class Mode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ACTIVE = "active"


def configured_mode() -> Mode:
    raw = os.getenv("SGLANG_PREFILL_COPY_AWARE_MODE", "off").strip().lower()
    try:
        return Mode(raw)
    except ValueError:
        return Mode.OFF


def configured_layer_id() -> int:
    try:
        return int(os.getenv("SGLANG_PREFILL_COPY_AWARE_LAYER", str(DEFAULT_LAYER_ID)))
    except ValueError:
        return DEFAULT_LAYER_ID


class PlanError(RuntimeError):
    pass


class SlotError(RuntimeError):
    pass


@dataclass(frozen=True)
class RoutingStats:
    counts: tuple[int, ...]
    by_source_counts: tuple[tuple[int, ...], ...]
    total_tasks: int
    invalid_tasks: int
    shared_tasks: int

    def quota_by_source(self, q: int) -> tuple[int, ...]:
        if q < 0 or q > self.total_tasks:
            raise PlanError(f"quota {q} outside [0,{self.total_tasks}]")
        remaining = q
        result = []
        for source_counts in self.by_source_counts:
            taken = min(remaining, sum(source_counts))
            result.append(taken)
            remaining -= taken
        if remaining:
            raise PlanError("source quota did not consume q")
        return tuple(result)


def collect_routing_stats(
    topk_ids_by_source: Sequence[torch.Tensor],
    *,
    num_logical_experts: int,
    num_token_non_padded=None,
    num_fused_shared_experts: int = 0,
) -> RoutingStats:
    """Count valid routing branches, excluding padding, invalid IDs and shared cols.

    ``topk_ids_by_source`` is an explicit list of input-owner tensors.  This is
    the M0 CPU contract.  The service currently supports one owner; multi-source
    tests use the same implementation.
    """
    if num_logical_experts <= 0:
        raise PlanError("num_logical_experts must be positive")
    counts = [0] * num_logical_experts
    by_source = []
    total = invalid = shared = 0
    for source_index, ids in enumerate(topk_ids_by_source):
        if ids.ndim != 2:
            raise PlanError("topk_ids must be 2-D")
        width = ids.shape[1]
        valid_width = width - num_fused_shared_experts
        if num_fused_shared_experts < 0 or valid_width < 0:
            raise PlanError("invalid shared expert width")
        rows = ids.shape[0]
        if num_token_non_padded is None:
            row_limit = rows
        elif isinstance(num_token_non_padded, int):
            row_limit = num_token_non_padded
        elif torch.is_tensor(num_token_non_padded):
            if num_token_non_padded.ndim == 0:
                row_limit = int(num_token_non_padded.item())
            elif num_token_non_padded.ndim == 1 and num_token_non_padded.dtype == torch.bool:
                row_limit = int(num_token_non_padded.sum().item())
            else:
                raise PlanError("unsupported num_token_non_padded tensor")
        else:
            raise PlanError("unsupported num_token_non_padded")
        if row_limit < 0 or row_limit > rows:
            raise PlanError(f"num_token_non_padded={row_limit} outside [0,{rows}]")
        source_counts = [0] * num_logical_experts
        for row in range(row_limit):
            for col in range(width):
                expert = int(ids[row, col].item())
                if num_fused_shared_experts and col >= valid_width:
                    shared += 1
                    continue
                if expert < 0 or expert >= num_logical_experts:
                    invalid += 1
                    continue
                source_counts[expert] += 1
                counts[expert] += 1
                total += 1
        by_source.append(tuple(source_counts))
    return RoutingStats(tuple(counts), tuple(by_source), total, invalid, shared)


class RankCostModel(Protocol):
    def predict(self, rank_expert_counts: Sequence[int]) -> float: ...


class Transport(Protocol):
    def is_cached(self, layer_id: int, logical_expert: int, weight_version: int, expert_tp_shard: int) -> bool: ...
    def prepare_copy(self, layer_id: int, logical_expert: int, weight_version: int, expert_tp_shard: int) -> None: ...


class LinearCostModel:
    """Small deterministic model used only by CPU tests / synthetic validation."""

    def __init__(self, fixed_us: float = 0.0, per_task_us: float = 1.0, per_active_expert_us: float = 0.0):
        self.fixed_us = fixed_us
        self.per_task_us = per_task_us
        self.per_active_expert_us = per_active_expert_us

    def predict(self, rank_expert_counts: Sequence[int]) -> float:
        return self.fixed_us + self.per_task_us * sum(rank_expert_counts) + self.per_active_expert_us * sum(v > 0 for v in rank_expert_counts)


@dataclass(frozen=True)
class Plan:
    step_id: int
    layer_id: int
    epoch: int
    action: str
    logical_expert: int
    weight_source_rank: int
    primary_compute_rank: int
    destination_rank: int
    destination_local_slot: int
    q_global: int
    q_by_source: tuple[int, ...]
    copy_required: bool
    layout_version: str
    weight_version: int
    estimated_gain_us: Optional[float]
    reason: str
    primary_physical_id: int
    destination_physical_id: int

    def to_dict(self):
        return {
            "step_id": self.step_id,
            "layer_id": self.layer_id,
            "epoch": self.epoch,
            "action": self.action,
            "logical_expert": self.logical_expert,
            "weight_source_rank": self.weight_source_rank,
            "primary_compute_rank": self.primary_compute_rank,
            "destination_rank": self.destination_rank,
            "destination_local_slot": self.destination_local_slot,
            "q_global": self.q_global,
            "q_by_source": list(self.q_by_source),
            "copy_required": self.copy_required,
            "layout_version": self.layout_version,
            "weight_version": self.weight_version,
            "estimated_gain_us": self.estimated_gain_us,
            "reason": self.reason,
            "primary_physical_id": self.primary_physical_id,
            "destination_physical_id": self.destination_physical_id,
        }


def quota_candidates(counts: RoutingStats, expert: int, source_load: float, destination_load: float) -> tuple[int, ...]:
    q0 = min(counts.counts[expert], max(0, int((source_load - destination_load) // 2)))
    return tuple(sorted({0, q0 // 2, q0}))


def remap_logical_to_physical(
    logical_ids_by_source: Sequence[torch.Tensor],
    *,
    primary_physical_ids: Sequence[int],
    plan: Plan,
    logical_expert: int,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Move exactly q_by_source stable prefix tasks; keep weights untouched."""
    if plan.q_global != sum(plan.q_by_source):
        raise PlanError("plan q_by_source does not sum to q_global")
    outputs = []
    source_offset = 0
    for source_index, ids in enumerate(logical_ids_by_source):
        if ids.ndim != 2:
            raise PlanError("topk_ids must be 2-D")
        valid_width = ids.shape[1] - num_fused_shared_experts
        result = ids.clone()
        quota = plan.q_by_source[source_index]
        taken = 0
        for row in range(ids.shape[0]):
            for col in range(valid_width):
                logical = int(ids[row, col].item())
                if logical < 0 or logical >= len(primary_physical_ids):
                    continue
                if logical == logical_expert:
                    if taken < quota:
                        result[row, col] = plan.destination_physical_id
                        taken += 1
                    else:
                        result[row, col] = plan.primary_physical_id
                else:
                    result[row, col] = int(primary_physical_ids[logical])
            for col in range(valid_width, ids.shape[1]):
                result[row, col] = ids[row, col]
        if taken != quota:
            raise PlanError(f"source {source_index} consumed {taken}, expected {quota}")
        outputs.append(result)
        source_offset += quota
    return tuple(outputs)


class SlotState(str, Enum):
    EMPTY = "empty"
    DRAINING = "draining"
    FILLING = "filling"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class SlotRecord:
    state: SlotState = SlotState.EMPTY
    logical_expert: Optional[int] = None
    weight_version: int = -1
    epoch: int = -1
    readers: int = 0


class SlotManager:
    def __init__(self):
        self._slots: dict[tuple[int, int, int], SlotRecord] = {}
        self._committed_epoch: dict[int, int] = {}

    def slot(self, layer_id: int, rank: int, local_slot: int) -> SlotRecord:
        return self._slots.setdefault((layer_id, rank, local_slot), SlotRecord())

    def begin_read(self, layer_id: int, rank: int, local_slot: int, epoch: int):
        rec = self.slot(layer_id, rank, local_slot)
        if rec.state is not SlotState.READY or rec.epoch != epoch:
            raise SlotError("dispatch references non-ready or stale slot")
        rec.readers += 1

    def end_read(self, layer_id: int, rank: int, local_slot: int):
        rec = self.slot(layer_id, rank, local_slot)
        if rec.readers <= 0:
            raise SlotError("end_read without reader")
        rec.readers -= 1

    def replace_and_publish(
        self,
        *,
        layer_id: int,
        rank: int,
        local_slot: int,
        logical_expert: int,
        weight_version: int,
        epoch: int,
        transport: Transport,
        expert_tp_shard: int = 0,
    ) -> SlotRecord:
        rec = self.slot(layer_id, rank, local_slot)
        if rec.readers:
            raise SlotError("cannot replace slot while old readers are active")
        rec.state = SlotState.DRAINING
        try:
            rec.state = SlotState.FILLING
            transport.prepare_copy(layer_id, logical_expert, weight_version, expert_tp_shard)
            rec.state = SlotState.PREPARING
            rec.logical_expert = logical_expert
            rec.weight_version = weight_version
            rec.epoch = epoch
            rec.state = SlotState.READY
        except Exception:
            rec.state = SlotState.FAILED
            raise
        return rec

    def commit_epoch(self, epoch: int, participants: Iterable[int], expected_participants: Iterable[int]):
        got = tuple(sorted(participants))
        expected = tuple(sorted(expected_participants))
        if got != expected:
            raise SlotError(f"epoch participants mismatch: got={got} expected={expected}")
        self._committed_epoch[epoch] = len(got)

    def validate_dispatch(self, layer_id: int, rank: int, local_slot: int, logical_expert: int, epoch: int):
        rec = self.slot(layer_id, rank, local_slot)
        if rec.state is not SlotState.READY:
            raise SlotError("slot not ready")
        if rec.logical_expert != logical_expert or rec.epoch != epoch or rec.weight_version < 0:
            raise SlotError("slot identity/version mismatch")
        if epoch not in self._committed_epoch:
            raise SlotError("epoch not committed by full EP group")


class Planner:
    def __init__(self, hot_factor: float = 2.0, max_hot_experts: int = 2, max_receivers: int = 3, max_actions: int = 1):
        self.hot_factor = hot_factor
        self.max_hot_experts = max_hot_experts
        self.max_receivers = max_receivers
        self.max_actions = max_actions

    def plan(
        self,
        *,
        stats: RoutingStats,
        step_id: int,
        layer_id: int,
        epoch: int,
        rank_expert_counts: Sequence[Sequence[int]],
        primary_rank_by_expert: Sequence[int],
        primary_local_slot_by_expert: Sequence[int],
        primary_physical_by_expert: Sequence[int],
        destination_rank: int,
        destination_local_slot: int,
        destination_physical_id: int,
        cost_model: Optional[RankCostModel],
        planner_cost_us: Optional[float],
        prepare_cost_us: Optional[float],
        dispatch_extra_us: Optional[float],
        min_gain_us: float,
        epsilon_us: float,
        slot_manager: Optional[SlotManager] = None,
        transport: Optional[Transport] = None,
        weight_version: int = 0,
        layout_version: str = "unknown",
        execute_copy: bool = True,
    ) -> Plan:
        old_max = max(cost_model.predict(v) for v in rank_expert_counts) if cost_model else None
        if stats.total_tasks == 0:
            return self._noop(step_id, layer_id, epoch, -1, primary_rank_by_expert[0] if primary_rank_by_expert else -1, destination_rank, destination_local_slot, 0, (), layout_version, weight_version, None, "empty_input", primary_physical_by_expert[0] if primary_physical_by_expert else -1, destination_physical_id)
        mean = stats.total_tasks / len(stats.counts)
        hot = [e for e, n in enumerate(stats.counts) if n > self.hot_factor * mean]
        hot = hot[: self.max_hot_experts]
        if not hot:
            return self._noop(step_id, layer_id, epoch, -1, -1, destination_rank, destination_local_slot, 0, (), layout_version, weight_version, None, "no_hot_expert", -1, destination_physical_id)
        if cost_model is None or planner_cost_us is None or prepare_cost_us is None or dispatch_extra_us is None:
            return self._noop(step_id, layer_id, epoch, hot[0], primary_rank_by_expert[hot[0]], destination_rank, destination_local_slot, 0, (), layout_version, weight_version, None, "unknown_cost", primary_physical_by_expert[hot[0]], destination_physical_id)
        if execute_copy and (slot_manager is None or transport is None):
            return self._noop(step_id, layer_id, epoch, hot[0], primary_rank_by_expert[hot[0]], destination_rank, destination_local_slot, 0, (), layout_version, weight_version, None, "no_slot_or_transport", primary_physical_by_expert[hot[0]], destination_physical_id)
        best = None
        source_rank = primary_rank_by_expert[hot[0]]
        if source_rank == destination_rank:
            return self._noop(step_id, layer_id, epoch, hot[0], source_rank, destination_rank, destination_local_slot, 0, (), layout_version, weight_version, old_max, "same_rank", primary_physical_by_expert[hot[0]], destination_physical_id)
        for expert in hot:
            source_rank = primary_rank_by_expert[expert]
            cached = transport.is_cached(layer_id, expert, weight_version, 0) if execute_copy else False
            effective_prepare_cost = 0.0 if cached else prepare_cost_us
            for q in quota_candidates(stats, expert, sum(rank_expert_counts[source_rank]), sum(rank_expert_counts[destination_rank])):
                if q == 0:
                    continue
                new = [list(v) for v in rank_expert_counts]
                new[source_rank][primary_local_slot_by_expert[expert]] -= q
                # destination vector is extended by one slot; its new expert group is appended.
                new[destination_rank] = list(new[destination_rank]) + [q]
                new_max = max(cost_model.predict(v) for v in new)
                gain = old_max - new_max - planner_cost_us - effective_prepare_cost - dispatch_extra_us
                if best is None or gain > best[0]:
                    best = (gain, expert, source_rank, q, cached)
        if best is None or best[0] <= epsilon_us or best[0] < min_gain_us:
            e = best[1] if best else hot[0]
            return self._noop(step_id, layer_id, epoch, e, primary_rank_by_expert[e], destination_rank, destination_local_slot, 0, (), layout_version, weight_version, best[0] if best else None, "below_threshold", primary_physical_by_expert[e], destination_physical_id)
        gain, expert, source_rank, q, cached = best
        q_by_source = stats.quota_by_source(q)
        if execute_copy:
            slot_manager.replace_and_publish(layer_id=layer_id, rank=destination_rank, local_slot=destination_local_slot, logical_expert=expert, weight_version=weight_version, epoch=epoch, transport=transport)
            copy_required = not cached
            reason = "gain_above_threshold"
        else:
            copy_required = False
            reason = "shadow_plan"
        return Plan(step_id, layer_id, epoch, "route", expert, source_rank, source_rank, destination_rank, destination_local_slot, q, q_by_source, copy_required, layout_version, weight_version, gain, reason, primary_physical_by_expert[expert], destination_physical_id)

    @staticmethod
    def _noop(step_id, layer_id, epoch, expert, primary_rank, destination_rank, destination_local_slot, q, q_by_source, layout_version, weight_version, gain, reason, primary_physical_id, destination_physical_id):
        return Plan(step_id, layer_id, epoch, "noop", expert, primary_rank, primary_rank, destination_rank, destination_local_slot, q, q_by_source, False, layout_version, weight_version, gain, reason, primary_physical_id, destination_physical_id)


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    ep_size: int
    local_slots_per_rank: int
    destination_rank: int
    destination_local_slot: int
    cost_model: RankCostModel
    transport: Transport
    planner_cost_us: float
    prepare_cost_us: float
    dispatch_extra_us: float
    min_gain_us: float
    epsilon_us: float
    weight_version: int = 0
    layout_version: str = "unknown"
    expert_tp_size: int = 1
    dp_size: int = 1
    eager: bool = True


class RuntimeController:
    def __init__(self, mode: Mode, layer_id: Optional[int] = None, planner: Optional[Planner] = None, service_config: Optional[ServiceRuntimeConfig] = None):
        self.mode = mode
        self.layer_id = configured_layer_id() if layer_id is None else layer_id
        self.planner = planner or Planner()
        self.service_config = service_config
        self.slots = SlotManager()
        self.last_plan: dict[int, Plan] = {}
        self.stats: dict[int, RoutingStats] = {}
        self.epoch = 1

    def apply_service_hook(self, *, topk_ids, topk_weights, expert_location_dispatch_info, num_token_non_padded, num_fused_shared_experts):
        cfg = self.service_config
        if cfg is None:
            return topk_ids, False
        if self.mode is Mode.ACTIVE and (cfg.expert_tp_size != 1 or cfg.dp_size != 1 or not cfg.eager):
            return topk_ids, False
        all_map = getattr(expert_location_dispatch_info, "partial_logical_to_all_physical_map", None)
        if all_map is None:
            return topk_ids, False
        num_logical = int(all_map.shape[0])
        stats = collect_routing_stats([topk_ids], num_logical_experts=num_logical, num_token_non_padded=num_token_non_padded, num_fused_shared_experts=num_fused_shared_experts)
        step_id = next_step_id()
        self.stats[step_id] = stats
        dispatch_primary = getattr(expert_location_dispatch_info, "partial_logical_to_rank_dispatch_physical_map", None)
        if dispatch_primary is None:
            dispatch_primary = all_map[:, 0]
        primary_physical = [int(x) for x in dispatch_primary.tolist()]
        primary_rank = [p // cfg.local_slots_per_rank for p in primary_physical]
        primary_local_slot = [p % cfg.local_slots_per_rank for p in primary_physical]
        rank_vectors = [[0] * cfg.local_slots_per_rank for _ in range(cfg.ep_size)]
        for expert, count in enumerate(stats.counts):
            rank = primary_rank[expert]
            slot = primary_physical[expert] % cfg.local_slots_per_rank
            if 0 <= rank < cfg.ep_size and 0 <= slot < cfg.local_slots_per_rank:
                rank_vectors[rank][slot] += count
        destination_physical = cfg.destination_rank * cfg.local_slots_per_rank + cfg.destination_local_slot
        plan = self.planner.plan(stats=stats, step_id=step_id, layer_id=self.layer_id, epoch=self.epoch, rank_expert_counts=rank_vectors, primary_rank_by_expert=primary_rank, primary_local_slot_by_expert=primary_local_slot, primary_physical_by_expert=primary_physical, destination_rank=cfg.destination_rank, destination_local_slot=cfg.destination_local_slot, destination_physical_id=destination_physical, cost_model=cfg.cost_model, planner_cost_us=cfg.planner_cost_us, prepare_cost_us=cfg.prepare_cost_us, dispatch_extra_us=cfg.dispatch_extra_us, min_gain_us=cfg.min_gain_us, epsilon_us=cfg.epsilon_us, slot_manager=self.slots, transport=cfg.transport, weight_version=cfg.weight_version, layout_version=cfg.layout_version, execute_copy=(self.mode is Mode.ACTIVE))
        self.last_plan[step_id] = plan
        if self.mode is not Mode.ACTIVE or plan.action != "route":
            return topk_ids, False
        remapped, = remap_logical_to_physical([topk_ids], primary_physical_ids=primary_physical, plan=plan, logical_expert=plan.logical_expert, num_fused_shared_experts=num_fused_shared_experts)
        return remapped, True

    def observe(self, *, step_id, epoch, topk_ids_by_source, num_logical_experts, num_fused_shared_experts, rank_expert_counts, primary_rank_by_expert, primary_physical_by_expert, destination_rank, destination_local_slot, destination_physical_id, cost_model, planner_cost_us, prepare_cost_us, dispatch_extra_us, min_gain_us, epsilon_us, transport, weight_version=0, layout_version="unknown"):
        stats = collect_routing_stats(topk_ids_by_source, num_logical_experts=num_logical_experts, num_fused_shared_experts=num_fused_shared_experts)
        self.stats[step_id] = stats
        plan = self.planner.plan(stats=stats, step_id=step_id, layer_id=self.layer_id, epoch=epoch, rank_expert_counts=rank_expert_counts, primary_rank_by_expert=primary_rank_by_expert, primary_local_slot_by_expert=[primary_physical_by_expert[e] % len(rank_expert_counts[primary_rank_by_expert[e]]) for e in range(len(primary_rank_by_expert))], primary_physical_by_expert=primary_physical_by_expert, destination_rank=destination_rank, destination_local_slot=destination_local_slot, destination_physical_id=destination_physical_id, cost_model=cost_model, planner_cost_us=planner_cost_us, prepare_cost_us=prepare_cost_us, dispatch_extra_us=dispatch_extra_us, min_gain_us=min_gain_us, epsilon_us=epsilon_us, slot_manager=self.slots, transport=transport, weight_version=weight_version, layout_version=layout_version)
        self.last_plan[step_id] = plan
        return plan


_RUNTIME: Optional[RuntimeController] = None
_STEP_COUNTER = 0


def register_runtime_controller(controller: Optional[RuntimeController]) -> None:
    global _RUNTIME
    _RUNTIME = controller


def get_runtime_controller() -> Optional[RuntimeController]:
    return _RUNTIME


def next_step_id() -> int:
    global _STEP_COUNTER
    _STEP_COUNTER += 1
    return _STEP_COUNTER


_LAST_SHADOW_STATS: Optional[RoutingStats] = None


def maybe_apply_prefill_copy_aware(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_location_dispatch_info,
    layer_id: int,
    num_token_non_padded=None,
    num_fused_shared_experts: int = 0,
):
    """Service hook called after logical top-k and before physical mapping.

    Returns ``(ids, applied)``. In M0:
    - off: immediate no-op;
    - shadow: count logical routes and cache stats, return original ids;
    - active: only supported when an explicitly registered runtime controller
      exists; otherwise fall back and record why.
    """
    global _LAST_SHADOW_STATS
    mode = configured_mode()
    if mode is Mode.OFF:
        return topk_ids, False
    if layer_id != configured_layer_id():
        return topk_ids, False
    if expert_location_dispatch_info is None:
        return topk_ids, False
    algorithm = getattr(expert_location_dispatch_info, "ep_dispatch_algorithm", None)
    if algorithm not in ("static", "dynamic"):
        return topk_ids, False
    # M0 active execution supports only an explicit static primary map. Dynamic
    # replica choice remains shadow-only until the physical-slot contract is
    # calibrated on hardware.
    if mode is Mode.ACTIVE and algorithm != "static":
        return topk_ids, False
    physical_map = getattr(expert_location_dispatch_info, "partial_logical_to_all_physical_map", None)
    if physical_map is None:
        return topk_ids, False
    num_logical = int(physical_map.shape[0])
    stats = collect_routing_stats(
        [topk_ids],
        num_logical_experts=num_logical,
        num_token_non_padded=num_token_non_padded,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    _LAST_SHADOW_STATS = stats
    if _RUNTIME is None:
        return topk_ids, False
    return _RUNTIME.apply_service_hook(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        expert_location_dispatch_info=expert_location_dispatch_info,
        num_token_non_padded=num_token_non_padded,
        num_fused_shared_experts=num_fused_shared_experts,
    )


def get_last_shadow_stats() -> Optional[RoutingStats]:
    return _LAST_SHADOW_STATS
