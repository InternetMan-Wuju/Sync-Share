#!/usr/bin/env python3
"""Generate patches/expert_distribution_phase_window.py from the v0.5.18 original.

Three additions (all opt-in via SGLANG_DUMP_EPLB_WINDOWS=1):
  (1) every window keeps the physical_to_logical_map that was in effect while it was recorded
  (2) auto dumps (EPLB rebalance, output_mode="object") are persisted too, not just manual ones
  (3) each step is tagged with its phase (prefill/decode) via two separate count buffers
"""
import pathlib, re, sys

base = pathlib.Path('/home/chengyetai/workspace/current_exp/pd_colloc_20261011/patches/v0518_orig/expert_distribution.py')
out  = pathlib.Path('/home/chengyetai/workspace/current_exp/pd_colloc_20261011/patches/expert_distribution_phase_window.py')
s = base.read_text()
n_edits = 0

# ---- Edit 0: imports ----
if re.search(r'^import os$', s, re.M) is None:
    s = s.replace('import math\n', 'import math\nimport os\n', 1)
    n_edits += 1

# ---- Edit A: base gatherer remembers the forward mode of the current pass ----
old = """    def __init__(self, expert_location_metadata: ExpertLocationMetadata, rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        pass
"""
new = """    def __init__(self, expert_location_metadata: ExpertLocationMetadata, rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank
        # [phase-window patch] phase of the pass currently being collected
        self._last_forward_mode = -1
        self._last_phase = "other"

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        # [phase-window patch] remember the phase of this forward pass.
        # Use SGLang's own predicates (authoritative); the raw value is diagnostic.
        fm = forward_batch.forward_mode
        self._last_forward_mode = int(fm.value)
        if fm.is_idle():
            self._last_phase = "idle"
        elif fm.is_decode():
            self._last_phase = "decode"
        elif fm.is_extend():
            self._last_phase = "prefill"
        else:
            self._last_phase = "other"
"""
assert old in s, 'Edit A anchor not found'
s = s.replace(old, new, 1); n_edits += 1

# ---- Edit B: GPU gatherer reports it in collect() ----
old = """        return dict(global_physical_count=global_physical_count)


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):"""
new = """        return dict(
            global_physical_count=global_physical_count,
            # [phase-window patch]
            forward_mode=getattr(self, "_last_forward_mode", -1),
            phase=getattr(self, "_last_phase", "other"),
        )


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):"""
assert old in s, 'Edit B anchor not found'
s = s.replace(old, new, 1); n_edits += 1

# ---- Edit D: unlock deepep auto for stat mode (stock raises NotImplementedError) ----
old = """            elif server_args.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata,
                    rank,
                    elastic_ep_enabled=server_args.elastic_ep_backend is not None,
                )
            else:
                raise NotImplementedError"""
new = """            elif server_args.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata,
                    rank,
                    elastic_ep_enabled=server_args.elastic_ep_backend is not None,
                )
            elif server_args.deepep_mode == "auto":
                # [phase-window patch] deepep auto = prefill normal + decode LL;
                # on_select_experts fires on the NPU topk path for BOTH, so the
                # router-side gatherer covers the whole window.
                return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
            else:
                raise NotImplementedError"""
assert old in s, 'Edit D anchor not found'
s = s.replace(old, new, 1); n_edits += 1

# ---- Edit C: replace _StatAccumulator with the two-buffer, always-persist version ----
start = s.index('class _StatAccumulator(_UtilizationRateAccumulatorMixin):')
end   = s.index('def _dump_to_file(name, data):')
new_stat = '''class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    # [phase-window patch] phases come from ForwardMode.is_idle()/is_decode()/is_extend()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (
            self._expert_location_metadata.num_layers,
            # Cannot use local_physical_count to support select_experts
            self._expert_location_metadata.num_physical_experts,
        )
        buffer_size = self._server_args.expert_distribution_recorder_buffer_size
        # [phase-window patch] one buffer per phase, so the phases can be summed
        # separately after the all_reduce (the all_reduce mixes ranks and steps).
        self._buf_by_phase = {
            key: _Buffer.init_new(
                item_shape=shape,
                buffer_size=buffer_size,
                dtype=torch.int32,
                device=self._server_args.device,
            )
            for key in ("prefill", "decode", "other")
        }
        self._first_dump = True
        self._pw_window_index = 0
        # [phase-window patch] append-order wall-clock per phase (kept <= buffer_size)
        self._ts_by_phase = {k: [] for k in ("prefill", "decode", "other")}
        self._n_steps_by_phase = {k: 0 for k in ("prefill", "decode", "other")}
        self._window_start_ts = None
        self._window_end_ts = None
        self._ts_keep = buffer_size if buffer_size and buffer_size > 0 else 4096
        self._mode_histogram = {}

    def _phase_key(self, single_pass_data: Dict) -> str:
        # "idle" steps carry no tokens -> return None meaning "do not buffer them"
        # (the buffer has fixed capacity, so skipping them buys coverage).
        phase = single_pass_data.get("phase", "other")
        if phase == "idle":
            return None
        return phase if phase in ("prefill", "decode") else "other"

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
        # Can optimize if overhead here is large
        key = self._phase_key(single_pass_data)
        # [phase-window patch] diagnostics: steps seen per raw ForwardMode value
        raw = int(single_pass_data.get("forward_mode", -1))
        self._mode_histogram[raw] = self._mode_histogram.get(raw, 0) + 1
        if key is None:
            return
        self._buf_by_phase[key].append(single_pass_data["global_physical_count"])
        # [phase-window patch] per-step wall-clock time
        now = time.time()
        self._ts_by_phase[key].append(now)
        if len(self._ts_by_phase[key]) > self._ts_keep:
            del self._ts_by_phase[key][: len(self._ts_by_phase[key]) - self._ts_keep]
        self._n_steps_by_phase[key] += 1
        if self._window_start_ts is None:
            self._window_start_ts = now
        self._window_end_ts = now

    def _logical_count_of_phase(self, key: str):
        """Per-step logical counts for this phase.

        NOTE: this must NOT trim rows. The circular buffer always yields exactly
        buffer_size rows on every rank, and the all_reduce below requires every
        rank to pass a tensor of the *same* shape -- trimming per rank made the
        shapes differ and HCCL aborted with
        "collective communication arguments inconsistent between ranks".
        Trimming happens later, only for the file written by rank 0."""
        all_steps = self._buf_by_phase[key].get_all()
        if all_steps.shape[0] == 0:
            return None
        return _convert_global_physical_count_to_logical_count(
            all_steps,
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )

    @staticmethod
    def _drop_empty_steps(t):
        """Drop the all-zero (not yet filled) rows of the circular buffer."""
        if t is None:
            return None
        nonempty = t.reshape(t.shape[0], -1).sum(dim=1) != 0
        t = t[nonempty]
        return t if t.shape[0] else None

    def reset(self):
        super().reset()
        for buf in self._buf_by_phase.values():
            buf.reset()
        # [phase-window patch]
        for k in self._ts_by_phase:
            self._ts_by_phase[k].clear()
            self._n_steps_by_phase[k] = 0
        self._window_start_ts = None
        self._window_end_ts = None
        self._mode_histogram = {}

    def dump(self, output_mode: _OutputMode):
        logical_of_phase = {
            key: self._logical_count_of_phase(key)
            for key in ("prefill", "decode", "other")
        }
        phase_keys = [k for k in ("prefill", "decode", "other") if logical_of_phase[k] is not None]
        phase_sizes = [logical_of_phase[k].shape[0] for k in phase_keys]
        present = [logical_of_phase[k] for k in phase_keys]
        if not present:
            present = [
                torch.zeros(
                    (
                        0,
                        self._expert_location_metadata.num_layers,
                        self._expert_location_metadata.num_logical_experts,
                    ),
                    dtype=torch.int32,
                    device=self._server_args.device,
                )
            ]
        logical_count_of_buffered_step = torch.cat(present, dim=0)

        if self._first_dump:
            self._first_dump = False
            torch.get_device_module().empty_cache()

        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )

        # [phase-window patch] persist the window AFTER the all_reduce, so the
        # counts are global (all_reduce exists precisely because each rank only
        # observes a fraction of the traffic). Slice the phases back out.
        if os.environ.get("SGLANG_DUMP_EPLB_WINDOWS", "0") == "1":
            window_kind = "eplb_rebalance" if output_mode == "object" else "manual_final"
            if self._rank == 0:
                global_of_phase, offset = {}, 0
                for k, n in zip(phase_keys, phase_sizes):
                    # trim here (rank-0-only, after the collective) -- never before it
                    global_of_phase[k] = self._drop_empty_steps(
                        logical_count_of_buffered_step[offset : offset + n]
                    )
                    offset += n
                _dump_to_file(
                    f"{self._pw_window_index:06d}_{time.time()}.pt",
                    dict(
                        num_layers=self._expert_location_metadata.num_layers,
                        num_logical_experts=self._expert_location_metadata.num_logical_experts,
                        num_physical_experts=self._expert_location_metadata.num_physical_experts,
                        ep_size=self._expert_location_metadata.ep_size,
                        # per-step counts, split by phase: [steps, layers, logical_experts]
                        logical_count_prefill=global_of_phase.get("prefill"),
                        logical_count_decode=global_of_phase.get("decode"),
                        logical_count_other=global_of_phase.get("other"),
                        # [phase-window patch] the layout in effect for THIS window
                        physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
                        # [phase-window patch] time axis
                        step_timestamps_by_phase={k: list(v) for k, v in self._ts_by_phase.items()},
                        n_steps_by_phase=dict(self._n_steps_by_phase),
                        mode_histogram=dict(self._mode_histogram),
                        window_start_timestamp=self._window_start_ts,
                        window_end_timestamp=self._window_end_ts,
                        window_kind=window_kind,
                        dump_index=self._pw_window_index,
                        dump_timestamp=time.time(),
                    ),
                )
            self._pw_window_index += 1

        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
            average_utilization_rate_over_window=self._get_global_average_utilization_rate(),
        )

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError

    def _get_global_average_utilization_rate(self):
        if not self._enable or math.isclose(
            self._server_args.eplb_min_rebalancing_utilization_threshold, 1.0
        ):
            return None

        if self._rank == 0:
            utilization_mean_rates = self._history.mean()
            window_index = self.window_sizes[-1]
            average_utilization_rate_over_window = (
                utilization_mean_rates[window_index]
                if window_index in utilization_mean_rates
                else 0
            )

            avg_rate_tensor = torch.tensor(
                [average_utilization_rate_over_window],
                dtype=torch.float32,
                device="cuda",
            )
        else:
            avg_rate_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
        torch.distributed.broadcast(avg_rate_tensor, src=0)
        return avg_rate_tensor.item()


'''
s = s[:start] + new_stat + s[end:]
n_edits += 1

out.write_text(s)
print(f'wrote {out}  ({len(s)} bytes, {n_edits} edits)')
