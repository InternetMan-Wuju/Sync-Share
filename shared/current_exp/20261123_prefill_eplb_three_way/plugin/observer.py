"""Record real layer49 routing and optional hidden/topk tensors.

Writing happens only on global rank 0 and is not used for formal timing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch


def _sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def install(module):
    out = Path(os.environ.get("CYT_OBSERVER_DIR", "/runexp/observer"))
    capture_hidden = os.environ.get("CYT_OBSERVER_CAPTURE_HIDDEN", "1") == "1"
    orig_init = module.DeepseekV2MoE.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if self.layer_id != 49:
            return
        original_forward = self.experts.forward
        self._cyt_observer_seq = 0

        def observed_experts_forward(hidden_states, topk_output=None, *fargs, **fkwargs):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            rows = int(hidden_states.shape[0]) if hasattr(hidden_states, "shape") else 0
            if topk_output is not None and rank >= 0 and rows > 10:
                seq = self._cyt_observer_seq
                self._cyt_observer_seq += 1
                out.mkdir(parents=True, exist_ok=True)
                ids = topk_output.topk_ids.detach()
                weights = topk_output.topk_weights.detach()
                physical = ids.reshape(-1).to(torch.int64).cpu()
                physical_counts = torch.bincount(physical, minlength=272)[:272]
                logical_ids = physical.clone()
                try:
                    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
                    meta = get_global_expert_location_metadata()
                    if meta is not None:
                        m = meta.physical_to_logical_map_cpu[self.layer_id]
                        logical_ids = m[physical]
                except Exception:
                    pass
                logical_counts = torch.bincount(logical_ids, minlength=256)[:256]
                meta_row = {
                    "kind": "layer49_route",
                    "at": time.time(),
                    "seq": seq,
                    "rank": rank,
                    "local_rows": rows,
                    "topk": int(ids.shape[-1]),
                    "physical_task_count": int(physical.numel()),
                    "logical_task_count": int(logical_ids.numel()),
                    "physical_counts": physical_counts.tolist(),
                    "logical_counts": logical_counts.tolist(),
                    "topk_ids_sha256": _sha_tensor(ids),
                    "topk_weights_sha256": _sha_tensor(weights),
                    "hidden_sha256": _sha_tensor(hidden_states),
                }
                if capture_hidden:
                    torch.save(hidden_states.detach().cpu(), out / f"rank{rank}_layer49_seq{seq}_hidden.pt")
                    torch.save(ids.detach().cpu(), out / f"rank{rank}_layer49_seq{seq}_topk_ids.pt")
                    torch.save(weights.detach().cpu(), out / f"rank{rank}_layer49_seq{seq}_topk_weights.pt")
                with (out / f"routes_rank{rank}.jsonl").open("a") as f:
                    f.write(json.dumps(meta_row, separators=(",", ":")) + "\n")
            return original_forward(hidden_states, topk_output, *fargs, **fkwargs)

        self.experts.forward = observed_experts_forward

    module.DeepseekV2MoE.__init__ = patched_init
