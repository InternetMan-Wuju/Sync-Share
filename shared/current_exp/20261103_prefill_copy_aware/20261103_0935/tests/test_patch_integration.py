#!/usr/bin/env python3
"""Static integration checks for the exact server-source patch."""
import hashlib
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'base/topk.py'; PATCHED=ROOT/'patched_tree/sglang/srt/hardware_backend/npu/moe/topk.py'
EXPECTED_BASE='f82f25bb343892b7b8b1eb833a4e6ca376a510ac74a3a935e0a1997dbd072f75'
class PatchIntegration(unittest.TestCase):
    def test_base_hash(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(),EXPECTED_BASE)
    def test_hook_is_between_logical_topk_and_physical_map(self):
        s=PATCHED.read_text()
        hook=s.index('maybe_apply_prefill_copy_aware(')
        fallback=s.index('topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)')
        self.assertLess(hook,fallback)
        self.assertIn('if get_is_extend_in_batch():',s)
        self.assertIn('if not copy_aware_applied:',s)
        self.assertIn('num_fused_shared_experts=topk_config.num_fused_shared_experts',s)
    def test_patch_contains_no_npu_execution(self):
        s=PATCHED.read_text()
        self.assertNotIn('launch_server',s)
        self.assertNotIn('torch.npu',s)
if __name__=='__main__': unittest.main(verbosity=2)
