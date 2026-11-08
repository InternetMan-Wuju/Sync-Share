#!/usr/bin/env python3
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ExternalInstall(unittest.TestCase):
    def test_no_source_copy_or_write(self):
        source=(ROOT/'m0_external/install.py').read_text()
        self.assertIn('import sglang.srt.hardware_backend.npu.moe.topk as npu_topk',source)
        self.assertIn('npu_topk.fused_topk_npu = _fused_wrapper',source)
        self.assertIn('npu_topk.topk_ids_logical_to_physical = _mapping_wrapper',source)
        self.assertNotIn('.write_text(',source)
        self.assertNotIn('open(',source)
    def test_launcher_installs_before_runpy(self):
        source=(ROOT/'m0_external/launch_server.py').read_text()
        self.assertLess(source.index('install()'),source.index('runpy.run_module'))
    def test_default_off_is_fast_passthrough(self):
        source=(ROOT/'m0_external/install.py').read_text()
        self.assertIn('if controller.configured_mode() is controller.Mode.OFF:',source)
        self.assertIn('return original(topk_ids, expert_location_dispatch_info',source)
if __name__=='__main__': unittest.main(verbosity=2)
