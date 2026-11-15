"""External bootstrap for every spawned worker; installed files stay intact."""
import os, runpy, sys, importlib.abc, importlib.machinery
if os.environ.get('SGLANG_READONLY_WEIGHT_MANIFEST'):
    runpy.run_path('/study/weights/loader/sitecustomize.py')
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'sglang.srt.models.deepseek_v2': return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None: return None
        base = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self, spec): return base.create_module(spec)
            def exec_module(self, module):
                base.exec_module(module)
                from runtime import install
                install(module)
        spec.loader = Loader()
        return spec
if os.environ.get('CYT_EP16_EXPERIMENT') == '1': sys.meta_path.insert(0, Hook())
