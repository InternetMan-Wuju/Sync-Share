"""Bootstrap a read-only layer49 routing observer in every SGLang worker."""
import importlib.abc
import importlib.machinery
import os
import sys


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.srt.models.deepseek_v2":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        base = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return base.create_module(spec)

            def exec_module(self, module):
                base.exec_module(module)
                import observer
                observer.install(module)

        spec.loader = Loader()
        return spec


if os.environ.get("CYT_LAYER49_OBSERVER") == "1":
    sys.meta_path.insert(0, Hook())
