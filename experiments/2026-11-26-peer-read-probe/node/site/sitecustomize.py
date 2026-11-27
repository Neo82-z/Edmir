"""Exploratory captures of other layers (2026-11-27). Loaded only when this folder is on PYTHONPATH.

The collector captures the gate/up input of one fixed layer per model, PROFILE["snapshot_layer"].
With EDMIR_SNAPSHOT_LAYER set, this hook changes that one value right after the collector module
is imported, in the parent and in every vLLM worker. The collector files stay byte-identical, so
the CPU receipts that bind their hashes remain valid; what the CPU step verified (assets,
tokenizers, engine arguments) does not depend on the layer. Every snapshot receipt records the
layer it was taken from. Captures made this way are exploratory and are kept apart from the
registered ones.
"""
import importlib.abc
import importlib.util
import os
import sys

_LAYER = os.environ.get("EDMIR_SNAPSHOT_LAYER")
_TARGET = "edmir.benchmark.real_moe"


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        module.PROFILE["snapshot_layer"] = int(_LAYER)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name!=_TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0,self)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if _LAYER:
    sys.meta_path.insert(0,_Finder())
