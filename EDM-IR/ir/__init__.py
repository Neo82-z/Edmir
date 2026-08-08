"""Compatibility entry point for the former ``ir`` package.

New code should import :mod:`edmir`. This shim remains for one transition
cycle so existing notebooks and experiment scripts keep working.
"""

import importlib
import sys
from warnings import warn

warn("import 'edmir' instead of the deprecated 'ir' package", DeprecationWarning, stacklevel=2)

from edmir import *  # noqa: F401,F403

for _module in ("analysis", "builder", "device", "graph", "ops", "printing", "uop"):
    sys.modules[f"ir.{_module}"] = importlib.import_module(f"edmir.{_module}")
