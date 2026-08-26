from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.runtime_capabilities import (  # noqa: E402
    NCCL_DEVICE_API_SYMBOLS,
    _evaluate_device_api,
    _parse_nccl_version,
)


class RuntimeCapabilitiesTest(unittest.TestCase):
    def test_nccl_version_macros_are_parsed(self) -> None:
        version = _parse_nccl_version(
            "#define NCCL_MAJOR 2\n"
            "#define NCCL_MINOR 28\n"
            "#define NCCL_PATCH 9\n"
            "#define NCCL_VERSION_CODE 22809\n"
        )
        self.assertEqual(version["version"], "2.28.9")
        self.assertEqual(version["version_code"], 22809)

    def test_device_api_gate_requires_version_header_symbols_and_compile(self) -> None:
        symbols = {name: True for name in NCCL_DEVICE_API_SYMBOLS}
        ready, blockers = _evaluate_device_api(
            version_code=22809,
            device_header=True,
            symbols=symbols,
            compile_ok=True,
        )
        self.assertTrue(ready)
        self.assertFalse(blockers)

        ready, blockers = _evaluate_device_api(
            version_code=22501,
            device_header=False,
            symbols={name: False for name in NCCL_DEVICE_API_SYMBOLS},
            compile_ok=False,
        )
        self.assertFalse(ready)
        self.assertTrue(any("older" in blocker for blocker in blockers))
        self.assertTrue(any("nccl_device.h" in blocker for blocker in blockers))
        self.assertTrue(any("missing Device API symbols" in blocker for blocker in blockers))


if __name__ == "__main__":
    unittest.main()
