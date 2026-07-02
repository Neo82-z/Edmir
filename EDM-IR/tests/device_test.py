from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir.device import (  # noqa: E402
    DeviceRegistry,
    DeviceSpec,
    arch_from_compute_capability,
    detect_device,
    format_device_spec,
    get_device_spec,
    known_device_specs,
    match_device_spec,
    register_device_spec,
    supported_nvidia_specs,
)


class DeviceSpecTest(unittest.TestCase):
    def test_builtin_hopper_and_blackwell_specs(self) -> None:
        h100 = get_device_spec("NVIDIA H100 80GB HBM3")
        self.assertEqual(h100.name, "H100")
        self.assertEqual(h100.vendor, "NVIDIA")
        self.assertEqual(h100.arch, "Hopper")
        self.assertEqual(h100.compute_capability, (9, 0))
        self.assertEqual(h100.cuda_arch, "sm90")
        self.assertTrue(h100.supports_tma)
        self.assertTrue(h100.supports_wgmma)
        self.assertTrue(h100.supports_fp8)

        rtx5090 = get_device_spec("GeForce RTX 5090")
        self.assertEqual(rtx5090.name, "RTX 5090")
        self.assertEqual(rtx5090.arch, "Blackwell")
        self.assertEqual(rtx5090.compute_capability, (12, 0))
        self.assertEqual(rtx5090.cuda_arch, "sm120")
        self.assertFalse(rtx5090.supports_nvlink)

    def test_registry_aliases_and_duplicate_protection(self) -> None:
        registry = DeviceRegistry()
        spec = DeviceSpec("Test GPU", arch="Hopper", compute_capability=(9, 0))
        registry.register(spec, aliases=("test-gpu", "sm90-test"))

        self.assertIs(registry.get("NVIDIA Test GPU"), spec)
        self.assertIs(registry.match("sm90-test dev0"), spec)

        with self.assertRaises(ValueError):
            registry.register(DeviceSpec("Test GPU"))

        updated = DeviceSpec("Test GPU", arch="Blackwell", compute_capability=(10, 0))
        registry.register(updated, replace_existing=True)
        self.assertIs(registry.get("test gpu"), updated)

    def test_arch_from_compute_capability(self) -> None:
        self.assertEqual(arch_from_compute_capability((9, 0)), "Hopper")
        self.assertEqual(arch_from_compute_capability((10, 0)), "Blackwell")
        self.assertEqual(arch_from_compute_capability((12, 0)), "Blackwell")
        self.assertIsNone(arch_from_compute_capability((8, 0)))
        self.assertIsNone(arch_from_compute_capability(None))

    def test_listing_helpers(self) -> None:
        names = {spec.name for spec in known_device_specs()}
        self.assertIn("H100", names)
        self.assertIn("B200", names)
        self.assertIn("RTX 5090", names)

        supported = supported_nvidia_specs()
        self.assertTrue(supported)
        self.assertTrue(all(spec.arch in {"Hopper", "Blackwell"} for spec in supported))

    def test_global_registration_helper(self) -> None:
        registry = DeviceRegistry()
        spec = DeviceSpec("Unit Test GPU", arch="Hopper", compute_capability=(9, 0))
        registered = register_device_spec(spec, aliases=("unit-test-gpu",), registry=registry)
        self.assertIs(registered, spec)
        self.assertIs(match_device_spec("unit test gpu", registry=registry), spec)

    def test_detect_device_is_optional(self) -> None:
        detected = detect_device()
        if self._testMethodName and any(arg in {"-v", "--verbose"} for arg in sys.argv):
            print("\n" + format_device_spec(detected))
        self.assertTrue(detected is None or isinstance(detected, DeviceSpec))


if __name__ == "__main__":
    unittest.main()
