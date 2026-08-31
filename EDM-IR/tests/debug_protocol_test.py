from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.debug_protocol import (  # noqa: E402
    build_debug_commands,
    evaluate_gpu_admission,
    parse_gpu_inventory,
    parse_gpu_processes,
)


class DebugProtocolTest(unittest.TestCase):
    def inventory(self):
        return parse_gpu_inventory(
            "0, NVIDIA GeForce RTX 4090, GPU-a, 24564, 12, 0\n"
            "1, NVIDIA GeForce RTX 4090, GPU-b, 24564, 15, 1\n"
            "2, NVIDIA GeForce RTX 4090, GPU-c, 24564, 20, 0\n"
            "3, NVIDIA GeForce RTX 4090, GPU-d, 24564, 18, 2\n"
        )

    def test_idle_four_4090_inventory_is_admitted(self) -> None:
        admission = evaluate_gpu_admission(
            self.inventory(),
            parse_gpu_processes(""),
            (0, 1, 2, 3),
            maximum_memory_used_mib=1024,
            maximum_utilization_percent=10,
        )
        self.assertTrue(admission.admitted)
        self.assertFalse(admission.reasons)

    def test_foreign_process_and_busy_memory_fail_closed(self) -> None:
        inventory = parse_gpu_inventory(
            "0, NVIDIA GeForce RTX 4090, GPU-a, 24564, 12000, 0\n"
            "1, NVIDIA GeForce RTX 4090, GPU-b, 24564, 15, 1\n"
            "2, NVIDIA GeForce RTX 4090, GPU-c, 24564, 20, 0\n"
            "3, NVIDIA GeForce RTX 4090, GPU-d, 24564, 18, 2\n"
        )
        admission = evaluate_gpu_admission(
            inventory,
            parse_gpu_processes("GPU-b, 4242, 256\n"),
            (0, 1, 2, 3),
            maximum_memory_used_mib=1024,
            maximum_utilization_percent=10,
        )
        self.assertFalse(admission.admitted)
        self.assertTrue(any("above" in reason for reason in admission.reasons))
        self.assertTrue(any("PID 4242" in reason for reason in admission.reasons))

    def test_wrong_gpu_or_missing_device_is_rejected(self) -> None:
        inventory = parse_gpu_inventory(
            "0, NVIDIA A100-SXM4-80GB, GPU-a, 81920, 0, 0\n"
            "1, NVIDIA GeForce RTX 4090, GPU-b, 24564, 0, 0\n"
            "2, NVIDIA GeForce RTX 4090, GPU-c, 24564, 0, 0\n"
        )
        admission = evaluate_gpu_admission(
            inventory,
            (),
            (0, 1, 2, 3),
            maximum_memory_used_mib=1024,
            maximum_utilization_percent=10,
        )
        self.assertFalse(admission.admitted)
        self.assertTrue(any("not an RTX 4090" in reason for reason in admission.reasons))
        self.assertTrue(any("absent" in reason for reason in admission.reasons))

    def test_plan_separates_cpu_gpu_and_requires_sm89_fused_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "EDM-IR"
            output = Path(directory) / "run"
            commands = build_debug_commands(root, output, python="python3", rounds=2)
        by_name = {command.name: command for command in commands}
        self.assertEqual(by_name["python-tests"].phase, "cpu")
        self.assertFalse(by_name["sm89-fused-fail-closed"].gpu_required)
        self.assertEqual(by_name["sm89-fused-fail-closed"].expected_returncodes, (1,))
        self.assertIn(
            "arch=sm_80", by_name["sm89-fused-fail-closed"].expected_output or ""
        )
        self.assertTrue(by_name["topology-probe"].gpu_required)
        self.assertIn("numerical-round-02", by_name)
        self.assertIn("reuse-round-02", by_name)

    def test_explicit_cuda_toolchain_adds_offline_build_and_reaches_gpu_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            commands = build_debug_commands(
                base / "EDM-IR",
                base / "run",
                python="python3",
                rounds=1,
                cuda_toolkit_root=base / "cuda",
                nccl_include_dir=base / "nccl" / "include",
                nccl_library=base / "nccl" / "lib" / "libnccl.so.2",
            )
        by_name = {command.name: command for command in commands}
        self.assertFalse(by_name["cuda-offline-build"].gpu_required)
        self.assertTrue(by_name["cuda-build"].gpu_required)
        configure = by_name["cuda-configure"].argv
        self.assertTrue(any(value.startswith("-DCUDAToolkit_ROOT=") for value in configure))
        self.assertTrue(any(value.startswith("-DNCCL_INCLUDE_DIR=") for value in configure))
        self.assertTrue(any(value.startswith("-DNCCL_LIBRARY=") for value in configure))


if __name__ == "__main__":
    unittest.main()
