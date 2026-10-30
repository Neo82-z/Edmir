from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    Target,
    TopologySnapshot,
    build_fused_program,
    enumerate_fused_schedule_space,
)


class FusedScheduleSpaceTest(unittest.TestCase):
    def test_grouped_n_exposes_traffic_versus_cta_supply(self) -> None:
        problem = A2AGemmProblem(4, 64, 256, 512)
        topology = TopologySnapshot(
            world_size=4,
            devices=("A100",) * 4,
            p2p=((True,) * 4,) * 4,
        )
        target = Target(
            "schedule-space",
            topology,
            arch="sm_80",
            gemm_provider="cutlass",
            communication_api="nccl_device",
        )
        points = enumerate_fused_schedule_space(
            build_fused_program(problem, chunk_rows=64),
            target,
            gemm_kernels=("sm80_64x128x32", "sm80_64x128x32_splitk8"),
            n_tiles_per_cta=(1, 2, 4),
        )
        by_family = {point.family: point for point in points}

        self.assertEqual(set(by_family), {
            "direct",
            "direct_groupedn2",
            "splitk8",
            "splitk8_groupedn2",
            "direct_groupedn4",
            "splitk8_groupedn4",
        })
        self.assertTrue(by_family["direct"].execution_ready)
        self.assertTrue(by_family["direct_groupedn2"].execution_ready)
        self.assertFalse(by_family["direct_groupedn4"].execution_ready)
        self.assertEqual(by_family["direct"].traffic.remote_a_replay, 4)
        self.assertEqual(by_family["direct_groupedn2"].traffic.remote_a_replay, 2)
        self.assertEqual(by_family["direct"].traffic.estimated_ctas_per_op, 4)
        self.assertEqual(by_family["direct_groupedn2"].traffic.estimated_ctas_per_op, 2)
        self.assertEqual(by_family["splitk8"].traffic.estimated_ctas_per_op, 32)
        self.assertEqual(
            by_family["splitk8_groupedn2"].traffic.estimated_ctas_per_op, 16
        )

    def test_illegal_split_k_point_is_retained_with_reason(self) -> None:
        problem = A2AGemmProblem(4, 64, 128, 256)
        topology = TopologySnapshot(4, ("A100",) * 4, ((True,) * 4,) * 4)
        target = Target(
            "schedule-space",
            topology,
            arch="sm_80",
            gemm_provider="cutlass",
            communication_api="nccl_device",
        )
        point = enumerate_fused_schedule_space(
            build_fused_program(problem, chunk_rows=64),
            target,
            gemm_kernels=("sm80_64x128x32_splitk8",),
            n_tiles_per_cta=(1,),
        )[0]
        self.assertFalse(point.legal)
        self.assertFalse(point.execution_ready)
        self.assertEqual(point.family, "splitk8")
        self.assertIn("K divisible by 256", point.reason)


if __name__ == "__main__":
    unittest.main()
