from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    FusedEnvironmentFingerprint,
    FusedEvidence,
    FusedKRule,
    FusedSelectionProfile,
    FusedShapeEnvelope,
    JsonBackend,
    TopologySnapshot,
    select_fused_a2a_gemm,
)


def environment() -> FusedEnvironmentFingerprint:
    return FusedEnvironmentFingerprint(
        arch="sm_80",
        device="NVIDIA A100-SXM4-80GB",
        world_size=4,
        link_label="NV12",
        driver_version="580.126.16",
        cuda_version="12.8.93",
        nccl_version="v2.28.9-1",
        nccl_commit="nccl-commit",
        cutlass_version="v4.7.0",
        cutlass_commit="cutlass-commit",
    )


def topology() -> TopologySnapshot:
    return TopologySnapshot(
        world_size=4,
        devices=tuple("NVIDIA A100-SXM4-80GB" for _ in range(4)),
        p2p=tuple(tuple(True for _ in range(4)) for _ in range(4)),
        link_labels=tuple(
            tuple("X" if src == dst else "NV12" for dst in range(4))
            for src in range(4)
        ),
    )


def profile() -> FusedSelectionProfile:
    return FusedSelectionProfile(
        profile_id="test-a100-profile",
        source="test-heldout-grid",
        environment=environment(),
        envelope=FusedShapeEnvelope(
            rows_per_peer=(32, 64, 128),
            n=(64, 128, 256),
            k=(256, 512, 1024, 2048, 4096),
        ),
        rule=FusedKRule(max_direct_k=1024, min_split_k=2048),
        evidence=FusedEvidence(
            train_problems=5,
            heldout_problems=40,
            heldout_direction_accuracy=1.0,
            heldout_max_selection_regret=0.0,
            heldout_host_wins=0,
        ),
    )


def problem(k: int = 512) -> A2AGemmProblem:
    return A2AGemmProblem(world_size=4, rows_per_peer=64, k=k, n=128)


class FusedSelectionTest(unittest.TestCase):
    def select(self, candidate: A2AGemmProblem, **kwargs):
        return select_fused_a2a_gemm(
            candidate,
            topology(),
            profile(),
            environment(),
            **kwargs,
        )

    def test_requires_explicit_opt_in_and_emits_host_fallback(self) -> None:
        result = self.select(problem())
        self.assertFalse(result.admitted)
        self.assertEqual(result.selected_family, "host")
        self.assertEqual(result.program.strategy, "sequential")
        self.assertEqual(result.target.communication_api, "host_nccl")
        self.assertIn("not explicitly enabled", result.reason)
        self.assertIsNone(result.traffic)
        self.assertIn("fused_selection", result.program.metadata)

    def test_selects_direct_inside_exact_measured_envelope(self) -> None:
        result = self.select(problem(512), allow_experimental_fused=True)
        self.assertTrue(result.admitted)
        self.assertEqual(result.selected_family, "direct")
        self.assertEqual(result.program.strategy, "fused")
        self.assertEqual(result.target.gemm_kernel, "sm80_64x128x32")
        self.assertEqual(result.target.communication_api, "nccl_device")
        self.assertEqual(result.traffic["split_k_slices"], 1)
        artifact = JsonBackend().emit(result.program, result.target)
        self.assertEqual(artifact.metadata["gemm_kernel"], "sm80_64x128x32")

    def test_selects_split_k_inside_exact_measured_envelope(self) -> None:
        result = self.select(problem(4096), allow_experimental_fused=True)
        self.assertTrue(result.admitted)
        self.assertEqual(result.selected_family, "splitk8")
        self.assertEqual(result.target.gemm_kernel, "sm80_64x128x32_splitk8")
        self.assertEqual(result.traffic["split_k_slices"], 8)

    def test_refuses_shape_extrapolation(self) -> None:
        result = self.select(problem(8192), allow_experimental_fused=True)
        self.assertEqual(result.selected_family, "host")
        self.assertIn("outside the measured envelope", result.reason)

    def test_refuses_environment_mismatch(self) -> None:
        result = select_fused_a2a_gemm(
            problem(),
            topology(),
            profile(),
            replace(environment(), cutlass_commit="different"),
            allow_experimental_fused=True,
        )
        self.assertEqual(result.selected_family, "host")
        self.assertIn("fingerprint", result.reason)

    def test_unmeasured_architecture_is_a_fallback_not_a_schema_error(self) -> None:
        observed = replace(
            environment(),
            arch="sm_90",
            device="NVIDIA H100 80GB HBM3",
            link_label="NV18",
        )
        observed_topology = replace(
            topology(),
            devices=tuple(observed.device for _ in range(4)),
            link_labels=tuple(
                tuple("X" if src == dst else "NV18" for dst in range(4))
                for src in range(4)
            ),
        )
        result = select_fused_a2a_gemm(
            problem(),
            observed_topology,
            profile(),
            observed,
            allow_experimental_fused=True,
        )
        self.assertEqual(result.selected_family, "host")
        self.assertEqual(result.target.arch, "sm_90")
        self.assertIn("fingerprint", result.reason)

    def test_refuses_topology_without_exact_link_identity(self) -> None:
        unlabeled = replace(topology(), link_labels=())
        result = select_fused_a2a_gemm(
            problem(),
            unlabeled,
            profile(),
            environment(),
            allow_experimental_fused=True,
        )
        self.assertEqual(result.selected_family, "host")
        self.assertIn("link labels are missing", result.reason)

    def test_refuses_insufficient_heldout_evidence(self) -> None:
        weak = replace(
            profile(), evidence=replace(profile().evidence, heldout_problems=10)
        )
        result = select_fused_a2a_gemm(
            problem(),
            topology(),
            weak,
            environment(),
            allow_experimental_fused=True,
        )
        self.assertEqual(result.selected_family, "host")
        self.assertIn("at least 30", result.reason)

    def test_environment_and_profile_round_trip(self) -> None:
        self.assertEqual(
            FusedEnvironmentFingerprint.from_dict(environment().to_dict()),
            environment(),
        )
        self.assertEqual(FusedSelectionProfile.from_dict(profile().to_dict()), profile())

    def test_profile_cannot_open_admission_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit admission"):
            replace(profile(), default_compiler_admission=True).validate()

    def test_retained_profile_matches_heldout_evaluation(self) -> None:
        edmir_root = Path(__file__).resolve().parents[1]
        repo_root = edmir_root.parent
        retained = FusedSelectionProfile.from_dict(
            json.loads(
                (edmir_root / "examples/a100_fused_selection_profile.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        evaluation_path = repo_root / retained.source
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        self.assertEqual(
            retained.rule.max_direct_k, evaluation["rule"]["max_direct_k"]
        )
        self.assertEqual(
            retained.rule.min_split_k, evaluation["rule"]["min_split_k"]
        )
        self.assertEqual(
            retained.evidence.heldout_problems,
            evaluation["heldout"]["problems"],
        )
        self.assertEqual(
            retained.evidence.heldout_direction_accuracy,
            evaluation["heldout"]["direction_accuracy"],
        )
        self.assertEqual(
            retained.evidence.heldout_max_selection_regret,
            evaluation["heldout"]["max_selection_regret"],
        )
        self.assertEqual(
            retained.evidence.heldout_host_wins,
            evaluation["heldout"]["host_wins"],
        )


if __name__ == "__main__":
    unittest.main()
