from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .model import A2AGemmProblem, Program, TopologySnapshot
from .schedule import build_fused_program, build_sequential_program
from .host_baseline import PER_SOURCE_CUBLAS, host_baseline
from .target import Target


SCHEMA_FUSED_ENVIRONMENT = "edmir.fused_environment.v1"
SCHEMA_FUSED_SELECTION_PROFILE = "edmir.fused_selection_profile.v1"
SCHEMA_FUSED_SELECTION_REPORT = "edmir.fused_selection_report.v1"


def _check_schema(value: dict[str, Any], expected: str) -> None:
    schema = value.get("schema")
    if schema not in (None, expected):
        raise ValueError(f"unsupported schema {schema!r}; expected {expected!r}")


def _positive_values(name: str, values: tuple[int, ...]) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive values")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class FusedEnvironmentFingerprint:
    """Exact hardware/software identity of one measured fused-kernel profile."""

    arch: str
    device: str
    world_size: int
    link_label: str
    driver_version: str
    cuda_version: str
    nccl_version: str
    nccl_commit: str
    cutlass_version: str
    cutlass_commit: str

    def validate(self) -> None:
        if self.world_size < 2:
            raise ValueError("fused environment world_size must be >= 2")
        for field_name in (
            "arch",
            "device",
            "link_label",
            "driver_version",
            "cuda_version",
            "nccl_version",
            "nccl_commit",
            "cutlass_version",
            "cutlass_commit",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"fused environment {field_name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_FUSED_ENVIRONMENT, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FusedEnvironmentFingerprint":
        _check_schema(value, SCHEMA_FUSED_ENVIRONMENT)
        environment = cls(
            arch=str(value["arch"]),
            device=str(value["device"]),
            world_size=int(value["world_size"]),
            link_label=str(value["link_label"]),
            driver_version=str(value["driver_version"]),
            cuda_version=str(value["cuda_version"]),
            nccl_version=str(value["nccl_version"]),
            nccl_commit=str(value["nccl_commit"]),
            cutlass_version=str(value["cutlass_version"]),
            cutlass_commit=str(value["cutlass_commit"]),
        )
        environment.validate()
        return environment


@dataclass(frozen=True, slots=True)
class FusedShapeEnvelope:
    rows_per_peer: tuple[int, ...]
    n: tuple[int, ...]
    k: tuple[int, ...]
    dtype: str = "float16"
    layout: str = "row_major"

    def validate(self) -> None:
        _positive_values("rows_per_peer", self.rows_per_peer)
        _positive_values("n", self.n)
        _positive_values("k", self.k)
        if self.dtype != "float16" or self.layout != "row_major":
            raise ValueError("fused selector v0 supports only row-major float16 profiles")

    def contains(self, problem: A2AGemmProblem) -> bool:
        return (
            problem.rows_per_peer in self.rows_per_peer
            and problem.n in self.n
            and problem.k in self.k
            and problem.dtype == self.dtype
            and problem.layout == self.layout
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_per_peer": list(self.rows_per_peer),
            "n": list(self.n),
            "k": list(self.k),
            "dtype": self.dtype,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FusedShapeEnvelope":
        envelope = cls(
            rows_per_peer=tuple(int(item) for item in value["rows_per_peer"]),
            n=tuple(int(item) for item in value["n"]),
            k=tuple(int(item) for item in value["k"]),
            dtype=str(value.get("dtype", "float16")),
            layout=str(value.get("layout", "row_major")),
        )
        envelope.validate()
        return envelope


@dataclass(frozen=True, slots=True)
class FusedKRule:
    max_direct_k: int
    min_split_k: int
    direct_kernel: str = "sm80_64x128x32"
    split_kernel: str = "sm80_64x128x32_splitk8"

    def validate(self) -> None:
        if self.max_direct_k <= 0 or self.min_split_k <= 0:
            raise ValueError("fused K boundaries must be positive")
        if self.max_direct_k >= self.min_split_k:
            raise ValueError("direct and split-K fused ranges must not overlap")
        if self.direct_kernel != "sm80_64x128x32":
            raise ValueError("fused selector v0 direct profile requires sm80_64x128x32")
        if self.split_kernel != "sm80_64x128x32_splitk8":
            raise ValueError(
                "fused selector v0 split profile requires sm80_64x128x32_splitk8"
            )

    def family(self, k: int) -> str | None:
        if k <= self.max_direct_k:
            return "direct"
        if k >= self.min_split_k:
            return "splitk8"
        return None


@dataclass(frozen=True, slots=True)
class FusedEvidence:
    train_problems: int
    heldout_problems: int
    heldout_direction_accuracy: float
    heldout_max_selection_regret: float
    heldout_host_wins: int

    def validate(self) -> None:
        for name in ("train_problems", "heldout_problems", "heldout_host_wins"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"fused evidence {name} must be a nonnegative integer")
        for name in ("heldout_direction_accuracy", "heldout_max_selection_regret"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"fused evidence {name} must be finite")
        if not 0.0 <= self.heldout_direction_accuracy <= 1.0:
            raise ValueError("heldout_direction_accuracy must be in [0, 1]")
        if self.heldout_max_selection_regret < 0.0:
            raise ValueError("heldout_max_selection_regret must be non-negative")
        if self.heldout_host_wins > self.heldout_problems:
            raise ValueError("heldout_host_wins exceeds the held-out problem count")

    def admission_error(self) -> str | None:
        self.validate()
        if self.heldout_problems < 30:
            return "evidence requires at least 30 held-out problems"
        if self.heldout_direction_accuracy < 0.99:
            return "held-out direction accuracy is below 99%"
        if self.heldout_max_selection_regret > 0.01:
            return "held-out maximum selection regret exceeds 1%"
        if self.heldout_host_wins != 0:
            return "host won at least one held-out problem"
        return None


@dataclass(frozen=True, slots=True)
class FusedSelectionProfile:
    profile_id: str
    source: str
    environment: FusedEnvironmentFingerprint
    envelope: FusedShapeEnvelope
    rule: FusedKRule
    evidence: FusedEvidence
    default_compiler_admission: bool = False
    reference_host_baseline_id: str | None = None

    def validate(self) -> None:
        if not self.profile_id or not self.source:
            raise ValueError("fused profile id and source must not be empty")
        if self.default_compiler_admission:
            raise ValueError("measured fused profiles must require explicit admission")
        self.environment.validate()
        if self.environment.arch != "sm_80":
            raise ValueError("fused selector v0 profiles require arch=sm_80")
        self.envelope.validate()
        self.rule.validate()
        self.evidence.validate()
        if self.reference_host_baseline_id is not None:
            host_baseline(self.reference_host_baseline_id).validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_FUSED_SELECTION_PROFILE,
            "profile_id": self.profile_id,
            "source": self.source,
            "default_compiler_admission": self.default_compiler_admission,
            "reference_host_baseline_id": self.reference_host_baseline_id,
            "environment": self.environment.to_dict(),
            "envelope": self.envelope.to_dict(),
            "rule": asdict(self.rule),
            "evidence": asdict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FusedSelectionProfile":
        _check_schema(value, SCHEMA_FUSED_SELECTION_PROFILE)
        profile = cls(
            profile_id=str(value["profile_id"]),
            source=str(value["source"]),
            default_compiler_admission=bool(
                value.get("default_compiler_admission", False)
            ),
            reference_host_baseline_id=value.get("reference_host_baseline_id"),
            environment=FusedEnvironmentFingerprint.from_dict(value["environment"]),
            envelope=FusedShapeEnvelope.from_dict(value["envelope"]),
            rule=FusedKRule(
                max_direct_k=int(value["rule"]["max_direct_k"]),
                min_split_k=int(value["rule"]["min_split_k"]),
                direct_kernel=str(
                    value["rule"].get("direct_kernel", "sm80_64x128x32")
                ),
                split_kernel=str(
                    value["rule"].get(
                        "split_kernel", "sm80_64x128x32_splitk8"
                    )
                ),
            ),
            evidence=FusedEvidence(
                train_problems=int(value["evidence"]["train_problems"]),
                heldout_problems=int(value["evidence"]["heldout_problems"]),
                heldout_direction_accuracy=float(
                    value["evidence"]["heldout_direction_accuracy"]
                ),
                heldout_max_selection_regret=float(
                    value["evidence"]["heldout_max_selection_regret"]
                ),
                heldout_host_wins=int(value["evidence"]["heldout_host_wins"]),
            ),
        )
        profile.validate()
        return profile


@dataclass(frozen=True, slots=True)
class FusedCandidateDecision:
    family: str
    eligible: bool
    gemm_provider: str
    gemm_kernel: str
    communication_api: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FusedSelectionResult:
    program: Program
    target: Target
    profile: FusedSelectionProfile
    observed_environment: FusedEnvironmentFingerprint
    selected_family: str
    admitted: bool
    reason: str
    candidates: tuple[FusedCandidateDecision, ...]
    traffic: dict[str, int] | None = None

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_FUSED_SELECTION_REPORT,
            "profile": self.profile.to_dict(),
            "fallback_host_baseline": PER_SOURCE_CUBLAS.to_dict(),
            "observed_environment": self.observed_environment.to_dict(),
            "selected_family": self.selected_family,
            "admitted": self.admitted,
            "reason": self.reason,
            "problem": self.program.problem.to_dict(),
            "target": self.target.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "traffic": self.traffic,
        }


def _topology_error(
    topology: TopologySnapshot, environment: FusedEnvironmentFingerprint
) -> str | None:
    if topology.world_size != environment.world_size:
        return "topology world size does not match measured environment"
    if not topology.has_full_p2p():
        return "topology does not have full peer-to-peer connectivity"
    if any(device != environment.device for device in topology.devices):
        return "topology device identity does not match measured environment"
    if not topology.link_labels:
        return "topology link labels are missing"
    for src in range(topology.world_size):
        for dst in range(topology.world_size):
            if src != dst and topology.link_labels[src][dst] != environment.link_label:
                return "topology link label does not match measured environment"
    return None


def _candidate_decisions(
    selected: str, rejection: str | None, rule: FusedKRule
) -> tuple[FusedCandidateDecision, ...]:
    fused_reason = rejection or "not selected by measured K rule"
    return (
        FusedCandidateDecision(
            family="host",
            eligible=True,
            gemm_provider=PER_SOURCE_CUBLAS.gemm_provider,
            gemm_kernel=PER_SOURCE_CUBLAS.gemm_kernel,
            communication_api="host_nccl",
            reason="safe fallback" if selected != "host" else "selected fallback",
        ),
        FusedCandidateDecision(
            family="direct",
            eligible=selected == "direct",
            gemm_provider="cutlass",
            gemm_kernel=rule.direct_kernel,
            communication_api="nccl_device",
            reason="selected by measured K rule" if selected == "direct" else fused_reason,
        ),
        FusedCandidateDecision(
            family="splitk8",
            eligible=selected == "splitk8",
            gemm_provider="cutlass",
            gemm_kernel=rule.split_kernel,
            communication_api="nccl_device",
            reason="selected by measured K rule" if selected == "splitk8" else fused_reason,
        ),
    )


def select_fused_a2a_gemm(
    problem: A2AGemmProblem,
    topology: TopologySnapshot,
    profile: FusedSelectionProfile,
    observed_environment: FusedEnvironmentFingerprint,
    *,
    allow_experimental_fused: bool = False,
) -> FusedSelectionResult:
    """Choose host/direct/split-K without extrapolating beyond measured evidence."""

    problem.validate()
    topology.validate()
    profile.validate()
    observed_environment.validate()

    rejection: str | None = None
    if not allow_experimental_fused:
        rejection = "experimental fused admission was not explicitly enabled"
    elif observed_environment != profile.environment:
        rejection = "runtime environment fingerprint does not match fused profile"
    elif (error := _topology_error(topology, profile.environment)) is not None:
        rejection = error
    elif problem.world_size != profile.environment.world_size:
        rejection = "problem world size does not match measured environment"
    elif not profile.envelope.contains(problem):
        rejection = "problem shape, dtype, or layout is outside the measured envelope"
    elif (error := profile.evidence.admission_error()) is not None:
        rejection = error
    elif profile.reference_host_baseline_id != PER_SOURCE_CUBLAS.baseline_id:
        rejection = (
            "measured host baseline is missing or differs from the compiler's "
            "per-source cuBLAS fallback; matched baseline evidence is required"
        )

    family = None if rejection else profile.rule.family(problem.k)
    if rejection is None and family is None:
        rejection = "problem K falls in an unmeasured decision gap"

    if rejection is not None:
        selected = "host"
        program = build_sequential_program(problem)
        target = Target(
            name="host-nccl-fallback",
            topology=topology,
            arch=observed_environment.arch,
            gemm_provider=PER_SOURCE_CUBLAS.gemm_provider,
            gemm_kernel=PER_SOURCE_CUBLAS.gemm_kernel,
            communication_api="host_nccl",
        )
        admitted = False
        reason = rejection
        traffic = None
    else:
        assert family is not None
        selected = family
        kernel = (
            profile.rule.direct_kernel
            if family == "direct"
            else profile.rule.split_kernel
        )
        program = build_fused_program(problem, chunk_rows=problem.rows_per_peer)
        target = Target(
            name=f"profile-gated-{family}",
            topology=topology,
            arch=observed_environment.arch,
            gemm_provider="cutlass",
            gemm_kernel=kernel,
            communication_api="nccl_device",
        )
        target.validate_program(program)
        admitted = True
        reason = "all guards passed; selected by measured K rule"
        traffic = target.estimate_fused_traffic(program).to_dict()

    candidates = _candidate_decisions(selected, rejection, profile.rule)
    selection_metadata = {
        "profile_id": profile.profile_id,
        "profile_source": profile.source,
        "reference_host_baseline_id": profile.reference_host_baseline_id,
        "fallback_host_baseline": PER_SOURCE_CUBLAS.to_dict(),
        "selected_family": selected,
        "admitted": admitted,
        "reason": reason,
        "environment_fingerprint": observed_environment.to_dict(),
        "measured_envelope": profile.envelope.to_dict(),
        "traffic": traffic,
    }
    program = replace(
        program,
        metadata={**program.metadata, "fused_selection": selection_metadata},
    )
    return FusedSelectionResult(
        program=program,
        target=target,
        profile=profile,
        observed_environment=observed_environment,
        selected_family=selected,
        admitted=admitted,
        reason=reason,
        candidates=candidates,
        traffic=traffic,
    )


__all__ = [
    "FusedCandidateDecision",
    "FusedEnvironmentFingerprint",
    "FusedEvidence",
    "FusedKRule",
    "FusedSelectionProfile",
    "FusedSelectionResult",
    "FusedShapeEnvelope",
    "SCHEMA_FUSED_ENVIRONMENT",
    "SCHEMA_FUSED_SELECTION_PROFILE",
    "SCHEMA_FUSED_SELECTION_REPORT",
    "select_fused_a2a_gemm",
]
