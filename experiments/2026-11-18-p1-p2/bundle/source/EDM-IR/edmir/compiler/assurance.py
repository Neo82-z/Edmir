from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Mapping

from .model import A2AGemmProblem, Program
from .verify import verify_program


SCHEMA_EVIDENCE_RECORD = "edmir.evidence_record.v1"

Matrix = tuple[tuple[float, ...], ...]


def _matrix_shape(matrix: Matrix, *, name: str) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return len(matrix), columns


def _tile_coordinates(tile: str) -> tuple[int, int, int]:
    try:
        src, dst, chunk = tile.split(".")
        return int(src[1:]), int(dst[1:]), int(chunk[1:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid tile identity {tile!r}") from exc


def _tile_ids(problem: A2AGemmProblem, chunk_rows: int) -> tuple[str, ...]:
    if chunk_rows <= 0 or problem.rows_per_peer % chunk_rows:
        raise ValueError("chunk_rows must be a positive divisor of rows_per_peer")
    chunks = problem.rows_per_peer // chunk_rows
    return tuple(
        f"s{src}.d{dst}.c{chunk}"
        for src in range(problem.world_size)
        for dst in range(problem.world_size)
        for chunk in range(chunks)
    )


def reference_gemm(
    left: Matrix,
    right: Matrix,
    *,
    split_k_slices: int = 1,
    tile_n: int | None = None,
    n_tiles_per_group: int = 1,
) -> Matrix:
    """Small executable GEMM semantics used for compiler assurance tests.

    ``split_k_slices`` changes the reduction order without changing the K
    coverage. ``n_tiles_per_group`` changes only the order in which adjacent N
    regions are visited. Neither option is a performance implementation.
    """

    m, k = _matrix_shape(left, name="left")
    right_k, n = _matrix_shape(right, name="right")
    if k != right_k:
        raise ValueError("GEMM inner dimensions do not match")
    if split_k_slices <= 0 or split_k_slices > k:
        raise ValueError("split_k_slices must be in [1, K]")
    if n_tiles_per_group <= 0:
        raise ValueError("n_tiles_per_group must be positive")
    if tile_n is None:
        tile_n = n
    if tile_n <= 0:
        raise ValueError("tile_n must be positive")

    output = [[0.0 for _ in range(n)] for _ in range(m)]
    group_width = tile_n * n_tiles_per_group
    for n_start in range(0, n, group_width):
        n_end = min(n, n_start + group_width)
        partials = [
            [[0.0 for _ in range(n_end - n_start)] for _ in range(m)]
            for _ in range(split_k_slices)
        ]
        for split in range(split_k_slices):
            k_start = k * split // split_k_slices
            k_end = k * (split + 1) // split_k_slices
            for row in range(m):
                for column in range(n_start, n_end):
                    accumulator = 0.0
                    for inner in range(k_start, k_end):
                        accumulator += left[row][inner] * right[inner][column]
                    partials[split][row][column - n_start] = accumulator
        for row in range(m):
            for column in range(n_start, n_end):
                output[row][column] = sum(
                    partial[row][column - n_start] for partial in partials
                )
    return tuple(tuple(row) for row in output)


@dataclass(frozen=True, slots=True)
class ReferenceFixture:
    source_tiles: Mapping[str, Matrix]
    weights: Mapping[int, Matrix]
    seed: int

    def validate(self, problem: A2AGemmProblem, chunk_rows: int) -> None:
        expected_tiles = set(_tile_ids(problem, chunk_rows))
        if set(self.source_tiles) != expected_tiles:
            raise ValueError("reference source tile coverage is incomplete or contains extras")
        if set(self.weights) != set(range(problem.world_size)):
            raise ValueError("reference weight coverage must match every destination rank")
        for tile, matrix in self.source_tiles.items():
            rows, columns = _matrix_shape(matrix, name=f"source tile {tile}")
            if rows != chunk_rows or columns != problem.k:
                raise ValueError(f"source tile {tile} has the wrong shape")
        for rank, matrix in self.weights.items():
            rows, columns = _matrix_shape(matrix, name=f"weight rank {rank}")
            if rows != problem.k or columns != problem.n:
                raise ValueError(f"weight rank {rank} has the wrong shape")


def make_reference_fixture(
    problem: A2AGemmProblem,
    chunk_rows: int,
    *,
    seed: int = 0,
) -> ReferenceFixture:
    """Create deterministic non-integer inputs without requiring NumPy."""

    problem.validate()
    rng = random.Random(seed)

    def value() -> float:
        # Avoid integer-only fixtures; those can hide reduction-order mistakes.
        return rng.uniform(-1.0, 1.0)

    source_tiles = {
        tile: tuple(
            tuple(value() for _ in range(problem.k))
            for _ in range(chunk_rows)
        )
        for tile in _tile_ids(problem, chunk_rows)
    }
    weights = {
        rank: tuple(
            tuple(value() for _ in range(problem.n))
            for _ in range(problem.k)
        )
        for rank in range(problem.world_size)
    }
    fixture = ReferenceFixture(source_tiles=source_tiles, weights=weights, seed=seed)
    fixture.validate(problem, chunk_rows)
    return fixture


def expected_reference_outputs(
    problem: A2AGemmProblem,
    chunk_rows: int,
    fixture: ReferenceFixture,
) -> dict[str, Matrix]:
    fixture.validate(problem, chunk_rows)
    return {
        tile: reference_gemm(fixture.source_tiles[tile], fixture.weights[_tile_coordinates(tile)[1]])
        for tile in _tile_ids(problem, chunk_rows)
    }


def execute_reference_program(
    program: Program,
    fixture: ReferenceFixture,
) -> dict[str, Matrix]:
    """Execute a legal Program as an observable, sequential reference machine."""

    report = verify_program(program.problem, program)
    report.require_valid()
    fixture.validate(program.problem, program.chunk_rows)
    buffer_map = program.buffer_map()
    op_map = program.op_map()
    values: dict[str, Matrix | bool] = {}
    for tile, matrix in fixture.source_tiles.items():
        values[f"input.{tile}"] = matrix
    for rank, matrix in fixture.weights.items():
        values[f"weight.r{rank}"] = matrix

    for op_id in report.dependence.topological_order:
        op = op_map[op_id]
        if op.kind == "transfer":
            values[op.writes[0]] = values[op.reads[0]]
        elif op.kind == "barrier":
            if any(name not in values for name in op.reads):
                raise ValueError(f"barrier {op.id} observed an unpublished input")
            values[op.writes[0]] = True
        elif op.kind in {"gemm", "fused_transfer_gemm"}:
            matrices = [
                (buffer_map[name].role, values[name])
                for name in op.reads
                if buffer_map[name].role != "token"
            ]
            left = next(
                value for role, value in matrices if role in {"input", "staging"}
            )
            right = next(value for role, value in matrices if role == "weight")
            if not isinstance(left, tuple) or not isinstance(right, tuple):
                raise ValueError(f"op {op.id} did not receive matrix inputs")
            values[op.writes[0]] = reference_gemm(left, right)
        else:  # The verifier should make this unreachable.
            raise ValueError(f"unsupported reference op kind {op.kind!r}")

    outputs: dict[str, Matrix] = {}
    for buffer in program.buffers:
        if buffer.role != "output":
            continue
        value = values.get(buffer.name)
        if not isinstance(value, tuple):
            raise ValueError(f"output buffer {buffer.name} was not produced")
        outputs[buffer.name.removeprefix("output.")] = value
    return outputs


@dataclass(frozen=True, slots=True)
class SemanticValidationReport:
    valid: bool
    checked_elements: int
    mismatched_elements: int
    max_abs_error: float
    max_rel_error: float
    errors: tuple[str, ...] = ()

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("semantic validation failed: " + "; ".join(self.errors))


def compare_reference_outputs(
    expected: Mapping[str, Matrix],
    observed: Mapping[str, Matrix],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> SemanticValidationReport:
    if atol < 0 or rtol < 0:
        raise ValueError("numerical tolerances must be non-negative")
    errors: list[str] = []
    if set(expected) != set(observed):
        errors.append("output tile coverage differs from the reference")
    checked = 0
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    for tile in sorted(set(expected) & set(observed)):
        try:
            expected_shape = _matrix_shape(expected[tile], name=f"expected {tile}")
            observed_shape = _matrix_shape(observed[tile], name=f"observed {tile}")
        except ValueError as error:
            errors.append(str(error))
            continue
        if expected_shape != observed_shape:
            errors.append(f"output tile {tile} has shape {observed_shape}, expected {expected_shape}")
            continue
        for expected_row, observed_row in zip(expected[tile], observed[tile]):
            for expected_value, observed_value in zip(expected_row, observed_row):
                checked += 1
                if not math.isfinite(expected_value) or not math.isfinite(observed_value):
                    mismatches += 1
                    continue
                absolute = abs(observed_value - expected_value)
                relative = absolute / max(abs(expected_value), 1e-30)
                max_abs = max(max_abs, absolute)
                max_rel = max(max_rel, relative)
                if absolute > atol + rtol * abs(expected_value):
                    mismatches += 1
    if mismatches:
        errors.append(f"{mismatches} numerical elements exceed tolerance")
    return SemanticValidationReport(
        valid=not errors,
        checked_elements=checked,
        mismatched_elements=mismatches,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        errors=tuple(errors),
    )


def validate_program_semantics(
    program: Program,
    fixture: ReferenceFixture,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> SemanticValidationReport:
    verification = verify_program(program.problem, program)
    if not verification.valid:
        return SemanticValidationReport(
            valid=False,
            checked_elements=0,
            mismatched_elements=0,
            max_abs_error=0.0,
            max_rel_error=0.0,
            errors=verification.errors,
        )
    expected = expected_reference_outputs(program.problem, program.chunk_rows, fixture)
    observed = execute_reference_program(program, fixture)
    return compare_reference_outputs(expected, observed, atol=atol, rtol=rtol)


def _require_digest(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class MeasurementInterval:
    samples: int
    median_us: float
    lower_us: float
    upper_us: float

    def validate(self) -> None:
        if self.samples <= 0:
            raise ValueError("measurement interval requires at least one sample")
        if not 0.0 <= self.lower_us <= self.median_us <= self.upper_us:
            raise ValueError("measurement interval must satisfy 0 <= lower <= median <= upper")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    problem: A2AGemmProblem
    family: str
    environment_hash: str
    code_commit: str
    protocol_hash: str
    source_hash: str
    numerical_check_passed: bool
    interval: MeasurementInterval

    def validate(self) -> None:
        self.problem.validate()
        if not self.family:
            raise ValueError("evidence family must not be empty")
        _require_digest("environment_hash", self.environment_hash)
        _require_digest("protocol_hash", self.protocol_hash)
        _require_digest("source_hash", self.source_hash)
        if re.fullmatch(r"[0-9a-f]{7,40}", self.code_commit) is None:
            raise ValueError("code_commit must be a 7-40 character lowercase Git hash")
        self.interval.validate()

    @property
    def binding_key(self) -> tuple[object, ...]:
        return (
            self.problem,
            self.environment_hash,
            self.code_commit,
            self.protocol_hash,
            self.source_hash,
        )

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema": SCHEMA_EVIDENCE_RECORD,
            "problem": self.problem.to_dict(),
            "family": self.family,
            "environment_hash": self.environment_hash,
            "code_commit": self.code_commit,
            "protocol_hash": self.protocol_hash,
            "source_hash": self.source_hash,
            "numerical_check_passed": self.numerical_check_passed,
            "interval": asdict(self.interval),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceRecord":
        schema = value.get("schema")
        if schema not in (None, SCHEMA_EVIDENCE_RECORD):
            raise ValueError(
                f"unsupported schema {schema!r}; expected {SCHEMA_EVIDENCE_RECORD!r}"
            )
        record = cls(
            problem=A2AGemmProblem.from_dict(value["problem"]),
            family=str(value["family"]),
            environment_hash=str(value["environment_hash"]),
            code_commit=str(value["code_commit"]),
            protocol_hash=str(value["protocol_hash"]),
            source_hash=str(value["source_hash"]),
            numerical_check_passed=bool(value["numerical_check_passed"]),
            interval=MeasurementInterval(
                samples=int(value["interval"]["samples"]),
                median_us=float(value["interval"]["median_us"]),
                lower_us=float(value["interval"]["lower_us"]),
                upper_us=float(value["interval"]["upper_us"]),
            ),
        )
        record.validate()
        return record


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    minimum_samples: int = 30
    minimum_gain_ratio: float = 0.0

    def validate(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if not 0.0 <= self.minimum_gain_ratio < 1.0:
            raise ValueError("minimum_gain_ratio must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    selected_family: str
    reason: str
    conservative_gain_ratio: float | None = None


def admit_evidence(
    host: EvidenceRecord,
    candidate: EvidenceRecord,
    policy: AdmissionPolicy = AdmissionPolicy(),
) -> AdmissionDecision:
    """Admit only target-bound evidence whose intervals clear a safety margin."""

    policy.validate()
    host.validate()
    candidate.validate()
    if host.family != "host":
        raise ValueError("host evidence must use family='host'")
    if candidate.family == "host":
        raise ValueError("candidate evidence must describe a non-host family")
    if host.binding_key != candidate.binding_key:
        return AdmissionDecision(False, "host", "evidence binding mismatch")
    if not host.numerical_check_passed or not candidate.numerical_check_passed:
        return AdmissionDecision(False, "host", "numerical validation is missing or failed")
    if min(host.interval.samples, candidate.interval.samples) < policy.minimum_samples:
        return AdmissionDecision(False, "host", "insufficient measurement samples")
    if host.interval.lower_us <= 0.0:
        return AdmissionDecision(False, "host", "host interval has no positive lower bound")
    conservative_gain = 1.0 - candidate.interval.upper_us / host.interval.lower_us
    # The contract uses a strict separation: touching confidence bounds are
    # still ambiguous evidence and must deterministically fall back to host.
    if conservative_gain <= policy.minimum_gain_ratio:
        return AdmissionDecision(
            False,
            "host",
            "candidate confidence interval does not clear the admission margin",
            conservative_gain,
        )
    return AdmissionDecision(
        True,
        candidate.family,
        "target-bound evidence clears correctness and confidence gates",
        conservative_gain,
    )


__all__ = [
    "AdmissionDecision",
    "AdmissionPolicy",
    "EvidenceRecord",
    "Matrix",
    "MeasurementInterval",
    "ReferenceFixture",
    "SCHEMA_EVIDENCE_RECORD",
    "SemanticValidationReport",
    "admit_evidence",
    "compare_reference_outputs",
    "execute_reference_program",
    "expected_reference_outputs",
    "make_reference_fixture",
    "reference_gemm",
    "validate_program_semantics",
]
