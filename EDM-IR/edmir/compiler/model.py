from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


SCHEMA_PROBLEM = "edmir.problem.v1"
SCHEMA_PROFILE = "edmir.profile.v1"
SCHEMA_TOPOLOGY = "edmir.topology.v1"
SCHEMA_PROGRAM = "edmir.program.v1"
SCHEMA_LEGACY_PLAN = "edmir.plan.v1"

DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}

BUFFER_ROLES = frozenset({"input", "weight", "staging", "output", "token"})
OP_KINDS = frozenset({"transfer", "gemm", "barrier"})
QUEUE_KINDS = frozenset({"comm", "compute", "control"})


def _check_schema(value: dict[str, Any], expected: str) -> None:
    schema = value.get("schema")
    if schema not in (None, expected):
        raise ValueError(f"unsupported schema {schema!r}; expected {expected!r}")


@dataclass(frozen=True, slots=True)
class A2AGemmProblem:
    """Static equal-sized row All2All followed by destination-local GEMM."""

    world_size: int
    rows_per_peer: int
    k: int
    n: int
    dtype: str = "float16"
    layout: str = "row_major"
    name: str = "a2a_gemm"

    def validate(self) -> None:
        if self.world_size < 2:
            raise ValueError("world_size must be >= 2")
        if min(self.rows_per_peer, self.k, self.n) <= 0:
            raise ValueError("rows_per_peer, k, and n must be positive")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.layout != "row_major":
            raise ValueError("v0.1 supports only row_major layout")

    @property
    def local_rows(self) -> int:
        return self.world_size * self.rows_per_peer

    @property
    def bytes_per_row(self) -> int:
        return self.k * DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_PROBLEM, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "A2AGemmProblem":
        _check_schema(value, SCHEMA_PROBLEM)
        problem = cls(**{key: item for key, item in value.items() if key != "schema"})
        problem.validate()
        return problem


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    world_size: int
    devices: tuple[str, ...]
    p2p: tuple[tuple[bool, ...], ...]
    link_labels: tuple[tuple[str, ...], ...] = ()
    raw_topology: str = ""

    def validate(self) -> None:
        if self.world_size < 1:
            raise ValueError("topology world_size must be positive")
        if len(self.devices) != self.world_size:
            raise ValueError("topology device count does not match world_size")
        if len(self.p2p) != self.world_size or any(len(row) != self.world_size for row in self.p2p):
            raise ValueError("topology p2p matrix must be world_size x world_size")
        if self.link_labels and (
            len(self.link_labels) != self.world_size
            or any(len(row) != self.world_size for row in self.link_labels)
        ):
            raise ValueError("topology link label matrix must be empty or world_size x world_size")

    def has_full_p2p(self) -> bool:
        return all(
            src == dst or self.p2p[src][dst]
            for src in range(self.world_size)
            for dst in range(self.world_size)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_TOPOLOGY,
            "world_size": self.world_size,
            "devices": list(self.devices),
            "p2p": [list(row) for row in self.p2p],
            "link_labels": [list(row) for row in self.link_labels],
            "raw_topology": self.raw_topology,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TopologySnapshot":
        _check_schema(value, SCHEMA_TOPOLOGY)
        topology = cls(
            world_size=int(value["world_size"]),
            devices=tuple(str(item) for item in value["devices"]),
            p2p=tuple(tuple(bool(item) for item in row) for row in value["p2p"]),
            link_labels=tuple(tuple(str(item) for item in row) for row in value.get("link_labels", ())),
            raw_topology=str(value.get("raw_topology", "")),
        )
        topology.validate()
        return topology


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Replaceable analytic cost interface populated by measurements."""

    comm_latency_us: float
    comm_bandwidth_gbps: float
    gemm_launch_us: float
    gemm_tflops: float
    sync_us: float = 0.0
    local_copy_us: float = 0.0
    source: str = "synthetic"

    def validate(self) -> None:
        if self.comm_latency_us < 0 or self.gemm_launch_us < 0 or self.sync_us < 0:
            raise ValueError("latencies must be non-negative")
        if self.comm_bandwidth_gbps <= 0 or self.gemm_tflops <= 0:
            raise ValueError("bandwidth and throughput must be positive")

    def comm_us(self, size_bytes: int, *, local: bool = False) -> float:
        if local:
            return self.local_copy_us
        return self.comm_latency_us + size_bytes / (self.comm_bandwidth_gbps * 1_000.0)

    def gemm_us(self, m: int, n: int, k: int) -> float:
        return self.gemm_launch_us + (2.0 * m * n * k) / (self.gemm_tflops * 1_000_000.0)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_PROFILE, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationProfile":
        _check_schema(value, SCHEMA_PROFILE)
        profile = cls(**{key: item for key, item in value.items() if key != "schema"})
        profile.validate()
        return profile


@dataclass(frozen=True, slots=True)
class Buffer:
    """Physical storage whose read/write hazards induce dependencies."""

    name: str
    rank: int
    role: str
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class Op:
    """One executable action; no dependency edges are stored in the IR."""

    id: str
    kind: str
    rank: int
    queue: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    tile: str | None = None
    rows: int = 0
    peer_rank: int | None = None

    def resources(self) -> tuple[str, ...]:
        if self.kind == "transfer" and self.peer_rank is not None:
            ranks = tuple(dict.fromkeys((self.peer_rank, self.rank)))
            return tuple(f"rank{rank}.comm" for rank in ranks)
        return (f"rank{self.rank}.{self.queue}",)


@dataclass(frozen=True, slots=True)
class Program:
    """Ordered Tile/communication program; dependencies are derived on demand."""

    problem: A2AGemmProblem
    strategy: str
    chunk_rows: int
    buffer_count: int
    buffers: tuple[Buffer, ...]
    ops: tuple[Op, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def buffer_map(self) -> dict[str, Buffer]:
        return {buffer.name: buffer for buffer in self.buffers}

    def op_map(self) -> dict[str, Op]:
        return {op.id: op for op in self.ops}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_PROGRAM,
            "problem": self.problem.to_dict(),
            "strategy": self.strategy,
            "chunk_rows": self.chunk_rows,
            "buffer_count": self.buffer_count,
            "buffers": [asdict(buffer) for buffer in self.buffers],
            "ops": [asdict(op) for op in self.ops],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Program":
        schema = value.get("schema")
        if schema == SCHEMA_LEGACY_PLAN or (schema is None and "tasks" in value):
            return cls._from_legacy_plan(value)
        _check_schema(value, SCHEMA_PROGRAM)
        return cls(
            problem=A2AGemmProblem.from_dict(value["problem"]),
            strategy=str(value["strategy"]),
            chunk_rows=int(value["chunk_rows"]),
            buffer_count=int(value["buffer_count"]),
            buffers=tuple(Buffer(**buffer) for buffer in value.get("buffers", ())),
            ops=tuple(
                Op(
                    **{
                        **op,
                        "reads": tuple(op.get("reads", ())),
                        "writes": tuple(op.get("writes", ())),
                    }
                )
                for op in value.get("ops", ())
            ),
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def _from_legacy_plan(cls, value: dict[str, Any]) -> "Program":
        """Rebuild old compiler-generated plan.v1 files without preserving edges."""

        from .schedule import build_pipelined_program, build_sequential_program

        problem = A2AGemmProblem.from_dict(value["problem"])
        strategy = str(value["strategy"])
        if strategy == "sequential":
            program = build_sequential_program(problem)
        elif strategy == "pipelined":
            program = build_pipelined_program(
                problem,
                chunk_rows=int(value["chunk_rows"]),
                buffer_count=int(value["buffer_count"]),
            )
        else:
            raise ValueError(f"unsupported legacy strategy: {strategy}")
        return replace(
            program,
            metadata={**program.metadata, **dict(value.get("metadata", {})), "loaded_from": SCHEMA_LEGACY_PLAN},
        )


@dataclass(frozen=True, slots=True)
class CompilerOptions:
    chunk_rows: tuple[int, ...] = ()
    buffer_counts: tuple[int, ...] = (2,)
    min_gain_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateResult:
    program: Program
    makespan_us: float
    ecr: float
    edmr: float

    @property
    def plan(self) -> Program:
        """Compatibility name used by the first prototype."""

        return self.program


@dataclass(frozen=True, slots=True)
class CompileResult:
    chosen: CandidateResult
    candidates: tuple[CandidateResult, ...]
    fallback_reason: str | None = None


# A narrow source compatibility alias. New code should use Program.
SchedulePlan = Program


__all__ = [
    "A2AGemmProblem",
    "Buffer",
    "CalibrationProfile",
    "CandidateResult",
    "CompileResult",
    "CompilerOptions",
    "Op",
    "Program",
    "SchedulePlan",
    "TopologySnapshot",
]
