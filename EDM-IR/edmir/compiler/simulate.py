from __future__ import annotations

from dataclasses import dataclass

from edmir.analysis import ExposureResult, analyze_exposure
from edmir.builder import EDMBuilder
from edmir.ops import (
    COMM_P2P,
    COMPUTE_GEMM,
    DATA_DEPENDENCY,
    EVENT_WAIT,
    PHASE_BOUNDARY,
    STREAM_ORDER,
)
from edmir.uop import MovementDesc

from .model import CalibrationProfile, Op, Program
from .verify import verify_program


@dataclass(frozen=True, slots=True)
class OpTiming:
    op_id: str
    start_us: float
    end_us: float

    @property
    def task_id(self) -> str:
        return self.op_id


@dataclass(frozen=True, slots=True)
class SimulationResult:
    strategy: str
    makespan_us: float
    timings: tuple[OpTiming, ...]
    exposure: ExposureResult


def _duration_us(op: Op, program: Program, profile: CalibrationProfile) -> float:
    if op.kind == "transfer":
        return profile.comm_us(
            op.rows * program.problem.bytes_per_row,
            local=op.peer_rank == op.rank,
        )
    if op.kind == "gemm":
        return profile.gemm_us(op.rows, program.problem.n, program.problem.k)
    if op.kind == "barrier":
        return profile.sync_us
    raise ValueError(f"unknown op kind: {op.kind}")


def simulate_program(program: Program, profile: CalibrationProfile) -> SimulationResult:
    report = verify_program(program.problem, program)
    report.require_valid()
    profile.validate()
    ops = program.op_map()
    predecessors = report.dependence.predecessors()

    end_times: dict[str, float] = {}
    resource_end: dict[str, float] = {}
    timings: dict[str, OpTiming] = {}
    for op_id in report.dependence.topological_order:
        op = ops[op_id]
        inputs_ready = max((end_times[src] for src in predecessors[op_id]), default=0.0)
        resources_ready = max((resource_end.get(name, 0.0) for name in op.resources()), default=0.0)
        start = max(inputs_ready, resources_ready)
        end = start + _duration_us(op, program, profile)
        end_times[op_id] = end
        for resource in op.resources():
            resource_end[resource] = end
        timings[op_id] = OpTiming(op_id, start, end)

    makespan = max(end_times.values(), default=0.0)
    builder = EDMBuilder(default_phase="a2a_gemm_simulation")
    begin = builder.phase_begin("a2a_gemm_simulation", 0.0)
    uops = {}
    for op in program.ops:
        timing = timings[op.id]
        if op.kind == "transfer":
            uop = builder.comm(
                op.id,
                timing.start_us,
                timing.end_us,
                op=COMM_P2P,
                device=f"rank:{op.rank}",
                stream="+".join(op.resources()),
                movement=MovementDesc(
                    kind="all_to_all_tile",
                    size_bytes=op.rows * program.problem.bytes_per_row,
                    src_ranks=() if op.peer_rank is None else (op.peer_rank,),
                    dst_ranks=(op.rank,),
                ),
            )
        elif op.kind == "gemm":
            uop = builder.compute(
                op.id,
                timing.start_us,
                timing.end_us,
                op=COMPUTE_GEMM,
                device=f"rank:{op.rank}",
                stream=op.resources()[0],
            )
        else:
            uop = builder.uop(
                EVENT_WAIT,
                op.id,
                timing.start_us,
                timing.end_us,
                device=f"rank:{op.rank}",
                stream=op.resources()[0],
            )
        uops[op.id] = uop
    end = builder.phase_end("a2a_gemm_simulation", makespan)

    dependency_kinds: dict[tuple[str, str], set[str]] = {}
    for dependency in report.dependence.dependencies:
        dependency_kinds.setdefault((dependency.src, dependency.dst), set()).add(dependency.kind)
    for (src, dst), kinds in dependency_kinds.items():
        edge_kind = STREAM_ORDER if kinds == {"queue"} else DATA_DEPENDENCY
        builder.edge(uops[src], uops[dst], edge_kind, derived_from=sorted(kinds))

    successors = {src for src, _dst in dependency_kinds}
    predecessor_ids = {dst for _src, dst in dependency_kinds}
    for op in program.ops:
        if op.id not in predecessor_ids:
            builder.edge(begin, uops[op.id], PHASE_BOUNDARY)
        if op.id not in successors:
            builder.edge(uops[op.id], end, PHASE_BOUNDARY)

    return SimulationResult(
        strategy=program.strategy,
        makespan_us=makespan,
        timings=tuple(timings[op.id] for op in program.ops),
        exposure=analyze_exposure(builder.build()),
    )


# Compatibility aliases retained for the first prototype.
simulate_schedule = simulate_program
TaskTiming = OpTiming


__all__ = [
    "OpTiming",
    "SimulationResult",
    "TaskTiming",
    "simulate_program",
    "simulate_schedule",
]
