from __future__ import annotations

from dataclasses import replace

from .model import (
    A2AGemmProblem,
    CalibrationProfile,
    CandidateResult,
    CompileResult,
    CompilerOptions,
    Program,
    TopologySnapshot,
)
from .schedule import build_pipelined_program, build_sequential_program
from .simulate import simulate_program
from .verify import verify_program


def _candidate_result(program: Program, profile: CalibrationProfile) -> CandidateResult:
    verify_program(program.problem, program).require_valid()
    simulation = simulate_program(program, profile)
    return CandidateResult(
        program=program,
        makespan_us=simulation.makespan_us,
        ecr=simulation.exposure.ecr,
        edmr=simulation.exposure.edmr,
        device_makespan_us=simulation.device_makespan_us,
        overhead_us=simulation.overhead_us,
    )


def _default_chunks(problem: A2AGemmProblem) -> tuple[int, ...]:
    candidates = {problem.rows_per_peer}
    candidates.update(
        value
        for value in range(16, problem.rows_per_peer, 16)
        if problem.rows_per_peer % value == 0
    )
    return tuple(sorted(candidates))


def _selection_key(item: CandidateResult) -> tuple[float, int, int, str, int]:
    # Cost ties are resolved deterministically in favor of a smaller launch
    # surface and then more user-bounded staging buffers.
    return (
        item.makespan_us,
        len(item.program.ops),
        -item.program.buffer_count,
        item.program.strategy,
        -item.program.chunk_rows,
    )


def compile_a2a_gemm(
    problem: A2AGemmProblem,
    profile: CalibrationProfile,
    topology: TopologySnapshot,
    options: CompilerOptions | None = None,
) -> CompileResult:
    problem.validate()
    profile.validate()
    topology.validate()
    if topology.world_size != problem.world_size:
        raise ValueError("topology world_size does not match problem")
    options = options or CompilerOptions()
    min_gain_ratio = (
        profile.selection_min_gain_ratio
        if options.min_gain_ratio is None
        else options.min_gain_ratio
    )
    if min_gain_ratio < 0:
        raise ValueError("min_gain_ratio must be non-negative")
    if any(value <= 0 for value in options.buffer_counts):
        raise ValueError("buffer counts must be positive")
    if any(value <= 0 or problem.rows_per_peer % value for value in options.chunk_rows):
        raise ValueError("every chunk_rows candidate must be a positive divisor of rows_per_peer")

    sequential = _candidate_result(build_sequential_program(problem), profile)
    candidates = [sequential]
    fallback_reason: str | None = None

    if not topology.has_full_p2p():
        fallback_reason = "pipeline disabled: topology lacks full peer-to-peer connectivity"
    else:
        chunks = options.chunk_rows or _default_chunks(problem)
        for chunk_rows in chunks:
            for buffer_count in options.buffer_counts:
                program = build_pipelined_program(
                    problem,
                    chunk_rows=chunk_rows,
                    buffer_count=buffer_count,
                )
                candidates.append(_candidate_result(program, profile))

    best = min(candidates, key=_selection_key)
    if best.program.strategy != "sequential":
        gain_ratio = (sequential.makespan_us - best.makespan_us) / sequential.makespan_us
        if gain_ratio <= min_gain_ratio:
            best = sequential
            fallback_reason = (
                f"pipeline fallback: predicted gain {gain_ratio:.6f} does not exceed "
                f"minimum {min_gain_ratio:.6f}"
            )
    elif fallback_reason is None:
        fallback_reason = "pipeline fallback: sequential schedule has the lowest predicted makespan"

    best = replace(
        best,
        program=replace(
            best.program,
            metadata={
                **best.program.metadata,
                "predicted_makespan_us": best.makespan_us,
                "predicted_ecr": best.ecr,
                "predicted_edmr": best.edmr,
                "predicted_device_makespan_us": best.device_makespan_us,
                "predicted_runtime_overhead_us": best.overhead_us,
                "selection_min_gain_ratio": min_gain_ratio,
                "fallback_reason": fallback_reason,
            },
        ),
    )
    return CompileResult(chosen=best, candidates=tuple(candidates), fallback_reason=fallback_reason)


__all__ = ["compile_a2a_gemm"]
