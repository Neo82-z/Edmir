from __future__ import annotations

from dataclasses import dataclass

from .dependence import DependenceResult, derive_dependencies
from .model import A2AGemmProblem, Program
from .remote_publication import validate_remote_publication


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    errors: tuple[str, ...]
    dependence: DependenceResult

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("invalid program: " + "; ".join(self.errors))


def _expected_tiles(problem: A2AGemmProblem, chunk_rows: int) -> set[str]:
    chunks = problem.rows_per_peer // chunk_rows
    return {
        f"s{src}.d{dst}.c{chunk}"
        for src in range(problem.world_size)
        for dst in range(problem.world_size)
        for chunk in range(chunks)
    }


def _tile_coordinates(tile: str) -> tuple[int, int, int]:
    try:
        src, dst, chunk = tile.split(".")
        return int(src[1:]), int(dst[1:]), int(chunk[1:])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid tile identity {tile!r}") from exc


def verify_program(problem: A2AGemmProblem, program: Program) -> VerificationReport:
    errors: list[str] = []
    try:
        problem.validate()
    except ValueError as exc:
        errors.append(str(exc))
    if program.problem != problem:
        errors.append("program problem does not match verifier problem")
    if program.strategy not in {"sequential", "pipelined", "fused"}:
        errors.append(f"unknown strategy {program.strategy!r}")
    if program.chunk_rows <= 0 or problem.rows_per_peer % program.chunk_rows:
        errors.append("chunk_rows must divide rows_per_peer")
    if program.buffer_count < 1:
        errors.append("buffer_count must be positive")

    dependence = derive_dependencies(program)
    errors.extend(dependence.errors)
    publication = validate_remote_publication(program)
    errors.extend(publication.errors)
    buffer_map = program.buffer_map()
    expected = _expected_tiles(problem, program.chunk_rows) if program.chunk_rows > 0 else set()
    transfers: dict[str, list] = {}
    gemms: dict[str, list] = {}
    fused_ops: dict[str, list] = {}
    barriers = []
    for op in program.ops:
        if op.kind == "transfer":
            if op.tile is None:
                errors.append(f"transfer {op.id} has no tile identity")
            else:
                transfers.setdefault(op.tile, []).append(op)
        elif op.kind == "gemm":
            if op.tile is None:
                errors.append(f"GEMM {op.id} has no tile identity")
            else:
                gemms.setdefault(op.tile, []).append(op)
        elif op.kind == "barrier":
            barriers.append(op)
        elif op.kind == "fused_transfer_gemm":
            if op.tile is None:
                errors.append(f"fused op {op.id} has no tile identity")
            else:
                fused_ops.setdefault(op.tile, []).append(op)

    if program.strategy == "fused":
        if transfers or gemms or barriers:
            errors.append("fused program must contain only fused_transfer_gemm ops")
        if set(fused_ops) != expected:
            errors.append("fused tile coverage is incomplete or contains extras")
    else:
        if fused_ops:
            errors.append("unfused program must not contain fused_transfer_gemm ops")
        if set(transfers) != expected:
            errors.append("transfer tile coverage is incomplete or contains extras")
        if set(gemms) != expected:
            errors.append("GEMM tile coverage is incomplete or contains extras")

    chunks = problem.rows_per_peer // program.chunk_rows if program.chunk_rows > 0 else 0
    for tile in expected:
        if program.strategy == "fused":
            tile_fused = fused_ops.get(tile, ())
            if len(tile_fused) != 1:
                errors.append(f"tile {tile} must have exactly one fused op")
                continue
            fused = tile_fused[0]
            try:
                src, dst, chunk = _tile_coordinates(tile)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if chunk < 0 or chunk >= chunks:
                errors.append(f"tile {tile} has invalid chunk index")
            if fused.peer_rank != src or fused.rank != dst:
                errors.append(f"fused op {fused.id} metadata does not match {tile}")
            if fused.rows != program.chunk_rows:
                errors.append(f"tile {tile} row count does not match chunk_rows")
            inputs = [buffer_map.get(name) for name in fused.reads]
            source_inputs = [
                buffer for buffer in inputs
                if buffer is not None and buffer.role == "input"
            ]
            weights = [
                buffer for buffer in inputs
                if buffer is not None and buffer.role == "weight"
            ]
            if len(fused.reads) != 2 or len(source_inputs) != 1 or source_inputs[0].rank != src:
                errors.append(f"fused op {fused.id} reads an invalid source buffer")
            if len(weights) != 1 or weights[0].rank != dst:
                errors.append(f"fused op {fused.id} reads an invalid weight buffer")
            outputs = [buffer_map.get(name) for name in fused.writes]
            if len(outputs) != 1 or outputs[0] is None or (
                outputs[0].role != "output" or outputs[0].rank != dst
            ):
                errors.append(f"fused op {fused.id} writes an invalid output buffer")
            continue

        tile_transfers = transfers.get(tile, ())
        tile_gemms = gemms.get(tile, ())
        if len(tile_transfers) != 1 or len(tile_gemms) != 1:
            errors.append(f"tile {tile} must have exactly one transfer and one GEMM")
            continue
        transfer = tile_transfers[0]
        gemm = tile_gemms[0]
        try:
            src, dst, chunk = _tile_coordinates(tile)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if chunk < 0 or chunk >= chunks:
            errors.append(f"tile {tile} has invalid chunk index")
        if transfer.peer_rank != src or transfer.rank != dst:
            errors.append(f"transfer {transfer.id} metadata does not match {tile}")
        if gemm.peer_rank != src or gemm.rank != dst:
            errors.append(f"GEMM {gemm.id} metadata does not match {tile}")
        if transfer.rows != program.chunk_rows or gemm.rows != program.chunk_rows:
            errors.append(f"tile {tile} row count does not match chunk_rows")
        shared_stage = set(transfer.writes) & set(gemm.reads)
        if len(shared_stage) != 1:
            errors.append(f"early read: GEMM {gemm.id} does not consume transfer output")
        else:
            stage = buffer_map.get(next(iter(shared_stage)))
            if stage is None or stage.role != "staging" or stage.rank != dst:
                errors.append(f"tile {tile} uses an invalid staging buffer")
        if len(transfer.reads) != 1:
            errors.append(f"transfer {transfer.id} must read one source tile")
        elif (source := buffer_map.get(transfer.reads[0])) is None or (
            source.role != "input" or source.rank != src
        ):
            errors.append(f"transfer {transfer.id} reads an invalid source buffer")
        if len(transfer.writes) != 1:
            errors.append(f"transfer {transfer.id} must write one staging buffer")
        weights = [buffer_map.get(name) for name in gemm.reads if buffer_map.get(name) is not None]
        weights = [buffer for buffer in weights if buffer.role == "weight"]
        if len(weights) != 1 or weights[0].rank != dst:
            errors.append(f"GEMM {gemm.id} reads an invalid weight buffer")
        outputs = [buffer_map.get(name) for name in gemm.writes]
        if len(outputs) != 1 or outputs[0] is None or (
            outputs[0].role != "output" or outputs[0].rank != dst
        ):
            errors.append(f"GEMM {gemm.id} writes an invalid output buffer")

    staging_per_rank: dict[int, int] = {rank: 0 for rank in range(problem.world_size)}
    for buffer in program.buffers:
        if buffer.role == "staging" and buffer.rank in staging_per_rank:
            staging_per_rank[buffer.rank] += 1
    for rank, count in staging_per_rank.items():
        if count > program.buffer_count:
            errors.append(
                f"rank {rank} declares {count} staging buffers but buffer_count is {program.buffer_count}"
            )

    if program.strategy == "fused" and any(
        buffer.role == "staging" for buffer in program.buffers
    ):
        errors.append("fused program must not declare staging buffers")

    if program.strategy == "sequential":
        if len(barriers) != problem.world_size:
            errors.append("sequential program requires one readiness barrier per rank")
        for rank in range(problem.world_size):
            rank_barriers = [barrier for barrier in barriers if barrier.rank == rank]
            if len(rank_barriers) != 1:
                errors.append(f"sequential rank {rank} requires exactly one readiness barrier")
                continue
            barrier = rank_barriers[0]
            expected_inputs = {
                write
                for transfer_ops in transfers.values()
                for transfer in transfer_ops
                if transfer.rank == rank
                for write in transfer.writes
            }
            if set(barrier.reads) != expected_inputs:
                errors.append(f"barrier {barrier.id} does not wait for every incoming tile")
            if barrier.writes != (f"ready.r{rank}",):
                errors.append(f"barrier {barrier.id} writes an invalid readiness token")
        for gemm_ops in gemms.values():
            for gemm in gemm_ops:
                if f"ready.r{gemm.rank}" not in gemm.reads:
                    errors.append(f"sequential GEMM {gemm.id} bypasses the readiness barrier")
    elif barriers:
        errors.append("pipelined program must not contain a global readiness barrier")

    errors = list(dict.fromkeys(errors))
    return VerificationReport(valid=not errors, errors=tuple(errors), dependence=dependence)


# Compatibility name retained for the first prototype.
verify_schedule = verify_program


__all__ = ["VerificationReport", "verify_program", "verify_schedule"]
