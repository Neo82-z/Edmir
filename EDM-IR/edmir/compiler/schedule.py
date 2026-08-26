from __future__ import annotations

from .model import A2AGemmProblem, Buffer, DTYPE_BYTES, Op, Program


class ProgramBuilder:
    """Small direct builder: declare buffers and ops, never dependency edges."""

    def __init__(
        self,
        problem: A2AGemmProblem,
        *,
        strategy: str,
        chunk_rows: int,
        buffer_count: int,
    ) -> None:
        problem.validate()
        if chunk_rows <= 0 or problem.rows_per_peer % chunk_rows:
            raise ValueError("chunk_rows must be a positive divisor of rows_per_peer")
        if buffer_count < 1:
            raise ValueError("buffer_count must be positive")
        self.problem = problem
        self.strategy = strategy
        self.chunk_rows = chunk_rows
        self.buffer_count = buffer_count
        self._buffers: dict[str, Buffer] = {}
        self._ops: list[Op] = []

    def buffer(self, name: str, *, rank: int, role: str, size_bytes: int = 0) -> str:
        buffer = Buffer(name=name, rank=rank, role=role, size_bytes=size_bytes)
        previous = self._buffers.get(name)
        if previous is not None and previous != buffer:
            raise ValueError(f"conflicting definitions for buffer {name}")
        self._buffers.setdefault(name, buffer)
        return name

    def transfer(self, *, tile: str, src: int, dst: int, stage: str, rows: int) -> Op:
        source = self.buffer(
            f"input.{tile}",
            rank=src,
            role="input",
            size_bytes=rows * self.problem.bytes_per_row,
        )
        op = Op(
            id=f"transfer.{tile}",
            kind="transfer",
            rank=dst,
            peer_rank=src,
            queue="comm",
            reads=(source,),
            writes=(stage,),
            tile=tile,
            rows=rows,
        )
        self._ops.append(op)
        return op

    def gemm(
        self,
        *,
        tile: str,
        src: int,
        dst: int,
        stage: str,
        rows: int,
        gate: str | None = None,
    ) -> Op:
        weight = self.buffer(
            f"weight.r{dst}",
            rank=dst,
            role="weight",
            size_bytes=self.problem.k * self.problem.n * DTYPE_BYTES[self.problem.dtype],
        )
        output = self.buffer(
            f"output.{tile}",
            rank=dst,
            role="output",
            size_bytes=rows * self.problem.n * DTYPE_BYTES[self.problem.dtype],
        )
        reads = (stage, weight) if gate is None else (stage, weight, gate)
        op = Op(
            id=f"gemm.{tile}",
            kind="gemm",
            rank=dst,
            peer_rank=src,
            queue="compute",
            reads=reads,
            writes=(output,),
            tile=tile,
            rows=rows,
        )
        self._ops.append(op)
        return op

    def fused_transfer_gemm(
        self,
        *,
        tile: str,
        src: int,
        dst: int,
        rows: int,
    ) -> Op:
        """Declare one device-initiated transfer plus destination-local GEMM.

        The fused op reads the remote input tile and the destination weight
        directly. It intentionally has no staging buffer: this is the Program
        semantic boundary that a future NCCL Device API backend must implement.
        """

        source = self.buffer(
            f"input.{tile}",
            rank=src,
            role="input",
            size_bytes=rows * self.problem.bytes_per_row,
        )
        weight = self.buffer(
            f"weight.r{dst}",
            rank=dst,
            role="weight",
            size_bytes=self.problem.k * self.problem.n * DTYPE_BYTES[self.problem.dtype],
        )
        output = self.buffer(
            f"output.{tile}",
            rank=dst,
            role="output",
            size_bytes=rows * self.problem.n * DTYPE_BYTES[self.problem.dtype],
        )
        op = Op(
            id=f"fused_transfer_gemm.{tile}",
            kind="fused_transfer_gemm",
            rank=dst,
            peer_rank=src,
            queue="compute",
            reads=(source, weight),
            writes=(output,),
            tile=tile,
            rows=rows,
        )
        self._ops.append(op)
        return op

    def barrier(self, *, rank: int, waits_for: tuple[str, ...]) -> str:
        token = self.buffer(f"ready.r{rank}", rank=rank, role="token")
        self._ops.append(
            Op(
                id=f"barrier.r{rank}",
                kind="barrier",
                rank=rank,
                queue="control",
                reads=waits_for,
                writes=(token,),
            )
        )
        return token

    def build(self, *, metadata: dict | None = None) -> Program:
        return Program(
            problem=self.problem,
            strategy=self.strategy,
            chunk_rows=self.chunk_rows,
            buffer_count=self.buffer_count,
            buffers=tuple(self._buffers.values()),
            ops=tuple(self._ops),
            metadata=dict(metadata or {}),
        )


def _rounds(problem: A2AGemmProblem, chunk_rows: int):
    chunks = problem.rows_per_peer // chunk_rows
    for step in range(problem.world_size):
        for chunk in range(chunks):
            for dst in range(problem.world_size):
                src = (dst + step) % problem.world_size
                tile = f"s{src}.d{dst}.c{chunk}"
                yield step, chunk, dst, src, tile


def build_sequential_program(problem: A2AGemmProblem) -> Program:
    builder = ProgramBuilder(
        problem,
        strategy="sequential",
        chunk_rows=problem.rows_per_peer,
        buffer_count=problem.world_size,
    )
    records: list[tuple[str, int, int, str]] = []
    stages_by_rank: dict[int, list[str]] = {rank: [] for rank in range(problem.world_size)}

    for _step, _chunk, dst, src, tile in _rounds(problem, problem.rows_per_peer):
        stage = builder.buffer(
            f"stage.r{dst}.slot{src}",
            rank=dst,
            role="staging",
            size_bytes=problem.rows_per_peer * problem.bytes_per_row,
        )
        builder.transfer(tile=tile, src=src, dst=dst, stage=stage, rows=problem.rows_per_peer)
        records.append((tile, src, dst, stage))
        stages_by_rank[dst].append(stage)

    gates = {
        rank: builder.barrier(rank=rank, waits_for=tuple(stages_by_rank[rank]))
        for rank in range(problem.world_size)
    }
    for tile, src, dst, stage in records:
        builder.gemm(
            tile=tile,
            src=src,
            dst=dst,
            stage=stage,
            rows=problem.rows_per_peer,
            gate=gates[dst],
        )
    return builder.build(metadata={"peer_order": "cyclic", "dependencies": "derived"})


def build_pipelined_program(
    problem: A2AGemmProblem,
    *,
    chunk_rows: int,
    buffer_count: int = 2,
) -> Program:
    builder = ProgramBuilder(
        problem,
        strategy="pipelined",
        chunk_rows=chunk_rows,
        buffer_count=buffer_count,
    )
    chunks = problem.rows_per_peer // chunk_rows
    current_group: tuple[int, int] | None = None
    pending: list[tuple[str, int, int, str]] = []

    def emit_gemms() -> None:
        for tile, src, dst, stage in pending:
            builder.gemm(tile=tile, src=src, dst=dst, stage=stage, rows=chunk_rows)
        pending.clear()

    for step, chunk, dst, src, tile in _rounds(problem, chunk_rows):
        group = (step, chunk)
        if current_group is not None and group != current_group:
            emit_gemms()
        current_group = group
        slot = (step * chunks + chunk) % buffer_count
        stage = builder.buffer(
            f"stage.r{dst}.slot{slot}",
            rank=dst,
            role="staging",
            size_bytes=chunk_rows * problem.bytes_per_row,
        )
        builder.transfer(tile=tile, src=src, dst=dst, stage=stage, rows=chunk_rows)
        pending.append((tile, src, dst, stage))
    emit_gemms()
    return builder.build(metadata={"peer_order": "cyclic", "dependencies": "derived"})


def build_fused_program(
    problem: A2AGemmProblem,
    *,
    chunk_rows: int,
) -> Program:
    """Lower All2All->GEMM tiles to an explicit fused Program prototype.

    This function defines semantics and legality only. It does not enter the
    cost-model candidate set until a measured fused kernel and executable
    backend are available.
    """

    builder = ProgramBuilder(
        problem,
        strategy="fused",
        chunk_rows=chunk_rows,
        buffer_count=1,
    )
    for _step, _chunk, dst, src, tile in _rounds(problem, chunk_rows):
        builder.fused_transfer_gemm(tile=tile, src=src, dst=dst, rows=chunk_rows)
    return builder.build(
        metadata={
            "peer_order": "cyclic",
            "dependencies": "derived",
            "fusion": "device_initiated_transfer_plus_gemm",
            "requires_communication_api": "nccl_device",
        }
    )


# Compatibility function names retained for callers of the first prototype.
build_sequential_schedule = build_sequential_program
build_pipelined_schedule = build_pipelined_program


__all__ = [
    "ProgramBuilder",
    "build_fused_program",
    "build_pipelined_program",
    "build_pipelined_schedule",
    "build_sequential_program",
    "build_sequential_schedule",
]
