"""Smallest complete 2-rank All2All -> GEMM Program.

Run from EDM-IR with ``python examples/minimal_program.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (
    A2AGemmProblem,
    CalibrationProfile,
    ProgramBuilder,
    derive_dependencies,
    simulate_program,
    verify_program,
)


problem = A2AGemmProblem(world_size=2, rows_per_peer=16, k=128, n=128)
builder = ProgramBuilder(
    problem,
    strategy="pipelined",
    chunk_rows=16,
    buffer_count=2,
)

for step in range(problem.world_size):
    for dst in range(problem.world_size):
        src = (dst + step) % problem.world_size
        tile = f"s{src}.d{dst}.c0"
        stage = builder.buffer(f"stage.r{dst}.slot{step}", rank=dst, role="staging")
        builder.transfer(tile=tile, src=src, dst=dst, stage=stage, rows=16)
        builder.gemm(tile=tile, src=src, dst=dst, stage=stage, rows=16)

program = builder.build(metadata={"example": "minimal"})
report = verify_program(problem, program)
report.require_valid()

print("ops")
for op in program.ops:
    print(f"  {op.id}: {op.kind} reads={op.reads} writes={op.writes}")

print("derived dependencies")
for dependency in derive_dependencies(program).dependencies:
    print(f"  {dependency.src} -> {dependency.dst}: {dependency.kind}")

profile = CalibrationProfile(
    comm_latency_us=1.0,
    comm_bandwidth_gbps=50.0,
    gemm_launch_us=0.5,
    gemm_tflops=100.0,
)
simulation = simulate_program(program, profile)
print(f"makespan_us={simulation.makespan_us:.3f}")
