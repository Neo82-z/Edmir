"""Run the complete frontend -> Dataflow -> Program -> C++ backend pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    CalibrationProfile,
    CompilerOptions,
    CompilerPipeline,
    CppManifestBackend,
    Target,
    TopologySnapshot,
    build_a2a_gemm_module,
)


problem = A2AGemmProblem(world_size=4, rows_per_peer=64, k=4096, n=4096)
module = build_a2a_gemm_module(problem)
topology = TopologySnapshot(
    world_size=4,
    devices=("example-gpu",) * 4,
    p2p=tuple(tuple(True for _ in range(4)) for _ in range(4)),
)
target = Target("example-cuda", topology, arch="sm_89")
profile = CalibrationProfile(
    comm_latency_us=1.0,
    comm_bandwidth_gbps=50.0,
    gemm_launch_us=0.5,
    gemm_tflops=100.0,
    source="synthetic-example-not-a-measurement",
)

result = CompilerPipeline().compile(
    module,
    target=target,
    profile=profile,
    backend=CppManifestBackend(),
    options=CompilerOptions(chunk_rows=(16, 32, 64), buffer_counts=(1, 2, 4)),
)

print("passes=" + ",".join(item.name for item in result.pass_trace))
print(f"matched={result.match.all_to_all.id}->{result.match.gemm.id}")
print(f"strategy={result.compile_result.chosen.program.strategy}")
print("files=" + ",".join(result.artifact.files))
