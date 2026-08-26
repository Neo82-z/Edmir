# Edmir C++ runtime

The runtime consumes the `ProgramView` emitted in `edmir_program.hpp`. It does
not rebuild compiler dependencies or choose another schedule.

```text
generated ProgramView
  -> host validation + topological order
  -> logical-rank device map
  -> per-rank comm / compute / control streams
  -> dependency CUDA events
  -> NCCL send/recv or local D2D copy
  -> selected row-major GEMM provider (cuBLAS or CUTLASS)
```

The host-only library contains the stable ABI and validator. The optional CUDA
library currently implements one-process, single-node execution for static
equal-size `All2All -> GEMM` Programs.

## Host-only build

This path needs only a C++20 compiler and is suitable for a laptop or CI:

```bash
cmake -S runtime -B build/runtime -DEDMIR_BUILD_TESTS=ON
cmake --build build/runtime -j
ctest --test-dir build/runtime --output-on-failure
```

## Generate a Program

From `EDM-IR/`:

```bash
python -m edmir frontend examples/a2a_gemm_problem.json \
  --out /tmp/a2a-gemm.dataflow.json

python -m edmir compile-graph /tmp/a2a-gemm.dataflow.json \
  --profile examples/synthetic_profile.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16,32,64 --buffers 1,2,4 \
  --target cuda --arch sm_89 --gemm-provider cublas --backend cpp \
  --out-dir /tmp/edmir-artifact
```

## CUDA server build and smoke execution

CUDA Toolkit, cuBLAS, and NCCL development headers/libraries must already be
installed. `EDMIR_GENERATED_DIR` enables a small runner for the generated
header:

```bash
cmake -S runtime -B build/runtime-cuda \
  -DEDMIR_ENABLE_CUDA=ON \
  -DEDMIR_GENERATED_DIR=/tmp/edmir-artifact
cmake --build build/runtime-cuda -j

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./build/runtime-cuda/edmir_run_generated
```

If NCCL is outside the default search path, pass
`-DNCCL_INCLUDE_DIR=/path/to/include -DNCCL_LIBRARY=/path/to/libnccl.so`.

## Optional CUTLASS provider

`gemm_provider` belongs to the compiler target and is materialized in
`ProgramView`; it is not a new IR operation. Thus a verified Program can be
emitted twice and compared with identical transfers, buffers, streams, events,
and dependencies:

```bash
git clone --depth 1 --branch v4.7.0 \
  https://github.com/NVIDIA/cutlass.git /opt/cutlass

python -m edmir compile-graph /tmp/a2a-gemm.dataflow.json \
  --profile examples/synthetic_profile.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16,32,64 --buffers 1,2,4 \
  --target a100 --arch sm_80 --gemm-provider cutlass --backend cpp \
  --out-dir /tmp/edmir-cutlass-artifact

cmake -S runtime -B build/runtime-cutlass \
  -DEDMIR_ENABLE_CUDA=ON \
  -DEDMIR_ENABLE_CUTLASS=ON \
  -DEDMIR_CUTLASS_DIR=/opt/cutlass \
  -DEDMIR_GENERATED_DIR=/tmp/edmir-cutlass-artifact
cmake --build build/runtime-cutlass -j
```

The deliberately narrow v0.1 specialization is FP16, row-major, `sm_80`, with
K and N divisible by eight. The compiler and runtime both reject unsupported
combinations. It uses the host-side `cutlass::gemm::device::Gemm` interface
with a three-member Ampere tensor-op family. `auto` is resolved in generated
artifacts to `64x128x32` for lowered GEMM M at most 96 and `128x128x32`
otherwise; `128x64x32` remains available for explicit experiments. All use a
`16x8x16` instruction, three stages, and alignment eight. See the official
[CUTLASS GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api.md)
and [SM80 example](https://github.com/NVIDIA/cutlass/blob/main/examples/45_dual_gemm/dual_gemm.cu).

The smoke runner fills every external input and weight buffer with zero,
executes the selected Program, and requires all output bytes to remain zero.
It checks allocation, communicator creation, stream/event dependency lowering,
transfers, and GEMM dispatch. It is not a performance benchmark or a substitute
for numerical comparison.

For a non-zero correctness check, generate a small `float32` or `float16`
Program from `examples/runtime_validation_problem.json` or
`examples/runtime_validation_problem_fp16.json`, rebuild with that artifact,
and run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./build/runtime-cuda/edmir_validate_generated
```

The numerical runner initializes deterministic inputs and weights, interprets
the generated dependency graph on the CPU, executes the same Program through
the CUDA backend, and compares every output element. Keeping this runner driven
by `ProgramView` makes it exercise transfers, buffer reuse, scheduling, and GEMM
lowering together instead of testing a hand-written operation sequence.

`edmir_benchmark_generated` performs five warmups followed by thirty measured
`execute()`/`synchronize()` iterations. It reports min, median, p95, max, and
mean host-observed latency plus the compiler-visible dispatch/event/reuse
feature counts. `--warmup` and `--iterations` override the defaults. It also
exercises event and buffer reuse across repeated Program launches.

`edmir_benchmark_gemm` measures the exact FP16 `cublasGemmEx` call used by the
runtime and emits one JSON object per device and shape. For example:

```bash
./build/runtime-cuda/edmir_benchmark_gemm \
  --m 16,32,64,128 --n 4096 --k 4096 \
  --warmup 10 --trials 7 --iterations 100
```

`edmir_benchmark_copy` similarly measures the local `cudaMemcpyAsync` D2D path
used for self tiles and emits JSON lines keyed by `size_bytes`.

By default logical rank `r` maps to visible CUDA ordinal `r`. Embedding code can
instead pass `CudaBackendOptions{.devices = {...}}`:

```cpp
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir_program.hpp"

edmir::runtime::CudaBackend runtime(
    edmir_generated::program(),
    edmir::runtime::CudaBackendOptions{.devices = {2, 3, 0, 1}});
runtime.prepare();
// Populate input.* and weight.r* buffers with copy_from_host().
runtime.execute();
runtime.synchronize();
// Read output.* buffers with copy_to_host().
```

## Deliberate v0.1 limits

- one host process controls all GPUs;
- one NCCL communicator and three streams per logical rank;
- each Program buffer owns a separate allocation;
- GEMM provider/kernel selection is static and materialized in `ProgramView`;
- no CUDA Graph capture, memory pool, online CUTLASS autotuning, Device
  API, GIN, multi-node bootstrap, or dynamic MoE routing.

The next backend step is provider-specific calibration followed by trace
comparison. CuTe device communication remains a later lowering boundary and is
not hidden inside this provider substitution.
