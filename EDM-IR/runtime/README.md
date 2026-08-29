# Edmir C++ runtime

The runtime consumes the `ProgramView` emitted in `edmir_program.hpp`. It does
not rebuild compiler dependencies or choose another schedule.

```text
generated ProgramView
  -> host validation + topological order
  -> logical-rank device map
  -> per-rank comm / compute / control streams
  -> dependency CUDA events
  -> host path: NCCL send/recv or local D2D copy, then cuBLAS/CUTLASS
  -> fused path: NCCL symmetric window, LSA remote read, CUDA GEMM reference
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

## NCCL Device API fused reference

This path requires NCCL 2.28 or newer, `nccl_device.h`, full single-node P2P,
SM80, FP16, and row-major input. Probe the exact NCCL prefix first:

```bash
NCCL_HOME=/opt/nccl-2.28.9/build \
python -m edmir probe-runtime \
  --out /tmp/runtime-capabilities.json --require-fused

python -m edmir frontend examples/fused_runtime_validation_problem.json \
  --out /tmp/fused.dataflow.json
python -m edmir lower-fused /tmp/fused.dataflow.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 8 --arch sm_80 \
  --gemm-provider cuda_reference --backend cpp \
  --out-dir /tmp/fused-artifact

cmake -S runtime -B build/runtime-fused \
  -DEDMIR_ENABLE_CUDA=ON \
  -DEDMIR_ENABLE_NCCL_DEVICE=ON \
  -DEDMIR_GENERATED_DIR=/tmp/fused-artifact \
  -DNCCL_INCLUDE_DIR=/opt/nccl-2.28.9/build/include \
  -DNCCL_LIBRARY=/opt/nccl-2.28.9/build/lib/libnccl.so
cmake --build build/runtime-fused -j

LD_LIBRARY_PATH=/opt/nccl-2.28.9/build/lib \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./build/runtime-fused/edmir_validate_generated
```

The runtime uses `ncclMemAlloc` for every source input, registers a symmetric
window for each fused tile, and launches one destination kernel that obtains
the remote pointer through `ncclGetLsaPointer`. `lsa_naive_f16` is a numerical
and lifecycle reference only. With `EDMIR_ENABLE_CUTLASS=ON`, the same pointer
is substituted into CUTLASS `GemmKernel::Params` inside a custom device wrapper;
CUTLASS still owns its predicated tensor-core mainloop and epilogue.

The CUTLASS path is an executable research kernel, not yet an admitted compiler
candidate. A standard `(M tile, N tile)` grid reloads remote A for every N tile,
so `edmir_benchmark_generated` reports `remote_input_bytes`,
`remote_a_replay`, and `estimated_remote_a_load_bytes`. These are static traffic
features, not latency predictions. The explicit
The `sm80_64x128x32_splitk{2,4,8,16}` identities add CUTLASS serial split-K
semaphore workspace and report `split_k_slices` plus
`estimated_ctas_per_op`.  A split count `q` requires K divisible by `32q` and
is available only for fused Device API Programs.  The identities remain
separate so the experiment can measure CTA supply instead of hiding it behind
an internal heuristic.

## Optional CUTLASS provider

`gemm_provider` belongs to the compiler target and is materialized in
`ProgramView`; it is not a new IR operation. Thus a verified Program can be
emitted twice and compared with identical transfers, buffers, streams, events,
and dependencies:

```bash
git clone --depth 1 --branch v4.7.0 \
  https://github.com/NVIDIA/cutlass.git /opt/cutlass

python -m edmir fit-gemm \
  --base-profile examples/synthetic_profile.json \
  --log /path/to/cutlass_gemm_measurements.jsonl \
  --gemm-provider cutlass --gemm-kernel auto \
  --out /tmp/cutlass-profile.json

python -m edmir compile-graph /tmp/a2a-gemm.dataflow.json \
  --profile /tmp/cutlass-profile.json \
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

## Nsight dependency validation

Pass `--nvtx` to `edmir_run_generated` or `edmir_benchmark_generated` to emit
NVTX3 ranges for each execute iteration and stable Program op ID. Annotation is
disabled by default and does not change the Program ABI or calibrated benchmark
path.

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=edmir-trace \
  ./build/runtime-cuda/edmir_benchmark_generated \
  --warmup 3 --iterations 1 --nvtx
nsys export --type sqlite --force-overwrite=true \
  --output=edmir-trace.sqlite edmir-trace.nsys-rep
python -m edmir.nsys_export edmir-trace.sqlite --out edmir-trace-export
python -m edmir validate-trace /path/to/program.json \
  edmir-trace-export/edm_cuda_events.csv \
  --out edmir-trace-validation.json
```

The validator chooses the last mapped iteration by default so NCCL first-use
initialization cannot masquerade as steady Program work. It checks per-device
op coverage, event kinds, and every derived dependency on the GPU devices
shared by predecessor and successor. Nsight tracing perturbs host dispatch, so
absolute trace span and overlap must not replace unprofiled latency baselines.

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
- Device API support is limited to single-node LSA remote reads and immutable
  host-populated source tiles;
- no CUDA Graph capture, memory pool, online CUTLASS autotuning, GIN,
  multi-node bootstrap, or dynamic MoE routing.

The next backend step is a persistent/grouped-N CuTe schedule that can reuse
remote A while retaining enough CTAs for occupancy, followed by M/K boundary
calibration. The scalar reference remains a separate provider so an optimized
implementation cannot silently inherit its identity or measurements.

The compiler can already serialize and analyze that future schedule through
`fused_schedule.n_tiles_per_cta`. Values above one intentionally stop before
C++ runtime emission. Runtime support is complete only when the CUDA kernel,
descriptor ABI, numerical validator, and benchmark runner all consume the same
contract; changing the traffic estimate alone does not make the optimization
available.
