# A100 NCCL Device API fused-reference validation

## Question

Can the exact Edmir `fused_transfer_gemm` Program contract cross the Python
lowering, generated C++ ABI, NCCL Device API resource lifecycle, and a real
four-GPU numerical run without falling back to host NCCL plus GEMM?

This slice answers **yes for correctness**, not for performance.

## Fixed environment

- 4x NVIDIA A100-SXM4-80GB, compute capability 8.0;
- all GPU pairs reported P2P access and `NV12` by `nvidia-smi topo -m`;
- CUDA 12.8;
- official NCCL tag `v2.28.9-1`, commit
  `dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`;
- NCCL was built in `/root/edmir-deps/nccl-2.28.9/build` with only SM80 code;
- the system NCCL 2.25.1 installation was not replaced.

The complete machine-readable probe is
`runtime-capabilities-nccl2289.json`.

## Build and gates

```bash
make -C /root/edmir-deps/nccl-2.28.9 -j32 src.build \
  CUDA_HOME=/usr/local/cuda-12.8 \
  NVCC_GENCODE='-gencode=arch=compute_80,code=sm_80'

NCCL_HOME=/root/edmir-deps/nccl-2.28.9/build \
python -m edmir probe-runtime \
  --out runtime-capabilities-nccl2289.json --require-fused
```

The probe compiled, linked, and executed a tiny program against the selected
NCCL prefix. It reported NCCL 2.28.9, the required Device API headers/symbols,
four GPUs, full P2P, and an open fused-runtime gate.

The repository was then lowered with:

```bash
python -m edmir frontend examples/fused_runtime_validation_problem.json \
  --out /tmp/fused.dataflow.json
python -m edmir lower-fused /tmp/fused.dataflow.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 8 --arch sm_80 \
  --gemm-provider cuda_reference --backend cpp \
  --out-dir /root/edmir-fused-runtime-artifact
```

The C++ runtime was configured with `EDMIR_ENABLE_CUDA=ON`,
`EDMIR_ENABLE_NCCL_DEVICE=ON`, the generated directory above, and explicit
include/library paths to the isolated NCCL build.

## Results

1. The unmodified official `examples/06_device_api/01_allreduce` example ran
   with four ranks and passed its sum-to-six check. See
   `official-device-api-allreduce.log`.
2. Edmir validated 16 fused ops and 8,192 output elements against the CPU
   interpreter with `max_abs_error=0` and `max_rel_error=0`. See
   `edmir-validation.log`.
3. A five-warmup, 30-iteration runtime sanity run reported median 0.2844 ms and
   p95 0.3112 ms. See `edmir-benchmark.log`.

The third result is deliberately **not** a baseline or speedup claim. It has no
identical-shape unfused comparison, uses a scalar naive GEMM, and is not fitted
into `CalibrationProfile`.

## What the reference establishes

- Python lowering emits the intended staging-free source-input/local-weight
  access contract.
- The generated C++ ABI preserves fused op, Device API, provider, and kernel
  identity.
- The runtime can create symmetric allocations/windows, resolve an LSA remote
  pointer inside the destination CUDA kernel, execute, synchronize, unregister,
  and free resources correctly.
- Cost-based compilation remains closed to fusion until an optimized kernel is
  measured on train and held-out shapes.

## Next experiment

Replace scalar `lsa_naive_f16` with a separately identified CuTe/CUTLASS kernel,
then collect identical-shape unfused and fused correctness, steady latency,
Nsight kernel-boundary, and held-out direction/regret evidence.

Official references:

- [NCCL Device API user guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/usage/deviceapi.html)
- [NCCL Device API examples](https://github.com/NVIDIA/nccl/tree/master/docs/examples/06_device_api)
- [NCCL device header](https://github.com/NVIDIA/nccl/blob/master/src/include/nccl_device.h)
