# A100 CUTLASS provider substitution (2026-08-26)

This experiment isolates the GEMM implementation behind one verified EDM-IR
Program. The compiler emits byte-identical `program.json` files for cuBLAS and
CUTLASS; only `target.json` and the generated provider enum differ.

## Frozen scope

- 4x NVIDIA A100-SXM4-80GB, all GPU pairs reported as `NV12`;
- CUDA toolkit 12.8.93, driver 580.126.16, NCCL 2.25.1;
- CUTLASS v4.7.0 at commit `dcf215af68a2d08d305076c152a06f201728cd53`;
- FP16 row-major GEMM, SM80, K and N divisible by eight;
- CUTLASS tensor-op kernel: threadblock `128x128x32`, warp `64x64x32`,
  instruction `16x8x16`, three stages, alignment eight;
- the same NCCL transfers, streams, buffers, events, and dependency schedule
  for both providers.

The numerical gate uses deterministic non-zero inputs and a CPU FP32
reference. Performance uses the two held-out shapes from the preceding cost
model milestone, the already-selected `c=rows_per_peer, b=4` schedule, five
independent processes per provider, ten warmups, and 50 measured iterations.
Provider order alternates between processes.

Regenerate the machine-readable result with:

```bash
python3 summarize_results.py
```

## Result

Both providers passed deterministic non-zero validation at 8,192 and 65,536
checked output elements with zero observed absolute and relative error. Across
five independent processes per provider, the median of process medians was:

| held-out shape | cuBLAS (ms) | CUTLASS (ms) | CUTLASS reduction |
|---|---:|---:|---:|
| `r96,k4096,n4096` | 0.6213 | 0.6121 | 1.48% |
| `r64,k2048,n4096` | 0.5635 | 0.5357 | 4.93% |

Every one of the five paired process medians favored CUTLASS for both shapes.
The immutable remote archive is `edmir-cutlass-provider-20260826.tar.gz`; its
verified SHA-256 is
`de4d469500aa92aa67d049bc2d00a686e10b4431a3228d8eaa86e252571d613e`.

This milestone establishes a controlled backend substitution, not a general
CUTLASS autotuner. Unsupported combinations are rejected by the compiler and
runtime instead of silently falling back.
