# A100 CUTLASS kernel-family selection (2026-08-26)

This milestone follows the single-kernel provider substitution and keeps the
same Dataflow and Program IR. `gemm_kernel` is target policy: `auto` is resolved
by the compiler backend to a concrete enum before `target.json` and
`edmir_program.hpp` are emitted. The runtime never performs hidden autotuning.

## Kernel family and protocol

All kernels use FP16 row-major SM80 tensor operations, instruction shape
`16x8x16`, K tile 32, three stages, and alignment eight:

- `sm80_128x128x32`, warp `64x64x32`;
- `sm80_64x128x32`, warp `32x64x32`;
- `sm80_128x64x32`, warp `64x32x32`.

Each variant passed a deterministic CPU-reference validation over 65,536
non-zero output elements. The sweep used three independent processes per
kernel and shape, rotating execution order, with ten warmups and 100 measured
iterations. The final matched cuBLAS/`auto` comparison used five independent
processes per provider with alternating order and the same iteration counts.

## Result

The `64x128x32` kernel won both held-out small-M shapes. The compiler therefore
resolves CUTLASS `auto` to that kernel when the lowered GEMM M (`chunk_rows`) is
at most 96, and retains `128x128x32` above that boundary. This is a deliberately
narrow A100 policy, not a cross-architecture claim.

| held-out shape | cuBLAS (ms) | CUTLASS auto (ms) | reduction |
|---|---:|---:|---:|
| `r64,k2048,n4096,c64,b4` | 0.5670 | 0.5228 | 7.80% |
| `r96,k4096,n4096,c96,b4` | 0.6200 | 0.6029 | 2.76% |

The values are medians of five process medians; all five paired processes
favored CUTLASS for both shapes. Run `python3 summarize_results.py` to rebuild
the CSV and JSON summaries from raw logs.

The immutable remote archive is
`edmir-cutlass-kernel-family-20260826.tar.gz`; its verified SHA-256 is
`b5345d7b258bbe2dad4305e8baeb0868d6aa125a2d119ab93bb57ed1c76af71a`.
