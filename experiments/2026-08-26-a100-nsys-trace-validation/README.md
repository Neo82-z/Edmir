# A100 Nsight dependency validation (2026-08-26)

This experiment checks whether the CUDA runtime executes the exact dependency
semantics derived from a selected EDM-IR Program. Optional NVTX3 ranges label
each `CudaBackend::execute()` call and every stable Program op ID. Nsight CUDA
runtime correlations then map kernels and D2D copies back to those op IDs.

## Protocol

The two held-out CUTLASS Programs from the provider-specific cost-model stage
were captured with three warmup executions and one measured execution. The
validator defaults to the last mapped iteration, avoiding the first-use NCCL
initialization: iteration 0 contains 768 extra 152-byte H2D events, while every
steady iteration contains exactly 40 kernels and four local D2D copies.

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=trace \
  ./edmir_benchmark_generated --warmup 3 --iterations 1 --nvtx
nsys export --type sqlite --force-overwrite=true \
  --output=trace.sqlite trace.nsys-rep
python -m edmir.nsys_export trace.sqlite --out trace-export
python -m edmir validate-trace program.json \
  trace-export/edm_cuda_events.csv --out trace-validation.json
```

The validator requires every non-barrier op on every expected logical device,
checks GEMM/transfer event kinds, derives dependencies from Program buffers,
and verifies predecessor completion before successor start on each shared GPU.

## Result

| case | op coverage | GPU events | dependency-device checks | min slack | violations |
|---|---:|---:|---:|---:|---:|
| `r64,k2048,n4096` | 32/32 | 44 | 52 | 37.024 us | 0 |
| `r96,k4096,n4096` | 32/32 | 44 | 52 | 20.832 us | 0 |

Each case checks 36 queue constraints and 16 read-after-write constraints.
Remote transfers map to two NCCL kernels (source and destination GPU), local
transfers to one D2D copy, and each GEMM to one CUTLASS kernel.

Across three alternating 100-iteration process pairs, NVTX increased r64 from
0.5208 to 0.5503 ms (5.66%) and r96 from 0.5977 to 0.6218 ms (4.03%). Keeping
annotation disabled by default therefore matters. Under Nsight, the single
measured iterations expanded further to 1.2683 and 1.3612 ms. Consequently, the
traces are used as ordering/coverage evidence; their absolute span and
communication-compute overlap are not performance baselines.

The capture diagnostic also reports that Nsight Systems 2025.1.1 used CUDA
12.8 tracing libraries with a newer installed driver. Capture health, CUDA,
NVTX, all four A100s, and all required event tables were present.

Run `python3 summarize_results.py` to rebuild the overhead table and summary.
The raw reports, SQLite databases, readable exports, Programs, and logs are in
`edmir-nsys-trace-validation-20260826.tar.gz`; SHA-256:
`c795c49b566ba3eec9f31ffc300f5b2b45fa2a9338eb166b5c178e3a63d67dc5`.
