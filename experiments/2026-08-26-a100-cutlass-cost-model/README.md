# Provider-specific CUTLASS cost model (2026-08-26)

This experiment tests whether replacing isolated GEMM samples is sufficient to
predict end-to-end `NCCL + CUTLASS` Programs. It reuses the exact 27 training
and 23 held-out schedule candidates from the cuBLAS milestone while emitting
new CUTLASS-auto target artifacts.

The important hypothesis is intentionally falsifiable: isolated CUTLASS GEMMs
are slower than cuBLAS for the measured small-M shapes, while the matched
end-to-end Programs are faster. A provider-tagged profile prevents accidental
cross-provider calibration, and a fresh runtime residual fit determines
whether host dispatch features explain the inversion or an explicit
communication/compute contention feature is required.

## Protocol

The isolated GEMM sweep measured nine exact shapes on all four A100s. The
end-to-end sweep then re-emitted the original 27 training and 23 held-out
schedule candidates with `gemm_provider=cutlass` and `gemm_kernel=auto`. Each
candidate used five warmups and 30 measured iterations. Runtime overhead was
fit on training rows only; held-out rows were never used to fit coefficients.

Reproduce the candidate sweep with:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/llm-serving/EDM-IR
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 run_provider_sweep.py \
  --code /root/llm-serving/EDM-IR \
  --prepared /root/edmir-cost-model/prepared \
  --topology /root/edmir-cost-model/input/topology.json \
  --cutlass /root/cutlass \
  --out /root/edmir-cutlass-cost-model
```

## Result

| metric | train | held-out |
|---|---:|---:|
| candidates | 27 | 23 |
| MAPE | 7.81% | 8.81% |
| direction accuracy | 100.00% | 90.48% |
| mean selection regret | 0 | 0 |

Across all five problem groups, including both held-out shapes, the model
selected the measured oracle: a full-row chunk with four staging buffers. The
overall pairwise direction accuracy was 95.56%. The fitted non-negative
runtime residual retained only `event_wait_us=3.17675`; the other four
coefficients remained zero.

The final compiler-in-the-loop run used the complete Dataflow IR through C++
backend path. It selected CUTLASS `sm80_64x128x32`, `chunk_rows=rows_per_peer`,
and four buffers for both held-out cases:

| held-out shape | scan median | final 100-iteration median | p95 |
|---|---:|---:|---:|
| `r64,k2048,n4096` | 519.6 us | 516.8 us | 531.0 us |
| `r96,k4096,n4096` | 594.0 us | 595.9 us | 623.1 us |

The original default search only considered two buffers. This experiment
exposed that search-space omission, so the compiler now enumerates buffer
depths 1, 2, and 4 plus every 16-row divisor chunk. The model did not require a
new contention feature: its held-out selection regret is already zero, and
adding one would be unsupported complexity rather than an evidence-based
extension.

Run `python3 summarize_results.py` to regenerate the standalone GEMM
comparison, selection table, and machine-readable summary. The immutable raw
archive is `edmir-cutlass-cost-model-20260826.tar.gz`; its verified SHA-256 is
`ac4e51e6fc4469952ef63643b4945a195c1fdb9c550cbff61e9dc23fc8580a6a`.
