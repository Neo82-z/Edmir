# A100 fused-kernel held-out grid

## Question

Does the direct-versus-split-K boundary learned at `M64 N64` preserve its
choice direction across unseen M and N shapes?

## Design

The training slice is `../2026-08-26-a100-fused-k-boundary/results.csv`. It has
five K values at `M64 N64`. This slice holds out the other eight points in the
Cartesian product:

```text
M = {32, 64, 128}
N = {64, 128, 256}
K = {256, 512, 1024, 2048, 4096}
```

For each of the 40 held-out problems, the runner measured host NCCL + CUTLASS,
direct LSA + CUTLASS, and LSA + CUTLASS serial split-K=8. That gives 120
held-out timing rows and 135 rows including training. Every point uses 30
warmups and 300 synchronous measured executions. Each held-out M/N directory
also retains all generated headers, build logs, timing logs, and three non-zero
K=256 correctness checks.

## Result

`python -m edmir evaluate-fused` fit the only boundary supported by the training
data:

```text
K <= 1024  -> direct
K >= 2048  -> splitk8
```

The result on the full grid is:

| split | problems | direction accuracy | mean regret | max regret | host wins |
|:---|---:|---:|---:|---:|---:|
| train | 5 | 100% | 0% | 0% | 0 |
| held-out | 40 | 100% | 0% | 0% | 0 |

Mean latency across the nine M/N combinations at each K was:

| K | host ms | direct ms | splitK8 ms | winner |
|---:|---:|---:|---:|:---|
| 256 | 0.5809 | **0.2972** | 0.4149 | direct |
| 512 | 0.5828 | **0.3134** | 0.4194 | direct |
| 1024 | 0.5853 | **0.3981** | 0.4366 | direct |
| 2048 | 0.5883 | 0.6009 | **0.4607** | splitK8 |
| 4096 | 0.6452 | 1.0260 | **0.5173** | splitK8 |

Selecting the better fused family reduced median latency relative to host by
33.7% on average across the 45 problems; the observed range was 16.1% to 50.2%.

## Admission decision

The evaluator intentionally writes `default_compiler_admission=false`. This
grid demonstrates generalization over its measured A100/NV12 envelope; it does
not prove portability to another topology, architecture, NCCL version, M/N/K
outside the grid, or a grouped-N/persistent kernel. The next compiler step is a
profile-gated selector carrying the environment fingerprint and refusing
extrapolation, not a global hard-coded K threshold.

`evaluation.json` is the machine-readable report. The per-directory
`results.csv` files and retained logs are authoritative for the measurements.

## Compiler integration

The follow-up selector is now implemented in
`EDM-IR/edmir/compiler/fused_selection.py` and exposed as
`python -m edmir compile-fused`. The retained profile admits only the exact
device/software fingerprint and the 45 measured M/N/K points above, requires
an explicit `--allow-experimental-fused`, and otherwise emits an auditable host
fallback. The source measurements and evaluation artifact remain unchanged.
