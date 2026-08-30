# Eight-A100 transfer experiment

This directory is the complete evidence bundle for the pre-registered P4 to
P8 transfer test.  The run used source commit
`72c01fd40d16210f667bd2c7f3697815f190c872` and the frozen manifest in
`extracted/experiments/2026-08-29-a100-8gpu-transfer-plan/manifest.json`.

## Environment

- 8 NVIDIA A100-SXM4-80GB GPUs;
- driver 570.172.08 and CUDA 12.8.93;
- full pairwise `NV12` topology with CUDA P2P read/write access enabled;
- NCCL `v2.28.9-1`, commit
  `dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`;
- CUTLASS `v4.7.0`, commit
  `dcf215af68a2d08d305076c152a06f201728cd53`.

Pairwise unidirectional P2P writes are mostly 269--277 GB/s.  Bidirectional
results form two tiers, roughly 417--422 and 518--522 GB/s, despite the common
`NV12` label.  GPU-initiated peer latency is approximately 2.8--3.4 us.  With
P2P disabled, bandwidth falls to about 20 GB/s and latency rises to 23--24 us.

At 1 GiB, the eight-rank NCCL bus-bandwidth measurements are:

| Collective | Out-of-place | In-place |
|:--|--:|--:|
| AllReduce | 229.05 GB/s | 229.07 GB/s |
| AllGather | 197.31 GB/s | 220.46 GB/s |
| ReduceScatter | 202.85 GB/s | 215.92 GB/s |
| AllToAll | 210.93 GB/s | 216.48 GB/s |

All NCCL and CUDA P2P correctness checks pass.

## Frozen matrix

The measured matrix contains 23 shapes and 253 candidate rows: host,
direct, split-K `q in {2,4,8,16}`, and the corresponding grouped-N variants.
The first calibration shape validated all 11 families numerically.  The
remaining frozen sweeps used 30 warmups and 300 timed iterations without
duplicating the same family validation.  Every shape has all 11 candidates.

The P8 oracle contains no host winner:

| Family | Oracle shapes |
|:--|--:|
| split-K 2 | 5 |
| split-K 4 | 9 |
| split-K 8 | 3 |
| grouped-N split-K 4 | 2 |
| grouped-N split-K 8 | 4 |

Representative former P4 host regions flip to fused schedules:

| M | K | N | P8 oracle | Oracle (ms) | Host (ms) | Host / oracle |
|--:|--:|--:|:--|--:|--:|--:|
| 64 | 4096 | 2048 | grouped-N split-K 4 | 1.7945 | 2.6050 | 1.452x |
| 64 | 8192 | 2048 | grouped-N split-K 8 | 2.7867 | 2.9435 | 1.056x |
| 256 | 8192 | 512 | grouped-N split-K 8 | 2.7125 | 3.7670 | 1.389x |
| 512 | 8192 | 512 | grouped-N split-K 4 | 3.4733 | 5.0696 | 1.460x |

## Transfer result

The zero-shot selector is the rule frozen on P4; no P8 result is used to
change its decisions.  The four-shape calibration protocol fits the existing
two-stage model only on the four pre-registered P8 calibration shapes and is
then frozen for the 19 held-out shapes.

| Protocol | Direction accuracy | Exact family | Mean regret | Max regret |
|:--|--:|--:|--:|--:|
| P4 to P8 zero-shot, all 23 | 91.30% | 56.52% | 11.60% | 64.14% |
| P4 to P8 zero-shot, held-out 19 | 89.47% | 57.89% | 10.87% | 64.14% |
| Four-shape P8 calibration, held-out 19 | 100.00% | 15.79% | 15.02% | 64.35% |

Four-point calibration fixes host-versus-fused direction but does not solve
family selection.  It increases held-out regret, so neither selector passes
the pre-registered 0.5% mean-regret and 1% maximum-regret gates.  This rejects
both the P4 frozen rule and the current four-point replacement as production
selectors.

## Boundary replication

Every shape whose initial top-two gap was below 2% was repeated with 100
warmups and 2,000 timed iterations.  All 6/6 winners remain stable.  Ten of
the twelve repeated candidates also have per-shape numerical validation.  The
two large `M=64, K=8192, N=1024` candidates rely on the already completed
family validation because the redundant CPU reference exceeded the experiment
window; this exception is recorded in the CSV rather than hidden.

The repeat-confirmed accounting leaves the scientific result unchanged.  In
particular, the transfer failure is not a near-tie artifact.

## Profiling limitation

Nsight Compute 2025.1.1 was available, but the cloud host denied hardware
performance-counter access on all eight devices with `ERR_NVGPUCTRPERM`.
No counter-derived claim is made.  The attempted command and failure are
retained under `extracted/results/p8-analysis/ncu/`.

## Artifact map

- `edmir-p8-a100-20260830.tar.gz`: immutable transfer archive, SHA-256
  `95446fda8dc12b2580b67c4f125e49c17730185514349a2a3d30109205cf830a`;
- `extracted/logs/`: environment, build, P2P, NCCL, frozen-run, scoring, and
  replication logs;
- `extracted/results/p8-analysis/shape_oracles.csv`: one oracle per shape;
- `extracted/results/p8-analysis/transfer_score.json`: untouched initial
  300-iteration accounting;
- `extracted/results/p8-analysis/transfer_score_with_repeats.json`: fixed
  decisions with high-statistics medians substituted at registered boundaries;
- `extracted/results/p8-analysis/replication_summary.json`: initial versus
  repeated winners and gaps;
- `extracted/PACKAGE_CONTENTS.txt`: packaged-file inventory.

