# Four-A100 fused LSA + GEMM schedule boundary

Status: **measured on 2026-08-29; grouped-N remains experimental and is not
profile-admitted**.

## Question

For fused LSA + GEMM, which schedule

```text
(g = N tiles retained per CTA, q = split-K slices)
```

minimizes latency while balancing remote-A replay against CTA supply? The
compiler-side accounting is

```text
rho_N = ceil(ceil(N / T_N) / g)
remote_A_bytes_est = remote_input_bytes * rho_N
CTA_supply_per_remote_op = ceil(M / T_M) * rho_N * q
```

`g` can reduce remote bytes. `q` cannot; it only changes concurrency and adds
split-K reduction cost. The experiment tests both claims and the separate
host-versus-fused admission decision.

## Reproducibility envelope

- 4 x NVIDIA A100-SXM4-80GB, every GPU pair reported as `NV12`, full P2P;
- driver 570.195.03, CUDA 12.8.93;
- isolated NCCL 2.28.9 with Device API enabled;
- CUTLASS commit `dcf215af68a2d08d305076c152a06f201728cd53`;
- EDM-IR commit `f584d7d` plus the direct-script import fix committed with
  this result;
- FP16, 30 warmups and 300 timed iterations for the main surface;
- 100 warmups and 2,000 iterations for the grouped-N crossover replication.

The compact raw archive is
`raw/edmir-a100-4gpu-20260829-compact-v2.tar.gz`, SHA-256
`262ef0947a7b943200fe3a884b20c29be156fca055014a09981364e6ea133d02`.
It retains CSVs, manifests, environment/topology fingerprints, correctness and
run logs, NCCL output, P2P output, and held-out decisions. Rebuildable binaries
and per-candidate build trees are excluded.

## Measurement coverage and correctness

- 358 timed candidate rows including the smoke gate and high-statistic repeat;
- 41 distinct base-family shapes over rows-per-peer 32--512, N 128--2048,
  and K 512--8192, plus the K=256 smoke point;
- host, direct, and split-K q in {2,4,8,16};
- grouped-N g=2 paired with direct and each split-K family on four N sweeps;
- 61 per-kernel correctness runs, all with `max_abs_error=0` and
  `max_rel_error=0`;
- every NCCL validation reported zero wrong elements.

## Split-K boundary at rows-per-peer=64, N=512

All values are median milliseconds. `speedup` compares the best candidate with
the host NCCL + local CUTLASS pipeline.

| K | host | direct | q=2 | q=4 | q=8 | q=16 | oracle | speedup |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 512 | 0.5649 | 0.3148 | 0.3128 | 0.3369 | 0.4411 | 0.7016 | q=2 | 1.806x |
| 1024 | 0.5811 | 0.4037 | 0.3426 | 0.3492 | 0.4581 | 0.7155 | q=2 | 1.696x |
| 2048 | 0.5826 | 0.6079 | 0.4433 | 0.4003 | 0.4844 | 0.7448 | q=4 | 1.455x |
| 4096 | 0.6370 | 1.0334 | 0.6482 | 0.5079 | 0.5467 | 0.7851 | q=4 | 1.254x |
| 8192 | 0.8427 | 1.8886 | 1.0863 | 0.7214 | 0.6556 | 0.8742 | q=8 | 1.285x |

This is the central split-K result: q does not reduce remote bytes. It improves
long-K latency only until the CTA grid is large enough to cover the SMs; larger
q then loses to extra reduction and synchronization work. Increasing either M
or N increases the base CTA grid and moves the oracle back toward smaller q.

## Grouped-N result

The g=2 kernel halves both estimated remote-A replay and CTA supply. Across 100
matched main-sweep pairs it won only four times; the geometric conclusion is
therefore not "reuse is always faster". Most points lose because the wider
CUTLASS tile and reduced grid outweigh the saved LSA loads.

The only useful crossover was independently repeated at
rows-per-peer=64, N=1024, K=8192:

| candidate | replay | CTA/op | median ms | p95 ms |
|:---|---:|---:|---:|---:|
| split-K q=4, g=1 | 8 | 32 | 0.7767 | 0.8381 |
| split-K q=8, g=1 | 8 | 64 | 0.8430 | 0.9261 |
| split-K q=4, g=2 | 4 | 16 | 0.8604 | 0.9036 |
| split-K q=8, g=2 | 4 | 32 | **0.7685** | 0.8459 |

Grouped q=8 is 1.1% faster than the best ungrouped schedule while restoring the
same 32-CTA supply as ungrouped q=4. This is evidence of a narrow interaction,
not sufficient evidence to admit g=2 globally.

## Host admission boundary

Among 41 base-family shapes, the oracle counts were direct=5, q2=14, q4=15,
q8=4, and host=3. Fused winners averaged 1.467x over host and reached 1.878x,
but host won these compute-heavy points:

| rows/peer | N | K | host ms | best fused ms | host advantage |
|---:|---:|---:|---:|---:|---:|
| 64 | 2048 | 4096 | 0.6453 | 0.7209 | 1.117x |
| 64 | 2048 | 8192 | 1.0471 | 1.1937 | 1.140x |
| 512 | 512 | 8192 | 1.5298 | 1.8079 | 1.182x |

The compiler therefore needs two decisions: host versus fused admission, then
direct/split/grouped schedule selection inside the admitted fused region.

## Held-out result: the old K-only rule is rejected

A per-K rule frozen on rows-per-peer=64, N=512 selected q2/q2/q4/q4/q8 for
K=512/1024/2048/4096/8192. On 30 held-out M/N problems it achieved:

- exact-family accuracy: 66.7%;
- mean selection regret: 2.15%;
- maximum selection regret: 19.47%;
- selected schedule faster than host: 28/30;
- mean selected speedup over host: 1.454x.

This clears the evidence-count threshold but fails the 99% direction and 1%
maximum-regret admission gates. The result falsifies K-only selection. The next
cost model must include the base CTA grid from M and N, q, grouped replay, and
an explicit host-admission term.

## Communication baselines

NCCL 2.28.9, four GPUs, 1 KiB--1 GiB, five warmups and 20 iterations:

| collective at 1 GiB | algorithm BW | bus BW |
|:---|---:|---:|
| AllReduce | 139.04 GB/s | 208.56 GB/s |
| AllGather | 249.79 GB/s | 187.35 GB/s |
| ReduceScatter | 265.90 GB/s | 199.42 GB/s |
| AllToAll | 269.82 GB/s | 202.37 GB/s |

Small-message latency was approximately 24--31 us after initialization. The
CUDA P2P sample reported 273--276 GB/s unidirectional writes, mostly 518--519
GB/s bidirectional bandwidth (some GPU-2 pairs were 425--427 GB/s), and roughly
3.0--3.4 us GPU-initiated peer latency. P2P-disabled copies were only 15--18
GB/s at roughly 24.6 us.

## Decision

- Keep g=2 executable as an experimental cost-model candidate and regression
  test; do not profile-admit it yet.
- Replace the K-only selector with a two-stage admission + CTA-supply model.
- Use the measured four-A100 topology and communication curves as the
  single-node control when the project moves to eight-card inference.

## Local two-stage selector checkpoint

`two_stage_selector_report.json` is a deterministic leave-one-sweep-group-out
evaluation over the nine M/N sweeps (346 candidate rows and 41 complete
problems).  The model predicts the host baseline first, admits fusion only when
the predicted fused winner is lower, and then chooses among direct,
q={2,4,8,16}, and the measured g=2 variants.  Its features expose FLOPs,
unique remote bytes, local bytes, rho_N, Gamma, K/q, and split reduction work.

The first checkpoint is deliberately **not admitted**:

| model | direction | mean regret | max regret | host regressions | fused coverage |
|:---|---:|---:|---:|---:|---:|
| full | 87.8% | 4.19% | 29.24% | 3 | 94.7% |
| no admission | 92.7% | 3.18% | 19.47% | 3 | 100.0% |
| no Gamma | 87.8% | 4.21% | 29.24% | 3 | 94.7% |
| no rho_N | 92.7% | 2.13% | 24.24% | 2 | 97.4% |
| no reduction | 87.8% | 9.75% | 57.88% | 1 | 89.5% |
| K-only regression | 82.9% | 31.67% | 77.92% | 3 | 89.5% |

The K-only row is the leave-one-group-out regression ablation, not the earlier
single-surface lookup rule whose separate 30-point evaluation reported 2.15%
mean regret. Both fail admission, but under different train/test protocols.

This result is more useful than an in-sample fit.  Removing reduction work is
clearly harmful, but the nearly unchanged no-Gamma result and improved
no-rho_N mean regret show that this surface is too correlated to interpret
individual ridge coefficients causally.  The frozen eight-GPU protocol varies
both fixed-per-peer and fixed-per-rank-row regimes to break that collinearity.
No coefficient from this checkpoint is connected to the default compiler path.
