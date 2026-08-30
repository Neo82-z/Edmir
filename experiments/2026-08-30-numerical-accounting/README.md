# Numerical accounting before the eight-GPU run

Status: **derived only from the completed four-A100 measurements and the
pre-registered eight-A100 shape matrix**. No eight-GPU latency is present.

This ledger fixes the units used by the model, selector, and paper. Every
latency-facing quantity is per destination rank:

- `M` is the balanced number of routed rows from one source rank to one
  destination rank;
- `batch_rows_per_rank = P*M` is the post-All2All GEMM row count on one
  destination;
- `batch_rows_global = P^2*M` is the routed-row count summed across all
  destinations;
- neither quantity is the serving request batch. For a balanced dispatch with
  `Q` token rows and top-`r` routing, `P^2*M = Q*r` before capacity padding.

The optimistic lower bound is

```text
T_compute = F / Pi
T_mem     = max(L_remote / BW_LSA, L_local / BW_HBM)
T         >= max(T_compute, T_mem) + H_sync + H_reduce
```

For a fused candidate, `F = 2*P*M*K*N` and
`L_remote = (P-1)*M*K*S*rho_N`. `L_local` is a compulsory-byte lower bound:
local A, one copy of B, one output write, and the extra output read/write
boundaries of serial split-K. It does not claim that caches deliver perfect
reuse. The host baseline stages unique remote A first, so its accounting floor
is `T_remote + max(T_compute, T_HBM)` rather than a fused overlap claim.

## Four-GPU result

The ledger covers 346 candidates and 41 complete shapes. The full oracle is
host=3, direct=5, split-K 2=14, split-K 4=14, split-K 8=4, and grouped-N
split-K 8=1.

- The optimistic bottleneck is remote traffic for 39/41 oracle shapes and
  compute for 2/41.
- The median oracle latency is 0.4568 ms, while the median critical-path floor
  is 0.01148 ms.
- The median floor explains only 3.29% of measured latency; the median
  unmodeled remainder is 0.4109 ms.
- The A100 balance points used by this calculation are 1138.7 FLOP/B for LSA
  and 153.0 FLOP/B for HBM.

Therefore "communication-bound" has a precise but limited meaning here. Remote
bytes determine the asymptotic scaling slope. At the measured finite batches,
fixed launch/device synchronization, CTA underfill, short-kernel efficiency,
and serial split reduction dominate the distance from the roofline. The next
run must test both the slope and the overhead regime; a peak-bandwidth quotient
alone is not a latency predictor.

For the measured `P=4, M=64, K=8192, N=1024` boundary:

| family | rho_N | Gamma | remote MiB/rank | local MiB/rank | floor ms | measured ms |
|---|---:|---:|---:|---:|---:|---:|
| host | 1 unique | 32 | 3 | 23.5 | 0.0254 | 0.9441 |
| split-K 4 | 8 | 32 | 24 | 20.5 | 0.0918 | 0.7640 |
| grouped-N split-K 8 | 4 | 32 | 12 | 24.5 | 0.0459 | 0.7599 |

The last two candidates have the same CTA supply. Grouping halves replay, yet
their measured latencies differ by only 0.5%. That gap is direct evidence that
the present point is not yet a saturated LSA-bandwidth experiment.

## Eight-GPU questions

The existing frozen matrix already contains the required controls, so no new
shape axis is added. It projects 253 candidates over 23 shapes: 228 candidate
floors are remote, 14 HBM, and 11 compute.

1. **Fixed per-peer rows:** keep `M` fixed. Per-rank work doubles from P4 to
   P8, while unique remote bytes grow by 7/3.
2. **Fixed per-rank routed rows:** halve `M`, keeping `P*M` fixed. Compute is
   fixed while unique remote bytes grow by 7/6.
3. **Supply control:** compare schedules with equal `Gamma` but different
   `(g,q)` to isolate replay from CTA supply.
4. **Admission boundary:** repeat the three P4 host-win regions and report host
   versus the complete fused oracle.
5. **Breakdown:** on the registered profiling subset, retain kernel count,
   per-kernel time, NVLink/LSA traffic when available, DRAM traffic, and achieved
   occupancy. These observations refine `H_sync` and `H_reduce`; they do not
   create new top-level terms.

At `M=64, K=8192, N=1024`, grouped-N split-K 8 projects from a 0.0459 ms P4
remote floor to 0.1072 ms at P8. Holding per-rank routed rows fixed with `M=32`
projects 0.0536 ms. These are the two falsifiable scaling targets, not predicted
end-to-end latencies.

## Operator scope

The next run remains All2All-to-GEMM only. Bias, gated activation, or another
GEMM epilogue may later increase `F` while removing intermediate local bytes.
RMSNorm adds reduction/synchronization behavior. Attention and KV-cache access
change `L_local` (or `L_remote` if sharded). Router packing and metadata change
fixed overhead. None is admitted as a new schedule axis until the current
communication/supply boundary transfers to P8; when added, it enters the same
`F`, `L_remote`, `L_local`, `H_sync`, and `H_reduce` decomposition.

## Artifacts

- `candidate_accounting.csv`: every measured P4 candidate.
- `shape_oracles.csv`: one measured oracle row per P4 shape.
- `projection_accounting.csv`: lower-bound terms for the frozen P8 matrix.
- `summary.json`: hashes, ceilings, counts, and aggregate results.

Regenerate with `EDM-IR/tools/build_numerical_ledger.py`; the input paths and
their aggregate SHA-256 are recorded in `summary.json`.
