# Four-A100 split-K supply boundary

## Question

With remote-A bytes and N-tile replay held fixed, how many serial split-K CTAs
are needed to hide LSA peer-load latency, and does a rule frozen on one
`(M,N)` pair retain low selection regret on unseen shapes?

This experiment extends the fused CUTLASS family from the earlier fixed
`splitK=8` implementation to explicit `q in {2,4,8,16}` identities.  The
extension changes CTA supply and partial-output reduction only; it does not
change `remote_input_bytes`, `remote_a_replay`, or
`estimated_remote_a_load_bytes`.

## Fixed environment

- four NVIDIA A100-SXM4-80GB GPUs, full pairwise `NV12`;
- driver 570.195.03 and CUDA 12.8.93;
- isolated NCCL `v2.28.9-1`, commit `dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`;
- CUTLASS `v4.7.0`, commit `dcf215af68a2d08d305076c152a06f201728cd53`;
- Edmir base commit `d5db18c`, plus `raw/experiment.patch`;
- FP16, row-major, four ranks, 30 warmups and 300 synchronous timed
  executions per candidate.

RunPod did not permit application clock locking or hardware performance
counter access.  The failed lock and Nsight Compute probes are retained in the
raw logs.  Family selection therefore uses unprofiled medians and independent
process repeats, not profiler timings or p95 values.

## Protocol

The training sweep fixes `(M,N)=(64,64)`, evaluates
`K in {512,1024,2048,4096,8192}`, and compares host NCCL+CUTLASS, direct LSA,
and split-K `q in {2,4,8,16}`.  Four held-out pairs repeat the complete grid:

```text
(32,64), (128,64), (64,256), (128,256)
```

There are 150 measured points.  At `K=512`, all six families were checked
against the non-zero CPU reference for both the training shape and one
held-out shape.  All 12 validations report zero maximum absolute and relative
error.  Python tests, the host C++ ABI test, and all five CUDA build/CTest
passes also succeed.

## Training boundary

| K | Frozen family | Median (ms) | Host (ms) | Host / selected | Reduction |
|---:|:---|---:|---:|---:|---:|
| 512 | splitK2 | 0.2976 | 0.5542 | 1.862x | 46.30% |
| 1024 | splitK4 | 0.3329 | 0.5652 | 1.698x | 41.10% |
| 2048 | splitK4 | 0.3845 | 0.5653 | 1.470x | 31.98% |
| 4096 | splitK4 | 0.4901 | 0.6338 | 1.293x | 22.67% |
| 8192 | splitK8 | 0.6142 | 0.8246 | 1.343x | 25.52% |

The optimum moves upward with K but is not monotone in the simplistic sense
that more CTAs are always better.  `q=16` never wins: reduction and semaphore
overhead dominate before its extra supply becomes useful.  The direct family
also stops winning much earlier than the previous direct-versus-splitK8 study
suggested because `q=2` and `q=4` fill the missing middle of the schedule
space.

## Held-out selection

The exact family learned on `(64,64)` matches the held-out finite-search oracle
on 14 of 20 problems (70%).  Exact-family accuracy understates the useful
result because every mismatch is near a boundary:

- mean selection regret: **0.43%**;
- maximum selection regret: **2.47%**;
- selected schedule beats host: **20/20** problems;
- mean `host / selected`: **1.535x**.

This is the desired selector behavior: preserve direction and low regret
without pretending that a discrete family label is stable at a near tie.

## Independent-process repeats

Five fresh processes were run for each near-boundary pair.  At `K=4096`,
splitK4 has a median of process medians of 0.4876 ms versus 0.4931 ms for
splitK8.  At `K=1024`, splitK2 and splitK4 are effectively tied at 0.3342 and
0.3333 ms.  At `K=512`, splitK2 remains slightly ahead of direct at 0.3059
versus 0.3081 ms.

## Artifact map

- `raw/results/q-sweep-*/results.csv`: authoritative 150 measurements;
- `raw/results/heldout-summary/heldout_decisions.csv`: frozen decisions and
  per-problem regret;
- `raw/results/boundary-repeats/`: independent-process summaries;
- `raw/results/runtime-capabilities.json`: NCCL Device API and P2P gate;
- `raw/logs/`: environment, dependency, test, clock-lock, profiler, and repeat
  logs;
- `raw/SHA256SUMS`: checksums for every extracted artifact;
- `edmir-a100-splitk-q-20260829.tar.gz`: transfer archive, SHA-256
  `e9a0ff7a080e4206d11b688dd3f8e4799beafd84dbd46d21414399df1b5ad8a6`.

## Consequence

The compiler should expose split count as a schedule decision, not encode
`splitK=8` as a Boolean alternative to direct fusion.  The next mechanism
experiment is grouped/persistent N-tile reuse, followed by BF16 replication;
neither claim is made by this FP16 run.
