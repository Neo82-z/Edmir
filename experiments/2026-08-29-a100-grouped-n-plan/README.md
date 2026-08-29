# A100 grouped-N remote-A reuse plan

Status: **design only; all latency and correctness result cells remain empty**.

## Question

For fused LSA + GEMM, which pair

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
serial split-K reduction cost. Neither axis is admitted by a latency formula.

## Candidate matrix

- host NCCL + identical local CUTLASS GEMM;
- direct LSA + CUTLASS: `(g=1, q=1)`;
- current split-K family: `(g=1, q in {2,4,8,16})`;
- grouped-N direct: `(g in {2,4}, q=1)`;
- grouped-N plus split-K: `(g in {2,4}, q in {2,4,8,16})`.

The current source implements `g=2` by doubling the CUTLASS N-threadblock;
`g=4` remains analysis-only. Neither grouped family is profile-admitted before
the new source compiles and passes the protocol below on the pinned A100 stack.

## Shape grid

Use four A100 SXM GPUs and retain exact topology/software fingerprints.

```text
M = rows_per_peer in {32, 64, 128}
K in {256, 512, 1024, 2048, 4096}
N in {128, 256, 512, 1024}
dtype = FP16 first; BF16 is a separate replication
```

Planning command:

```bash
python tools/plan_grouped_n_sweep.py \
  --problem INPUT_PROBLEM.json \
  --topology INPUT_TOPOLOGY.json \
  --kernels sm80_64x128x32,sm80_64x128x32_splitk2,sm80_64x128x32_splitk4,sm80_64x128x32_splitk8,sm80_64x128x32_splitk16 \
  --n-tiles-per-cta 1,2,4 \
  --out results/schedule-space.json
```

After the source build passes, the executable smoke matrix adds
`--grouped-n-values 2` to `tools/run_fused_kernel_sweep.py`.

The first server invocation is deliberately small and correctness-first:

```bash
python tools/run_fused_kernel_sweep.py \
  --runtime runtime \
  --topology INPUT_TOPOLOGY.json \
  --cutlass /opt/cutlass \
  --nccl-home /opt/nccl-device \
  --out results/g2-smoke \
  --rows-per-peer 64 --n 256 --k-values 256 \
  --split-k-values 2 --grouped-n-values 2 \
  --warmup 2 --iterations 5 --validate-first
```

Only after all five host/direct/split-K/grouped candidates build and validate
does the full grid begin.

## Measurement protocol

1. Run numerical validation before timing every new `(g,q)` family.
2. Use identical generated inputs and weights for host and fused candidates.
3. Warm up 30 iterations and record at least 300 measured iterations.
4. Retain min/median/p95/max/mean, CUDA/NCCL/CUTLASS commits, clocks, power
   state, topology, and the generated target/program artifacts.
5. Train a selection rule on the designated M64 subset; use the other M/N
   pairs only for held-out direction accuracy and regret.
6. Admit no grouped-N point unless all exact-shape numerical checks pass,
   held-out direction accuracy is at least 99%, maximum selection regret is at
   most 1%, and host has no unmodeled win inside the claimed envelope.

## Result table (intentionally empty)

| M | K | N | g | q | rho_N | CTA/op | median ms | p95 ms | max abs error | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |

## Stop conditions before the timed sweep

- grouped-N kernel compiles in the pinned A100 runtime during a short build-only gate;
- C++ descriptor carries the schedule contract end to end;
- invalid `g`, shape alignment, and unsupported runtime combinations fail
  before launch;
- CPU-only tests and host C++ validation are green;
- the runner emits one machine-readable row per candidate and never overwrites
  retained results.
