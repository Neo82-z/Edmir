# A100 fused-kernel K/traffic boundary

## Question

With N-tile replay fixed at one, when should Edmir use a one-CTA direct LSA
CUTLASS kernel, an eight-CTA serial split-K kernel, or the host NCCL + CUTLASS
baseline?

This is a schedule-boundary experiment. It does not admit fused kernels into
the default compiler cost model.

## Controls

- 4x A100-SXM4-80GB, full pairwise `NV12` topology;
- CUDA 12.8, isolated NCCL 2.28.9, CUTLASS 4.7.0;
- FP16 row-major, world size 4, `M=64` rows per peer, `N=64`;
- K in `{256, 512, 1024, 2048, 4096}`;
- identical `sm80_64x128x32` tensor-core mainloop for host and direct paths;
- `sm80_64x128x32_splitk8` changes only the fused K partition/concurrency;
- 30 warmups and 300 synchronous measured executions per point.

`EDM-IR/tools/run_fused_kernel_sweep.py` generated every Program artifact,
rebuilt the benchmark against that header, checked the reported kernel identity,
and checkpointed every row. The first K point for all three families was also
compared against the CPU interpreter: each checked 65,536 non-zero output
elements with zero maximum absolute and relative error.

## Results

| K | unique remote A | host ms | direct ms | splitK8 ms | best |
|---:|---:|---:|---:|---:|:---|
| 256 | 384 KiB | 0.5766 | **0.2983** | 0.3970 | direct |
| 512 | 768 KiB | 0.5803 | **0.3120** | 0.4005 | direct |
| 1024 | 1.5 MiB | 0.5815 | **0.3918** | 0.4195 | direct |
| 2048 | 3 MiB | 0.5853 | 0.5982 | **0.4445** | splitK8 |
| 4096 | 6 MiB | 0.6339 | 1.0227 | **0.4914** | splitK8 |

At K=256, direct fusion is 48.3% below the host baseline and split-K is slower
than direct because its extra partition/reduction work is not yet repaid. At
K=2048, the one-CTA direct kernel reaches parity with host, while split-K is
24.1% below host. At K=4096, split-K is 22.5% below host and 51.9% below direct.

The sparse sweep locates the observed transition only between K=1024 and
K=2048, equivalently between 1.5 and 3 MiB of unique non-local A input. It does
not justify interpolating an exact threshold or extrapolating to a different M,
N, topology, CUDA/NCCL version, or kernel tile.

## Compiler consequence

The measured winner is piecewise even when replay is fixed:

```text
small remote A  -> direct LSA, avoid split-K reduction overhead
large remote A  -> split-K, expose enough CTAs to cover remote-load latency
large N/replay  -> neither result is sufficient; grouped-N reuse is a new axis
```

Edmir therefore keeps `direct` and `splitk8` as separate kernel identities and
exports remote bytes, replay, split count, and CTA count as cost-model features.
Automatic fused selection remains closed until held-out M/N shapes reproduce
the direction of these choices.

The authoritative measurements are `results.csv`; `manifest.json`, all 15
generated headers, build logs, benchmark logs, and the three correctness logs
are retained alongside it.
