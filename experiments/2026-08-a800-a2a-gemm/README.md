# 4xA800 static All2All -> GEMM evaluation

Status: **design only; no performance result has been collected**.

这个实验只评价 EDM-IR v0.1 的静态、等长、单节点 `All2All -> GEMM` case。所有 `TBD` 必须由归档命令、环境快照和 raw log 填写；`EDM-IR/examples/` 中的 synthetic profile 不得进入实测结果表。

## Research questions

- **RQ1 — metric validity:** 反事实暴露量是否比总通信时长和单条诊断路径上的通信时长更好地预测一次变换的实际收益？
- **RQ2 — compiler correctness and selection:** verifier 是否拒绝非法计划，校准模拟器是否能选择接近有限搜索 oracle 的合法计划？
- **RQ3 — performance regimes:** 在 4xA800 的实际拓扑上，哪些 shape、chunk 和 buffer 配置使流水 helpful、neutral 或 harmful？

## Platform record

| Field | Value |
|---|---|
| GPUs | 4x NVIDIA A800 |
| GPU PCI bus IDs | TBD |
| `nvidia-smi topo -m` | TBD: archive under `environment/` |
| `nvidia-smi topo -p2p p` | TBD: archive under `environment/` |
| NVLink/PCIe fabric | TBD: infer only from captured topology and counters |
| CPU/NUMA placement | TBD |
| Driver / CUDA / NCCL | TBD |
| PyTorch / Triton / Syncopate commits | TBD / not applicable |
| Clock and power policy | TBD |

Before any timed run, create an actual topology artifact:

```bash
cd EDM-IR
python -m edmir probe \
  --out ../experiments/2026-08-a800-a2a-gemm/environment/topology.json
```

## Workload matrix

Primary sweep:

| Dimension | Values |
|---|---|
| world size | 4 |
| dtype | FP16, BF16 |
| GEMM `K`, `N` | 4096, 8192 |
| rows per source-destination block `m` | 16, 64, 256, 1024 |
| chunk rows | 16, 32, 64, 128, full block; only divisors of `m` |
| staging buffers | 1, 2, 4 |

Calibrate on the same rank group and clock policy used by the end-to-end run:

- communication: message size, median, p95, effective bandwidth;
- GEMM: each `(m, n, k, dtype)` used by the sweep, including chunked `m`;
- overhead: launch, synchronization, program/runtime bookkeeping where observable.

## Baselines

1. **Sequential:** NCCL All2All followed by the matching GEMM.
2. **Fixed pipeline:** one preregistered legal chunk/buffer choice.
3. **EDM-IR selected:** lowest predicted legal candidate above the configured gain threshold.
4. **Oracle:** fastest measured candidate in the finite sweep; used only to measure regret.
5. **Syncopate:** only if its public path can express identical tensors, dtype, and consumer semantics; otherwise report `N/A` with the incompatibility.

## Protocol

1. Archive topology, clocks, software versions, git revisions, and the exact device ordering.
2. Collect calibration curves and fit a versioned `edmir.profile.v1`; never fit on end-to-end test observations that will be reported as held-out evaluation.
3. Compile every problem with the same candidate grid and archive all Programs, including rejected candidates and fallback reasons.
4. Run verifier and numerical equivalence before timing a GPU Program.
5. Use warmups and repeated timed trials; record the count, seed, median, p95, and raw samples.
6. Randomize candidate execution order where practical, and periodically rerun the sequential control to detect drift.
7. Classify helpful/neutral/harmful using a threshold fixed before reading the results; record that threshold here: **TBD**.

## Blank result tables

### RQ1: metric validity

| Predictor | Direction accuracy | Spearman rho | Rank error |
|---|---:|---:|---:|
| Total communication | TBD | TBD | TBD |
| Selected-path communication | TBD | TBD | TBD |
| Counterfactual ECR / EDMR | TBD | TBD | TBD |

Required synthetic sanity cases: uniquely exposed communication; communication hidden behind longer compute; tied critical paths with communication on only one path; and a pipeline that removes exposure but adds greater overhead.

### RQ2: selection

| Regime | Predicted choice | Measured best | Regret | Numerically correct |
|---|---|---|---:|---|
| Small-`m` | TBD | TBD | TBD | TBD |
| Balanced | TBD | TBD | TBD | TBD |
| Compute-heavy | TBD | TBD | TBD | TBD |

### RQ3: end-to-end

| Strategy | Median latency | p95 | Speedup | Max error | ECR | EDMR |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | TBD | TBD | 1.00x | TBD | TBD | TBD |
| Fixed pipeline | TBD | TBD | TBD | TBD | TBD | TBD |
| EDM-IR selected | TBD | TBD | TBD | TBD | TBD | TBD |
| Syncopate, if matched | TBD / N/A | TBD / N/A | TBD / N/A | TBD / N/A | TBD / N/A | TBD / N/A |

## Required artifact layout

```text
environment/  topology and version snapshots
calibration/  raw communication/GEMM samples and fitted profile
programs/     compiler outputs, verifier reports, fallback reasons
raw/          per-trial logs; never hand-edit
results/      derived CSV/JSON tables and plotting inputs
commands/     exact runnable scripts or command transcripts
```

Do not generalize this experiment to dynamic MoE routing, other GPU generations/fabrics, or multiple nodes without separate evidence.
