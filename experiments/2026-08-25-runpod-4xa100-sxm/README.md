# RunPod 4x A100 SXM: EDM-IR backend validation

Date: 2026-08-25 (Asia/Shanghai)

This is a time-bounded, single-node validation of EDM-IR's CUDA/NCCL/cuBLAS
backend.  The raw logs, compiler inputs, generated program, parsed tables, and
numerical checks are retained so that the conclusions do not depend on the
RunPod instance remaining available.

## Questions answered

1. Is the rented machine a coherent four-GPU SXM/NVLink node?
2. Does EDM-IR discover the topology and calibrate its communication model?
3. Can a lowered program build and run through the CUDA/NCCL/cuBLAS runtime?
4. Does the generated GPU result agree with a CPU interpreter on non-zero data?
5. Does the compiler's selected schedule beat the sequential baseline?

## Environment

| Item | Observed value |
| --- | --- |
| GPU | 4x NVIDIA A100-SXM4-80GB |
| GPU topology | Full mesh `NV12` between every pair |
| CUDA driver/runtime | Driver 580.126.16; driver API 13.0; toolkit 12.8.93 |
| PyTorch | 2.8.0+cu128 |
| NCCL | 2.25.1+cuda12.8 |
| Host | Ubuntu 24.04.3, 2x AMD EPYC 7742, about 2 TiB RAM |

The complete inventory is in `results/00_environment.txt` and
`results/72_extended_inventory.txt`.  The corrected machine-readable topology
is `results/a100_sxm_topology_fixed.json`.

## Main measurements

### CUDA P2P

CUDA's `p2pBandwidthLatencyTest` reports:

- unidirectional P2P bandwidth: approximately 270--276 GB/s for every pair;
- bidirectional P2P bandwidth: approximately 516--519 GB/s;
- P2P GPU latency: approximately 2.9--3.1 us;
- with P2P disabled: approximately 15--20 GB/s and 24.5 us.

All peer reads, writes, atomics, and twelve NVLinks per GPU were reported as
supported.  See `results/25_p2p_bandwidth_latency.txt`.

### NCCL

`nccl-tests` swept 1 KiB through 512 MiB (factor 2, five warmups, twenty
iterations).  The test matrix contains AllReduce, AllGather, and AllToAll on
two, three, and four GPUs, plus SendRecv for all six GPU pairs.  Every row has
`#wrong = 0`.

At 512 MiB on four GPUs:

| Operation | Time (us) | Algorithm BW (GB/s) | Bus BW (GB/s) |
| --- | ---: | ---: | ---: |
| AllReduce | 3868.08 | 138.80 | 208.19 |
| AllGather | 2282.95 | 235.17 | 176.37 |
| AllToAll | 2090.28 | 256.84 | 192.63 |

The six pairwise 512 MiB SendRecv results are tightly grouped at
203.70--206.03 GB/s.  Parsed curves are in `derived/nccl_curves.csv`; the raw
files are `results/30_*.txt` and `results/pair_*_default_sendrecv.txt`.

### GEMM calibration probe

The FP16 probe uses `[m, 4096] x [4096, 4096]` on each GPU.  Representative
results range from about 27.7 us / 19.4 TFLOP/s at `m=16` to about 38.4 us /
111.7 TFLOP/s at `m=128`.  The four GPUs are symmetric.  See
`results/73_gemm_microbench.txt`.

## Compiler/runtime validation

The original 39 Python tests passed on the node.  After adding the topology
regression and schedule tie-break tests, the final suite passes 43/43.
The C++ runtime builds with CUDA 12.8 and passes its host-side CTest.

Two independent correctness checks passed:

- a four-GPU generated FP16 smoke test checked 8 MiB of output;
- a deterministic non-zero FP32 problem compared 8192 output elements with a
  CPU interpreter and obtained `max_abs_error=0` and `max_rel_error=0`.

The second check is intentionally small and exact.  It verifies lowering,
buffer initialization, communication order, GEMM dispatch, synchronization,
and result collection without relying on an all-zero input.

## Topology bug found during the run

The original Linux probe used `nvidia-smi topo -p2p p`.  That mode reports
PCIe P2P capability, so it returned `NS` even though NVLink reads and writes
were available.  EDM-IR consequently serialized this full-NVLink node as a
diagonal-only P2P topology and disabled the intended pipeline search.

The probe now combines `nvidia-smi topo -p2p r` and `-p2p w`, and obtains link
labels such as `NV12` from `nvidia-smi topo -m`.  Three regression tests cover
the distinction.  Both the pre-fix and corrected JSON files are retained for
auditability.

## Schedule selection

The measured NCCL profile and corrected topology were used to compare one
sequential schedule with nine pipelined candidates.  Each runtime measurement
uses five warmups and thirty timed iterations.

| Strategy | Chunk rows | Buffers | Ops | Median (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 64 | 4 | 36 | 0.6322 |
| Pipelined | 16 | 1 | 128 | 2.9010 |
| Pipelined | 16 | 2 | 128 | 2.9737 |
| Pipelined | 16 | 4 | 128 | 2.9872 |
| Pipelined | 32 | 1 | 64 | 1.3430 |
| Pipelined | 32 | 2 | 64 | 1.3645 |
| Pipelined | 32 | 4 | 64 | 1.3129 |
| Pipelined | 64 | 1 | 32 | 0.6644 |
| Pipelined | 64 | 2 | 32 | 0.6422 |
| **Pipelined** | **64** | **4** | **32** | **0.5812** |

The selected schedule is about 8.1% faster than the sequential median in this
sweep.  A separate final run measured a 0.5855 ms median.  Repeated independent
runs put sequential at 0.6206--0.6331 ms and the selected candidate at
0.5831--0.5905 ms.

The experiment also exposes a model limitation: 16- and 32-row chunks look
attractive if host launch and event costs are omitted, but are much slower in
the runtime.  Equal predicted makespans are now resolved by preferring fewer
operations and then more buffers, both within the user-bounded search space.
This is a conservative tie-break, not a claim that the current cost model has
learned launch overhead.

## Reproduce the derived tables

From this directory:

```bash
python3 summarize_results.py
```

This regenerates:

- `derived/nccl_curves.csv`: all 360 parsed NCCL rows;
- `derived/benchmark_summary.csv`: all 36 captured benchmark rows;
- `derived/summary.json`: topology, selected schedule, and largest-message
  summary.

Core EDM-IR checks from the repository root:

```bash
python3 -m unittest discover -s EDM-IR/tests -p '*_test.py'
cmake -S EDM-IR/runtime -B /tmp/edmir-runtime \
  -DEDMIR_GENERATED_DIR="$PWD/EDM-IR/generated"
cmake --build /tmp/edmir-runtime -j
ctest --test-dir /tmp/edmir-runtime --output-on-failure
```

The exact selected compiler output is preserved as
`results/71_selected_program.json`, `results/71_selected_target.json`, and
`results/71_edmir_program.hpp`.

## Limitations

- This is one single-node sample, not a multi-node or provider comparison.
- Nsight Systems was absent.  Nsight Compute was installed, but RunPod's host
  disabled GPU performance counters (`ERR_NVGPUCTRPERM`), so kernel-counter
  attribution is not available.
- The generated communication profile contains measured NCCL curves, but its
  current scalar GEMM and synchronization terms remain inherited from the
  seed profile.  Shape-specific GEMM measurements are retained separately for
  the next calibration-model revision.
- RDMA devices were visible but were not evaluated; they are irrelevant to
  the single-node conclusion here.
- The speedup applies to this small A2A+GEMM validation problem and should not
  be generalized to end-to-end model serving without a workload study.

## Artifact map

- `results/00_*`--`25_*`: environment, builds, correctness, and CUDA P2P.
- `results/30_*` and `results/pair_*`: raw NCCL sweeps.
- `results/40_*`--`61_*`: profiling, initial selection, bug discovery, and
  corrected selection.
- `results/62_*`--`74_*`: repeated benchmarks, final tests, generated code,
  profiler limitation, inventory, and GEMM probe.
- `results/a100_sxm_measured_profile.json`: measured communication profile.
- `derived/`: compact machine-readable tables suitable for analysis/plots.
- `metadata.json`: provider, time window, and cost ceiling without credentials.
- `SHA256SUMS`: integrity manifest for the retained artifacts.
