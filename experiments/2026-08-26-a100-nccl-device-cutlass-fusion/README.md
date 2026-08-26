# A100 NCCL Device API + CUTLASS fusion boundary

## Question

Can Edmir replace the scalar LSA correctness kernel with a real CUTLASS
tensor-core kernel without splitting communication and GEMM into two launches,
and where does direct remote-A execution beat the host-NCCL baseline?

## Implementation

NCCL 2.28 exposes `ncclGetLsaPointer` only in device code. The runtime therefore
does not derive or copy an internal NCCL pointer on the host. Instead, a custom
CUDA wrapper:

1. receives the registered `ncclWindow_t` and ordinary CUTLASS kernel params;
2. resolves the source A pointer with `ncclGetLsaPointer` in device code;
3. resets only `GemmKernel::Params::ref_A`;
4. invokes CUTLASS's own predicated SM80 tensor-core mainloop and epilogue.

The generated Program still contains one `fused_transfer_gemm` per tile and no
Program-visible staging buffer.

## Environment and fixed controls

- same 4x A100-SXM4-80GB server and isolated NCCL 2.28.9 build as the fused
  reference experiment;
- CUTLASS v4.7.0, commit `dcf215a`;
- FP16 row-major, world size 4, chunk rows equal to rows per peer;
- kernel `sm80_64x128x32`, threadblock N = 128;
- host path uses the identical CUTLASS kernel after grouped NCCL send/recv;
- each reported point uses synchronous `execute()`/`synchronize()` timing.

The source problem files are in `problems/`; `results.csv` retains medians and
the exact warmup/iteration counts. `commands.sh` records the server-side build,
lowering, validation, benchmark, and Nsight command shapes; the two transcript
files retain the reported measurements rather than only the derived table.

## Correctness and kernel boundary

The `M8 K64 N64` four-rank Program passed 8,192 non-zero element comparisons
against the CPU interpreter with zero absolute and relative error. A zero-data
smoke run also covered `M64 K4096 N4096` and 8 MiB of generated outputs.

Nsight Systems over three warmups and one measured `M64 K4096 N4096` iteration
reported exactly 64 `lsa_cutlass_gemm` instances: 16 Program ops times four
executions. No separate host-NCCL transfer kernel is part of that fused path.
The baseline trace contains separate NCCL SendRecv and CUTLASS kernels; the
normalized summary is in `nsys-kernel-summary.csv`.

## Result

For `M8 K64 N64`, CUTLASS fusion reduced median runtime from 0.5176 ms to
0.2972 ms (42.6%). This is a low-latency development point, not an LLM-shape
claim.

For `M64 K4096`, the fused path lost at every N:

| N | A replay | host ms | fused ms | fused overhead |
|---:|---:|---:|---:|---:|
| 64 | 1 | 0.5597 | 1.0203 | +82.3% |
| 128 | 1 | 0.5665 | 1.0257 | +81.1% |
| 256 | 2 | 0.5655 | 1.0426 | +84.4% |
| 512 | 4 | 0.5645 | 1.0389 | +84.0% |
| 1024 | 8 | 0.5659 | 1.0514 | +85.8% |
| 2048 | 16 | 0.5650 | 1.0725 | +89.8% |
| 4096 | 32 | 0.5891 | 1.3770 | +133.7% |

Each non-local A tile is 512 KiB and total unique remote input is 6 MiB. The
standard CUTLASS grid reloads A once per N tile, reaching an estimated 192 MiB
of remote A load traffic at N=4096. More importantly, fusion already loses at
replay one: SM-initiated LSA reads of a large A tile do not behave like a copy
engine moving A once to local HBM.

That negative result was not an LSA bandwidth verdict. At `N=64`, the direct
kernel has only one CTA per fused op, so one CTA must pull the whole 512 KiB A
tile. A separately identified CUTLASS serial split-K=8 variant holds remote
bytes and replay constant while raising concurrency to eight CTAs per op. Its
median was 0.4937 ms: 51.6% below the one-CTA fused kernel and 11.8% below the
0.5597 ms host baseline. A non-zero `M8 K256 N64` validation matched 8,192 CPU
reference elements with zero absolute and relative error, covering the split-K
semaphore/reduction path.

The serial split-K workspace is zero-initialized before first use. CUTLASS's
final partition resets each semaphore after reduction, so repeated executions
on the same per-destination stream do not require a host-side reset between
iterations.

## Compiler consequence

Direct LSA+CUTLASS is a legal kernel family, not a default optimization. Fused
admission must condition on at least:

- unique remote A bytes;
- N-tile replay factor;
- launch/event savings;
- CTA count and split-K concurrency;
- measured LSA-vs-copy-engine behavior for the exact topology.

The current simulator continues to reject fused costs. The next schedule search
must jointly vary split-K and the number of N tiles grouped per persistent CTA:
split-K restores concurrency without increasing total A bytes, while grouped-N
reuse reduces A replay but also reduces CTA-level parallelism.

Official references:

- [NCCL 2.28 Device API guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/usage/deviceapi.html)
- [NCCL Device API examples](https://github.com/NVIDIA/nccl/tree/master/docs/examples/06_device_api)
- [CUTLASS GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api.md)
