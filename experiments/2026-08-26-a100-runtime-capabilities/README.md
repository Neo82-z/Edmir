# A100 fused-runtime capability gate (2026-08-26)

This slice answers one question: can the current 4x A100 node execute the
`fused_transfer_gemm` Program contract using the installed NCCL Device API?

## Command

The repository was synchronized to `/root/llm-serving/EDM-IR`, then probed
without changing the driver, CUDA, NCCL, clocks, or topology:

```bash
cd /root/llm-serving/EDM-IR
PYTHONPATH=. python3 -m edmir probe-runtime \
  --out /root/edmir-runtime-capabilities-20260826.json
```

The probe records GPU identities, CUDA compiler, NCCL headers/version/symbols,
an optional header compilation check, and the read/write P2P plus transport
topology.

## Result

| gate | observed | status |
|---|---|---:|
| GPUs | 4x NVIDIA A100-SXM4-80GB, compute capability 8.0 | pass |
| P2P | complete 4x4 read/write matrix | pass |
| links | NV12 between every GPU pair | pass |
| CUDA | 12.8, nvcc 12.8.93 | pass |
| NCCL | 2.25.1 (`NCCL_VERSION_CODE=22501`) | fail |
| `nccl_device.h` | absent | fail |
| Window/DevComm/LSA symbols | all absent | fail |

`ncclMemAlloc` is present, but symmetric allocation alone is not the Device
API. NVIDIA documents device-initiated communication as beginning with NCCL
2.28 and requires a device communicator plus a registered symmetric window;
NVIDIA's current examples include both `nccl.h` and `nccl_device.h`.

Primary references:

- [NCCL 2.28 device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2289/user-guide/docs/usage/deviceapi.html)
- [NCCL Device API overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html)
- [Official NCCL Device API examples](https://github.com/NVIDIA/nccl/tree/master/docs/examples/06_device_api)

## Decision

The fused runtime gate is closed by the installed NCCL development/runtime
stack, not by A100 peer connectivity. Edmir must keep selecting the measured
host-NCCL plus CUTLASS/cuBLAS path on this node. The legality-checked fused JSON
prototype remains useful, but extending the C++ ABI or assigning it an
unmeasured cost here would create a false executable claim.

The next fused experiment requires NCCL 2.28 or newer with exported
`nccl_device.h`, successful CUDA header compilation, and the same topology
probe. `runtime-capabilities.json` is the complete machine-readable record.
