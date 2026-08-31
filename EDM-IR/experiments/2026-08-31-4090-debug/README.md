# RTX 4090 debug protocol

This is a portability and correctness protocol, not a paper performance
experiment. It exercises the host NCCL + cuBLAS backend on four SM89 devices
and checks that the SM80 NCCL Device API + CUTLASS path remains fail-closed.

The runner is deliberately non-destructive:

- it writes only a fresh child of `--out-root`;
- it records environment commands and every command outcome in `manifest.json`;
- it never stops a container, kills a process, or deletes an existing file;
- it skips all GPU work unless four requested RTX 4090s have no compute process
  and pass the configured memory/utilization thresholds;
- it repeats the idle check before every GPU command, so a newly arrived job
  stops the remaining debug phases without being killed or disturbed;
- it marks all captured latency as diagnostic-only (`paper_evidence=false`).

Plan the run without touching a GPU:

```bash
python tools/run_4090_debug.py \
  --out-root /data/neo-edmir-debug --run-id planned-sm89
```

Execute three correctness/reuse rounds after the four GPUs are idle:

```bash
python tools/run_4090_debug.py \
  --out-root /data/neo-edmir-debug \
  --run-id sm89-$(date -u +%Y%m%dT%H%M%SZ) \
  --devices 0,1,2,3 --rounds 3 \
  --cuda-toolkit-root /path/to/cuda \
  --nccl-include-dir /path/to/nccl/include \
  --nccl-library /path/to/nccl/lib/libnccl.so.2 \
  --execute --allow-gpu
```

The first server pass should use the defaults: at most 1 GiB memory use and at
most 10% instantaneous utilization on every requested GPU. A rejection is a
safe outcome; inspect `gpu_admission.reasons` instead of overriding the gate or
disturbing another workload.

The CPU phase runs the Python suite, host runtime tests, SM89 host codegen, and
the expected SM89 fused-lowering rejection. When all three explicit toolchain
paths are supplied, it also compiles and links the CUDA host runtime without
opening a GPU context. The GPU phase records the observed
topology/runtime capabilities, regenerates the host artifact against that
topology, builds the CUDA runtime, runs its tests, then repeats numerical and
runtime-reuse checks. No result from this directory should enter the paper
without a separately frozen performance protocol.

## 2026-08-31 server preflight

The protocol was exercised on `ubuntuserver` at compiler commit `0954859`.
This is diagnostic evidence only and remains excluded from the paper.

- Hardware: four RTX 4090 GPUs, driver `580.126.18`.
- Topology: GPU 0/1 and GPU 2/3 share NUMA-local host domains; cross-domain
  links are `SYS`. There are no NVLink edges.
- CUDA P2P: every off-diagonal read and write entry reports `CNS` (chipset not
  supported). This node cannot exercise the A100 LSA path.
- Toolchain: CUDA Toolkit 13.0.48, cuBLAS 13, and the PyTorch environment's
  NCCL 2.27.5 headers/library. NCCL 2.27.5 is sufficient for host NCCL but not
  for the NCCL 2.28 Device API.
- Compiler/runtime: 129 Python tests passed; host C++ tests passed; SM89 host
  code generation passed; the SM80 CUTLASS fused lowering failed closed as
  expected; the generated CUDA host runtime compiled, linked, and passed its
  host test without opening a GPU context.
- Admission: GPU execution was skipped before `topology-probe`. All four GPUs
  exceeded the 1 GiB idle threshold and seven foreign vLLM compute processes
  were present. No process, container, or pre-existing file was changed.

The authoritative server record is
`/data/edmir/results/server-toolchain-gated-0954859-20260831T145800/manifest.json`.
The manifest, key logs, and generated fixture artifact retained in Git are under
[`server-preflight-0954859/`](server-preflight-0954859/); transient CMake build
products are intentionally excluded.
The transferred archive
`server-toolchain-gated-0954859.tar.gz` has SHA-256
`61f4c47bdefc811f4b16e987b5f21dd40e4c815bcbbfacb69cdb866e82cbed37`.
The next run should reuse the same explicit toolchain arguments after the four
GPUs are genuinely free; lowering and build work need not be redesigned.
