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
  --devices 0,1,2,3 --rounds 3 --execute --allow-gpu
```

The first server pass should use the defaults: at most 1 GiB memory use and at
most 10% instantaneous utilization on every requested GPU. A rejection is a
safe outcome; inspect `gpu_admission.reasons` instead of overriding the gate or
disturbing another workload.

The CPU phase runs the Python suite, host runtime tests, SM89 host codegen, and
the expected SM89 fused-lowering rejection. The GPU phase records the observed
topology/runtime capabilities, regenerates the host artifact against that
topology, builds the CUDA runtime, runs its tests, then repeats numerical and
runtime-reuse checks. No result from this directory should enter the paper
without a separately frozen performance protocol.
