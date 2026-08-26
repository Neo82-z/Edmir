# EDM-IR cost-model validation: ready-to-run design

This experiment is the gate before adding CUTLASS/CuTe or NCCL Device API
backends. It tests whether the existing host-launched NCCL + cuBLAS backend can
select schedules using measured costs rather than a tie-break heuristic.

## What changed in the compiler

The cost model now has two explicit layers:

1. **Device layer:** directed NCCL message-size curves, local D2D-copy curves,
   and dtype/shape-specific GEMM latency curves. GEMM samples are interpolated only along `m` for an
   identical `(dtype, n, k)` family; unmatched families use the analytic
   fallback.
2. **Runtime layer:** a non-negative residual over five observable features:
   iteration constant, Program op dispatches, CUDA event records, CUDA event
   waits, and buffer-reuse dependencies. The feature counts match the current
   C++ runtime implementation and are printed by its benchmark.

Runtime coefficients are fitted only from `train` rows. `heldout` problems are
evaluated once, without refitting. The report contains MAPE, direction accuracy
against the sequential baseline, and schedule-selection regret.

The archived 2026-08-25 A100 sweep is useful only as a retrospective sanity
check: after adding its measured GEMM shapes and fitting all ten rows, training
MAPE falls from 67.2% to about 7.85% and the measured-best candidate is selected.
Because all ten rows belong to one problem and were used for fitting, this is
not evidence of generalization.

## Fixed experiment matrix

`design.json` generates 50 Programs:

| Split | Problem families | Candidates |
| --- | --- | ---: |
| Train | `r=32/64/128, k=n=4096` | 27 |
| Held out | `r=96, k=n=4096`; `r=64, k=2048, n=4096` | 23 |

Each family includes sequential plus legal pipelined schedules over 1, 2, and 4
staging buffers. Nine exact cuBLAS GEMM shapes and every local-copy size are measured. The runtime sweep
defaults to five warmups and thirty measured iterations per Program and writes
each completed row with `fsync`, so an interrupted hourly instance can resume.

## Before starting the paid machine

The local, no-GPU gate is:

```bash
cd /path/to/llm-serving
python3 -m unittest discover -s EDM-IR/tests -p '*_test.py'

export PYTHONPATH="$PWD/EDM-IR"
python3 EDM-IR/tools/prepare_cost_model_experiment.py \
  --design experiments/2026-08-cost-model-validation/design.json \
  --profile experiments/2026-08-25-runpod-4xa100-sxm/results/a100_sxm_measured_profile.json \
  --topology experiments/2026-08-25-runpod-4xa100-sxm/results/a100_sxm_topology_fixed.json \
  --arch sm_80 \
  --out /tmp/edmir-cost-model-prepared

python3 EDM-IR/tools/run_cost_model_experiment.py \
  --prepared /tmp/edmir-cost-model-prepared \
  --base-profile experiments/2026-08-25-runpod-4xa100-sxm/results/a100_sxm_measured_profile.json \
  --runtime EDM-IR/runtime \
  --out /tmp/edmir-cost-model-dry-run \
  --dry-run
```

Expected: 49 tests pass, 50 candidates are materialized, and the dry-run says
`gpu_not_touched=true`.

## Paid-machine sequence

Use one 4x A100 SXM 80 GB node. Do not use A100 PCIe, mixed GPUs, MIG, or a
partial allocation. First capture the actual topology:

```bash
cd /workspace/llm-serving
export PYTHONPATH="$PWD/EDM-IR"
mkdir -p /workspace/edmir-cost-model/input
python3 -m edmir probe \
  --out /workspace/edmir-cost-model/input/topology.json
python3 - <<'PY'
import json
p=json.load(open('/workspace/edmir-cost-model/input/topology.json'))
assert all(i == j or p['p2p'][i][j] for i in range(4) for j in range(4))
assert all(label == 'NV12' for i,row in enumerate(p['link_labels']) for j,label in enumerate(row) if i != j)
print('topology_gate=pass')
PY
```

Re-measure all six directed-pair proxies on the actual node rather than assuming
the archived RunPod curve applies:

```bash
bash experiments/2026-08-cost-model-validation/run_pairwise_nccl.sh \
  /workspace/nccl-tests/build/sendrecv_perf \
  /workspace/edmir-cost-model/input/nccl-pairs

python3 -m edmir fit-nccl \
  --base-profile EDM-IR/examples/synthetic_profile.json \
  --logs /workspace/edmir-cost-model/input/nccl-pairs \
  --out /workspace/edmir-cost-model/input/comm-profile.json
```

Prepare against the captured topology and run the checkpointed sweep:

```bash
python3 EDM-IR/tools/prepare_cost_model_experiment.py \
  --design experiments/2026-08-cost-model-validation/design.json \
  --profile /workspace/edmir-cost-model/input/comm-profile.json \
  --topology /workspace/edmir-cost-model/input/topology.json \
  --arch sm_80 \
  --out /workspace/edmir-cost-model/prepared

python3 EDM-IR/tools/run_cost_model_experiment.py \
  --prepared /workspace/edmir-cost-model/prepared \
  --base-profile /workspace/edmir-cost-model/input/comm-profile.json \
  --runtime EDM-IR/runtime \
  --out /workspace/edmir-cost-model/results \
  --devices 0,1,2,3 --warmup 5 --iterations 30 \
  2>&1 | tee /workspace/edmir-cost-model/run.log
```

Re-running the last command skips candidates already present in
`runtime_measurements.csv`.

## Decision gate

Do not add another backend until all of the following are reported on held-out
problems:

- numerical/runtime tests pass;
- held-out MAPE is at most 15%;
- raw direction accuracy is at least 85%;
- a train-only conservative gain threshold is reported together with policy
  direction accuracy;
- mean schedule-selection regret is at most 5%;
- the selected candidate is repeated in three independent benchmark processes.

If MAPE fails but direction and regret pass, the compiler may proceed as a
ranking model while the paper reports prediction error honestly. If direction
or regret fails, inspect residuals by plan feature before changing the IR or
adding a backend.

## Files to download before stopping the instance

Download the complete `/workspace/edmir-cost-model` directory. The irreplaceable
outputs are `gemm_measurements.jsonl`, `local_copy_measurements.jsonl`, `runtime_measurements.csv`,
`device_profile.json`, `calibrated_profile.json`, `evaluation.json`, all logs,
the captured topology, and the six raw NCCL pair logs. Then verify the archive
locally before stopping the hourly instance.
