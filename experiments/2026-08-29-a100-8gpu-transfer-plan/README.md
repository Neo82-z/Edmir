# Eight-A100 transfer protocol

Status: **frozen before observing any eight-GPU latency**.

This experiment is not a larger repetition of the four-GPU sweep. It asks
whether a model whose variables have physical meaning transfers when world
size changes.  The two scaling regimes separate effects that a single P=4
surface cannot identify:

- fixed per-peer M keeps each rank's tile shape unchanged while the routed rows
  per destination, B_rank=P*M, double;
- fixed per-rank routed rows halves M from the corresponding four-GPU point,
  keeping B_rank constant while the number of peers and unique remote bytes
  change. B_rank is not a serving request batch.

The frozen matrix contains 23 shapes and 253 timed candidates: host, direct,
q={2,4,8,16}, and their executable g=2 variants. Four shapes are labeled for
few-shot calibration and 19 remain held out. The set includes the P=4 host-win
anchors, the M=64/N=512 split boundary, and the M=64/N=1024/K=8192 grouped
crossover under both scaling regimes.

`zero_shot_predictions.json` records all 23 choices and predicted candidate
latencies before P=8 measurement. It also records the hashes of the P=4 input
CSVs and this manifest. The source model failed the P=4 admission gates, so
these predictions are an explicit transfer hypothesis, not a production
profile. After correcting all model traffic features to per-rank units, the
current zero-shot choices are host=2, q2=9, q4=11, and q8=1.

## Required order

1. Capture the eight-GPU environment, `nvidia-smi topo -m`, P2P matrix, NCCL
   version/commit, CUTLASS commit, driver, clocks, and power limits.
2. Reject the node if it is not eight A100 SM80 GPUs with full P2P/NVLink.
3. Run the frozen matrix without refitting or inspecting partial latency.
4. Score the recorded zero-shot choices against the measured oracle.
5. Refit only the four `calibration` shapes, then score the unchanged
   `heldout` shapes.
6. Run the registered ablations and profile the three named boundary sweeps.
   Retain kernel count, per-kernel time, DRAM traffic, LSA/NVLink traffic when
   the profiler exposes it, and achieved occupancy.
7. Repeat only points whose top two medians differ by less than 2%, using 100
   warmups and 2,000 iterations.

The same gates apply to zero-shot and few-shot reports: direction accuracy at
least 99%, no host regression, mean regret at most 0.5%, and maximum regret at
most 1%. A failed gate keeps compiler admission closed; it does not invalidate
the measurement.

## Entrypoint

First print and preserve the commands:

```bash
python EDM-IR/tools/run_frozen_fused_experiment.py \
  --manifest experiments/2026-08-29-a100-8gpu-transfer-plan/manifest.json \
  --runtime EDM-IR/runtime \
  --topology /path/to/observed-8gpu-topology.json \
  --cutlass /path/to/cutlass \
  --nccl-home /path/to/nccl-device \
  --out /path/to/results
```

After reviewing `run_record.json`, add `--execute`. Candidate order is
deterministically shuffled by the seed frozen for each sweep. The runner
requires eight identified A100s with a complete NVLink/P2P matrix, captures
`nvidia-smi`, topology, CUDA, CUTLASS, and NCCL-library evidence, performs
correctness checks, and never updates the frozen prediction file.
