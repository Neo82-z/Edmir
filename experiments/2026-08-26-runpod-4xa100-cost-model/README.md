# 4x A100 SXM held-out cost-model validation (2026-08-26)

This directory is the complete local copy of the paid RunPod experiment used
to close EDM-IR Milestone 6. The immutable remote archive is
`edmir-cost-model-20260826.tar.gz`; its verified SHA-256 is
`ed3e68e47a969f72c34906bbf4aab0ff729a8a1a20da0adc27819e2e1ecda7c4`.

## Platform and protocol

- 4x NVIDIA A100-SXM4-80GB, every off-diagonal topology entry `NV12`;
- CUDA toolkit 12.8.93, driver 580.126.16, PyTorch 2.8.0+cu128, NCCL 2.25.1;
- six pairwise NCCL SendRecv curves from 1 KiB to 512 MiB;
- nine exact cuBLAS FP16 GEMM shapes and all required local D2D copy sizes;
- 50 generated NCCL+cuBLAS Programs: 27 training and 23 held-out candidates;
- five warmups and 30 timed iterations per candidate;
- three independent benchmark processes for each held-out sequential baseline
  and selected schedule.

The first runner invocation exposed a parser bug (`p95_ms` contained a digit),
so it stopped before the first runtime row. The corrected retry is preserved in
`logs/07_run_retry.txt`; no partial candidate measurements from the failed run
entered the dataset.

## Result

The runtime residual is fitted only on the 27 training rows. A conservative
9.35% predicted-gain threshold is also learned from training labels and stored
in the calibration profile. On 23 untouched candidate rows from two held-out
problem families, the resulting model obtains:

- 7.97% MAPE;
- 90.48% raw direction accuracy and 95.24% policy direction accuracy;
- zero mean schedule-selection regret;
- the measured oracle (`chunk_rows = rows_per_peer`, four buffers) for both
  held-out families.

The three-process repetitions preserve the selected schedule's advantage:
6.18% lower median latency for `r=96, k=n=4096` and 5.42% for
`r=64, k=2048, n=4096`. These are shape-held-out results on one topology and
one backend; they are not evidence of cross-hardware or cross-backend
generalization.

After the train-only fallback policy was integrated, the final source tree was
synced back to the same A100 instance. All 49 Python tests passed there, and a
fresh compiler invocation selected `c96/b4` and `c64/b4` for the two held-out
problems with the 9.35% profile threshold. The exact Programs and console logs
are retained in `final-validation/` and `logs/08_*`--`logs/09_*`; their local
SHA-256 values were checked against the remote copies before handoff.

Regenerate the compact, paper-ready files with:

```bash
python3 summarize_results.py
```

This writes `derived/candidate_predictions.csv`, `derived/problem_summary.csv`,
`derived/repeat_process_medians.csv`, and `derived/summary.json` from the
archived raw measurements and v2 evaluation.
