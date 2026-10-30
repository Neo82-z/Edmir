# Routed offline readiness — 2026-09-09

CPU-only engineering rehearsal. **No GPU measurements or new paper performance
results.** D7 experimental scope, candidate gates and historical data are unchanged.

The `artifacts/` directory contains deterministic toy routes, synthetic timing
records/statistics, CPU semantic checks, full unit-test output, and fresh TLC
safe/mutation runs. Synthetic timing data are labeled throughout and rejected by
the measured-paper exporter. `ready_for_paid_experiments` remains false.

Reproduce from `EDM-IR` with a new output directory:

```bash
python3 tools/prepare_routed_benchmark.py \
  --out-dir build/routed-offline-recheck \
  --jar /absolute/path/tla2tools.jar
```

This run uses the official TLA+ v1.7.4 release JAR (TLC 2.19), SHA-256
`936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`.
The JAR is not vendored. Model/config/raw-output hashes are recorded per run.
CPU fixtures cover 216 schedule/layout combinations. The four safe model bounds
and seven deliberate mutation witnesses are small bounded checks, not CUDA or
unbounded correctness proofs.

See `EDM-IR/docs/ROUTED_OFFLINE_READINESS.md` for the remaining real-trace,
BF16 backend, same-node hardware, and manifest-signoff obligations. This record
does not imply those obligations have been fulfilled.
