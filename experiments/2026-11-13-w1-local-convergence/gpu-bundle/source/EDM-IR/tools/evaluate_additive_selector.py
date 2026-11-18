#!/usr/bin/env python3
"""CPU-only nested replay with the unchanged previous acceptance gate."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
from edmir.compiler.additive_latency_selection import AdditiveLatencySelector
from edmir.compiler.execution_cost_selection import direction_diagnostics
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from evaluate_execution_cost_selector import decision_gate, summarize, PRIORITY, CONTROLS

METHOD = "additive_latency"
REFERENCE = ROOT/"experiments/2026-10-21-execution-cost-revision/verified/results.json"


def evaluate(data, protocol, reference):
    key = (lambda r: r.group_id) if protocol == "primary" else (lambda r: r.k)
    records = {tuple(r["shape"]): r for r in reference[protocol]["decisions"]}
    rows, fits = [], []
    for held in sorted({key(r) for r in data.rows}):
        train = replace(data, rows=tuple(r for r in data.rows if key(r) != held))
        test = replace(data, rows=tuple(r for r in data.rows if key(r) == held))
        assert not {r.shape for r in train.rows} & {r.shape for r in test.rows}
        model = AdditiveLatencySelector.fit(train)
        fits.append({"heldout": held, "model": model.to_dict()})
        for s in test.validate().values():
            first = next(iter(s.values()))
            c = {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in s.items()}
            selected = model.select(c, baseline=data.baseline, allow_extrapolation=True)
            times = {n: r.measured_ms for n, r in s.items()}
            previous = records[first.shape]
            assert times == previous["times_ms"]
            rows.append({"shape": list(first.shape), "heldout": held, "times_ms": times,
                         "choices": {**previous["choices"], METHOD: selected["choice"]},
                         "scores": {METHOD: selected["scores"]},
                         "predictions": selected,
                         "diagnostics": {**previous["diagnostics"], METHOD: direction_diagnostics(times, selected["choice"], tuple(selected["scores"]))}})
    return {"metrics": {**reference[protocol]["metrics"], METHOD: summarize(rows, METHOD)}, "decisions": rows, "fits": fits}


def main(out):
    if out.exists():
        raise ValueError("use a fresh result path")
    start = time.monotonic()
    data, _, _, _ = load_identified_selector_dataset(ROOT/"experiments/2026-10-20-offline-predictor/data")
    ref = json.loads(REFERENCE.read_text())
    # Frozen prior controls are copied, not silently recomputed or changed.
    for path, digest in ref["source_sha256"].items():
        assert sha256(ROOT/path) == digest, path
    primary, stress = [evaluate(data, p, ref) for p in ("primary", "stress")]
    gate = decision_gate(primary, stress, METHOD)
    model = AdditiveLatencySelector.fit(data)
    sources = [Path(__file__), ROOT/"EDM-IR/tools/evaluate_execution_cost_selector.py",
               *(ROOT/"EDM-IR/edmir/compiler"/n for n in (
                   "additive_latency_selection.py", "execution_cost_selection.py", "factorized_selection.py",
                   "measurement_dataset.py", "relative_selection.py", "two_stage_selection.py", "host_baseline.py"))]
    result = {"schema": "edmir.additive-selector-replay.v1", "method": METHOD,
              "status": "adaptively developed on old data; nested retrospective replay, not independent validation",
              "primary": primary, "stress": stress, "gate": gate, "fitted_model": model.to_dict(),
              "prior_results_sha256": sha256(REFERENCE), "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
              "dataset_sha256": sha256(ROOT/"experiments/2026-10-20-offline-predictor/data/manifest.json"),
              "new_independent_labels": 0, "gpu_operations": 0, "compiler_admission": False,
              "elapsed_seconds": time.monotonic()-start}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"gate": gate, "primary": primary["metrics"][METHOD], "stress": stress["metrics"][METHOD],
                      "priority": [{"shape": r["shape"], "choice": r["choices"][METHOD], "regret": r["diagnostics"][METHOD]["regret_all_observed"],
                                    "predicted_ms": r["predictions"]["predicted_ms"]} for r in primary["decisions"] if tuple(r["shape"]) in PRIORITY],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    return gate["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    sys.exit(0 if main(p.parse_args().out) else 2)
