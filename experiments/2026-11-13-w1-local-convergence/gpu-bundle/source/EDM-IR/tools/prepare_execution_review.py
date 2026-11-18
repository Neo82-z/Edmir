#!/usr/bin/env python3
"""Generate a bounded local execution review from existing, hashed evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.execution_ledger import FAMILIES, execution_ledgers
from edmir.compiler.evidence_selection import assess_candidates
from edmir.compiler.measurement_dataset import P4_ENV, load_selector_dataset, sha256

ROOT = Path(__file__).resolve().parents[2]
# Existing four priorities, plus the already identified M512 failure.
SHAPES = ((4,64,1024,512), (4,64,4096,2048), (4,64,8192,512),
          (4,256,8192,512), (4,512,4096,512))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def prepare(out: Path, root=ROOT):
    started = time.monotonic()
    data = root/"experiments/2026-10-20-offline-predictor/data"
    p4, _, _, manifest = load_selector_dataset(data)
    records = [json.loads(line) for line in (data/"selector_measurements.jsonl").read_text().splitlines()]
    records = [r for r in records if r['environment_id'] == P4_ENV]
    # Check retained raw CSV identity as well as the normalized bundle. These
    # hashes establish archive identity, not contemporary hardware correctness.
    sources = {r['source']['path']: r['source']['sha256'] for r in records}
    for path, expected in sources.items():
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts or sha256(root/relative) != expected:
            raise ValueError("historical source mismatch: "+path)
    audit_path = root/"experiments/2026-10-21-round-two-error-audit/verified/outer_errors.json"
    audit = json.loads(audit_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for shape in SHAPES:
        name = 'p%d-m%d-k%d-n%d' % shape
        ledgers = execution_ledgers(shape, environment_id=P4_ENV)
        observations = {}
        for r in records:
            m = r['measurement']
            if (m['world_size'], m['rows_per_peer'], m['k'], m['n']) == shape:
                observations[m['family']] = {"median_us": m['measured_ms']*1000,
                    "reported_timing": r['reported_timing'], "source": r['source'],
                    "status": "historical_development_observation_not_new_validation",
                    "current_source_identity_matched": False}
        best = min(v['median_us'] for v in observations.values())
        report = {"schema": "edmir.execution-review-case.v1", "shape": list(shape),
                  "ledgers": ledgers, "decision": assess_candidates(ledgers),
                  "historical_observations": observations,
                  "historical_best_observed": sorted(n for n, v in observations.items() if v['median_us'] == best),
                  "unmeasured_candidates": sorted(set(FAMILIES)-set(observations)),
                  "historical_errors": [r for r in audit['inventory'] if tuple(r['shape']) == shape],
                  "historical_times_used_for_selection": False}
        write_json(out/(name+'.json'), report)
        cases.append({"shape": list(shape), "path": name+'.json',
                      "decision": report['decision']['status'], "choice": None,
                      "historical_best_observed": report['historical_best_observed'],
                      "unmeasured_candidates": report['unmeasured_candidates']})

    # Exercise all old shapes using static descriptors, never fit or import a
    # saved model. Historical coverage is reported rather than scored as a gain.
    all_shapes = sorted({r.shape for r in p4})
    counts = 0
    for shape in all_shapes:
        if time.monotonic()-started > 60:
            raise TimeoutError("local review exceeded its 60-second work budget")
        ledgers = execution_ledgers(shape, environment_id=P4_ENV)
        decision = assess_candidates(ledgers)
        if decision['choice'] is not None or decision['retained_candidates'] != sorted(FAMILIES):
            raise AssertionError("counts alone manufactured a performance choice")
        counts += len(ledgers)

    diagnosis = {
        "schema": "edmir.mechanism-diagnosis.v1",
        "M512": {
            "shape": [4,512,4096,512],
            "confirmed_model_defect": "q4-q2 expression is affine in K at fixed M,N; q-independent K tail cancels; archived fit crosses too early",
            "confirmed_source_facts": ["q4 doubles CTA count against q2", "K mainloop per CTA halves",
                "serial epilogues per output tile rise from 2 to 4", "additional FP16 output read/write is explicit"],
            "not_established": ["how much of the 43.9 us gap is semaphore waiting", "compiled occupancy or memory contention as the unique cause"],
            "minimal_discriminating_comparison": "reuse the existing M512/K4096 anchor and M512/K3072 diagnostic; compare fixed q2/q4 under the same source and timing mode; no parameter sweep",
            "measurement_sequence": ["unprofiled whole-program timings and full-output correctness first",
                "separate CUDA timeline for host submission and serialized GEMM duration",
                "only if needed, inspect compiled resource use and kernel stalls for the same q2/q4 pair"],
            "falsification": "if the gap appears in CPU submission but not device kernel durations, a kernel-only semaphore explanation is insufficient",
        },
        "large_N": {
            "shape": [4,64,4096,2048],
            "confirmed_model_defect": "fused loses separate M,N geometry; host basis changes discretely under small training-data interventions",
            "confirmed_source_facts": ["n_tiles=16 at g1, so logical A replay differs from unique remote payload",
                "g2 halves n_tiles and doubles threads per CTA; compiled residency remains unknown",
                "host stages inputs before per-source GPU GEMMs; fused reads A through peer pointers"],
            "not_established": ["physical NVLink/HBM traffic inferred from logical replay", "which component caused the historical 79.5 us host/q2 gap"],
            "minimal_discriminating_comparison": "reuse the large-N anchor host/direct/q2; reuse the already frozen equal-tile M64/N1536 and M192/N512 pair at K3072 to separate M,N geometry",
            "measurement_sequence": ["same legacy implementation for reproducing old-model errors",
                "prepared metadata as a separately identified control, never mixed into legacy labels",
                "if the device gap remains, inspect memory traffic separately from primary timing"],
            "falsification": "if equal-tile shapes differ while measured traffic is similar, total-traffic-only attribution is insufficient; if metadata mode changes the gap, include CPU execution costs",
        },
        "grouped_N_scope": {
            "confirmed": "archived P4 grouped-N observations are at M64; other M values are not validated by that coverage",
            "policy": "static legal candidates remain unresolved; absent grouped measurements are not filled with host or neighboring-shape labels",
            "minimal_discriminating_comparison": "within the existing four priorities compare g1/g2 at the same q and K, including M64 and M256; keep counts, resources and full timing separate",
        },
        "baseline_contract": {
            "identity": "host_nccl_cutlass_sm80_per_source_v1", "metadata_mode": "legacy",
            "meaning": "NCCL transfers plus P sequential CUTLASS GPU GEMMs per destination; not CPU GEMM",
            "stronger_baselines": "explicit host_compiler_cublas / host_merged_cutlass / host_merged_cublaslt families have separate execution accounts; library internals remain unknown and cuBLASLt needs the recorded runtime algorithm before timing evidence is accepted",
        },
        "ascend_transfer_boundary": "transfer the reasoning about transactions, resource lifetimes and critical paths; do not transfer NPU timings or presume CUDA has the same bottleneck",
        "new_training_runs": 0, "new_gpu_runs": 0,
    }
    write_json(out/'mechanism_diagnosis.json', diagnosis)
    summary = {"schema": "edmir.execution-review.v1", "cases": cases,
               "old_shapes_checked": len(all_shapes), "candidate_ledgers_checked": counts,
               "scope": "static local execution review, not predictor accuracy validation",
               "new_training_runs": 0, "new_gpu_runs": 0, "independent_validation": False,
               "elapsed_seconds": time.monotonic()-started,
               "source_evidence": {**sources,
                   str(audit_path.relative_to(root)): sha256(audit_path),
                   str((data/'manifest.json').relative_to(root)): sha256(data/'manifest.json'),
                   str((data/'selector_measurements.jsonl').relative_to(root)): sha256(data/'selector_measurements.jsonl')},
               "files": {p.name: sha256(p) for p in sorted(out.glob('*.json'))}}
    write_json(out/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.out)
    print(json.dumps({k:result[k] for k in ('old_shapes_checked','candidate_ledgers_checked','elapsed_seconds')},indent=2))
