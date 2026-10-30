#!/usr/bin/env python3
"""Reproduce public trace accounting and audit existing decision margins; no fit."""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EDM-IR'))
from edmir.compiler.overlap_evidence import analyze_deepseek_trace, decision_margin


def main():
    out = ROOT/'experiments/2026-10-28-overlap-evidence'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protected = json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/protected-before.json').read_text())
    for path, expected in protected.items():
        assert sha(ROOT/path) == expected, path
    started = time.monotonic()
    traces = {}
    for mode in ('decode','prefill'):
        p = out/f'external/deepseek-ai_profile-data/{mode}.json'
        traces[mode] = {**analyze_deepseek_trace(json.loads(p.read_text())), 'source_sha256': sha(p)}
    (out/'public-trace-analysis.json').write_text(json.dumps(traces,indent=2)+'\n')

    path = ROOT/'experiments/2026-10-28-calibrated-structure/run-01/results.json'
    original = json.loads(path.read_text())
    audit = {}
    for name, variant in original['variants'].items():
        shapes = []
        for d in variant['decisions']:
            actual = d['measured_us'];pred = {n:math.exp(v) for n,v in d['predicted_log_us'].items()}
            # Family optimum is selected using measurements for diagnosis only,
            # never used for inference or retraining.
            host = min((n for n in actual if n.startswith('host')),key=lambda n:(actual[n],n))
            fused = min((n for n in actual if not n.startswith('host')),key=lambda n:(actual[n],n))
            chosen = d['provisional_choice'];best = min(actual,key=lambda n:(actual[n],n))
            shapes.append({'shape':d['shape'],'host':host,'fused':fused,
                'host_vs_fused':decision_margin(actual[host],actual[fused],pred[host],pred[fused]),
                'provisional_choice':chosen,'measured_best':best,'chosen':d['chosen'],
                'exact_choice_wrong':actual[chosen]>actual[best], 'regret':d['provisional_regret'],
                'winner_margin_us':actual[chosen]-actual[best],
                'host_fused_direction_wrong':d['host_error'] or d['fused_error']})
        audit[name] = {'shapes':shapes,'wrong_choices':sum(d['exact_choice_wrong'] for d in shapes),
            'host_fused_direction_errors':sum(d['host_fused_direction_wrong'] for d in shapes),
            'max_regret':variant['max_provisional_regret'],'mean_regret':variant['mean_provisional_regret'],
            'mean_shape_mape':variant['mean_shape_mape'],'selection_coverage':variant['selection_coverage']}
    (out/'decision-margin-audit.json').write_text(json.dumps({'source_sha256':sha(path),'variants':audit},indent=2)+'\n')
    for path, expected in protected.items():
        assert sha(ROOT/path)==expected, path
    summary = {'new_training':False,'new_gpu_cost':0,'frozen_six_used':False,
        'protected_unchanged':True,'seconds':time.monotonic()-started,
        'external_trace_events':sum(t['trace_event_count'] for t in traces.values()),
        'not_claimed':['A100 speedup','M512 fixed','network lifetime equals kernel lifetime','occupancy from CTA grid'],
        'primary_metrics':['host_fused_direction_errors','wrong_choices','max_regret','mean_regret'],
        'secondary_metrics':['mean_shape_mape'],
        'variants':{k:{n:v for n,v in a.items() if n!='shapes'} for k,a in audit.items()}}
    (out/'review.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
