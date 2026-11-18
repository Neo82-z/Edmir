#!/usr/bin/env python3
"""Inspect the fixed review without training, tuning, or touching holdouts."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'EDM-IR'), str(Path(__file__).parent)]
import numpy as np
from edmir.compiler import calibrated_candidate_selection as scorer
from review_calibrated_structure import aggregate, DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    results = json.loads((run / 'results.json').read_text())
    data = json.loads(DATA.read_text())
    output = {'analysis_only': True, 'new_training': False, 'frozen_six_used': False,
              'results_sha256': hashlib.sha256((run / 'results.json').read_bytes()).hexdigest(),
              'prior_only': {}, 'variants': {}, 'host_basis_rank': []}
    for basis in ('original', 'workload'):
        decisions = []
        for fold in range(4):
            artifact = json.loads((run / f'p4-calibrated-{basis}-adamw-fold{fold}.json').read_text())
            model = deepcopy(artifact['model'])
            model['state'] = {k: np.zeros_like(v).tolist() for k, v in model['state'].items()}
            shapes = {tuple(d['shape']) for d in artifact['evaluation']['decisions']}
            rows = [{**r, 'split': 'evaluation'} for r in data['p4']['rows'] if tuple(r['shape']) in shapes]
            decisions.extend(scorer.evaluate(model, rows)['decisions'])
            if basis == 'workload':
                hosts = [r for r in data['p4']['rows'] if r['is_host'] and r['shape'] in model['training_shapes']]
                matrix = np.asarray([scorer.host_workload_features(r) for r in hosts])
                output['host_basis_rank'].append({'fold': fold, 'rows': len(hosts), 'columns': matrix.shape[1],
                    'rank': int(np.linalg.matrix_rank(matrix)), 'fit': model['prior']['host']})
        output['prior_only'][basis] = aggregate(decisions)
    for name, result in results['variants'].items():
        host_errors = []
        pair_wrong = pair_total = 0
        for d in result['decisions']:
            for candidate, actual in d['measured_us'].items():
                if candidate.startswith('host'):
                    host_errors.append(abs(math.exp(d['predicted_log_us'][candidate]) / actual - 1))
            names = sorted(d['measured_us'])
            for i, a in enumerate(names):
                for b in names[i+1:]:
                    actual_gap = d['measured_us'][a] - d['measured_us'][b]
                    if actual_gap == 0:
                        continue
                    predicted_gap = d['predicted_log_us'][a] - d['predicted_log_us'][b]
                    pair_total += 1
                    pair_wrong += actual_gap * predicted_gap <= 0
        folds = [json.loads(p.read_text()) for p in sorted(run.glob(name + '-fold*.json'))]
        detail = {'max_host_relative_error': max(host_errors), 'pair_direction_wrong': pair_wrong,
                  'pair_direction_total': pair_total, 'train_shape_mape_by_fold': [f['training_diagnostic']['mean_shape_mape'] for f in folds],
                  'selected_seeds': [f['model']['selected_seed'] for f in folds]}
        if name.startswith('p4'):
            shape = [4, 512, 4096, 512]
            d = next(d for d in result['decisions'] if d['shape'] == shape)
            fit = next(f['model'] for f in folds if any(v['shape'] == shape for v in f['evaluation']['decisions']))
            surface = sorted([r for r in data['p4']['rows'] if r['shape'] == shape], key=lambda r: r['candidate_id'])
            priors = scorer.prior_scores(fit['prior'], surface)
            detail['M512_decomposition'] = {r['candidate_id']: {'prior_us': math.exp(float(priors[i])),
                 'estimated_us': math.exp(d['predicted_log_us'][r['candidate_id']]), 'measured_us': r['latency_us']}
                 for i, r in enumerate(surface)}
        else:
            d = next(d for d in result['decisions'] if d['shape'][1] == 256)
            q2 = next(k for k in d['measured_us'] if k.startswith('splitk2@'))
            q8 = next(k for k in d['measured_us'] if k.startswith('splitk8@'))
            detail['M256_ratio'] = {'estimated_q2_over_q8': math.exp(d['predicted_log_us'][q2]-d['predicted_log_us'][q8]),
                'measured_q2_over_q8': d['measured_us'][q2]/d['measured_us'][q8], 'mape': d['mape'],
                'q2_support': d['support'][q2], 'chosen': d['chosen'], 'provisional_choice': d['provisional_choice']}
        output['variants'][name] = detail

    # Reuse all four already-trained folds; no full-data fit and no GPU.
    cases = []
    for fold in range(4):
        a = json.loads((run / f'p4-calibrated-workload-muon-fold{fold}.json').read_text())
        for d in a['evaluation']['decisions']:
            rows = [r for r in data['p4']['rows'] if r['shape'] == d['shape']]
            cases.append((a['model'], rows))
    deadline = time.monotonic() + 30
    scorer.old.prior_model.in_training_hull.cache_clear()
    started = time.perf_counter()
    for model, rows in cases:
        scorer.predict(model, rows)
    cold = time.perf_counter() - started
    samples = []
    for _ in range(10):
        if time.monotonic() > deadline:
            raise TimeoutError('bounded inference review')
        start = time.perf_counter()
        for model, rows in cases:
            scorer.predict(model, rows)
        samples.append(time.perf_counter() - start)
    output['inference'] = {'shapes': len(cases), 'candidate_rows': sum(len(rows) for _, rows in cases),
        'cold_seconds': cold, 'warm_median_seconds': statistics.median(samples), 'warm_max_seconds': max(samples),
        'repeats': len(samples), 'cache': str(scorer.old.prior_model.in_training_hull.cache_info()),
        'scope': 'local CPU; all P4 held-out fold predictions plus candidate support; not kernel timing'}
    (run / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'inference': output['inference'], 'prior_only': {k: {n: v for n, v in x.items() if n != 'decisions'}
                       for k, x in output['prior_only'].items()}}, indent=2))


if __name__ == '__main__':
    main()
