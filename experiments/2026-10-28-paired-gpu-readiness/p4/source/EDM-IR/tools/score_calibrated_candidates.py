#!/usr/bin/env python3
"""Inspect conditional latency estimates and explicit candidate support."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'EDM-IR'), str(Path(__file__).parent)]
from edmir.compiler import calibrated_candidate_selection as scorer
from review_precision_scoring import precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--shape', nargs=4, type=int, required=True, metavar=('P', 'M', 'K', 'N'))
    args = parser.parse_args()
    envelope = json.loads(args.model.read_text())
    model = envelope.get('model', envelope)
    data = json.loads(args.data.read_text()).get('p' + str(args.shape[0]), {})
    rows = [r for r in data.get('rows', []) if r['shape'] == args.shape]
    if not rows:
        parser.error('shape is absent from supplied observations; do not invent labels or correctness')
    source = hashlib.sha256(args.data.read_bytes()).hexdigest()
    for r in data.get('excluded', []):
        if r['shape'] != args.shape:
            continue
        failed = {'shape': args.shape, 'candidate_id': r['candidate_id'],
            'execution_context': rows[0]['execution_context'], 'source_id': source,
            'precision': precision(r['candidate_id'].split('@')[0]),
            'validity': {'status': 'failed', 'source_id': source, 'scope': 'failure in supplied development dataset'}}
        failed['validity']['contract_sha256'] = scorer.old.evidence_contract(failed)
        rows.append(failed)
    try:
        result = scorer.predict(model, rows)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({**result, 'estimated_us': {k: math.exp(v) for k, v in result['predicted_log_us'].items()},
        'training_shape': args.shape in model['training_shapes'], 'prediction_semantics': model['prediction_semantics'],
        'compiler_admission': False,
        'note': 'Provisional choice includes unsupported candidates for diagnosis. It is not an admitted decision or a calibrated probability.'}, indent=2))


if __name__ == '__main__':
    main()
