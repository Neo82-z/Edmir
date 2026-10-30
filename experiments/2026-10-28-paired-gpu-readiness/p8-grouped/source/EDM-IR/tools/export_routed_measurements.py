#!/usr/bin/env python3
"""Export validated real normal-point records. No synthetic or zero filling."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.measurement import require_paper_evidence, summarize


def export_payload(batches, campaign):
    require_paper_evidence(batches, campaign=campaign)
    identities = [(b['trace_hash'],b['family'],b['pairing_id']) for b in batches]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate timing identity; repeated runs need distinct pairing IDs')
    return {'schema':'edmir.routed-observations.v1', 'evidence_kind':'cuda_event_measurement',
            'protocol_hash':campaign['protocol_hash'],
            'scope':'Normal-point operator observations only; not selector or serving results.',
            'rows':[{**{k:b[k] for k in ('trace_hash','environment_hash','source_hash','pairing_id',
                                         'world_size','phase','family')}, **summarize(b)} for b in batches]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a = p.parse_args()
    source, manifest = a.input.read_bytes(),a.campaign.read_bytes()
    payload = export_payload(json.loads(source),json.loads(manifest))
    payload['input_sha256'] = hashlib.sha256(source).hexdigest()
    payload['campaign_sha256'] = hashlib.sha256(manifest).hexdigest()
    # Validate everything before creating an output directory.
    a.out_dir.mkdir(parents=True,exist_ok=False)
    (a.out_dir/'observations.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(f'Exported {len(payload["rows"])} normal-point observations; no deployment decision inferred.')


if __name__ == '__main__':
    main()
