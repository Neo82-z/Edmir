"""CPU-only reconstruction of the sealed inputs; no model weights or downloads."""
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'bundle/source/EDM-IR'),str(HERE/'bundle/source/EDM-IR/tools')]
from prepare_selector_gpu import write_inputs
records=json.loads((HERE/'bundle/inputs.json').read_text())
for tag,seeds in records.items():
    for seed,expected in seeds.items():
        folder=HERE/'inputs'/tag/seed
        if folder.exists():raise ValueError('choose an unused inputs directory')
        actual=write_inputs(expected['shape'],folder,seed=int(seed))
        for key in ['shape','input_seed','input_bytes','reference_bytes','input_sha256','reference_sha256']:
            if actual[key]!=expected[key]:raise ValueError('reconstruction mismatch: '+tag+'/'+seed+'/'+key)
        print(tag,seed,'verified',flush=True)

