#!/usr/bin/env python3
"""Cross-check reported shortlists using the existing independent scorer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler.candidate_budget import rank_predictions,shortlist,score

p=argparse.ArgumentParser();p.add_argument('--review',required=True,type=Path)
p.add_argument('--analysis',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
if a.out.exists():p.error('fresh output required')
read=lambda f:json.loads(f.read_text())
review=read(a.review);analysis=read(a.analysis);groups={}
for stage in review['stages']:
    for r in stage['configurations']:
        if r['primary_timing_label'] and not r['quality_flags']:
            groups.setdefault((stage['stage'],tuple(r['shape'])),[]).append(r)
count=permutations=0
for d in analysis['decisions']:
    if d['scope']!='stable':continue
    rows=groups[d['stage'],tuple(d['shape'])]
    if d['selection_scope']=='fused':rows=[r for r in rows if not r['family'].startswith('host')]
    if d['selection_scope']=='compiler_host_fused':rows=[r for r in rows if r['family']=='host' or not r['family'].startswith('host')]
    predictions=[{'candidate_id':r['candidate_id'],'predicted_us':r['forecast_errors'][d['model']]['predicted_us']} for r in rows]
    ranked=rank_predictions(predictions)
    chosen=shortlist(ranked,d['budget']);result=score(rows,chosen)
    assert chosen==d['ranked_shortlist']
    assert math.isclose(result['choice_loss_fraction'],d['choice_loss'],rel_tol=1e-12,abs_tol=1e-15)
    for seed in range(10):
        changed=list(predictions);random.Random(seed).shuffle(changed)
        assert rank_predictions(changed)==ranked;permutations+=1
    count+=1
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.review,a.analysis,Path(__file__),ROOT/'EDM-IR/edmir/compiler/candidate_budget.py']}
a.out.write_text(json.dumps({'decisions_cross_checked':count,'permutation_checks':permutations,
    'all_passed':True,'training_or_refit':False,'inputs_sha256':hashes},indent=2)+'\n')
print(json.dumps({'decisions_cross_checked':count,'permutation_checks':permutations,'all_passed':True}))
