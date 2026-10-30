#!/usr/bin/env python3
"""Inspect support and the strong two-host baseline without further fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_gpu_evidence import REPO, read, sha
from edmir.compiler.mechanism_prediction import HOSTS, basis
from edmir.compiler.prediction_support import linear_target_support


def build_audit(run):
    seal=read(run/"analysis-seal.json")
    for name in ("configurations.json","selection.json","models.json","counterexamples.json"):
        if sha(run/name)!=seal["outputs"][name]:raise ValueError("revision run seal mismatch")
    datasets=read(run/"configurations.json");host_support=[]
    for context in ("old","primary"):
        rows=datasets[context]
        for target in rows:
            if target["family"] not in HOSTS:continue
            train=[r for r in rows if r["family"]==target["family"] and r["shape"]!=target["shape"] and r["eligible_for_calibration"]]
            if len({tuple(r['shape']) for r in train})<2:continue
            names,vector=basis(target["shape"],target["family"],"cutlass_depth")
            support=linear_target_support([basis(r['shape'],r['family'],'cutlass_depth')[1] for r in train],vector)
            host_support.append({"context":context,"shape":target["shape"],"candidate_id":target["candidate_id"],
                                 "training_shapes":[r['shape'] for r in train],"feature_names":names,**support})
    matched=[]
    for key,rows in read(run/"selection.json").items():
        rows=[r for r in rows if r['matched_two_hosts']]
        if not rows:continue
        policies={}
        for policy in ('model_top1','model_top2','two_merged_hosts'):
            vals=[r['policies'][policy]['choice_loss_fraction'] for r in rows]
            policies[policy]={'mean_loss':st.mean(vals),'max_loss':max(vals)}
        matched.append({'evaluation':key,'shapes':[r['shape'] for r in rows], 'policies':policies,
                        'same_candidate_support':True,'ideal_replay_not_deployment':True})
    # Group coefficient support is audited separately from any timing label.
    return {'host_linear_support':host_support,'two_host_matched_comparison':matched,
            'fit_performed':False,'causal_hardware_attribution':False,'compiler_admission':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('fresh output file required')
    result=build_audit(a.run)
    result['sources']={str(f.resolve()):sha(f) for f in (Path(__file__),REPO/'EDM-IR/edmir/compiler/prediction_support.py',a.run/'analysis-seal.json')}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'host_targets':len(result['host_linear_support']),
                      'outside_training_span':sum(not r['target_in_training_row_span'] for r in result['host_linear_support'])}))


if __name__=='__main__':main()
