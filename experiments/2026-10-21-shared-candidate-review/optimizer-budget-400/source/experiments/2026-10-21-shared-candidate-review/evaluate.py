"""Replay all predeclared models under the existing unmodified boundary gates."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.shared_candidate_selection import SharedCandidateSelector, VARIANTS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from run_selector_gpu import write_json
import evaluate_traffic_selector as evaluation


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def add_gates(primary,stress,method):
    from evaluate_execution_cost_selector import decision_gate
    gate=decision_gate(primary,stress,method)
    gate['checks'].update({
        'M512_primary_best_observed':all(r['diagnostics'][method]['regret_all_observed']<=1e-12 for r in primary['decisions'] if r['shape'][1]==512),
        'M512_leave_K_best_observed':all(r['diagnostics'][method]['regret_all_observed']<=1e-12 for r in stress['decisions'] if r['shape'][1]==512),
        'primary_mean_and_max_no_worse_than_accepted':all(primary['metrics'][method][k]<=primary['metrics']['additive_latency'][k]+1e-12 for k in ('mean_regret','max_regret')),
        'stress_mean_and_max_no_worse_than_accepted':all(stress['metrics'][method][k]<=stress['metrics']['additive_latency'][k]+1e-12 for k in ('mean_regret','max_regret')),
    })
    gate['passed']=all(gate['checks'].values())
    return gate


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    reference=json.loads(evaluation.REFERENCE.read_text())
    for path,value in reference['source_sha256'].items():
        assert sha256(ROOT/path)==value,path
    previous=ROOT/'experiments/2026-10-21-host-boundary-repair'
    review=load('shared_boundary_review',previous/'review.py')
    assessment=load('shared_boundary_assessment',previous/'assessment.py')
    variants=[];start=time.monotonic()
    for method in VARIANTS:
        started=time.monotonic();fit_audits=[]
        def progress(stage):
            if time.monotonic()-started>600:raise TimeoutError('variant exceeded 600 seconds')
            state={'method':method,'stage':stage,'elapsed_seconds':time.monotonic()-start}
            write_json(out/'progress.json',state);print(json.dumps(state),flush=True)
        class Policy:
            @staticmethod
            def fit(dataset):
                if time.monotonic()-started>600:raise TimeoutError('variant exceeded 600 seconds')
                model=SharedCandidateSelector.fit(dataset,method=method)
                fit_audits.append(model.training_audit)
                return model
        evaluation.TrafficLatencySelector=Policy;evaluation.METHOD=method
        review.TrafficLatencySelector=Policy;review.METHOD=method
        progress('primary');primary=evaluation.evaluate(data,'primary',reference)
        progress('leave_K');stress=evaluation.evaluate(data,'stress',reference)
        result={'method':method,'primary':primary,'stress':stress,'gate':add_gates(primary,stress,method)}
        progress('group_deletion');checks={'deletion':review.deletion(data,primary)}
        progress('previous_attacks');checks['previous_attacks']=review.replay_previous_attack(data)
        progress('one_percent_stress');checks['boundary_stress']=review.boundary_stress(data)
        result['review']=checks;result['assessment']=assessment.assess(result,checks)
        result['full_fit']=Policy.fit(data).to_dict()
        selected=[a['starts'][a['selected_start']] for a in fit_audits]
        result['optimization']={'fits_including_cache_hits':len(selected),
            'selected_iteration_limit':sum(r['status']==1 for r in selected),
            'selected_max_gradient':max(r['gradient_max_abs'] for r in selected),
            'all_starts':fit_audits}
        result['elapsed_seconds']=time.monotonic()-started
        variants.append(result);write_json(out/(method+'.json'),result)
        progress('variant_complete')
        print(json.dumps({'method':method,'assessment':result['assessment'],
            'primary':primary['metrics'][method],'stress':stress['metrics'][method],
            'optimization':{k:v for k,v in result['optimization'].items() if k!='all_starts'}},indent=2),flush=True)
    paths=[Path(__file__),HERE/'design.json',previous/'review.py',previous/'assessment.py',
           ROOT/'EDM-IR/tools/evaluate_traffic_selector.py',ROOT/'EDM-IR/tools/evaluate_execution_cost_selector.py']
    paths.extend(ROOT/'EDM-IR/edmir/compiler'/p for p in ('shared_candidate_selection.py','additive_latency_selection.py',
        'traffic_latency_selection.py','factorized_selection.py','relative_selection.py','measurement_dataset.py',
        'host_baseline.py','two_stage_selection.py','execution_cost_selection.py'))
    write_json(out/'results.json',{'variants':variants,'compiler_admission':False,'new_independent_labels':0,'gpu_operations':0,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
        'dataset_manifest_sha256':sha256(ROOT/'experiments/2026-10-20-offline-predictor/data/manifest.json'),
        'prior_results_sha256':sha256(evaluation.REFERENCE),'elapsed_seconds':time.monotonic()-start})
    write_json(out/'progress.json',{'state':'complete','elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
