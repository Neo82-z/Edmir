"""Run the fixed local gates for both predeclared experimental policies."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from candidate import Candidate, ROOT
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from evaluate_execution_cost_selector import decision_gate, PRIORITY
import evaluate_traffic_selector as evaluation


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main(out):
    if out.exists():raise ValueError('fresh output required')
    start=time.monotonic()
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    reference=json.loads(evaluation.REFERENCE.read_text())
    for path,digest in reference['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    previous=ROOT/'experiments/2026-10-21-host-boundary-repair'
    review=load_file('prior_boundary_review',previous/'review.py')
    assessment=load_file('prior_assessment',previous/'assessment.py')
    variants=[]
    for average in (False,True):
        method='separate_resources_average' if average else 'separate_resources'
        class Policy:
            @staticmethod
            def fit(data):
                model=Candidate.fit(data,average=average)
                model.to_dict=model.audit
                return model
        # Reuse the exact previous fold construction, controls and stresses.
        evaluation.TrafficLatencySelector=Policy;evaluation.METHOD=method
        review.TrafficLatencySelector=Policy;review.METHOD=method
        primary,stress=[evaluation.evaluate(data,p,reference) for p in ('primary','stress')]
        gate=decision_gate(primary,stress,method)
        gate['checks'].update({
            'M512_primary_best_observed':all(r['diagnostics'][method]['regret_all_observed']<=1e-12 for r in primary['decisions'] if r['shape'][1]==512),
            'M512_leave_K_best_observed':all(r['diagnostics'][method]['regret_all_observed']<=1e-12 for r in stress['decisions'] if r['shape'][1]==512),
            'primary_mean_and_max_no_worse_than_accepted':all(primary['metrics'][method][k]<=primary['metrics']['additive_latency'][k]+1e-12 for k in ('mean_regret','max_regret')),
            'stress_mean_and_max_no_worse_than_accepted':all(stress['metrics'][method][k]<=stress['metrics']['additive_latency'][k]+1e-12 for k in ('mean_regret','max_regret')),
        });gate['passed']=all(gate['checks'].values())
        result={'method':method,'primary':primary,'stress':stress,'gate':gate}
        checks={'deletion':review.deletion(data,primary),'previous_attacks':review.replay_previous_attack(data),'boundary_stress':review.boundary_stress(data)}
        result['review']=checks;result['assessment']=assessment.assess(result,checks)
        result['full_fit']=Policy.fit(data).audit();variants.append(result)
        print(json.dumps({'method':method,'assessment':result['assessment'],
              'primary':primary['metrics'][method],'stress':stress['metrics'][method],
              'boundary_failures':[r for r in checks['boundary_stress']['records'] if r['regret']>0],
              'priority_deletions':[{'shape':r['shape'],'counts':r['choice_counts'],'worst':r['worst_deletion_regret']} for r in checks['deletion'] if tuple(r['shape']) in PRIORITY or r['shape'][1]==512]},indent=2),flush=True)
    paths=[Path(__file__),Path(__file__).with_name('candidate.py'),Path(__file__).with_name('design.json'),
           ROOT/'EDM-IR/edmir/compiler/resource_geometry.py',previous/'review.py',previous/'assessment.py',
           ROOT/'EDM-IR/tools/evaluate_traffic_selector.py',ROOT/'EDM-IR/tools/evaluate_execution_cost_selector.py',
           ROOT/'EDM-IR/edmir/compiler/traffic_latency_selection.py']
    out.write_text(json.dumps({'variants':variants,'new_independent_labels':0,'gpu_operations':0,
                 'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths},'elapsed_seconds':time.monotonic()-start},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
