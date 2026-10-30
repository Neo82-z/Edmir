#!/usr/bin/env python3
"""Independent constrained-solver check and bounded stability diagnostics."""
import argparse
from collections import Counter
from dataclasses import replace
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, lsq_linear
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.traffic_latency_selection import TrafficLatencySelector, host_features, fused_features, _solve, training_rows_weights
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from evaluate_traffic_selector import METHOD, PRIORITY
REVIEW_SHAPES = PRIORITY | {(4,512,k,512) for k in (2048,4096,8192)}


def matrix(train,model,host):
    rows=sorted((r for r in train.rows if r.is_host==host),key=lambda r:(r.shape,r.family))
    fit=model.host if host else model.fused
    x=np.array([host_features(r,model.input_threshold,model.depth_threshold) if host else fused_features(r,model.capacity) for r in rows])/fit.scales
    counts={k:len(s)-1 for k,s in train.validate().items()}
    weights=np.array([1. if host else 1./counts[r.grouped_shape] for r in rows])
    y=np.array([r.measured_ms for r in rows])
    if not host:
        relative=np.mean(y)/y
        weights=weights*relative**2/np.mean(relative**2)
    a=x*np.sqrt(weights[:,None]);b=y*np.sqrt(weights)
    penalty=np.eye(x.shape[1])*math.sqrt(model.host_ridge if host else model.ridge);penalty[0,0]=0
    A=np.vstack((a,penalty));B=np.r_[b,np.zeros(len(penalty))]
    # Independent bounded least-squares algorithm (BVLS) vs production NNLS.
    alt=lsq_linear(A,B,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=1000)
    coef=np.array(fit.coefficients)*fit.scales
    if not alt.success or np.max(np.abs(A@alt.x-A@coef))>1e-8:
        raise AssertionError('independent BVLS solution disagrees')
    H=A.T@A;J=x.T*weights
    active=coef>1e-10
    response=np.zeros_like(J)
    response[active]=np.linalg.solve(H[np.ix_(active,active)],J[active])
    np.testing.assert_allclose(response@y,coef,atol=1e-9,rtol=1e-9)
    # z_active >= 0 and gradient_inactive >= 0 define this NNLS active region.
    conditions=np.vstack((response[active],(H@response-J)[~active]))
    return rows,response,conditions,{'solver':'scipy lsq_linear BVLS', 'prediction_difference':float(np.max(np.abs(A@alt.x-A@coef))),
                                  'active':active.tolist(),'hessian_condition':float(np.linalg.cond(H))}


def constrained_attack(scores,influence,conditions,condition_values,target):
    size=len(influence[target]);a=[];b=[]
    for n in scores:
        if n!=target:
            a.append(np.r_[influence[target]-influence[n],0.]);b.append(scores[n]-scores[target]-1e-7)
    # Conditions remain in the same active region; this makes linearity exact.
    a.extend(np.c_[-conditions,np.zeros(len(conditions))]);b.extend(condition_values)
    a.extend(np.c_[np.eye(size),-np.ones(size)]);a.extend(np.c_[-np.eye(size),-np.ones(size)]);b.extend([0.]*(2*size))
    a=np.array(a);b=np.array(b);obj=np.r_[np.zeros(size),1.]
    result=linprog(obj,A_ub=a,b_ub=b,bounds=[(None,None)]*size+[(0.,.5)],method='highs',
                   options={'time_limit':5.,'maxiter':10000,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:return {'success':False,'status':int(result.status),'reason':result.message},None
    primal=max(0.,float(np.max(a@result.x-b)))
    dual=float(np.max(np.abs(obj-a.T@result.ineqlin.marginals-result.lower.marginals-result.upper.marginals)))
    dv=float(b@result.ineqlin.marginals+.5*result.upper.marginals[-1])
    gap=abs(float(result.fun)-dv)
    comp=float(np.max(np.abs(result.ineqlin.marginals*(b-a@result.x))))
    sign=max(0.,float(np.max(result.ineqlin.marginals)), -float(result.lower.marginals[-1]),float(result.upper.marginals[-1]),
             float(np.max(np.abs(result.lower.marginals[:-1]))),float(np.max(np.abs(result.upper.marginals[:-1]))))
    boundcomp=max(abs(float(result.lower.marginals[-1]*result.x[-1])),abs(float(result.upper.marginals[-1]*(.5-result.x[-1]))))
    if max(primal,dual,gap,comp,sign,boundcomp)>1e-7:raise AssertionError('LP primal/dual certificate failed')
    return {'success':True,'epsilon_fractional':float(result.fun),'primal_violation':primal,'dual_residual':dual,'duality_gap':gap,
            'complementarity':max(comp,boundcomp),'dual_sign_error':sign},result.x[:-1]


def sensitivity(data):
    results=[]
    for surface in data.validate().values():
        first=next(iter(surface.values()))
        if first.shape not in REVIEW_SHAPES:continue
        train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=first.group_id))
        model=TrafficLatencySelector.fit(train)
        candidates={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in surface.items()}
        chosen=model.select(candidates,baseline=data.baseline,allow_extrapolation=True)
        rows=list(train.rows);index={(r.shape,r.family):i for i,r in enumerate(rows)};conditions=[];numerical=[];influence={}
        for host in (True,False):
            sub,R,C,check=matrix(train,model,host);numerical.append(check)
            ids=[index[(r.shape,r.family)] for r in sub];y=np.array([r.measured_ms for r in sub])
            full=np.zeros((len(C),len(rows)));full[:,ids]=C*y
            conditions.extend(full)
            fit=model.host if host else model.fused
            for n in chosen['predicted_ms']:
                r=candidates[n]
                if r.is_host!=host:continue
                feat=np.array(host_features(r,model.input_threshold,model.depth_threshold) if host else fused_features(r,model.capacity))/fit.scales
                v=np.zeros(len(rows));v[ids]=feat@R*y;influence[n]=v
                assert abs(np.sum(v)-chosen['predicted_ms'][n])<1e-9
        C=np.array(conditions);base=C@np.ones(len(rows));targets=[]
        for n in chosen['predicted_ms']:
            regret=surface[n].measured_ms/min(r.measured_ms for r in surface.values())-1
            if regret<=.05:continue
            cert,delta=constrained_attack(chosen['predicted_ms'],influence,C,base,n)
            entry={'target':n,'observed_regret':regret,**cert}
            if delta is not None:
                perturbed=replace(train,rows=tuple(replace(r,measured_ms=r.measured_ms*(1+float(d))) for r,d in zip(rows,delta)))
                fixed_fits=[]
                changed={(r.shape,r.family):r.measured_ms for r in perturbed.rows}
                for is_host in (True,False):
                    original_rows, original_weights=training_rows_weights(list(train.validate().values()),is_host)
                    basis=[host_features(r,model.input_threshold,model.depth_threshold) if is_host else fused_features(r,model.capacity) for r in original_rows]
                    params=model.host if is_host else model.fused
                    fixed_fits.append(_solve(basis,[changed[(r.shape,r.family)] for r in original_rows],original_weights,params.names,
                                             model.host_ridge if is_host else model.ridge))
                h,f=fixed_fits
                fixed=replace(model,host=h,fused=f).select(candidates,baseline=data.baseline,allow_extrapolation=True)
                assert fixed['choice']==n
                assert max(abs(fixed['predicted_ms'][k]-v-float(influence[k]@delta)) for k,v in chosen['predicted_ms'].items())<1e-8
                refit=TrafficLatencySelector.fit(perturbed)
                actual=refit.select(candidates,baseline=data.baseline,allow_extrapolation=True)
                entry.update(full_refit_choice=actual['choice'],full_refit_capacity=refit.capacity,full_refit_ridge=refit.ridge,
                             full_refit_confirms_target=actual['choice']==n,witness_fractional_changes=delta.tolist())
            targets.append(entry)
        witnesses=[t for t in targets if t['success'] and t['full_refit_confirms_target']]
        smallest=min(witnesses,key=lambda t:t['epsilon_fractional']) if witnesses else None
        results.append({'shape':first.shape,'choice':chosen['choice'],'numerical':numerical,
                        'training_catalog':[{'shape':r.shape,'family':r.family} for r in rows], 'targets':targets,
                        'smallest_confirmed_witness':{k:v for k,v in smallest.items() if k!='witness_fractional_changes'} if smallest else None})
    return {'interpretation':'LP is exact only with frozen training weights, active sets and hyperparameters. Complete refits recompute weights and all inner selections; only confirmed witnesses count. Conditional LP radii are not global robustness guarantees or noise probabilities.', 'records':results}


def deletion(data,primary):
    result={tuple(r['shape']):{'shape':r['shape'],'original_choice':r['choices'][METHOD],'deletions':[]} for r in primary['decisions']}
    groups=sorted({r.group_id for r in data.rows})
    for held in groups:
        test=replace(data,rows=tuple(r for r in data.rows if r.group_id==held))
        for deleted in groups:
            if deleted==held:continue
            train=replace(data,rows=tuple(r for r in data.rows if r.group_id not in (held,deleted)))
            model=TrafficLatencySelector.fit(train)
            for s in test.validate().values():
                r=next(iter(s.values()));c={n:ScheduleCandidate.from_measurement(v,data.environment_id) for n,v in s.items()}
                choice=model.select(c,baseline=data.baseline,allow_extrapolation=True)['choice']
                result[r.shape]['deletions'].append({'removed_group':deleted,'choice':choice,'regret':s[choice].measured_ms/min(v.measured_ms for v in s.values())-1})
    for r in result.values():
        r['all_deletions_keep_choice']=all(d['choice']==r['original_choice'] for d in r['deletions'])
        r['worst_deletion_regret']=max(d['regret'] for d in r['deletions'])
        r['choice_counts']=dict(Counter(d['choice'] for d in r['deletions']))
    return list(result.values())


def replay_previous_attack(data):
    old=json.loads((ROOT/'experiments/2026-10-21-additive-selector-repair/verified/review_results.json').read_text())
    result=[]
    for record in old['sensitivity']['records']:
        shape=tuple(record['shape']);target=record['smallest_confirmed_witness']['target']
        witness=next(t for t in record['targets'] if t['target']==target)
        by_id={(tuple(r['shape']),r['family']):d for r,d in zip(record['training_catalog'],witness['witness_fractional_changes'])}
        test=next(s for s in data.validate().values() if s['host'].shape==shape)
        group=test['host'].group_id;train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=group))
        assert set(by_id)=={(r.shape,r.family) for r in train.rows}
        changed=replace(train,rows=tuple(replace(r,measured_ms=r.measured_ms*(1+by_id[(r.shape,r.family)])) for r in train.rows))
        model=TrafficLatencySelector.fit(changed)
        candidates={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in test.items()}
        choice=model.select(candidates,baseline=data.baseline,allow_extrapolation=True)['choice']
        result.append({'shape':shape,'old_wrong_target':target,'old_epsilon_fractional':witness['epsilon_fractional'],
                       'new_choice':choice,'new_regret':test[choice].measured_ms/min(r.measured_ms for r in test.values())-1})
    return result


def boundary_stress(data):
    # Fixed seeds and budget. A successful finite stress suite is not a proof
    # covering the continuous label box or unmeasured hardware variation.
    shape=(4,64,4096,2048)
    surface=next(s for s in data.validate().values() if s['host'].shape==shape)
    train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=surface['host'].group_id))
    candidates={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in surface.items()}
    rng=np.random.default_rng(20261021);directions=[]
    for i in range(32):directions.append((f'seeded_signs_{i}',rng.choice([-1.,1.],len(train.rows))))
    directions.extend([('host_up_fused_down',np.array([1. if r.is_host else -1. for r in train.rows])),
                       ('host_down_fused_up',np.array([-1. if r.is_host else 1. for r in train.rows]))])
    records=[]
    for name,direction in directions:
        changed=replace(train,rows=tuple(replace(r,measured_ms=r.measured_ms*(1+.01*float(v))) for r,v in zip(train.rows,direction)))
        model=TrafficLatencySelector.fit(changed)
        choice=model.select(candidates,baseline=data.baseline,allow_extrapolation=True)['choice']
        records.append({'perturbation':name,'choice':choice,'regret':surface[choice].measured_ms/min(r.measured_ms for r in surface.values())-1})
    return {'shape':shape,'fractional_bound':.01,'seed':20261021,'records':records,'all_correct':all(r['regret']==0 for r in records)}


def main(results_path,out):
    if out.exists():raise ValueError('fresh output required')
    start=time.monotonic();results=json.loads(results_path.read_text())
    for path,digest in results['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    review={'sensitivity':sensitivity(data),'deletion':deletion(data,results['primary']),
            'previous_attacks':replay_previous_attack(data),'boundary_stress':boundary_stress(data),
            'results_sha256':sha256(results_path),'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))},
            'elapsed_seconds':time.monotonic()-start}
    out.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'sensitivity':[{'shape':r['shape'],'witness':r['smallest_confirmed_witness']} for r in review['sensitivity']['records']],
                      'previous_attacks':review['previous_attacks'],'boundary_stress_all_correct':review['boundary_stress']['all_correct'],
                      'stable_after_deletion':sum(r['all_deletions_keep_choice'] for r in review['deletion']),
                      'priority_deletion':[{'shape':r['shape'],'counts':r['choice_counts'],'worst_regret':r['worst_deletion_regret']} for r in review['deletion'] if tuple(r['shape']) in PRIORITY],
                      'elapsed_seconds':review['elapsed_seconds']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.results,a.out)
