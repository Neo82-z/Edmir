"""Exact design aliases and conditional LP prediction ambiguity certificates.

The LP error envelope is model discrepancy on training data. It is not a
hardware-noise estimate, confidence interval, or guarantee outside this basis.
Held-out times are used only after solving to report decision loss.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog, lsq_linear
from candidate import Candidate, ROOT, fused_features, host_features, FUSED_NAMES, HOST_NAMES, fit_path
from edmir.compiler.resource_geometry import ResourceGeometry, axis_interaction
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.traffic_latency_selection import fused_features as old_fused_features, training_rows_weights
from predict_traffic_schedule import static_candidates


def certified_lp(c, a, b, bounds):
    a,b,c=np.asarray(a,dtype=float),np.asarray(b,dtype=float),np.asarray(c,dtype=float)
    result=linprog(c,A_ub=a,b_ub=b,bounds=bounds,method='highs',options={
        'time_limit':5.,'maxiter':10000,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.status==2:return None,{'status':'infeasible','solver_status':2}
    if not result.success:raise RuntimeError(result.message)
    low=np.array([(-np.inf if lo is None else lo) for lo,hi in bounds])
    high=np.array([(np.inf if hi is None else hi) for lo,hi in bounds])
    primal=max(0.,float(np.max(a@result.x-b)),float(np.max(low-result.x)),float(np.max(result.x-high)))
    dual=float(np.max(np.abs(c-a.T@result.ineqlin.marginals-result.lower.marginals-result.upper.marginals)))
    dv=float(b@result.ineqlin.marginals + np.sum(low[np.isfinite(low)]*result.lower.marginals[np.isfinite(low)]) + np.sum(high[np.isfinite(high)]*result.upper.marginals[np.isfinite(high)]))
    gap=abs(float(result.fun)-dv)
    sign=max(0.,float(np.max(result.ineqlin.marginals)),float(np.max(-result.lower.marginals)),float(np.max(result.upper.marginals)))
    if max(primal,dual,gap,sign)>1e-7:raise AssertionError('LP numerical certificate failed')
    return result.x,{'status':'optimal','primal_violation':primal,'dual_residual':dual,'duality_gap':gap,'dual_sign_violation':sign}


def design(train,host,param):
    rows=sorted([r for r in train.rows if r.is_host==host],key=lambda r:(r.shape,r.family))
    features=lambda r:host_features(r,*param) if host else fused_features(ScheduleCandidate.from_measurement(r,train.environment_id),param)
    raw=np.array([features(r) for r in rows]);y=np.array([r.measured_ms for r in rows])
    scale=np.maximum(np.sqrt(np.mean(raw*raw,0)),1e-9);x=raw/scale
    singular=np.linalg.svd(x/np.maximum(np.linalg.norm(x,axis=0),1e-12),compute_uv=False)
    a=np.vstack((np.c_[x,-y],np.c_[-x,-y]));b=np.r_[y,-y]
    sol,certificate=certified_lp(np.r_[np.zeros(x.shape[1]),1.],a,b,[(0.,None)]*(x.shape[1]+1))
    if sol is None:raise AssertionError('minimax should always be feasible')
    error=float(sol[-1]);observed=float(np.max(np.abs(x@sol[:-1]/y-1)))
    if abs(observed-error)>1e-7:raise AssertionError('minimax replay failed')
    names=HOST_NAMES if host else FUSED_NAMES
    return x,y,scale,{'rows':len(y),'columns':x.shape[1],'rank':int(np.linalg.matrix_rank(x)),
                     'zero_columns':[name for name,col in zip(names,raw.T) if np.max(np.abs(col))==0],
                     'normalized_singular_values':singular.tolist(),'minimax_relative_error':error,
                     'minimax_coefficients':(sol[:-1]/scale).tolist(),'certificate':certificate}


def ambiguity(train,surface):
    model=Candidate.fit(train)
    hp,hr,hfit=model.host_fits[0];fp,fr,ffit=model.fused_fits[0]
    hx,hy,hs,hd=design(train,True,hp);fx,fy,fs,fd=design(train,False,fp)
    # Independently verify the actual ridge NNLS fits using BVLS.
    checks=[]
    for host,param,ridge,fit in ((True,hp,hr,hfit),(False,fp,fr,ffit)):
        ss=[{n:s[n] for n in sorted(s)} for _,s in sorted(train.validate().items())]
        rows,w=training_rows_weights(ss,host)
        x=np.array([host_features(r,*param) if host else fused_features(ScheduleCandidate.from_measurement(r,train.environment_id),param) for r in rows])/fit.scales
        w=np.array(w);y=np.array([r.measured_ms for r in rows]);pen=np.eye(x.shape[1])*ridge**.5;pen[0,0]=0.
        A=np.vstack((x*w[:,None],pen));B=np.r_[y*w,np.zeros(x.shape[1])]
        alt=lsq_linear(A,B,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=1000)
        difference=float(np.max(np.abs(A@(np.array(fit.coefficients)*fit.scales)-A@alt.x)))
        if not alt.success or difference>1e-8:raise AssertionError('independent solver mismatch')
        checks.append({'path':'host' if host else 'fused','max_prediction_difference':difference,'kkt_violation':fit.kkt_max_violation})
    candidates={n:ScheduleCandidate.from_measurement(r,train.environment_id) for n,r in surface.items()}
    baseline=model.select(candidates,baseline=train.baseline,allow_extrapolation=True)
    H,F=hx.shape[1],fx.shape[1];width=H+F
    vectors={}
    for n in baseline['predicted_ms']:
        r=candidates[n]
        vectors[n]=np.r_[np.array(host_features(r,*hp))/hs,np.zeros(F)] if r.is_host else np.r_[np.zeros(H),np.array(fused_features(r,fp))/fs]
    records=[]
    for allowance in (0.,.01):
        he,fe=hd['minimax_relative_error']+allowance,fd['minimax_relative_error']+allowance
        a=np.vstack((np.c_[hx,np.zeros((len(hx),F))],np.c_[-hx,np.zeros((len(hx),F))],
                     np.c_[np.zeros((len(fx),H)),fx],np.c_[np.zeros((len(fx),H)),-fx]))
        b=np.r_[hy*(1+he),-hy*(1-he),fy*(1+fe),-fy*(1-fe)]
        winners=[]
        for target in vectors:
            # Maximize a common strict margin against every other candidate.
            # This avoids presenting a nearly exact tie as a large ambiguity.
            aa=np.vstack((np.c_[a,np.zeros(len(a))],
                          [np.r_[vectors[target]-v,1.] for n,v in vectors.items() if n!=target]))
            bb=np.r_[b,[0.]*(len(vectors)-1)]
            solution,certificate=certified_lp(np.r_[np.zeros(width),-1.],aa,bb,[(0.,None)]*width+[(0.,None)])
            if solution is None or solution[-1]<=1e-7:continue
            coef=solution[:-1];margin=float(solution[-1])
            scores={n:float(v@coef) for n,v in vectors.items()}
            selected=min(scores,key=lambda n:(scores[n],n!='host',n))
            if selected!=target:raise AssertionError('winning-witness replay failed')
            herr=float(np.max(np.abs(hx@coef[:H]/hy-1)));ferr=float(np.max(np.abs(fx@coef[H:]/fy-1)))
            if herr>he+1e-7 or ferr>fe+1e-7:raise AssertionError('witness outside training envelope')
            if abs(min(v-scores[target] for n,v in scores.items() if n!=target)-margin)>1e-7:
                raise AssertionError('margin certificate does not replay')
            winners.append({'choice':target,'max_winning_margin_ms':margin,'predicted_ms':scores,'host_max_training_relative_error':herr,
                            'fused_max_training_relative_error':ferr,'host_coefficients':(coef[:H]/hs).tolist(),
                            'fused_coefficients':(coef[H:]/fs).tolist(),'certificate':certificate,
                            'heldout_regret_reported_after_solving':surface[target].measured_ms/min(r.measured_ms for r in surface.values())-1})
        records.append({'extra_model_error_allowance':allowance,'possible_winners':winners})
    return {'shape':surface['host'].shape,'training_shapes':model.training_shapes,'host_parameters':hp,'fused_pressure_scale':fp,
            'ridge_choice':baseline['choice'],'host_design':hd,'fused_design':fd,'independent_solver_checks':checks,'ambiguity':records}


def main(out):
    if out.exists():raise ValueError('fresh output required')
    start=time.monotonic();data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    a=static_candidates(64,8192,2048,data.environment_id)['splitk4']
    b=static_candidates(256,8192,512,data.environment_id)['splitk4']
    assert old_fused_features(a,32)==old_fused_features(b,32)
    assert fused_features(a,32)!=fused_features(b,32)
    ga,gb=ResourceGeometry.from_candidate(a),ResourceGeometry.from_candidate(b)
    assert ga.logical_a_load_bytes==gb.logical_a_load_bytes and ga.logical_b_load_bytes==gb.logical_b_load_bytes
    rows=[s['host'] for s in data.validate().values()]
    x=np.array([[r.k/4096,r.k/4096*r.rows_per_peer/64,r.k/4096*r.n/512,r.k/4096*r.rows_per_peer/64*r.n/512] for r in rows])
    null=np.array([1.,-1.,-1.,1.]);assert np.max(np.abs(x@null))==0
    assert all(axis_interaction(ScheduleCandidate.from_measurement(r,data.environment_id))==0 for r in rows)
    records=[]
    for shape,deleted in [((4,512,4096,512),None),((4,512,4096,512),'m256-n512-longk'),
                          ((4,64,4096,2048),None),((4,64,4096,2048),'m256-n512-longk')]:
        surface=next(s for s in data.validate().values() if s['host'].shape==shape)
        excluded={surface['host'].group_id,deleted}
        train=replace(data,rows=tuple(r for r in data.rows if r.group_id not in excluded))
        r=ambiguity(train,surface);r['additional_deleted_group']=deleted;records.append(r)
        print(json.dumps({'shape':shape,'deleted':deleted,'host_minimax':r['host_design']['minimax_relative_error'],
             'fused_minimax':r['fused_design']['minimax_relative_error'],
             'possible_winners':[[w['choice'] for w in e['possible_winners']] for e in r['ambiguity']]},indent=2),flush=True)
    sources=[Path(__file__),Path(__file__).with_name('candidate.py'),ROOT/'EDM-IR/edmir/compiler/resource_geometry.py']
    result={'scope':'Conditional model-discrepancy analysis, not hardware uncertainty or independent accuracy',
            'old_alias':{'shape_a':a.shape,'shape_b':b.shape,'a':ga.to_dict(),'b':gb.to_dict(),
                         'new_feature_vectors_differ':True,'old_feature_vectors_equal':True,
                         'interpretation':'Different working sets and reuse directions; equal total logical A/B loads. Timing difference cannot be attributed to aggregate replay bytes alone.'},
            'axes_alias':{'columns':['K','K*M/64','K*N/512','K*M/64*N/512'],'rank':int(np.linalg.matrix_rank(x)),
                          'columns_count':4,'null_vector':null.tolist(),'max_null_residual':float(np.max(np.abs(x@null))),
                          'zero_interaction_on_all_41_shapes':True,
                          'interpretation':'The existing domain guard must remain. Coefficients of cross-M/N effects cannot be recovered from these axes.'},
            'records':records,'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},'elapsed_seconds':time.monotonic()-start}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
