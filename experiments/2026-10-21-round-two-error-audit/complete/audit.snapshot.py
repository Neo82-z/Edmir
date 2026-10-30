"""Bounded posthoc audit of the frozen second-round model; no model promotion."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.decision_loss_selection import DecisionLossSelector, features, tensors, objective, FIXED_LOSS
from edmir.compiler.traffic_latency_selection import HOST_NAMES,FUSED_NAMES,host_features,fused_features
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from predict_traffic_schedule import static_candidates
from run_selector_gpu import write_json
OLD=ROOT/'experiments/2026-10-21-decision-loss-iterations'
NAMES=tuple('host.'+n for n in HOST_NAMES)+tuple('fused.'+n for n in FUSED_NAMES)
LOWER=1e-10


def basis(model):
    return {k:getattr(model.prior,k) for k in ('input_threshold','depth_threshold','capacity')}


def raw(row,geometry):
    return np.asarray(features(row,SimpleNamespace(**geometry),1.,'time'))


def canonical(data):
    return replace(data,rows=tuple(sorted(data.rows,key=lambda r:(r.shape,r.family,r.group_id))))


def arrays(data,geometry):
    surfaces=[s for _,s in sorted(data.validate().items())]
    size=max(map(len,surfaces));x=np.zeros((len(surfaces),size,len(NAMES)))
    y=np.zeros((len(surfaces),size));w=np.zeros_like(y);mask=np.zeros_like(y,dtype=bool)
    names=[];shape=[]
    for i,s in enumerate(surfaces):
        ns=tuple(sorted(s));names.append(ns);shape.append(s['host'].shape)
        for j,n in enumerate(ns):
            x[i,j]=raw(ScheduleCandidate.from_measurement(s[n],data.environment_id),geometry)
            y[i,j]=s[n].measured_ms;mask[i,j]=True;w[i,j]=.5 if n=='host' else .5/(len(s)-1)
    scales=np.maximum(np.sqrt(np.sum(x*x*w[...,None],axis=(0,1))/len(surfaces)),.1)
    return x/scales,y,w,mask,scales,names,shape


def explicit_system(x,y,w,mask):
    """Independently enumerate every pair (no weighted-variance shortcut)."""
    count=len(x);a=[];b=[];h=[];d=[]
    for i in range(count):
        j=int(mask[i].sum());norm=float(y[i,:j].min());best=int(np.argmin(y[i,:j]))
        pairs=[(p,q) for p in range(j) for q in range(p+1,j)]
        mass=sum(w[i,p]*w[i,q] for p,q in pairs)
        for p in range(j):
            scale=math.sqrt(w[i,p]/count)/norm
            a.append(x[i,p]*scale);b.append(y[i,p]*scale)
            if p!=best:
                h.append((x[i,p]-x[i,best])/(norm*math.sqrt(count)))
                d.append((y[i,p]-y[i,best])/(norm*math.sqrt(count)))
        for p,q in pairs:
            scale=math.sqrt(w[i,p]*w[i,q]/mass/count)/norm
            a.append((x[i,p]-x[i,q])*scale);b.append((y[i,p]-y[i,q])*scale)
    a.extend(math.sqrt(FIXED_LOSS[2])*np.eye(x.shape[-1]));b.extend(np.zeros(x.shape[-1]))
    return np.asarray(a),np.asarray(b),np.asarray(h),np.asarray(d)


def explicit_objective(theta,system):
    a,b,h,d=system
    e=a@theta-b;hinge=np.maximum(d-h@theta,0.)
    return float(e@e+hinge@hinge),2*a.T@e-2*h.T@hinge


def polish(theta,system):
    """Solve the current piecewise-quadratic active region exactly, if valid."""
    a,b,h,d=system
    active=(d-h@theta)>0;free=theta>LOWER+1e-8
    aa=np.vstack((a,h[active]));bb=np.r_[b,d[active]]
    result=np.full_like(theta,LOWER)
    result[free]=np.linalg.solve(aa[:,free].T@aa[:,free],aa[:,free].T@(bb-aa[:,~free]@result[~free]))
    value,g=explicit_objective(result,system)
    pg=np.where((result<=1.01e-10)&(g>0),0.,g)
    valid=(np.min(result)>=LOWER-1e-12 and np.max(np.abs(pg))<1e-9)
    return result,{'valid':bool(valid),'projected_gradient':float(np.max(np.abs(pg))),
                   'free_parameters':int(free.sum()),'active_hinges':int(active.sum()),
                   'active_hessian_condition':float(np.linalg.cond(aa[:,free].T@aa[:,free]))}


def solve(data,geometry,initial=None):
    x,y,w,mask,scales,names,shapes=arrays(data,geometry);system=explicit_system(x,y,w,mask)
    if initial is None:initial=np.full(x.shape[-1],.05)
    fun=lambda t:explicit_objective(t,system)
    opt=minimize(fun,initial,method='SLSQP',jac=True,bounds=[(LOWER,None)]*len(initial),
                 options={'ftol':1e-14,'maxiter':2000})
    if not opt.success:raise ArithmeticError(str(opt.message))
    theta,check=polish(opt.x,system)
    if not check['valid']:raise ArithmeticError('active-region polish failed: '+str(check))
    f,g=fun(theta)
    # Strong convexity from ridge: minimize its global quadratic lower bound
    # over the actual coefficient bounds. The remaining objective gap is bounded.
    delta=np.maximum(LOWER-theta,-g/(2*FIXED_LOSS[2]))
    bound=max(0.,float(-g@delta-FIXED_LOSS[2]*(delta@delta)))
    return {'geometry':geometry,'parameters':theta.tolist(),'scales':scales.tolist(),
            'check':{**check,'success':True,'slsqp_iterations':int(opt.nit),'objective':f,
                     'global_objective_gap_upper_bound':bound},'training_shapes':[list(s) for s in shapes]}


def predict(fit,candidates):
    coef=np.asarray(fit['parameters'])/fit['scales']
    return {n:float(raw(r,fit['geometry'])@coef) for n,r in sorted(candidates.items())}


def choice(pred):return min(pred,key=lambda n:(pred[n],n!='host',n))


def signed_terms(fit,candidates,a,b):
    fa,fb=raw(candidates[a],fit['geometry']),raw(candidates[b],fit['geometry'])
    coef=np.asarray(fit['parameters'])/fit['scales'];gap=fa-fb
    values=gap*coef*1000
    return {'a':a,'b':b,'gap_us':float(values.sum()),
            'terms':[{'name':n,'feature_difference':float(dx),'coefficient_ms':float(c),'gap_us':float(v)}
                     for n,dx,c,v in zip(NAMES,gap,coef,values) if abs(v)>1e-8]}


def state(model):
    return {'geometry':basis(model),'parameters':list(model.parameters),'scales':list(model.scales),
            'host_cv_top':sorted(model.prior.host_cv_scores,key=lambda r:(r[-1],r[:-1]))[:6],
            'fused_cv_top':sorted(model.prior.cv_scores,key=lambda r:(r[-1],r[:-1]))[:4]}


def independent_case(train,model,candidates,selected,best):
    x,y,w,mask,scale,names,shapes=arrays(train,basis(model))
    xx,b,yy,mm,ww,norm,means,ss,ns=tensors(canonical(train),model.prior,'time')
    np.testing.assert_allclose(x,xx,rtol=0,atol=0);np.testing.assert_array_equal(y,yy)
    np.testing.assert_array_equal(mask,mm);np.testing.assert_array_equal(w,ww)
    system=explicit_system(x,y,w,mask);t=np.asarray(model.parameters)
    f,g=explicit_objective(t,system);old,oldg=objective(t,xx,b,yy,mm,ww,norm,*FIXED_LOSS)
    np.testing.assert_allclose(f,old,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(g,oldg,atol=1e-12)
    alt=solve(train,basis(model),initial=np.maximum(t*.8,LOWER));pred=predict(alt,candidates)
    original=model.select(candidates,baseline=train.baseline,allow_extrapolation=True)['predicted_ms']
    original={n:original[n] for n in pred}
    delta=np.maximum(LOWER-t,-g/(2*FIXED_LOSS[2]))
    bound=max(0.,float(-g@delta-FIXED_LOSS[2]*(delta@delta)))
    return {'alternative':alt,'original':state(model),'original_objective':f,
            'original_objective_gap_upper_bound':bound,'objective_improvement':f-alt['check']['objective'],
            'max_prediction_difference_us':max(abs(pred[n]-original[n])*1000 for n in pred),
            'original_choice':choice(original),'alternative_choice':choice(pred),
            'original_gap_us':1000*(original[selected]-original[best]),
            'alternative_gap_us':1000*(pred[selected]-pred[best]),
            'terms':signed_terms(alt,candidates,selected,best)}


def brief(fit,candidates):
    p=predict(fit,candidates)
    return {'geometry':fit['geometry'],'choice':choice(p),'predicted_ms':p,
            'best_fused_minus_host_us':1000*(min(v for n,v in p.items() if n!='host')-p['host'])}


def main(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def progress(stage):
        if time.monotonic()-start>600:raise TimeoutError('audit deadline')
        write_json(out/'progress.json',{'stage':stage,'elapsed_seconds':time.monotonic()-start})
        print(stage,flush=True)
    for relative,digest in json.loads((OLD/'delivery.json').read_text())['sha256'].items():
        assert sha256(ROOT/relative)==digest,relative
    v=json.loads((OLD/'verified/replay/traffic_gap.json').read_text())
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    surfaces={s['host'].shape:s for s in data.validate().values()}
    inventory=[];cases=[]
    progress('all outer errors and independent objective checks')
    for protocol in ('primary','stress'):
        for record in v[protocol]['decisions']:
            regret=record['diagnostics']['traffic_gap']['regret_all_observed']
            if regret<=1e-12:continue
            held=record['heldout'];shape=tuple(record['shape']);selected=record['choices']['traffic_gap']
            actual=record['times_ms'];best=min(actual,key=actual.get)
            model=DecisionLossSelector.from_dict(next(f['model'] for f in v[protocol]['fits'] if f['heldout']==held))
            train=replace(data,rows=tuple(r for r in data.rows if (r.group_id if protocol=='primary' else r.k)!=held))
            allc={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in surfaces[shape].items()}
            eligible=record['predictions']['predicted_ms']
            supported_best=min(eligible,key=actual.get)
            inventory.append({'protocol':protocol,'heldout':held,'shape':list(shape),'selected':selected,'observed_best':best,
                              'regret':regret,'observed_difference_us':1000*(actual[selected]-actual[best]),
                              'observed_best_eligible':best in eligible,'best_eligible':supported_best,
                              'regret_within_eligible':actual[selected]/actual[supported_best]-1})
            # All remaining errors, not just the large original priority point.
            c={n:allc[n] for n in eligible}
            check=independent_case(train,model,c,selected,supported_best)
            cases.append({'protocol':protocol,'heldout':held,'shape':list(shape),**check})
    write_json(out/'outer_errors.json',{'inventory':inventory,'independent_checks':cases})
    progress('all boundary deletion interventions')
    deletions=[]
    for shape in ((4,512,2048,512),(4,512,4096,512),(4,512,8192,512),(4,64,4096,2048)):
        s=surfaces[shape];held=s['host'].group_id
        train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=held))
        original=DecisionLossSelector.fit(train,method='traffic_gap')
        c={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items() if n in ('host','splitk2','splitk4')}
        records=[]
        for removed in sorted({r.group_id for r in train.rows}):
            changed=replace(train,rows=tuple(r for r in train.rows if r.group_id!=removed))
            fitted=DecisionLossSelector.fit(changed,method='traffic_gap')
            frozen=solve(changed,basis(original))
            records.append({'removed_group':removed,'retuned':brief(state(fitted),c),'frozen_geometry':brief(frozen,c),
                            'frozen_fit_check':frozen['check'],'host_cv_top':state(fitted)['host_cv_top']})
        deletions.append({'shape':list(shape),'original':brief(state(original),c),'records':records})
    write_json(out/'deletions.json',deletions)
    progress('34 label perturbations with frozen-geometry counterfactuals')
    shape=(4,64,4096,2048);s=surfaces[shape]
    train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=s['host'].group_id))
    model=DecisionLossSelector.fit(train,method='traffic_gap');geometry=basis(model)
    c={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items() if n in ('host','splitk2','splitk4')}
    rng=np.random.default_rng(20261021)
    directions=[(f'seeded_signs_{i}',rng.choice([-1.,1.],len(train.rows))) for i in range(32)]
    directions.extend([('host_up_fused_down',np.array([1. if r.is_host else -1. for r in train.rows])),
                       ('host_down_fused_up',np.array([-1. if r.is_host else 1. for r in train.rows]))])
    records=[];traces=[]
    for name,direction in directions:
        changed=replace(train,rows=tuple(replace(r,measured_ms=r.measured_ms*(1+.01*float(sign))) for r,sign in zip(train.rows,direction)))
        fitted=DecisionLossSelector.fit(changed,method='traffic_gap');frozen=solve(changed,geometry)
        records.append({'name':name,'retuned':brief(state(fitted),c),'frozen_geometry':brief(frozen,c),
                        'frozen_fit_check':frozen['check'],'host_cv_top':state(fitted)['host_cv_top']})
        old=next(r for r in v['review']['boundary_stress']['records'] if r['perturbation']==name)
        assert old['choice']==records[-1]['retuned']['choice']
        if old['regret']>1e-12:
            for epsilon in (0.,.0025,.005,.0075,.01):
                changed=replace(train,rows=tuple(replace(r,measured_ms=r.measured_ms*(1+epsilon*float(sign))) for r,sign in zip(train.rows,direction)))
                fitted=DecisionLossSelector.fit(changed,method='traffic_gap')
                traces.append({'name':name,'epsilon':epsilon,**brief(state(fitted),c),'host_cv_top':state(fitted)['host_cv_top']})
    write_json(out/'perturbations.json',{'original':brief(state(model),c),'records':records,'traces':traces})
    progress('K basis identifiability and cancellation')
    f=next(f for f in v['stress']['fits'] if f['heldout']==4096);model=DecisionLossSelector.from_dict(f['model'])
    train=replace(data,rows=tuple(r for r in data.rows if r.k!=4096))
    host=sorted((r for r in train.rows if r.is_host),key=lambda r:r.shape)
    a=np.array([host_features(r,2,.5)[3] for r in host]);b=np.array([host_features(r,2,1.)[3] for r in host])
    c=static_candidates(512,4096,512,data.environment_id)
    alternatives=[]
    for depth in (.5,1.):
        geometry={**basis(model),'depth_threshold':depth};fit=solve(train,geometry)
        alternatives.append({'geometry':geometry,'fit':fit,'prediction':brief(fit,{n:c[n] for n in ('host','splitk2','splitk4')})})
    geom=basis(model);curve=[];fit=state(model)
    for k in (512,1024,2048,3072,4096,5120,6144,8192):
        cc=static_candidates(512,k,512,data.environment_id)
        gap=signed_terms(fit,cc,'splitk4','splitk2')['gap_us']
        actual=surfaces.get((4,512,k,512))
        curve.append({'K':k,'predicted_q4_minus_q2_us':gap,
                      'observed_q4_minus_q2_us':1000*(actual['splitk4'].measured_ms-actual['splitk2'].measured_ms) if actual else None})
    slope=(curve[-1]['predicted_q4_minus_q2_us']-curve[0]['predicted_q4_minus_q2_us'])/(8192-512)
    crossing=512-curve[0]['predicted_q4_minus_q2_us']/slope
    rank={'train_K':sorted({r.k for r in train.rows}),'depth_05_equals_15_times_depth_1_max_abs':float(np.max(np.abs(a-1.5*b))),
          'normalized_columns_identical':bool(np.allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),atol=1e-14,rtol=0)),
          'host_inner_scores':state(model)['host_cv_top'],'alternatives_posthoc':alternatives,
          'q4_q2_curve':curve,'predicted_crossing_K':crossing}
    write_json(out/'identifiability.json',rank)
    original_deletion_errors=[{'shape':r['shape'],**d} for r in v['review']['deletion'] for d in r['deletions'] if d['regret']>1e-12]
    sources=[Path(__file__),HERE/'design.json',ROOT/'EDM-IR/edmir/compiler/decision_loss_selection.py',ROOT/'EDM-IR/edmir/compiler/traffic_latency_selection.py']
    write_json(out/'summary.json',{'outer_error_count':len(inventory),'independent_checks':len(cases),
        'all_alternate_choices_equal':all(r['alternative_choice']==r['original_choice'] for r in cases),
        'max_alternate_prediction_difference_us':max(r['max_prediction_difference_us'] for r in cases),
        'all_deletion_error_count':len(original_deletion_errors),'all_deletion_errors':original_deletion_errors,
        'N_perturbation_retuned_host':sum(r['retuned']['choice']=='host' for r in records),
        'N_perturbation_frozen_host':sum(r['frozen_geometry']['choice']=='host' for r in records),
        'old_results_sha256':sha256(OLD/'verified/replay/traffic_gap.json'),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},'gpu_operations':0,'compiler_admission':False,
        'elapsed_seconds':time.monotonic()-start})
    progress('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
