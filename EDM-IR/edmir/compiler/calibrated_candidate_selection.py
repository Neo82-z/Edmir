"""Latency-first candidate scoring with explicit data support.

Cost identities, empirical service coefficients, and measured labels remain
separate. This development model never replaces the admitted compiler policy.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import inspect
import math
import time

from . import precision_candidate_selection as old
from . import additive_latency_selection as additive
from .shared_candidate_selection import canonical_choice

SCHEMA = 'edmir.calibrated-candidate-selection.v1'
SETTINGS = {'epochs':1000, 'learning_rate':.003, 'weight_decay':.001,
            'temperature':.05, 'seeds':[0,1], 'gradient_clip':10., 'width':8,
            'max_seconds_per_seed':20., 'schedule':'linear decay to 1% of initial LR',
            'muon_adjust_lr_fn':'match_rms_adamw', 'network_dtype':'float32'}
HOST_NAMES = additive.HOST_NAMES + ('remote_input_MK', 'padded_gemm_work_MKN')


def calibrated_loss(score, target, mask, weights, temperature=.05):
    """All three data terms have zero gradient at score == log(measured time).

    Pair error is half the weighted squared error of all log-time differences,
    equal to the weighted variance of per-candidate log-time errors. CE is
    scaled by tau^2 to remove its leading inverse-temperature curvature.
    Expected decision regret is reported, not optimized as a timing target.
    """
    import torch
    _, metrics = old.losses(score,target,mask,weights,temperature=temperature)
    if not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
        raise ValueError('finite nonnegative weights required')
    if not bool(torch.allclose(weights.sum(1),torch.ones_like(weights.sum(1)))) or bool((weights[~mask]!=0).any()):
        raise ValueError('one unit of observed candidate mass per shape required')
    error = torch.where(mask,score,torch.zeros_like(score))-torch.where(mask,target,torch.zeros_like(target))
    centered = error-(weights*error).sum(1,keepdim=True)
    pair = (weights*centered.square()).sum(1).mean()
    value = metrics['anchor']+pair+temperature**2*metrics['kl_to_cost_target']
    return value,{**metrics,'symmetric_log_gap':pair,
                  'expected_regret_is_training_objective':False}


def host_workload_features(row):
    """Historical per-source FP16 CUTLASS only; counts are not service times."""
    if row['candidate_id']!='host' or row['shape'][0]!=4 or row['precision']['input']!='fp16':
        raise ValueError('host basis is specific to historical P4 per-source FP16 CUTLASS')
    p,m,k,n=row['shape']; mt=(m+63)//64; nt=(n+127)//128
    kk,nn=k/4096,n/512; lm=math.log2(mt)
    # Unique remote inputs per destination and padded work across its P GEMMs,
    # normalized by the explicit reference counts at P4/M64/K4096/N512.
    a=(p-1)*m*k/(3*64*4096)
    work=p*mt*nt*64*128*k/(4*64*512*4096)
    return (1.,kk,kk*kk,kk*lm,kk*kk*lm,kk*nn,kk*kk*nn,a,work)


def make_prior(rows, prior, kind):
    if kind not in ('original','workload'):raise ValueError('unknown host basis')
    if prior is None:
        if kind!='original':raise ValueError('no measured P8 service prior; do not transfer P4 coefficients')
        return None
    shapes={tuple(r['shape']) for r in rows}
    if shapes!={tuple(s) for s in prior['training_shapes']} or prior['environment_id']!=rows[0]['execution_context']:
        raise ValueError('cost prior must use exactly this training fold and context')
    if any(r['precision']['input']!='fp16' or r['precision']['output']!='fp16' for r in rows):
        raise ValueError('historical prior cannot be applied to another precision')
    out={'original':prior,'host_basis':kind,'host':None}
    if kind=='workload':
        hosts=sorted((r for r in rows if r['is_host']),key=lambda r:tuple(r['shape']))
        fit=additive._solve([host_workload_features(r) for r in hosts],
             [r['latency_us']/1000 for r in hosts],[1.]*len(hosts),HOST_NAMES,prior['ridge'])
        out['host']=asdict(fit)
        out['coefficient_meaning']='empirical end-to-end cost terms, not identified bandwidth or stage service time'
    return out


def prior_scores(prior,rows):
    import numpy as np
    if prior is None:return np.zeros(len(rows))
    score=old.prior_model.prior_scores(prior['original'],rows)
    if prior['host'] is not None:
        h=prior['host'];fit=additive.LatencyFit(tuple(h['names']),tuple(h['coefficients']),tuple(h['scales']),h['kkt_max_violation'])
        fit.validate(HOST_NAMES)
        for i,r in enumerate(rows):
            if r['is_host']:score[i]=math.log(1000*fit.predict(host_workload_features(r)))
    return np.asarray(score)


def make_optimizers(net,name):
    import torch
    if name not in ('adamw','muon'):raise ValueError('only AdamW or official PyTorch Muon')
    kw={'lr':SETTINGS['learning_rate'],'weight_decay':SETTINGS['weight_decay']}
    if name=='adamw':return [torch.optim.AdamW(net.parameters(),**kw)]
    scalar=[p for n,p in net.named_parameters() if n!='hidden.weight']
    return [torch.optim.Muon([net.hidden.weight],**kw,momentum=.95,nesterov=True,
                            adjust_lr_fn=SETTINGS['muon_adjust_lr_fn']),
            torch.optim.AdamW(scalar,**kw)]


def update_record(before,after):
    """Actual full update, including decay; no assumed radial orthogonality."""
    import torch
    out={}
    for name,w in before.items():
        z=after[name].detach();delta=z-w;n=float(torch.linalg.vector_norm(w));dn=float(torch.linalg.vector_norm(delta))
        tangent=delta-(delta*w).sum()/n**2*w if n else delta
        zn=float(torch.linalg.vector_norm(z))
        out[name]={'weight_norm':n,'update_norm':dn,'relative_update':dn/n if n else None,
                   'relative_tangent_update':float(torch.linalg.vector_norm(tangent))/n if n else None,
                   'angular_chord':float(torch.linalg.vector_norm(z/zn-w/n)) if n and zn else None,
                   'zero_norm':n==0}
    return out


def fit(rows,*,optimizer='adamw',objective='calibrated',prior=None,host_basis='original'):
    import numpy as np
    import torch
    if objective not in ('legacy','calibrated'):raise ValueError('unknown objective')
    started=time.monotonic();x,y,mask,w,means,scales,surfaces=old.tensors(rows);s,j,d=x.shape
    prior=make_prior(rows,prior,host_basis);offset=np.zeros_like(y)
    for i,surface in enumerate(surfaces):offset[i,:len(surface)]=prior_scores(prior,surface)
    a=np.concatenate((x,np.ones((s,j,1))),axis=-1).reshape(-1,d+1);wa=w.ravel()/s
    affine=np.linalg.solve(a.T@(a*wa[:,None])+.001*np.eye(d+1),a.T@(wa*(y-offset).ravel()))
    tx,ty,tw,to=[torch.tensor(v,dtype=torch.float32) for v in (x,y,w,offset)]
    tm=torch.tensor(mask,dtype=torch.bool);runs=[];states=[];original_threads=torch.get_num_threads()
    loss_fn=calibrated_loss if objective=='calibrated' else old.losses
    torch.set_num_threads(1)
    try:
        for seed in SETTINGS['seeds']:
            begin=time.monotonic()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);net=old._network(d,SETTINGS['width'])
            with torch.no_grad():
                net.affine.weight.copy_(torch.tensor(affine[:d],dtype=torch.float32)[None,:]);net.affine.bias.fill_(float(affine[-1]))
                net.hidden.weight.copy_(torch.tensor(np.random.default_rng(seed).normal(0,.15/math.sqrt(d),(d,SETTINGS['width'])).T,dtype=torch.float32))
                net.hidden.bias.zero_();net.readout.weight.zero_()
            optimizers=make_optimizers(net,optimizer);history=[]
            checkpoints={1,100,500,SETTINGS['epochs']}
            for epoch in range(1,SETTINGS['epochs']+1):
                if time.monotonic()-begin>SETTINGS['max_seconds_per_seed']:raise TimeoutError('local fit deadline')
                lr=SETTINGS['learning_rate']*(1-.99*(epoch-1)/max(SETTINGS['epochs']-1,1))
                for opt in optimizers:
                    for g in opt.param_groups:g['lr']=lr
                    opt.zero_grad(set_to_none=True)
                score=net(tx)+to;value,metrics=loss_fn(score,ty,tm,tw)
                if not bool(torch.isfinite(value)):raise FloatingPointError('nonfinite training objective')
                value.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),SETTINGS['gradient_clip'],error_if_nonfinite=True)
                before={n:p.detach().clone() for n,p in net.named_parameters()} if epoch in checkpoints else None
                for opt in optimizers:opt.step()
                if before is not None:
                    with torch.no_grad():change=float(torch.max(torch.abs((net(tx)+to-score)[tm])))
                    history.append({'epoch':epoch,'nominal_lr':lr,'pre_update_loss':float(value.detach()),
                                    'gradient_norm':float(norm),'max_log_time_update':change,
                                    'parameter_updates':update_record(before,dict(net.named_parameters()))})
            net.zero_grad(set_to_none=True);value,metrics=loss_fn(net(tx)+to,ty,tm,tw);value.backward()
            grad=math.sqrt(sum(float(p.grad.square().sum()) for p in net.parameters()))
            runs.append({'seed':seed,'objective':float(value.detach()),'metrics':{k:float(v.detach()) if hasattr(v,'detach') else v for k,v in metrics.items()},
                         'gradient_norm':grad,'history':history,'seconds':time.monotonic()-begin})
            states.append({n:p.detach().tolist() for n,p in net.state_dict().items()})
    finally:torch.set_num_threads(original_threads)
    selected=min(range(len(runs)),key=lambda i:(runs[i]['objective'],runs[i]['seed']))
    groups={}
    for r in rows:groups.setdefault(support_key(r),set()).add(tuple(r['shape']))
    return {'schema':SCHEMA,'features':list(old.base.FEATURES)+list(old.EXTRA),'means':means.tolist(),'scales':scales.tolist(),
            'state':states[selected],'width':SETTINGS['width'],'optimizer':optimizer,'objective':objective,
            'execution_context':rows[0]['execution_context'],'world_size':rows[0]['shape'][0],
            'supported_precision_contracts':sorted({old.precision_key(r) for r in rows}),
            'training_shapes':[ss[0]['shape'] for ss in surfaces],'training_sources':sorted({r['source_id'] for r in rows}),
            'candidate_support':{k:sorted(v) for k,v in groups.items()},'prior':prior,'host_basis':host_basis,
            'runs':runs,'selected_seed':runs[selected]['seed'],'settings':dict(SETTINGS),'seconds':time.monotonic()-started,
            'network_dtype':'float32','torch_version':torch.__version__,
            'optimizer_source_sha256':hashlib.sha256(inspect.getsource(importlib.import_module('torch.optim._muon')).encode()).hexdigest() if optimizer=='muon' else None,
            'optimizer_orthogonalization_dtype':'bfloat16' if optimizer=='muon' else None,
            'prediction_semantics':'estimated_log_latency' if objective=='calibrated' else 'legacy_mixed_utility_diagnostic',
            'compiler_admission':False,'calibration_verified_on_unseen_data':False}


def support_key(row):
    return row['candidate_id']+'|'+','.join(old.precision_key(row))


def predict(model,rows):
    import numpy as np
    if model['schema']!=SCHEMA:raise ValueError('wrong calibrated schema')
    # Reuse precision/identity/evidence validation and portable shared scoring.
    raw=old.predict({**model,'schema':old.SCHEMA,'prior':None},rows)
    surface=sorted([r for r in rows if r['candidate_id'] in raw['scores']],key=lambda r:r['candidate_id'])
    if not surface:return {'chosen':None,'provisional_choice':None,'predicted_log_us':{},'excluded':raw['excluded'],
        'support':{},'requires_validation':True,'compiler_admission':False}
    offset=prior_scores(model['prior'],surface)
    global_support=old.prior_model.in_training_hull(tuple(tuple(s) for s in model['training_shapes']),tuple(surface[0]['shape']))
    score={r['candidate_id']:float(offset[i]+(raw['scores'][r['candidate_id']] if model['prior'] is None or global_support else 0.))
           for i,r in enumerate(surface)}
    if not all(math.isfinite(v) and abs(v)<700 for v in score.values()):raise FloatingPointError('invalid log latency')
    support={}
    for r in surface:
        shapes=model['candidate_support'].get(support_key(r),[])
        if not shapes:reason='candidate_or_precision_absent_from_training'
        elif r['shape'] in [list(s) for s in shapes]:reason=None
        elif len(shapes)<2:reason='one_training_shape_only'
        elif not old.prior_model.in_training_hull(tuple(tuple(s) for s in shapes),tuple(r['shape'])):reason='outside_same_candidate_shape_hull'
        else:reason=None
        support[r['candidate_id']]={'supported':reason is None,'reason':reason,'training_shape_count':len(shapes)}
    provisional=canonical_choice(score)
    unsupported=any(not v['supported'] for v in support.values())
    unknown=any(v!='failed' for v in raw['excluded'].values())
    center=sum(score.values())/len(score)
    return {'chosen':None if unsupported or unknown else provisional,'provisional_choice':provisional,
            'predicted_log_us':score,'relative_log_scores':{k:v-center for k,v in score.items()},
            'support':support,'excluded':raw['excluded'],'requires_validation':unsupported or unknown,
            'residual_global_hull_support':global_support,'compiler_admission':False,
            'support_is_accuracy_guarantee':False,'all_measured_valid_candidates_retained':True}


def evaluate(model,rows,*,training_diagnostic=False):
    import numpy as np
    import torch
    surfaces=old.valid_rows(rows,'train' if training_diagnostic else 'evaluation');decisions=[]
    if not training_diagnostic and any(ss[0]['shape'] in model['training_shapes'] for ss in surfaces):raise ValueError('shape leakage')
    if training_diagnostic and any(ss[0]['shape'] not in model['training_shapes'] for ss in surfaces):raise ValueError('not training data')
    for surface in surfaces:
        pred=predict(model,surface);names=sorted(pred['predicted_log_us']);actual={r['candidate_id']:r['latency_us'] for r in surface}
        score=torch.tensor([[pred['predicted_log_us'][n] for n in names]],dtype=torch.float64)
        target=torch.tensor([[math.log(actual[n]) for n in names]],dtype=torch.float64)
        _,metrics=calibrated_loss(score,target,torch.ones_like(score,dtype=torch.bool),torch.ones_like(score)/len(names))
        chosen=pred['provisional_choice'];host=next(r['is_host'] for r in surface if r['candidate_id']==chosen)
        h=min(r['latency_us'] for r in surface if r['is_host']);f=min(r['latency_us'] for r in surface if not r['is_host'])
        decisions.append({'shape':surface[0]['shape'],**pred,'measured_us':actual,
            'provisional_regret':actual[chosen]/min(actual.values())-1,
            'admitted_regret':actual[pred['chosen']]/min(actual.values())-1 if pred['chosen'] is not None else None,
            'host_error':h<f and not host,'fused_error':f<h and host,
            'mape':float(torch.expm1(score-target).abs().mean()),'max_relative_error':float(torch.expm1(score-target).abs().max()),
            'cross_entropy':float(metrics['cross_entropy']),'kl':float(metrics['kl_to_cost_target']),
            'order_invariant':pred==predict(model,list(reversed(surface)))})
    return {'decisions':decisions,'mean_provisional_regret':float(np.mean([r['provisional_regret'] for r in decisions])),
            'max_provisional_regret':max(r['provisional_regret'] for r in decisions),
            'selection_coverage':sum(r['chosen'] is not None for r in decisions)/len(decisions),
            'mean_shape_mape':float(np.mean([r['mape'] for r in decisions])),
            'max_relative_error':max(r['max_relative_error'] for r in decisions),
            'mean_kl':float(np.mean([r['kl'] for r in decisions])),
            'host_errors':sum(r['host_error'] for r in decisions),'fused_errors':sum(r['fused_error'] for r in decisions),
            'all_order_invariant':all(r['order_invariant'] for r in decisions),'compiler_admission':False}
