"""Experimental precision-aware shared scores, masked cost/CE losses.

Kernel precision is metadata, not the scorer's training dtype. Unsupported
formats are rejected rather than borrowing FP16 labels. This does not modify
the compiler's admitted selector or the frozen v1 model.
"""
from __future__ import annotations
import math
import time
from collections import defaultdict
from . import shape_decision_model as base
from . import path_aware_selection as prior_model
from .shared_candidate_selection import canonical_choice
from .two_stage_selection import parse_schedule_family

SCHEMA = 'edmir.precision-candidate-selection.v1'
DION_REVISION = '7692479288f928a3d61d1cc58137b4dbe4f71000'
EXTRA = ('input_fraction_bits', 'input_max_exponent', 'weight_fraction_bits',
         'output_fraction_bits', 'accumulator_fraction_bits', 'partial_fraction_bits',
         'partial_unknown', 'accumulator_api_only', 'log_serial_fp16_or_bf16_writes')
FORMATS = {'fp16': (10, 15), 'bf16': (7, 127), 'fp32': (23, 127)}
SETTINGS = {'epochs':1000, 'learning_rate':.003, 'weight_decay':.001,
            'temperature':.05, 'seeds':[0,1], 'gradient_clip':10.,
            'width':8, 'max_seconds_per_seed':20., 'network_dtype':'float32',
            'dion_fraction':.5, 'dion_adjust_lr':'rms_norm', 'dion_betas':[.9,.999]}


def precision_key(row):
    p = row['precision']
    fields = ('input', 'weight', 'output', 'accumulator', 'partial', 'accumulator_evidence')
    if set(p) != set(fields):
        raise ValueError('explicit input/weight/output/accumulator/partial precision and evidence required')
    if any(p[k] not in FORMATS for k in fields[:4]):
        raise ValueError('unknown numerical format')
    if p['partial'] not in (*FORMATS, 'none', 'unknown'):
        raise ValueError('unknown intermediate storage precision')
    if p['accumulator_evidence'] not in ('source', 'api_requested'):
        raise ValueError('distinguish source accumulation from opaque library internals')
    return tuple(p[k] for k in fields)


def feature_vector(row):
    precision_key(row); p=row['precision']
    x=list(row['features'])
    if len(x)!=len(base.FEATURES) or x[0]!=float(row['is_host']):
        raise ValueError('geometry feature/host identity mismatch')
    family=row['candidate_id'].split('@')[0]
    q=1 if row['is_host'] else parse_schedule_family(family)[0]
    x += [FORMATS[p['input']][0], FORMATS[p['input']][1], FORMATS[p['weight']][0],
          FORMATS[p['output']][0], FORMATS[p['accumulator']][0],
          FORMATS[p['partial']][0] if p['partial'] in FORMATS else 0.,
          float(p['partial']=='unknown'), float(p['accumulator_evidence']=='api_requested'),
          math.log1p(q-1) if p['partial'] in ('fp16','bf16') else 0.]
    if not all(math.isfinite(v) for v in x): raise ValueError('nonfinite input features')
    return x


def valid_rows(rows, split):
    if any(r.get('validity',{}).get('status')!='passed' or
           not r['validity'].get('source_id') for r in rows):
        raise ValueError('training/evaluation needs observed correctness-eligible labels; no failed/unknown labels')
    for r in rows: feature_vector(r)
    return base.validate_rows(rows, split=split)


def tensors(rows):
    import numpy as np
    surfaces=valid_rows(rows,'train')
    if len(surfaces)<2: raise ValueError('whole-shape training needs at least two shapes')
    s,j,d=len(surfaces),max(map(len,surfaces)),len(base.FEATURES)+len(EXTRA)
    x,y,w=np.zeros((s,j,d)),np.zeros((s,j)),np.zeros((s,j))
    mask=np.zeros((s,j),bool)
    for i,surface in enumerate(surfaces):
        hosts=sum(r['is_host'] for r in surface)
        for k,r in enumerate(surface):
            x[i,k],y[i,k],mask[i,k]=feature_vector(r),math.log(r['latency_us']),True
            w[i,k]=.5/(hosts if r['is_host'] else len(surface)-hosts)
    means=np.sum(x*w[...,None],axis=(0,1))/s
    scales=np.maximum(np.sqrt(np.sum((x-means)**2*w[...,None],axis=(0,1))/s),.1)
    return np.where(mask[...,None],(x-means)/scales,0.),y,mask,w,means,scales,surfaces


def losses(score,target,mask,weights,*,ce_weight=1.,temperature=.05):
    """Full-information regret + anchor + old gap, optionally soft-target CE.

    Target weights = softmin(observed log times / temperature), not one-hot
    rankings or calibrated best-candidate probabilities. Invalid/padded slots
    have exactly zero mass and gradient; never invent their latency.
    """
    import torch
    if not math.isfinite(temperature) or temperature<=0 or ce_weight not in (0.,1.):
        raise ValueError('positive temperature and declared CE ablation required')
    if score.shape!=target.shape or mask.shape!=score.shape or weights.shape!=score.shape:
        raise ValueError('tensor shapes disagree')
    if mask.dtype!=torch.bool or not bool(mask.any(dim=1).all()):
        raise ValueError('at least one measured eligible candidate per surface')
    if not bool(torch.isfinite(score[mask]).all() and torch.isfinite(target[mask]).all()):
        raise ValueError('nonfinite measured values')
    # Sanitize before arithmetic, not after, so masked NaN labels cannot leak gradients.
    target=torch.where(mask,target,torch.zeros_like(target))
    score=torch.where(mask,score,torch.zeros_like(score))
    logp=torch.log_softmax((-score/temperature).masked_fill(~mask,-torch.inf),dim=1)
    logq=torch.log_softmax((-target/temperature).masked_fill(~mask,-torch.inf),dim=1)
    prob=logp.exp(); q=logq.exp().detach()
    safe_logp=torch.where(mask,logp,torch.zeros_like(logp))
    safe_logq=torch.where(mask,logq,torch.zeros_like(logq))
    ce=-(q*safe_logp).sum(dim=1).mean()
    entropy=-(q*safe_logq).sum(dim=1).mean()
    best=target.masked_fill(~mask,torch.inf).min(dim=1,keepdim=True).values
    regret=torch.expm1(torch.where(mask,target-best,torch.zeros_like(target)))
    expected=(prob*regret).sum(dim=1).mean()
    anchor=(weights*torch.where(mask,(score-target).square(),torch.zeros_like(score))).sum(dim=1).mean()
    # Vectorize the original best-gap objective, averaging exact tied minima.
    minima=mask & (target==best)
    violation=torch.relu(target[:,None,:]-target[:,:,None]-score[:,None,:]+score[:,:,None])
    pair=minima[:,:,None] & mask[:,None,:]
    gap=(torch.where(pair,violation.square(),torch.zeros_like(violation)).sum(dim=(1,2)) /
         (minima.sum(dim=1)*mask.sum(dim=1))).mean()
    total=anchor+expected+gap+ce_weight*ce
    return total,{'anchor':anchor,'expected_regret':expected,'gap':gap,'cross_entropy':ce,
                 'target_entropy':entropy,'kl_to_cost_target':ce-entropy}


def _network(d, width):
    import torch
    class SharedScore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.affine=torch.nn.Linear(d,1)
            self.hidden=torch.nn.Linear(d,width)
            self.readout=torch.nn.Linear(width,1,bias=False)
        def forward(self,x):
            return (self.affine(x)+self.readout(torch.tanh(self.hidden(x)))).squeeze(-1)
    return SharedScore()


def make_optimizer(net, name):
    import torch
    lr,wd=SETTINGS['learning_rate'],SETTINGS['weight_decay']
    if name=='adamw': return torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd,betas=(.9,.999))
    if name!='dion3': raise ValueError('undeclared optimizer')
    from dion import Dion3
    # Matrix orthogonalization only for hidden weights. Scalar output and
    # affine score weights/biases use the same AdamW algorithm as control.
    scalar=[p for n,p in net.named_parameters() if n!='hidden.weight']
    return Dion3([{'params':[net.hidden.weight],'algorithm':'nordion2'},
                  {'params':scalar,'algorithm':'adamw'}],lr=lr,weight_decay=wd,
                 fraction=SETTINGS['dion_fraction'],adjust_lr=SETTINGS['dion_adjust_lr'],
                 betas=tuple(SETTINGS['dion_betas']),use_triton=False,use_gram_newton_schulz=False)


def fit(rows, *, optimizer='adamw',ce_weight=1.,prior=None):
    import numpy as np
    import torch
    if optimizer not in ('adamw','dion3') or ce_weight not in (0.,1.):raise ValueError('undeclared ablation')
    started=time.monotonic();x,y,mask,w,means,scales,surfaces=tensors(rows);s,j,d=x.shape
    offset=np.zeros_like(y)
    if prior is not None:
        # Archive prior has no dtype field: explicit guard forbids applying it to BF16.
        if any(r['precision']['input']!='fp16' or r['precision']['output']!='fp16' for r in rows):
            raise ValueError('archival P4 prior supports only its original FP16 contract')
        if ({tuple(z) for z in prior['training_shapes']}!={tuple(z[0]['shape']) for z in surfaces}
            or prior['environment_id']!=rows[0]['execution_context']):
            raise ValueError('prior must use exactly the same training fold and context')
        for i,surface in enumerate(surfaces):offset[i,:len(surface)]=prior_model.prior_scores(prior,surface)
    a=np.concatenate((x,np.ones((s,j,1))),axis=-1).reshape(-1,d+1);wa=w.ravel()/s
    affine=np.linalg.solve(a.T@(a*wa[:,None])+.001*np.eye(d+1),a.T@(wa*(y-offset).ravel()))
    tx,ty,tw,to=[torch.tensor(v,dtype=torch.float32) for v in (x,y,w,offset)]
    tm=torch.tensor(mask,dtype=torch.bool);runs=[];states=[]
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for seed in SETTINGS['seeds']:
            begin=time.monotonic()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);net=_network(d,SETTINGS['width'])
            with torch.no_grad():
                net.affine.weight.copy_(torch.tensor(affine[:d],dtype=torch.float32)[None,:]);net.affine.bias.fill_(float(affine[-1]))
                net.hidden.weight.copy_(torch.tensor(np.random.default_rng(seed).normal(0,.15/math.sqrt(d),(d,SETTINGS['width'])).T,dtype=torch.float32))
                net.hidden.bias.zero_();net.readout.weight.zero_()
            opt=make_optimizer(net,optimizer);history=[]
            for epoch in range(1,SETTINGS['epochs']+1):
                if time.monotonic()-begin>SETTINGS['max_seconds_per_seed']:raise TimeoutError('bounded local fit deadline')
                opt.zero_grad(set_to_none=True)
                total,components=losses(net(tx)+to,ty,tm,tw,ce_weight=ce_weight,temperature=SETTINGS['temperature'])
                if not bool(torch.isfinite(total)):raise FloatingPointError('nonfinite objective')
                total.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),SETTINGS['gradient_clip'],error_if_nonfinite=True)
                # Official optimizer arithmetic, but no JIT on this tiny CPU
                # experiment. Avoid unbounded first-use compilation.
                with torch.compiler.set_stance('force_eager'):
                    opt.step()
                if epoch in (1,100,500,1000):history.append({'epoch':epoch,'pre_update_loss':float(total.detach()),'gradient_norm':float(norm)})
            with torch.no_grad():
                total,comp=losses(net(tx)+to,ty,tm,tw,ce_weight=ce_weight,temperature=SETTINGS['temperature'])
            runs.append({'seed':seed,'final_objective':float(total),'metrics':{k:float(v) for k,v in comp.items()},'history':history,'seconds':time.monotonic()-begin})
            states.append({n:v.detach().tolist() for n,v in net.state_dict().items()})
    finally:torch.set_num_threads(previous_threads)
    selected=min(range(len(runs)),key=lambda i:(runs[i]['final_objective'],runs[i]['seed']))
    return {'schema':SCHEMA,'features':list(base.FEATURES)+list(EXTRA),'means':means.tolist(),'scales':scales.tolist(),
            'state':states[selected],'width':SETTINGS['width'],'optimizer':optimizer,'ce_weight':ce_weight,'settings':dict(SETTINGS),
            'execution_context':rows[0]['execution_context'],'world_size':rows[0]['shape'][0],
            'supported_precision_contracts':sorted({precision_key(r) for r in rows}),
            'training_shapes':[z[0]['shape'] for z in surfaces],'training_sources':sorted({r['source_id'] for r in rows}),
            'prior':prior,'residual_support':'training_hull' if prior is not None else 'unrestricted_development_only',
            'runs':runs,'selected_seed':runs[selected]['seed'],'seconds':time.monotonic()-started,
            'torch_version':torch.__version__,'dion_revision':DION_REVISION if optimizer=='dion3' else None,
            'compiler_admission':False,'best_probability_calibrated':False}


def predict(model, rows):
    """Pure label-free scores, plus an explicit evidence gate; no automatic fallback."""
    import numpy as np
    if model['schema']!=SCHEMA or model['features']!=list(base.FEATURES)+list(EXTRA):raise ValueError('model schema/features differ')
    if not 1<=len(rows)<=14 or len({r['candidate_id'] for r in rows})!=len(rows) or len({tuple(r['shape']) for r in rows})!=1:
        raise ValueError('one bounded candidate surface required')
    rows=sorted(rows,key=lambda r:r['candidate_id']);retained=[];excluded={}
    for r in rows:
        if r['execution_context']!=model['execution_context'] or r['shape'][0]!=model['world_size']:
            raise ValueError('unseen execution context or GPU count')
        if precision_key(r) not in {tuple(k) for k in model['supported_precision_contracts']}:
            raise ValueError('unsupported precision contract: measured FP16 does not validate BF16')
        validity=r.get('validity',{})
        if validity.get('status')!='passed' or not validity.get('source_id'):
            excluded[r['candidate_id']]=validity.get('status','unknown');continue
        retained.append(r)
    if not retained:return {'scores':{},'chosen':None,'excluded':excluded,'requires_validation':True}
    x=(np.asarray([feature_vector(r) for r in retained])-model['means'])/model['scales'];state=model['state']
    score=(x@np.array(state['affine.weight']).T+state['affine.bias']+
           np.tanh(x@np.array(state['hidden.weight']).T+state['hidden.bias'])@np.array(state['readout.weight']).T).ravel()
    support=None
    if model['prior'] is not None:
        support=prior_model.in_training_hull(tuple(tuple(s) for s in model['training_shapes']),tuple(rows[0]['shape']))
        if not support:score=np.zeros_like(score)
        score+=prior_model.prior_scores(model['prior'],retained)
    scores=dict(zip((r['candidate_id'] for r in retained),map(float,score)))
    return {'scores':scores,'chosen':canonical_choice(scores),'excluded':excluded,'requires_validation':bool(excluded),'residual_geometric_support':support}


def evaluate(model, rows, *, training_diagnostic=False):
    import numpy as np
    import torch
    surfaces=valid_rows(rows,'train' if training_diagnostic else 'evaluation');decisions=[]
    if not training_diagnostic and any(s[0]['shape'] in model['training_shapes'] for s in surfaces):raise ValueError('whole-shape train/evaluation overlap')
    for surface in surfaces:
        p=predict(model,surface);names=sorted(p['scores']);by={r['candidate_id']:r for r in surface}
        score=torch.tensor([[p['scores'][n] for n in names]],dtype=torch.float64)
        target=torch.tensor([[math.log(by[n]['latency_us']) for n in names]],dtype=torch.float64)
        mask=torch.ones_like(score,dtype=torch.bool);weights=torch.ones_like(score)/len(names)
        _,metrics=losses(score,target,mask,weights,ce_weight=model['ce_weight'],temperature=SETTINGS['temperature'])
        actual={n:by[n]['latency_us'] for n in names};choice=p['chosen'];best=min(actual.values())
        h=min(r['latency_us'] for r in surface if r['is_host']);f=min(r['latency_us'] for r in surface if not r['is_host'])
        ranked=sorted(names,key=lambda n:(p['scores'][n],n));chosen_host=by[choice]['is_host']
        decisions.append({'shape':surface[0]['shape'],'chosen':choice,'regret':actual[choice]/best-1,
            'host_wins':h<f,'host_error':h<f and not chosen_host,'fused_error':f<h and chosen_host,
            'cross_entropy':float(metrics['cross_entropy']),'target_entropy':float(metrics['target_entropy']),
            'kl_to_cost_target':float(metrics['kl_to_cost_target']),'top2_contains_measured_best':any(actual[n]==best for n in ranked[:2]),
            'predicted_log_us':p['scores'],'measured_us':actual,'order_invariant':p==predict(model,list(reversed(surface))),
            'residual_geometric_support':p.get('residual_geometric_support')})
    return {'mean_regret':float(np.mean([r['regret'] for r in decisions])),'max_regret':max(r['regret'] for r in decisions),
            'host_errors':sum(r['host_error'] for r in decisions),'fused_errors':sum(r['fused_error'] for r in decisions),
            'mean_cross_entropy':float(np.mean([r['cross_entropy'] for r in decisions])),
            'mean_kl_to_cost_target':float(np.mean([r['kl_to_cost_target'] for r in decisions])),
            'decisions':decisions,'compiler_admission':False,'correctness_conditional_subset':True}
