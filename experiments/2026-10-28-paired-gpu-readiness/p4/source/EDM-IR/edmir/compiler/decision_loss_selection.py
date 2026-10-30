"""Local, retrospective decision-loss iterations. Not compiler admission.

A single candidate potential supplies all comparisons. The new convex loss
anchors times, fits signed gaps, and penalizes insufficient observed-best
margins. Only measured candidates enter the objective. Direct-time features
have nonnegative coefficients; this is an empirical extrapolation assumption,
not an identified service-time decomposition or a GPU performance guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import lru_cache
import copy
import math
import time

from .additive_latency_selection import AdditiveLatencySelector
from .traffic_latency_selection import TrafficLatencySelector, host_features, fused_features, HOST_NAMES, FUSED_NAMES
from .shared_candidate_selection import static_features, FEATURE_NAMES, canonical_choice
from .factorized_selection import IdentifiedMeasurements
from .host_baseline import PER_SOURCE_CUTLASS
from .measurement_dataset import P4_ENV
from .relative_selection import ScheduleCandidate

SCHEMA='edmir.decision-loss-selector.v1'
VARIANTS=('log_gap','traffic_gap','traffic_nested')
# All settings fixed before the outer replay. The nested alternative chooses
# using inner observed hard regret, never an outer boundary label.
LOSS_GRID=tuple((1.,margin,ridge) for margin in (0.,1.,4.) for ridge in (.0001,.001,.01))
FIXED_LOSS=(1.,1.,.001)
MAX_SECONDS=30.
MAX_ITER=1500


def objective(theta, x, base, targets, mask, weights, normalizers, pair_weight, margin_weight, ridge):
    """Convex quadratic + squared positive-part loss; exact analytic gradient.

    Per shape: balanced absolute error; balanced error on every signed pair;
    sum_c [delta_c - (s_c-s_best)]_+^2. Delta is the measured cost gap in
    model units. For direct times normalized by measured best, this last sum
    bounds squared hard-choice regret on the observed candidate set. For log
    times it bounds log1p(regret)^2. It is NOT a bound on an unseen workload.
    No averaging of margin terms by candidate count: that would weaken the
    stated bound. Every shape has equal mass.
    """
    import numpy as np
    count=x.shape[0]
    s=base+x@theta
    err=np.where(mask,(s-targets)/normalizers[:,None],0.)
    value=float(np.sum(weights*err*err)/count)
    gs=2*weights*err/normalizers[:,None]/count
    # Weighted all-pairs identity: sum_{i<j} w_i*w_j*(e_i-e_j)^2
    # equals weighted variance. Normalize by total pair mass per shape.
    avg=np.sum(weights*err,axis=1,keepdims=True)
    mass=(1-np.sum(weights*weights,axis=1,keepdims=True))/2
    variance=np.sum(weights*(err-avg)**2,axis=1,keepdims=True)
    value+=pair_weight*float(np.mean(variance/mass))
    gs+=pair_weight*2*weights*(err-avg)/mass/normalizers[:,None]/count
    best=np.argmin(np.where(mask,targets,np.inf),axis=1)
    idx=np.arange(count)
    margin=np.where(mask,np.maximum((targets-targets[idx,best,None]-s+s[idx,best,None])/normalizers[:,None],0.),0.)
    value+=margin_weight*float(np.sum(margin*margin)/count)
    gm=-2*margin_weight*margin/normalizers[:,None]/count
    gm[idx,best]-=gm.sum(axis=1)
    gs+=gm
    grad=np.einsum('sjd,sj->d',x,gs)+2*ridge*theta
    value+=ridge*float(theta@theta)
    if not math.isfinite(value) or not np.isfinite(grad).all():
        raise FloatingPointError('nonfinite decision objective')
    return value,grad


def features(row, prior, prediction, space):
    if not isinstance(row,ScheduleCandidate):raise TypeError('static inputs only')
    if space=='log':return (*static_features(row,prediction),1.)
    # Branch properties are execution attributes. All candidates are scored
    # by the same vector; neither position nor candidate-set size is an input.
    return ((*host_features(row,prior.input_threshold,prior.depth_threshold),*(0.,)*len(FUSED_NAMES)) if row.is_host
            else (*(0.,)*len(HOST_NAMES),*fused_features(row,prior.capacity)))


@lru_cache(maxsize=256)
def _prior(data,space):
    return (AdditiveLatencySelector if space=='log' else TrafficLatencySelector).fit(data)


def tensors(data,prior,space):
    import numpy as np
    surfaces=[{n:s[n] for n in sorted(s)} for _,s in sorted(data.validate().items())]
    rows=[]
    for surface in surfaces:
        candidates={n:ScheduleCandidate.from_measurement(r,P4_ENV) for n,r in surface.items()}
        pred=prior.select(candidates,baseline=PER_SOURCE_CUTLASS)['predicted_ms']
        ns=sorted(pred)
        rows.append((ns,[features(candidates[n],prior,pred[n],space) for n in ns],
                     [math.log(pred[n]) if space=='log' else 0. for n in ns],
                     [math.log(surface[n].measured_ms) if space=='log' else surface[n].measured_ms for n in ns]))
    shape=(len(rows),max(len(r[0]) for r in rows));d=len(rows[0][1][0])
    x=np.zeros((*shape,d));base=np.zeros(shape);target=np.zeros(shape)
    mask=np.zeros(shape,bool);weights=np.zeros(shape);normalizers=np.ones(shape[0])
    for i,(ns,xx,bb,yy) in enumerate(rows):
        j=len(ns)
        if j<2 or 'host' not in ns:raise ValueError('measured host and fused rows required')
        x[i,:j],base[i,:j],target[i,:j],mask[i,:j]=xx,bb,yy,True
        weights[i,:j]=[.5 if n=='host' else .5/(j-1) for n in ns]
        if space=='time':normalizers[i]=min(yy)
    means=np.sum(x*weights[...,None],axis=(0,1))/len(rows) if space=='log' else np.zeros(d)
    # Leave the affine intercept at one; it has its own regularization.
    if space=='log':means[-1]=0.
    scales=np.maximum(np.sqrt(np.sum((x-means)**2*weights[...,None],axis=(0,1))/len(rows)),.1)
    x=np.where(mask[...,None],(x-means)/scales,0.)
    return x,base,target,mask,weights,normalizers,means,scales,tuple(tuple(r[0]) for r in rows)


def _fit_fixed(data,space,loss,deadline):
    import numpy as np
    from scipy.optimize import minimize,nnls
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('decision fit deadline exceeded')
    check();prior=_prior(data,space);check()
    x,b,y,mask,w,norm,means,scales,names=tensors(data,prior,space)
    pair,margin,ridge=loss;d=x.shape[-1]
    # Initial anchor solution. The complete convex objective is minimized below.
    a=x.reshape(-1,d);weights=np.sqrt(w/len(x))/norm[:,None]
    aa=np.vstack((a*weights.reshape(-1,1),math.sqrt(ridge)*np.eye(d)))
    yy=np.r_[((y-b)*weights).ravel(),np.zeros(d)]
    bounds=[(1e-10,None)]*d if space=='time' else None
    theta=(np.maximum(nnls(aa,yy,maxiter=3000)[0],1e-10) if space=='time' else np.linalg.lstsq(aa,yy,rcond=None)[0])
    def loss_fn(t):
        check();return objective(t,x,b,y,mask,w,norm,*loss)
    initial=loss_fn(theta)[0]
    opt=minimize(loss_fn,theta,method='L-BFGS-B',jac=True,bounds=bounds,
                 options={'maxiter':MAX_ITER,'maxfun':4000,'maxls':40,'gtol':1e-9,'ftol':1e-15})
    value,g=loss_fn(opt.x)
    pg=np.where((opt.x<=1.01e-10)&(g>0),0.,g) if space=='time' else g
    violation=float(np.max(np.abs(pg)))
    if not opt.success or violation>1e-5 or value>initial+1e-10:
        raise ArithmeticError(f'uncertified convex fit: {opt.message}, KKT={violation}')
    audit={'space':space,'loss':list(loss),'initial_loss':initial,'final_loss':value,
           'iterations':int(opt.nit),'evaluations':int(opt.nfev),'success':bool(opt.success),
           'projected_gradient':violation,'training_candidates':names,'parameter_count':d,
           'global_suboptimality_upper_bound':float(g@g/(4*ridge)) if space=='log' else None}
    return DecisionLossSelector('log_gap' if space=='log' else 'traffic_gap',prior,tuple(map(float,means)),tuple(map(float,scales)),tuple(map(float,opt.x)),audit,())


def _fit(data,method,seconds):
    deadline=time.monotonic()+seconds
    if method!='traffic_nested':return _fit_fixed(data,'log' if method=='log_gap' else 'time',FIXED_LOSS,deadline)
    # Every inner fit refits its basis/scaler from the inner training fold,
    # including the traffic prior's parameter choice. No held-out-label reuse.
    groups=sorted({r.group_id for r in data.rows});records=[]
    for loss in LOSS_GRID:
        regrets=[];errors=[];folds=[]
        for held in groups:
            train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=held))
            test=replace(data,rows=tuple(r for r in data.rows if r.group_id==held))
            model=_fit_fixed(train,'time',loss,deadline)
            folds.append({'heldout':held,'training_shapes':model.prior.training_shapes,
                          'projected_gradient':model.training_audit['projected_gradient']})
            for surface in test.validate().values():
                candidates={n:ScheduleCandidate.from_measurement(r,P4_ENV) for n,r in surface.items()}
                out=model.select(candidates,baseline=data.baseline,allow_extrapolation=True)
                best=min(r.measured_ms for r in surface.values())
                regrets.append(surface[out['choice']].measured_ms/best-1)
                errors.extend(((out['predicted_ms'][n]-surface[n].measured_ms)/best)**2 for n in out['predicted_ms'])
        records.append({'loss':list(loss),'mean_regret':math.fsum(regrets)/len(regrets),'max_regret':max(regrets),
                        'relative_mse':math.fsum(errors)/len(errors),'folds':folds})
    best=min(records,key=lambda r:(r['mean_regret'],r['max_regret'],r['relative_mse'],r['loss']))
    model=_fit_fixed(data,'time',tuple(best['loss']),deadline)
    return replace(model,method=method,cv=tuple(records))


@lru_cache(maxsize=128)
def _cached(data,method):return _fit(data,method,MAX_SECONDS)


@dataclass(frozen=True)
class DecisionLossSelector:
    method:str
    prior:object
    means:tuple
    scales:tuple
    parameters:tuple
    training_audit:dict
    cv:tuple

    @classmethod
    def fit(cls,data,*,method='traffic_nested',timeout_seconds=MAX_SECONDS,use_cache=True):
        if method not in VARIANTS or type(use_cache) is not bool:raise ValueError('unknown fit configuration')
        if type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or timeout_seconds<=0:raise ValueError('positive finite timeout required')
        if not isinstance(data,IdentifiedMeasurements) or len(data.rows)>4096 or data.environment_id!=P4_ENV or data.baseline!=PER_SOURCE_CUTLASS:
            raise ValueError('bounded identified A100 P4/CUTLASS data required')
        canonical=replace(data,rows=tuple(sorted(data.rows,key=lambda r:(r.shape,r.family,r.group_id))))
        result=(_cached(canonical,method) if use_cache and timeout_seconds>=MAX_SECONDS else _fit(canonical,method,min(timeout_seconds,MAX_SECONDS)))
        result.validate()
        return replace(result,training_audit=copy.deepcopy(result.training_audit),cv=copy.deepcopy(result.cv))

    def validate(self):
        if self.method not in VARIANTS:raise ValueError('unknown method')
        self.prior.validate()
        space='log' if self.method=='log_gap' else 'time'
        d=len(FEATURE_NAMES)+1 if space=='log' else len(HOST_NAMES)+len(FUSED_NAMES)
        if not (len(self.means)==len(self.scales)==len(self.parameters)==d):raise ValueError('feature dimensions differ')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (*self.means,*self.scales,*self.parameters)) or any(v<=0 for v in self.scales):raise ValueError('invalid numeric state')
        if space=='time' and any(v<1e-10 for v in self.parameters):raise ValueError('positive time coefficients required')
        if self.training_audit['space']!=space or not self.training_audit['success'] or self.training_audit['projected_gradient']>1e-5:raise ValueError('uncertified fit')
        if self.method=='traffic_nested':
            if {tuple(r['loss']) for r in self.cv}!=set(LOSS_GRID):raise ValueError('incomplete inner search')
            best=min(self.cv,key=lambda r:(r['mean_regret'],r['max_regret'],r['relative_mse'],r['loss']))
            if list(best['loss'])!=list(self.training_audit['loss']):raise ValueError('wrong nested selection')
        elif self.cv or tuple(self.training_audit['loss'])!=FIXED_LOSS:raise ValueError('wrong fixed loss')

    def select(self,candidates,*,baseline,allow_extrapolation=False):
        import numpy as np
        previous=self.prior.select(candidates,baseline=baseline,allow_extrapolation=allow_extrapolation)
        names=sorted(previous['predicted_ms']);space=self.training_audit['space']
        raw=np.array([features(candidates[n],self.prior,previous['predicted_ms'][n],space) for n in names])
        out=((raw-self.means)/self.scales)@np.array(self.parameters)
        if space=='log':
            logs=np.log([previous['predicted_ms'][n] for n in names])+out
            if not np.isfinite(logs).all() or np.max(np.abs(logs))>80:raise FloatingPointError('invalid log prediction')
            out=np.exp(logs)
        if not np.isfinite(out).all() or np.min(out)<=0:raise FloatingPointError('invalid time prediction')
        pred=dict(zip(names,map(float,out)))
        return {'choice':canonical_choice(pred),'predicted_ms':pred,'scores':{n:math.log(t/pred['host']) for n,t in pred.items()},
                'eligible_for_compiler_admission':False}

    def to_dict(self):
        self.validate()
        return {'schema':SCHEMA,'method':self.method,'prior':self.prior.to_dict(),'means':list(self.means),'scales':list(self.scales),
                'parameters':list(self.parameters),'training_audit':copy.deepcopy(self.training_audit),'cv':copy.deepcopy(self.cv),
                'training_shapes':list(self.prior.training_shapes),'compiler_admission':False}

    @classmethod
    def from_dict(cls,obj):
        keys={'schema','method','prior','means','scales','parameters','training_audit','cv','training_shapes','compiler_admission'}
        if set(obj)!=keys or obj['schema']!=SCHEMA or obj['compiler_admission'] is not False:raise ValueError('invalid schema')
        prior=(AdditiveLatencySelector if obj['method']=='log_gap' else TrafficLatencySelector).from_dict(obj['prior'])
        model=cls(obj['method'],prior,tuple(obj['means']),tuple(obj['scales']),tuple(obj['parameters']),copy.deepcopy(obj['training_audit']),tuple(copy.deepcopy(obj['cv'])))
        model.validate()
        if [list(s) for s in obj['training_shapes']]!=[list(s) for s in prior.training_shapes]:raise ValueError('training provenance mismatch')
        return model
