"""Experimental host/fused boundary revision, separate from frozen v1.

Shared candidate scores with explicit path-by-shape interactions. Optional
cross-path gap loss and training-only class weighting are independent ablations.
No shape lookup table, winner threshold, default-compiler admission or new data.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

from . import shape_decision_model as base
from .shared_candidate_selection import forward, parameter_count, canonical_choice

SCHEMA = "edmir.path-aware-selection.v1"
INTERACTIONS = ("host_x_log_M", "host_x_log_K", "host_x_log_N",
                "host_x_log_M_log_K", "host_x_log_N_log_K", "host_x_log_M_log_N")
SETTINGS = {"epochs":1000,"learning_rate":.003,"weight_decay":.001,"temperature":.05,
            "gradient_clip":10.,"seeds":[0,1],"max_seconds":10.}


def feature_vector(row, representation):
    """Pure metadata function; latency, fold, winner and row position forbidden."""
    if representation not in ("original", "path_geometry"):
        raise ValueError("undeclared representation")
    x=list(row["features"])
    if len(x)!=len(base.FEATURES) or x[0] != float(row["is_host"]):
        raise ValueError("base features and host identity disagree")
    if representation=="path_geometry":
        _,m,k,n=row["shape"]
        lm,lk,ln=math.log2(m/64),math.log2(k/4096),math.log2(n/512)
        h=float(row["is_host"])
        x.extend(h*v for v in (lm,lk,ln,lm*lk,ln*lk,lm*ln))
    return x


def objective(theta,x,target,mask,weights,host_mask,shape_weights,width,*,cross_path=False,offset=None):
    """Original data loss plus optional symmetric log-gap MSE between paths.

    Every observed host/fused pair participates, including when softmin host
    weight is nearly zero. Shape/class weighting is an independent choice.
    """
    import numpy as np
    score=forward(theta,x,width)
    if offset is not None:score=score+offset
    mass=shape_weights/shape_weights.sum()
    error=np.where(mask,score-target,0.)
    anchor=float(np.sum(mass[:,None]*weights*error**2))
    grad=2*mass[:,None]*weights*error
    logits=np.where(mask,-score/SETTINGS["temperature"],-np.inf)
    logits-=logits.max(axis=1,keepdims=True)
    prob=np.exp(logits);prob/=prob.sum(axis=1,keepdims=True)
    best=np.min(np.where(mask,target,np.inf),axis=1,keepdims=True)
    regret=np.expm1(np.where(mask,target-best,0.))
    expected=np.sum(prob*regret,axis=1,keepdims=True)
    cost=float(np.sum(mass[:,None]*expected))
    grad-=mass[:,None]*prob*(regret-expected)/SETTINGS["temperature"]
    gap,cross=0.,0.
    for i in range(len(x)):
        minima=np.flatnonzero(mask[i] & (target[i]==best[i,0]))
        factor=mass[i]/(len(minima)*int(mask[i].sum()))
        for b in minima:
            v=np.where(mask[i],np.maximum(0.,target[i]-target[i,b]-score[i]+score[i,b]),0.)
            gap+=float(v@v)*factor
            dg=2*v*factor;grad[i]-=dg;grad[i,b]+=dg.sum()
        if cross_path:
            hosts=np.flatnonzero(mask[i]&host_mask[i]);fused=np.flatnonzero(mask[i]&~host_mask[i])
            if not len(hosts) or not len(fused):raise ValueError("observed cross-path pairs required")
            if cross_path=="frontier":
                # Compare the fastest predicted member of each path class,
                # against the fastest *measured* member in each class.
                hs=score[i,hosts].min();fs=score[i,fused].min()
                delta=fs-hs-(target[i,fused].min()-target[i,hosts].min())
                hi=hosts[score[i,hosts]==hs];fi=fused[score[i,fused]==fs]
                cross+=mass[i]*delta*delta
                grad[i,fi]+=2*mass[i]*delta/len(fi)
                grad[i,hi]-=2*mass[i]*delta/len(hi)
            else:
                delta=(score[i,fused][None,:]-score[i,hosts][:,None]
                       -target[i,fused][None,:]+target[i,hosts][:,None])
                factor=mass[i]/(len(hosts)*len(fused))
                cross+=float(np.sum(delta**2))*factor
                grad[i,fused]+=2*factor*delta.sum(axis=0)
                grad[i,hosts]-=2*factor*delta.sum(axis=1)
    grad*=mask
    value=anchor+cost+gap+cross;gradient=base._backward(theta,x,grad,width)
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite boundary objective")
    return value,gradient


def tensors(rows,representation,balanced):
    import numpy as np
    surfaces=base.validate_rows(rows,split="train")
    if not 2<=len(surfaces)<=256:raise ValueError("2..256 whole training shapes required")
    s,j,d=len(surfaces),max(map(len,surfaces)),len(feature_vector(surfaces[0][0],representation))
    x,y,w=np.zeros((s,j,d)),np.zeros((s,j)),np.zeros((s,j))
    mask,hosts=np.zeros((s,j),bool),np.zeros((s,j),bool)
    winning=[]
    for i,surface in enumerate(surfaces):
        nh=sum(r["is_host"] for r in surface)
        winning.append(min(r["latency_us"] for r in surface if r["is_host"])
                       <min(r["latency_us"] for r in surface if not r["is_host"]))
        for t,r in enumerate(surface):
            x[i,t],y[i,t]=feature_vector(r,representation),math.log(r["latency_us"])
            mask[i,t],hosts[i,t]=True,r["is_host"]
            w[i,t]=.5/(nh if r["is_host"] else len(surface)-nh)
    positive=sum(winning);shape_weights=np.ones(s)
    if balanced and 0<positive<s:
        shape_weights=np.array([s/(2*(positive if v else s-positive)) for v in winning])
    # Keep the representation/scaler invariant across loss/shape-weight ablations.
    means=np.sum(x*w[...,None],axis=(0,1))/s
    scales=np.maximum(np.sqrt(np.sum((x-means)**2*w[...,None],axis=(0,1))/s),.1)
    x=np.where(mask[...,None],(x-means)/scales,0.)
    return x,y,mask,w,hosts,shape_weights,means,scales,surfaces,positive


def fit(rows,*,representation="original",cross_path=False,balanced=False,width=8,prior=None,
        residual_support="unrestricted"):
    import numpy as np
    if (width not in (0,8) or not (type(cross_path) is bool or cross_path=="frontier")
        or type(balanced) is not bool):
        raise ValueError("bounded, predeclared ablation choices required")
    if residual_support not in ("unrestricted","training_hull") or (prior is None and residual_support!="unrestricted"):
        raise ValueError("hull-limited residual needs an explicit cost prior")
    begin=time.monotonic()
    def check():
        if time.monotonic()-begin>SETTINGS["max_seconds"]:raise TimeoutError("boundary fit exceeded deadline")
    x,y,mask,w,hosts,mass,means,scales,surfaces,positive=tensors(rows,representation,balanced)
    s,j,d=x.shape
    offset=np.zeros((s,j))
    if prior is not None:
        from .additive_latency_selection import AdditiveLatencySelector
        prior=AdditiveLatencySelector.from_dict(prior).to_dict()
        if ({tuple(v) for v in prior["training_shapes"]}!={tuple(v[0]["shape"]) for v in surfaces}
            or prior["environment_id"]!=rows[0]["execution_context"]):
            raise ValueError("prior must be fitted on exactly this training fold")
        for i,surface in enumerate(surfaces):
            offset[i,:len(surface)]=prior_scores(prior,surface)
    a=np.concatenate((x,np.ones((s,j,1))),axis=-1).reshape(-1,d+1)
    # Identical initialization for loss/shape-weight comparisons.
    wa=w.ravel()/s
    affine=np.linalg.solve(a.T@(a*wa[:,None])+.001*np.eye(d+1),a.T@(wa*(y-offset).ravel()))
    runs,params=[],[]
    for seed in (SETTINGS["seeds"] if width else [0]):
        theta=np.zeros(parameter_count(d,width));theta[:d+1]=affine
        if width:theta[d+1:d+1+d*width]=np.random.default_rng(seed).normal(0,.15/math.sqrt(d),d*width)
        first,second=np.zeros_like(theta),np.zeros_like(theta)
        initial=objective(theta,x,y,mask,w,hosts,mass,width,cross_path=cross_path,offset=offset)[0]
        checkpoints=[]
        for epoch in range(1,SETTINGS["epochs"]+1):
            check()
            value,grad=objective(theta,x,y,mask,w,hosts,mass,width,cross_path=cross_path,offset=offset)
            norm=float(np.linalg.norm(grad));grad*=min(1.,SETTINGS["gradient_clip"]/max(norm,1e-30))
            first=.9*first+.1*grad;second=.999*second+.001*grad*grad
            theta*=1-SETTINGS["learning_rate"]*SETTINGS["weight_decay"]
            theta-=SETTINGS["learning_rate"]*(first/(1-.9**epoch))/(np.sqrt(second/(1-.999**epoch))+1e-8)
            if epoch in (100,500,1000):checkpoints.append({"epoch":epoch,"pre_update_loss":value,"gradient_norm":norm})
        value,grad=objective(theta,x,y,mask,w,hosts,mass,width,cross_path=cross_path,offset=offset)
        runs.append({"seed":seed,"initial_loss":initial,"final_loss":value,
                     "gradient_norm":float(np.linalg.norm(grad)),"checkpoints":checkpoints,
                     "convergence_claim":False})
        params.append(theta.tolist())
    selected=min(range(len(runs)),key=lambda i:(runs[i]["final_loss"],runs[i]["seed"]))
    return {"schema":SCHEMA,"representation":representation,"cross_path":cross_path,
            "balanced_training_classes":balanced,"width":width,
            "features":list(base.FEATURES)+(list(INTERACTIONS) if representation=="path_geometry" else []),
            "means":means.tolist(),"scales":scales.tolist(),"theta":params[selected],
            "execution_context":rows[0]["execution_context"],"world_size":rows[0]["shape"][0],
            "training_shapes":[s[0]["shape"] for s in surfaces],
            "training_sources":sorted({r["source_id"] for r in rows}),"settings":dict(SETTINGS),
            "training_host_winners":positive,"training_shape_count":len(surfaces),
            "training_shape_weights":mass.tolist(),"runs":runs,"selected_start":selected,
            "prior":prior,"residual_support":residual_support,
            "seconds":time.monotonic()-begin,"compiler_admission":False}


def prior_scores(prior,rows):
    """Read-only raw cost scores on the same measured candidate set.

    This archive prior is empirical and P4-specific, not measured service
    times. No prior.select eligibility filtering may change the evaluation set.
    """
    from .additive_latency_selection import AdditiveLatencySelector,host_features,fused_features,_domain
    from .relative_selection import ScheduleCandidate
    from .two_stage_selection import parse_schedule_family
    parsed=AdditiveLatencySelector.from_dict(prior);scores=[]
    for row in rows:
        p,m,k,n=row["shape"];name=row["candidate_id"]
        q,g=parse_schedule_family(name)
        h=name=="host";replay=1 if h else (n+128*g-1)//(128*g)
        c=ScheduleCandidate(row["execution_context"],name,p,m,k,n,2*p*(p-1)*m*k,replay,q,g,
                            0 if h else ((m+63)//64)*replay*q)
        c.validate()
        if (row["execution_context"]!=parsed.environment_id or not _domain(c)
            or h!=row["is_host"] or (g==2 and m!=64)):
            raise ValueError("archival cost prior cannot be applied to this candidate/context")
        ms=parsed.host.predict(host_features(c)) if h else parsed.fused.predict(fused_features(c,parsed.capacity))
        scores.append(math.log(ms*1000))
    return scores


@lru_cache(maxsize=256)
def in_training_hull(training_shapes,shape):
    """Geometry support only, NOT an uncertainty probability or accuracy proof."""
    import numpy as np
    from scipy.optimize import linprog
    if not 2<=len(training_shapes)<=256 or any(s[0]!=shape[0] for s in training_shapes):
        raise ValueError("bounded same-world support set required")
    if shape in training_shapes:return True
    x=np.log2(np.asarray([s[1:] for s in training_shapes],dtype=float));q=np.log2(shape[1:])
    result=linprog(np.zeros(len(x)),A_eq=np.vstack((x.T,np.ones(len(x)))),b_eq=np.r_[q,1.],
                   bounds=(0,None),method="highs",options={"time_limit":.25})
    if result.status==2:return False
    if result.status!=0:raise RuntimeError("geometry support solve incomplete; do not guess support")
    if np.max(abs(x.T@result.x-q))>1e-7 or abs(result.x.sum()-1)>1e-7:
        raise ArithmeticError("invalid support certificate")
    return True


def predict(model,rows):
    import numpy as np
    expected=list(base.FEATURES)+(list(INTERACTIONS) if model["representation"]=="path_geometry" else [])
    if (model["schema"]!=SCHEMA or model["features"]!=expected or not 1<=len(rows)<=14
        or len({r["candidate_id"] for r in rows})!=len(rows) or len({tuple(r["shape"]) for r in rows})!=1):
        raise ValueError("one valid bounded candidate surface required")
    if any(r["execution_context"]!=model["execution_context"] or r["shape"][0]!=model["world_size"] for r in rows):
        raise ValueError("execution context differs from training")
    rows=sorted(rows,key=lambda r:r["candidate_id"])
    x=(np.asarray([feature_vector(r,model["representation"]) for r in rows])-model["means"])/model["scales"]
    correction=forward(np.asarray(model["theta"]),x,model["width"])
    if model.get("prior") is not None:
        if model["residual_support"]=="training_hull" and not in_training_hull(
                tuple(tuple(s) for s in model["training_shapes"]),tuple(rows[0]["shape"])):
            correction=np.zeros_like(correction)
        correction=correction+prior_scores(model["prior"],rows)
    scores=dict(zip((r["candidate_id"] for r in rows),map(float,correction)))
    canonical_choice(scores)
    return scores


def evaluate(model,rows,*,training_diagnostic=False):
    surfaces=base.validate_rows(rows,split="train" if training_diagnostic else "evaluation")
    if not training_diagnostic and any(s[0]["shape"] in model["training_shapes"] for s in surfaces):
        raise ValueError("whole-shape train/evaluation overlap")
    if training_diagnostic and any(s[0]["shape"] not in model["training_shapes"] for s in surfaces):
        raise ValueError("training diagnostic contains an untrained shape")
    decisions=[]
    for surface in surfaces:
        scores=predict(model,surface);chosen=canonical_choice(scores)
        measured={r["candidate_id"]:r["latency_us"] for r in surface}
        h=min(r["latency_us"] for r in surface if r["is_host"])
        f=min(r["latency_us"] for r in surface if not r["is_host"])
        chose_host=next(r["is_host"] for r in surface if r["candidate_id"]==chosen)
        support=(in_training_hull(tuple(tuple(s) for s in model["training_shapes"]),tuple(surface[0]["shape"]))
                 if model.get("prior") is not None and model["residual_support"]=="training_hull" else None)
        decisions.append({"shape":surface[0]["shape"],"chosen":chosen,"measured_us":measured,
                          "predicted_us":{n:math.exp(v) for n,v in scores.items()},
                          "host_wins":h<f,"host_error":h<f and not chose_host,
                          "fused_error":f<h and chose_host,"regret":measured[chosen]/min(measured.values())-1,
                          "residual_geometric_support":support,
                          "order_invariant":scores==predict(model,list(reversed(surface)))})
    return {"training_diagnostic":training_diagnostic,"mean_regret":sum(r["regret"] for r in decisions)/len(decisions),
            "max_regret":max(r["regret"] for r in decisions),"host_errors":sum(r["host_error"] for r in decisions),
            "fused_errors":sum(r["fused_error"] for r in decisions),"decisions":decisions}
