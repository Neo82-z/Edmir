"""Static information-loss checks; no fitting and no candidate promotion."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from audit import ROOT,HERE,OLD,NAMES,raw,state,arrays
from edmir.compiler.decision_loss_selection import DecisionLossSelector
from edmir.compiler.traffic_latency_selection import CAPACITIES,FUSED_NAMES,fused_features
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from predict_traffic_schedule import static_candidates
from run_selector_gpu import write_json


def main(out):
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    surfaces=list(data.validate().values());core=('direct','splitk2','splitk4','splitk8','splitk16')
    collisions=[]
    for capacity in CAPACITIES:
        groups=defaultdict(list)
        for s in surfaces:groups[tuple(tuple(fused_features(s[n],capacity)) for n in core)].append(s)
        for equal in groups.values():
            if len(equal)>1 and len({min(core,key=lambda n:s[n].measured_ms) for s in equal})>1:
                collisions.append({'capacity':capacity,'all_core_candidate_features_identical':True,
                    'shapes':[{'shape':s['host'].shape,'best_fused':min(core,key=lambda n:s[n].measured_ms),
                               'q4_minus_q2_us':1000*(s['splitk4'].measured_ms-s['splitk2'].measured_ms),
                               'times_ms':{n:s[n].measured_ms for n in core}} for s in equal]})
    v=json.loads((OLD/'verified/replay/traffic_gap.json').read_text());model=DecisionLossSelector.from_dict(v['full_fit'])
    full_errors=[];pair_weights=[]
    for s in surfaces:
        c={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items()}
        pred=model.select(c,baseline=data.baseline);best=min(s,key=lambda n:s[n].measured_ms)
        if pred['choice']!=best:
            full_errors.append({'shape':s['host'].shape,'selected':pred['choice'],'best':best,
                                'actual_extra_us':1000*(s[pred['choice']].measured_ms-s[best].measured_ms)})
    for total in (6,11):
        w=np.r_[.5,np.repeat(.5/(total-1),total-1)]
        mass=(1-w@w)/2
        pair_weights.append({'candidates':total,'host_fused_total_fraction':float(w[0]*sum(w[1:])/mass),
            'each_host_fused_fraction':float(w[0]*w[1]/mass),'each_fused_fused_fraction':float(w[1]*w[2]/mass)})
    # Pair differences are affine in K for any fixed M,N,g,capacity,coefficient
    # vector. In particular, the only explicitly nonlinear K term cancels.
    affine=[]
    for capacity in CAPACITIES:
        per_k=[]
        for k in (2048,4096,8192):
            c=static_candidates(512,k,512,data.environment_id)
            per_k.append(np.array(fused_features(c['splitk4'],capacity))-fused_features(c['splitk2'],capacity))
        residual=per_k[1]-(2*per_k[0]+per_k[2])/3
        affine.append({'capacity':capacity,'max_feature_affine_residual':float(np.max(np.abs(residual))),
                       'depth_tail_output_tiles_difference':float(per_k[1][FUSED_NAMES.index('depth_tail_output_tiles')])})
    paths=[Path(__file__),ROOT/'EDM-IR/runtime/src/nccl_device_fused_cutlass.cu',
           ROOT/'EDM-IR/runtime/src/cutlass_sm80_gemm.cuh',ROOT/'EDM-IR/runtime/src/cuda_backend.cpp']
    write_json(out,{'feature_collisions':collisions,'full_training_only_errors':full_errors,'pair_weights':pair_weights,
                    'q_pair_affine_checks':affine,'interpretation':'Static exact feature collisions and posthoc diagnostics; not a new trained model',
                    'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths}})


if __name__=='__main__':main(HERE/'verified/structure.json')
