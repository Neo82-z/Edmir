"""Post-hoc training-objective diagnostics; never used to select a model."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.shared_candidate_selection import SharedCandidateSelector,training_tensors,objective,VARIANTS,forward
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256


def main(results,out):
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    surfaces=[{n:s[n] for n in sorted(s)} for _,s in sorted(data.validate().items())]
    result=json.loads(results.read_text());records=[]
    for variant in result['variants']:
        m=SharedCandidateSelector.from_dict(variant['full_fit'])
        x,b,y,mask,w,*_=training_tensors(surfaces,m.prior)
        theta=np.asarray(m.parameters);width=VARIANTS[m.method][0]
        logt=b+forward(theta,x,width);error=np.where(mask,logt-y,0.)
        f1,_=objective(theta,x,b,y,mask,w,width,1)
        f0,_=objective(theta,x,b,y,mask,w,width,0)
        chosen=np.where(mask,logt,np.inf).argmin(1)
        best=np.min(np.where(mask,y,np.inf),1)
        regret=np.expm1(y[np.arange(len(surfaces)),chosen]-best)
        records.append({'method':m.method,'training_log_mse':float(np.sum(w*error**2)/len(surfaces)),
            'training_expected_regret':f1-f0,'training_discrete_mean_regret':float(regret.mean()),
            'training_discrete_max_regret':float(regret.max())})
    report={'scope':'Full-data training diagnostics; these are not held-out metrics',
            'records':records,'results_sha256':sha256(results),'source_sha256':sha256(Path(__file__))}
    with out.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.results,a.out)
