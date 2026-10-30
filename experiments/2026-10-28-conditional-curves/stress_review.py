"""No new stress partition selection: reuse the three existing development cuts."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import conditional_curve_selection as model
from review_calibrated_structure import aggregate, DATA


def main():
    out=HERE/'stress';out.mkdir(exist_ok=False)
    names=('all_M512','all_N2048','all_host_winners')
    (out/'design-before-fit.json').write_text(json.dumps({'cuts':names,'modes':['prior','host_tiles_fused_residual'],
        'source':'existing host-decision-audit stress partitions; no hyperparameter search',
        'frozen_six_used':False,'compiler_admission':False},indent=2)+'\n')
    data=json.loads(DATA.read_text())['p4']['rows'];result={}
    for name in names:
        src=ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{name}.json'
        old=json.loads(src.read_text());prior=old['model']['prior'];shapes={tuple(s) for s in prior['training_shapes']}
        train=[r for r in data if tuple(r['shape']) in shapes]
        test=[{**r,'split':'evaluation'} for r in data if tuple(r['shape']) not in shapes]
        result[name]={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
        for mode in ('prior','host_tiles_fused_residual'):
            fit=model.fit(train,prior,mode=mode);d=model.evaluate(fit,test)
            a=aggregate(d);a['wrong_choices']=sum(r['provisional_regret']>0 for r in d)
            a['host_interval_applied']=sum(v['host_interval_applied'] for r in d for v in r['curve_details'].values())
            a['fused_curve_applied']=sum(v['curve_applied'] for r in d for v in r['curve_details'].values())
            result[name][mode]=a
            (out/f'{name}-{mode}.json').write_text(json.dumps({'model':fit,'evaluation':a},indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({n:{m:{k:v for k,v in a.items() if k!='decisions'} for m,a in r.items() if isinstance(a,dict)} for n,r in result.items()},indent=2))


if __name__=='__main__':main()
