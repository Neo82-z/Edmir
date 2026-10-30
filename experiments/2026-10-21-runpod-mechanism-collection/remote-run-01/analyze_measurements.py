"""Validate downloaded measurements and compute descriptive paired contrasts."""
from pathlib import Path
import json,sys,statistics,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from run_mechanism_gpu import check_row,summarize
from independent_selector_validation import digest


def main():
    p=json.loads((HERE/'revision-01/package.json').read_text())['package']
    infos=json.loads((HERE/'revision-01/inputs.json').read_text())['workloads']
    results={}
    for stage in ('anchors','mechanisms','geometry'):
        directory=HERE/'results'/(stage+'-01')
        raw=directory/'measurements.jsonl'
        if not raw.exists():continue
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        jobs=[j for j in p['jobs'] if j['stage']==stage]
        if len(rows)!=len(jobs):raise ValueError('incomplete stage: '+stage)
        context=json.loads((directory/'context.json').read_text())
        run=json.loads((directory/'run.json').read_text())
        if context['package_sha256']!=digest(p) or context['stage']!=stage or run['context']!=context:
            raise ValueError('run context differs')
        for row,job in zip(rows,jobs):
            if row!=check_row(row,job,infos[job['workload_id']],p):raise ValueError('measurement identity changed')
            if row['runtime']!=context['runtime']:raise ValueError('row runtime differs')
        summary=summarize(p,rows,stage)
        if summary!=json.loads((directory/'summary.json').read_text()):raise ValueError('server/local summary disagreement')
        rawhash=hashlib.sha256(raw.read_bytes()).hexdigest()
        if json.loads((directory/'run.json').read_text())['raw_sha256']!=rawhash:raise ValueError('raw hash mismatch')
        windows=[]
        algorithms={};means={}
        for row in rows:
            key=str(row['shape'])+' '+row['candidate_id']
            algorithms.setdefault(key,set()).add(row['gemm_algorithm_id'])
            means.setdefault(key,[]).append(row['mean_ms'])
            v=row['samples_ms'];early=statistics.median(v[:20]);late=statistics.median(v[-20:])
            windows.append({'shape':row['shape'],'candidate':row['candidate_id'],'block':row['repetition'],
                            'last20_over_first20':late/early,'max_over_median':max(v)/statistics.median(v)})
        results[stage]={'summary':summary,'raw_sha256':rawhash,'processes':len(rows),
            'timed_samples':sum(len(r['samples_ms']) for r in rows),
            'checked_elements_per_before_or_after_pass':sum(r['checked_elements'] for r in rows),
            'max_abs_error':max(r['max_abs_error'] for r in rows),
            'max_normalized_error':max(r['max_rel_error'] for r in rows),
            'mean_of_process_means_ms':{k:statistics.mean(v) for k,v in means.items()},
            'algorithm_variation_flags':{k:sorted(v) for k,v in algorithms.items() if len(v)>1},
            'within_process_windows':windows,
            'within_process_drift_flags':[r for r in windows if not .9<=r['last20_over_first20']<=1.1]}
    obj={'stages':results,'new_training_runs':0,'independent_validation':False,'compiler_admission':False,
         'statistics_scope':'Descriptive process-block differences and ratios; no significance or confidence claims.'}
    (HERE/'validated-analysis.json').write_text(json.dumps(obj,indent=2)+'\n')
    for stage,obj in results.items():
        print(stage,obj['processes'],'processes',obj['timed_samples'],'samples',len(obj['within_process_drift_flags']),'within-process drift flags')
        for w in obj['summary']['workloads']:
            print(w['shape'],{k:round(v*1000,2) for k,v in w['median_of_process_medians_ms'].items()})


if __name__=='__main__':main()
