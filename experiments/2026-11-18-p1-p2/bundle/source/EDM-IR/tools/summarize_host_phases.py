#!/usr/bin/env python3
"""Validate host phase diagnostics without creating training latency labels."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.two_stage_selection import parse_schedule_family


def summarize(record):
    if record.get('kind')!='host_phase_diagnostic' or record.get('training_eligible') is not False:
        raise ValueError('only explicitly non-training phase diagnostics accepted')
    if record.get('correctness_passed') is not True or record.get('precision')!='fp16_fp32acc_fp16out':
        raise ValueError('FP16 correctness identity required')
    if record.get('dependency_policy')!='resource_queues_v1' or record.get('metadata_mode') not in ('legacy','prepared'):
        raise ValueError('execution identity required')
    shape=record.get('shape')
    if not isinstance(shape,list) or len(shape)!=4 or any(type(v) is not int or not 0<v<=1048576 for v in shape) or not 2<=shape[0]<=8:
        raise ValueError('bounded P,M,K,N required')
    family=record.get('family'); q,g=parse_schedule_family(family)
    if family=='host' and (q,g)!=(1,1): raise ValueError('invalid host')
    if family!='host' and not (family=='direct' or family=='direct_groupedn2' or family.startswith('splitk')):
        raise ValueError('unknown phase family')
    if shape[2]%(32*q) or shape[3]%8:
        raise ValueError('kernel geometry differs')
    if record.get('device_service_us') is not None:
        raise ValueError('host intervals cannot label device service')
    samples=record.get('samples_ns')
    if not isinstance(samples,list) or not 1<=len(samples)<=64:
        raise ValueError('bounded nonempty samples required')
    for s in samples:
        if not isinstance(s,dict) or set(s)!={'submit','completion_wait','wall'}:
            raise ValueError('exact phase schema required')
        if any(type(v) is not int or not 0<=v<=10**12 for v in s.values()):
            raise ValueError('bounded integer nanoseconds required')
        if s['submit']+s['completion_wait']!=s['wall']:
            raise ValueError('host partition identity violated')
    phases={k:{'mean_us':statistics.mean(s[k] for s in samples)/1000,
               'median_us':statistics.median(s[k] for s in samples)/1000}
            for k in ('submit','completion_wait','wall')}
    return {'kind':'host_phase_diagnostic_summary','training_eligible':False,
            'shape':shape,'family':family,'metadata_mode':record['metadata_mode'],
            'sample_count':len(samples),'phases':phases,
            'median_of_wall_is_computed_from_each_total':True,
            'separate_phase_medians_must_not_be_added':True,
            'device_service_us':None,
            'semantics':'submit includes API blocking and concurrent device progress; completion_wait is remaining host wait, not device duration',
            'deployment_identity_requires_external_build_and_run_receipt':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.input.stat().st_size>1048576: raise ValueError('diagnostic exceeds 1 MiB')
    result=summarize(json.loads(a.input.read_text()))
    with a.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
