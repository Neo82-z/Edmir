"""Work-equivalence and tail-boundary controls, not cache equivalence claims.

Same CTA work need not have the same unique operand footprint, stride, or reuse.
This module makes that distinction explicit and tests a fixed residual transfer.
It uses saved training observations only; no optimizer or kernel calibration.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

from . import axis_residual_selection as axis
from .dependency_wave_model import KernelCase

SCHEMA='edmir.shape-transfer-control.v1'
MODES=('work_equivalent','work_equivalent_with_M','tail_guarded_M')


def geometry(row):
    precision=axis.base.stage.curves.baseline.old.precision_key(row)
    if precision[:4]!=('fp16','fp16','fp16','fp32') or precision[-1]!='source':
        raise ValueError('geometry ledger is specific to source-verified FP16/FP32 accumulation')
    c=KernelCase.from_candidate(axis.base.stage.geometry(row),precision='fp16_fp32acc_fp16out')
    if precision[4]!=('none' if c.q==1 else 'fp16'):
        raise ValueError('explicit FP16 split-partial contract required')
    count=c.counts();mt=(c.m+63)//64;nt=(c.n+128*c.group_n-1)//(128*c.group_n)
    return {'shape':list(row['shape']),'q':c.q,'group_n':c.group_n,
            'full_tiles':c.m%64==0 and c.n%(128*c.group_n)==0,
            'M_tiles':mt,'N_tiles':nt,'depth':c.depth,'ctas':c.ctas,'chains':c.chains,
            'logical_A_load_bytes':count['logical_A_load_bytes_before_cache_reuse'],
            'logical_B_load_bytes':count['logical_B_load_bytes_before_cache_reuse'],
            'output_write_bytes':count['logical_output_write_bytes'],
            'extra_output_read_bytes':count['logical_extra_output_read_bytes'],
            'unique_A_bytes':2*c.m*c.k,'unique_B_bytes':2*c.k*c.n,
            'A_row_stride_bytes':2*c.k,'B_row_stride_bytes':2*c.n,
            'A_reuse_across_N_tiles':nt,'B_reuse_across_M_tiles':mt,
            'per_destination_source_GEMMs':c.world_size,
            'per_destination_unique_weight_bytes':2*c.k*c.n,
            'per_destination_addressed_input_bytes':2*c.world_size*c.m*c.k,
            'per_destination_remote_input_bytes':2*(c.world_size-1)*c.m*c.k,
            'per_destination_logical_B_load_bytes':c.world_size*count['logical_B_load_bytes_before_cache_reuse'],
            'weight_reuse_across_source_GEMMs':c.world_size,
            'program_locality_contract':'uniform per-source runtime, one weight.r buffer per destination',
            'padded_flops':2*c.chains*64*(128*c.group_n)*c.k,
            'valid_flops':2*c.m*c.n*c.k,
            'physical_cache_or_partition_mapping':None}


def work_key(row):
    if row['is_host']:return None
    d=geometry(row)
    if not d['full_tiles']:return None
    # Missing unique footprints and strides are deliberate experimental factors.
    return json.dumps([row['execution_context'],row['shape'][0],row['shape'][2],row['candidate_id'],
                       axis.base.stage.curves.baseline.old.precision_key(row),row.get('implementation_id'),
                       d['ctas'],d['padded_flops'],d['logical_A_load_bytes'],d['logical_B_load_bytes'],
                       d['output_write_bytes'],d['extra_output_read_bytes']],separators=(',',':'))


def assemble(saved_model,training_rows,mode):
    if mode not in MODES:raise ValueError('unknown shape transfer mode')
    base_axis=axis.assemble(saved_model,training_rows,'M_work_extension')
    index={}
    for r in training_rows:
        key=work_key(r)
        if key is None:continue
        prior=axis.base.transferred_value(saved_model['stage_prior'],r)
        index.setdefault(key,[]).append({'shape':list(r['shape']),'source_id':r['source_id'],
            'latency_us':r['latency_us'],'residual_us':r['latency_us']-prior,'geometry':geometry(r)})
    for items in index.values():items.sort(key=lambda a:(a['shape'],a['source_id']))
    return {'schema':SCHEMA,'mode':mode,'base':deepcopy(saved_model),'axis':base_axis,'index':index,
            'training_shapes':deepcopy(saved_model['training_shapes']),'compiler_admission':False}


def predict(model,rows):
    if model.get('schema')!=SCHEMA or model.get('mode') not in MODES:raise ValueError('unknown shape model')
    old=axis.base.predict(model['base'],rows)
    p=deepcopy(old) if model['mode']=='work_equivalent' else axis.predict(model['axis'],rows)
    p['shape_transfer_details']={}
    for row in sorted(rows,key=lambda r:r['candidate_id']):
        name=row['candidate_id']
        if name not in p['predicted_log_us']:continue
        if row['is_host']:
            p['shape_transfer_details'][name]={'applied':False,'reason':'host_has_different_communication_work'}
            continue
        if model['mode']=='tail_guarded_M':
            d=p['axis_details'][name]
            if d.get('applied'):
                shapes=[row['shape'],*[a['shape'] for t in d['transfers'] for a in t['anchors']]]
                if any(s[1]%64 for s in shapes):
                    p['predicted_log_us'][name]=old['predicted_log_us'][name]
                    p['curve_details'][name]=deepcopy(old['curve_details'][name])
                    p['support'][name]=deepcopy(old['support'][name])
                    p['axis_details'][name]={'applied':False,'rejected_transfer':d,'reason':'M_tail_regime_changed'}
                    p['shape_transfer_details'][name]={'applied':False,'reason':'M_tail_regime_changed'}
                    continue
            p['shape_transfer_details'][name]={'applied':False,'reason':'no_tail_guard_change'}
            continue
        detail=old['curve_details'][name]
        if detail['curve_applied'] or detail['host_interval_applied']:
            p['shape_transfer_details'][name]={'applied':False,'reason':'existing_K_route'}
            continue
        key=work_key(row)
        anchors=[a for a in model['index'].get(key,[]) if a['shape']!=row['shape']] if key else []
        if not anchors:
            p['shape_transfer_details'][name]={'applied':False,'reason':'no_full_tile_equal_work_anchor'}
            continue
        if len(anchors)>256:raise ValueError('bounded anchor table required')
        value=detail['prior_us']+math.fsum(a['residual_us'] for a in anchors)/len(anchors)
        if not math.isfinite(value) or value<=0:raise ValueError('invalid work-transfer time')
        query_geometry=geometry(row)
        p['predicted_log_us'][name]=math.log(value)
        p['curve_details'][name]={'prior_us':detail['prior_us'],'prediction_us':value,
            'curve_applied':False,'host_interval_applied':False,'shape_transfer_applied':True,
            'curve':None,'reason':'experimental_equal_work_transfer'}
        p['shape_transfer_details'][name]={'applied':True,'prediction_us':value,'anchors':anchors,
            'query_geometry':query_geometry,'cache_equivalence_verified':False,
            'varying_factors':['unique_A_bytes','unique_B_bytes','B_row_stride_bytes',
                              'A_reuse_across_N_tiles','B_reuse_across_M_tiles']}
        p['support'][name]={'supported':False,'reason':'equal_work_is_not_verified_cache_equivalence'}
        if 'axis_details' in p and p['axis_details'][name].get('applied'):
            p['axis_details'][name]={'applied':False,'superseded_transfer':p['axis_details'][name],
                                     'reason':'equal_work_control_has_precedence'}
    p['provisional_choice']=axis.base.stage.curves.baseline.canonical_choice(p['predicted_log_us']) if p['predicted_log_us'] else None
    p['requires_validation']=(not p['predicted_log_us'] or any(not v['supported'] for v in p['support'].values())
                              or any(v!='failed' for v in p['excluded'].values()))
    p['chosen']=None if p['requires_validation'] else p['provisional_choice']
    p['compiler_admission']=False
    return p
