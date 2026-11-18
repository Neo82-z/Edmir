"""Audit recovered P1/P2 records; no fitting or measured-label shortlist design."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib, json, statistics as st, sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'bundle/source/EDM-IR'),str(HERE/'bundle/source/EDM-IR/tools')]
from run_w1_gpu import validate_rows,done_raw_changed
from edmir.compiler import w1_candidate_selection as score
read=lambda p:json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def med(row):return st.median(row['samples_ms'])*1000
def choose(order,times):return min(order,key=lambda f:(times[f],0 if f in score.HOSTS else 1,f))

def main():
    rec=HERE/'recovered';out=rec/'results';plan=read(HERE/'bundle/plan.json')
    manifest=read(rec/'recovery-manifest.json')['files']
    for name,v in manifest.items():
        p=(rec/name).resolve()
        assert p.is_relative_to(rec.resolve()) and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
    for name,h in read(HERE/'protection-before.json').items():assert sha(name)==h,name
    for name in ('run.py','preregistered-predictions.json','frozen-primary-model.json'):
        assert sha(HERE/name)==sha(rec/name),name
    pred=read(HERE/'preregistered-predictions.json')['predictions']
    assert sha(HERE/'preregistered-predictions.json')==plan['predictions_sha256']
    context=read(out/'context.json');inputs=read(HERE/'bundle/inputs.json')
    assert context['package']==sha(HERE/'bundle/files.json')
    assert context['build']==sha(rec/'build/receipt.json')==sha(HERE/'build/receipt.json')
    for tag,h in read(HERE/'build/receipt.json')['binaries'].items():assert sha(rec/f'build/w1_{tag}_batch')==h
    assert context['plan']==sha(HERE/'bundle/plan.json')
    assert context['sealed_runner']==sha(HERE/'run.py')
    failed=set();failures=[];allrows=[];blocks=defaultdict(list);checks=defaultdict(list)
    for i,job in enumerate(plan['jobs']):
        p=out/f'collect-{i:03d}.json';r=read(p);tag=job['shape']
        expected=sorted(set([f for f in pred[tag]['frozen_ranking'] if (tag,f) not in failed][:3]+[f for f in plan['fixed_order'] if (tag,f) not in failed])) if job['phase']=='select' else [f for f in job['order'] if job['phase']=='check' or (tag,f) not in failed]
        assert r['job']==job and r['context']==sha(out/'context.json') and r['order']==expected
        assert r['excluded_by_numerical_check']==sorted(f for t,f in failed if t==tag)
        assert r['input']==inputs[tag][str(job['seed'])]
        assert not done_raw_changed(p,out)
        validate_rows(r['rows'],job,expected,plan['shapes'][tag],context['runtime'])
        times={}
        for row in r['rows']:
            allrows.append(dict(group=i,shape=tag,phase=job['phase'],row=row))
            if not row['correctness_passed']:
                failed.add((tag,row['family']));failures.append(dict(group=i,shape=tag,row=row))
            if job['phase']=='check':checks[tag].append((job['seed'],row,p))
            elif row['correctness_passed']:times[row['family']]=med(row)
        if times:blocks[tag].append(dict(group=i,phase=job['phase'],seed=job['seed'],order=r['order'],times_us=times,process_seconds=r['process_seconds']))
    assert read(out/'collect-complete.json')['groups']==len(plan['jobs'])
    audit=dict(files_verified=len(manifest),groups=len(plan['jobs']),correctness_rows=sum(x['phase']=='check' for x in allrows),timing_rows=sum('samples_ms' in x['row'] for x in allrows),samples=sum(len(x['row'].get('samples_ms',[])) for x in allrows),failed_candidates=sorted(failed),failures=failures,protected_files_unchanged=True,fit_performed=False)
    write(HERE/'recovery-audit.json',audit)
    secondary_p1=blocks['m512']
    balanced_plan=read(HERE/'balanced-p1-plan.json');balanced=rec/'results-p1-balanced'
    assert sha(rec/'balanced-p1-plan.json')==sha(HERE/'balanced-p1-plan.json')
    assert sha(rec/'run_balanced_p1.py')==sha(HERE/'run_balanced_p1.py')
    bc=read(balanced/'context.json')
    assert {k:v for k,v in bc.items() if k not in ('balanced_plan','balanced_runner')}==context
    assert bc['balanced_plan']==sha(HERE/'balanced-p1-plan.json') and bc['balanced_runner']==sha(HERE/'run_balanced_p1.py')
    assert read(balanced/'collect-complete.json')['groups']==6
    p1=[]
    for i,job in enumerate(balanced_plan['jobs']):
        path=balanced/f'collect-{i:03d}.json';r=read(path)
        assert r['job']==job and r['order']==job['order'] and r['context']==sha(balanced/'context.json')
        assert r['input']==inputs['m512'][str(job['seed'])]
        assert not done_raw_changed(path,balanced)
        validate_rows(r['rows'],job,job['order'],plan['shapes']['m512'],context['runtime'])
        assert all(x['correctness_passed'] for x in r['rows'])
        p1.append(dict(group=i,phase='balanced_pair',seed=job['seed'],order=r['order'],times_us={x['family']:med(x) for x in r['rows']},process_seconds=r['process_seconds']))
    audit.update(primary_p1_balanced_groups=6,primary_p1_balanced_timing_rows=24,primary_p1_balanced_samples=2400,original_p1_blocks_retained_as_secondary=True)
    write(HERE/'recovery-audit.json',audit)
    pairs={}
    for a,b in [('pipeline_serial','pipeline'),('splitk4','splitk4_groupedn2')]:
        ps=[dict(group=x['group'],a_us=x['times_us'][a],b_us=x['times_us'][b],speedup=x['times_us'][a]/x['times_us'][b],latency_reduction_pct=100*(1-x['times_us'][b]/x['times_us'][a])) for x in p1 if a in x['times_us'] and b in x['times_us']]
        pairs[a+'__'+b]=dict(blocks=ps,median_speedup=st.median(x['speedup'] for x in ps) if ps else None,b_faster_blocks=sum(x['speedup']>1 for x in ps),work_matched=(a=='pipeline_serial'))
    p2={};primary_regrets=[];fixed_regrets=[];host_regrets=[]
    for tag in pred:
        v=[x for x in blocks[tag] if x['phase']=='verify'];s=[x for x in blocks[tag] if x['phase']=='select']
        assert len(v)==3 and len(s)==1
        families=set.intersection(*(set(x['times_us']) for x in v))-{f for t,f in failed if t==tag}
        aggregate={f:st.median(x['times_us'][f] for x in v) for f in sorted(families)}
        oracle=choose(families,aggregate);host=choose(families & set(score.HOSTS),aggregate);fused=choose(families-set(score.HOSTS),aggregate)
        ranking=[f for f in pred[tag]['frozen_ranking'] if f in families]
        supported_oracle=choose(ranking,aggregate)
        fixed=[f for f in plan['fixed_order'] if f in families]
        results={}
        def evaluate(name,shortlist,selected):
            if selected not in aggregate:return dict(status='not_admitted',candidate=selected)
            br=[100*(x['times_us'][selected]/min(x['times_us'][f] for f in families)-1) for x in v]
            return dict(candidate=selected,shortlist=shortlist,oracle_retained=oracle in shortlist,aggregate_regret_pct=100*(aggregate[selected]/aggregate[oracle]-1),block_regrets_pct=br,mean_block_regret_pct=st.mean(br),worst_block_regret_pct=max(br))
        for k in (1,2,3):
            for name,order in [('model',ranking),('fixed_order',fixed)]:
                sl=order[:k]
                # A late numerical failure is a failed shortlist evaluation, never
                # permission to substitute a newly measured verification winner.
                if not sl or not set(sl)<=set(s[0]['times_us']):
                    results[f'{name}_top{k}']=dict(status='selection_incomplete',shortlist=sl)
                    continue
                selected=choose(sl,s[0]['times_us'])
                results[f'{name}_top{k}']=evaluate(name,sl,selected)
        results['fixed_q4_g2']=evaluate('fixed', ['splitk4_groupedn2'],'splitk4_groupedn2')
        host_choices=[f for f in ['host_merged_cublaslt','host_merged_cutlass'] if f in families]
        results['independent_host_choice']=evaluate('measured_host_choice',host_choices,choose(host_choices,s[0]['times_us']))
        results['strong_host_oracle']=evaluate('host', [host],host)
        primary=results['model_top1'];primary_regrets.append(primary['aggregate_regret_pct'])
        if 'aggregate_regret_pct' in results['fixed_q4_g2']:fixed_regrets.append(results['fixed_q4_g2']['aggregate_regret_pct'])
        host_regrets.append(results['strong_host_oracle']['aggregate_regret_pct'])
        hf_margin=(aggregate[fused]/aggregate[host]-1)*100
        errors=[]
        for a,b in combinations(ranking,2):
            measured=aggregate[a]-aggregate[b];estimated=pred[tag]['scores_us'][a]-pred[tag]['scores_us'][b]
            if measured*estimated<0:errors.append(dict(a=a,b=b,predicted_delta_us=estimated,measured_delta_us=measured))
        # Current-context checks are bound to their actual hashes, never the training context.
        required={str(seed):inputs[tag][str(seed)] for seed in plan['seeds']}
        queries=[]
        for old in pred[tag]['queries_for_geometry_only']:
            row=dict(old,execution_context=sha(out/'context.json'),source_id=context['build'])
            binding=score.contract(row);row['checks']=[]
            for seed,c,p in checks[tag]:
                if c['family']==row['candidate_id']:
                    row['checks'].append(dict(seed=seed,contract_sha256=binding,record_sha256=sha(p),input=required[str(seed)],checked_elements=c['checked_elements'],passed=c['correctness_passed']))
            queries.append(row)
        strict=score.predict(read(HERE/'frozen-primary-model.json')['model'],queries,required)
        p2[tag]=dict(shape=plan['shapes'][tag],aggregate_us=aggregate,oracle=oracle,oracle_block_winners=dict(Counter(choose(families,x['times_us']) for x in v)),best_host=host,best_fused=fused,host_advantage_pct=hf_margin,host_fused_class='host' if hf_margin>2 else 'fused' if hf_margin< -2 else 'near_tie',primary_direction_error=(hf_margin>2 and primary['candidate'] not in score.HOSTS) or (hf_margin< -2 and primary['candidate'] in score.HOSTS),evaluations=results,full_original_surface_present=len(families)==13,shape_mape_supported_pct=st.mean(100*abs(pred[tag]['scores_us'][f]/aggregate[f]-1) for f in ranking),pairwise_direction_errors=errors,strict_new_context_result=strict,verification_blocks=v,selection_block=s[0],search_cost_note='Selection process_seconds is measured for the whole union of model and fixed candidates, including setup/checks. Subset search wall time is not isolated; do not attribute it solely to model top-k.')
        # A conditional amortization of the actual joint evaluation workflow,
        # not an invented isolated top-k search cost or a measured deployment.
        p2[tag]['joint_selection_workflow_amortization_us']={str(n):{
            method:s[0]['process_seconds']*1e6/n+aggregate[value['candidate']]
            for method,value in results.items() if 'aggregate_regret_pct' in value}
            for n in (1,100,10000,1000000)}
        p2[tag]['regret_decomposition']=dict(supported_oracle=supported_oracle,
            candidate_support_coverage_loss_pct=100*(aggregate[supported_oracle]/aggregate[oracle]-1),
            ranking_loss_within_supported_candidates_pct=100*(aggregate[primary['candidate']]/aggregate[supported_oracle]-1),
            identity='1+total_regret = (1+support_coverage_loss)*(1+within_support_ranking_loss), each in fractions')
    directions={v['host_fused_class'] for v in p2.values()}
    metrics=dict(mean_regret_pct=st.mean(primary_regrets),max_regret_pct=max(primary_regrets),fixed_q4g2_mean_regret_pct=st.mean(fixed_regrets) if fixed_regrets else None,fixed_q4g2_max_regret_pct=max(fixed_regrets) if fixed_regrets else None,host_oracle_mean_regret_pct=st.mean(host_regrets),host_oracle_max_regret_pct=max(host_regrets),shape_balanced_mape_pct=st.mean(x['shape_mape_supported_pct'] for x in p2.values()),direction_errors=sum(x['primary_direction_error'] for x in p2.values()),bidirectional_over_2pct_coverage={'host','fused'}<=directions,full_candidate_oracle_gate=all(x['full_original_surface_present'] for x in p2.values()),topk_oracle_retention={str(k):sum(x['evaluations'][f'model_top{k}']['oracle_retained'] for x in p2.values()) for k in (1,2,3)})
    metrics['methods']={name:dict(mean_regret_pct=st.mean(x['evaluations'][name]['aggregate_regret_pct'] for x in p2.values()),max_regret_pct=max(x['evaluations'][name]['aggregate_regret_pct'] for x in p2.values())) for name in p2['b1']['evaluations'] if all('aggregate_regret_pct' in x['evaluations'][name] for x in p2.values())}
    metrics['exact_top1_hits']=sum(x['evaluations']['model_top1']['candidate']==x['oracle'] for x in p2.values())
    metrics['gate_conditions']=dict(mean_at_most_1pct=metrics['mean_regret_pct']<=1,max_at_most_3pct=metrics['max_regret_pct']<=3,both_directions=metrics['bidirectional_over_2pct_coverage'],full_candidate_surface=metrics['full_candidate_oracle_gate'],not_worse_than_fixed_order_top1=all(metrics['methods']['model_top1'][m]<=metrics['methods']['fixed_order_top1'][m] for m in ['mean_regret_pct','max_regret_pct']),not_worse_than_independently_selected_strong_host=all(metrics['methods']['model_top1'][m]<=metrics['methods']['independent_host_choice'][m] for m in ['mean_regret_pct','max_regret_pct']))
    metrics['all_reported_accuracy_coverage_gates_pass']=all(metrics['gate_conditions'].values())
    result=dict(audit=audit,p1=dict(pairs=pairs,blocks=p1,secondary_original_unbalanced_order_blocks=secondary_p1,primary_design=balanced_plan),p2=p2,summary=metrics,statistical_unit='shape and independent process block; 100 serial samples are not 100 independent experiments',strict_admission_and_head_transfer_separate=True,trace_audit='See trace-analysis.json; not assumed complete by this script',predictor_closed=True)
    write(HERE/'analysis.json',result)
    print(json.dumps({'audit':{k:v for k,v in audit.items() if k!='failures'},'summary':metrics,'p1':pairs},ensure_ascii=False))

if __name__=='__main__':main()
