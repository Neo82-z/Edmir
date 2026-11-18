#!/usr/bin/env python3
"""Read-only, standard-library analysis of recovered MoE artifacts. No model import."""
from pathlib import Path
from collections import Counter, defaultdict
import json, gzip, hashlib, statistics as st, math, sqlite3, csv
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'experiments/2026-11-17-real-moe-gpu'
RAW=RUN/'recovered'
def read(p): return json.loads(p.read_text())
def write(n,x): (HERE/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def quant(a,q):
 a=sorted(a)
 if not a:return None
 z=(len(a)-1)*q;i=int(z);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
def desc(a):return {'n':len(a),'min':min(a) if a else None,'median':st.median(a) if a else None,'p95':quant(a,.95),'max':max(a) if a else None}
def histdesc(h):
 n=sum(h.values());nz=n-h[0];mass=sum(m*c for m,c in h.items())
 return {'observations':n,'zero_fraction':h[0]/n if n else None,'nonzero_min':min((m for m in h if m),default=None),'max':max(h,default=0),'assignment_rows':mass,'nonzero_le128_fraction':sum(c for m,c in h.items() if 0<m<=128)/nz if nz else None,'bins':[{'lo':lo,'hi':hi,'count':sum(c for m,c in h.items() if lo<=m<=hi),'row_mass_fraction':sum(m*c for m,c in h.items() if lo<=m<=hi)/mass if mass else None} for lo,hi in [(0,0),(1,32),(33,63),(64,128),(129,512),(513,1000000)]],'histogram':{str(k):v for k,v in sorted(h.items())}}
protected=read(RUN/'protection-before.json')['files']
protection={'checked':len(protected),'mismatches':[n for n,h in protected.items() if sha(ROOT/n)!=h]}
manifest=read(RAW/'recovery-manifest.json')['files']
recovery={'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest.values()),'mismatches':[n for n,x in manifest.items() if (RAW/n).stat().st_size!=x['bytes'] or sha(RAW/n)!=x['sha256']]}
assert not protection['mismatches'] and not recovery['mismatches']
print('Integrity verified',protection['checked'],recovery['files'],flush=True)
# Baseline: only r0/r1/r2 formal blocks; warmups and profiled runs excluded.
blocks=[]; all_requests=[]; by_work=defaultdict(list); outputs=defaultdict(list)
for f in sorted((RAW/'baseline-attempt01').glob('r[012]-*.json')):
 d=read(f)
 if not isinstance(d,dict) or 'requests' not in d:continue
 req=d['requests']; all_requests+=req;work=d['block_id'].split('-',1)[1]
 b={'block':d['block_id'],'workload':work,'requests':len(req),'success':sum(r['success'] for r in req),'output_tokens':sum(r['output_tokens'] for r in req if r['success']),'seconds':d['seconds']}
 b['tps']=b['output_tokens']/d['seconds']
 assert math.isclose(b['tps'],d['successful_output_tokens_per_second'],rel_tol=1e-12)
 for field in ['ttft_seconds','tpot_seconds','wall_seconds']:
  a=[r[field]*1000 for r in req if r['success']]; b[field+'_ms']=desc(a)
 for r in req: outputs[(work,r['source_id'],r['prompt_token_ids_sha256'])].append(r['output_token_ids'])
 blocks.append(b);by_work[work].append(b)
perf={'blocks':blocks,'requests':len(all_requests),'success':sum(r['success'] for r in all_requests),'output_tokens':sum(r['output_tokens'] for r in all_requests),'workloads':{}}
for w,bs in by_work.items():
 perf['workloads'][w]={'tps':desc([b['tps'] for b in bs]),'tps_range_over_median':(max(b['tps'] for b in bs)-min(b['tps'] for b in bs))/st.median(b['tps'] for b in bs)}
 for m in ['ttft_seconds_ms','tpot_seconds_ms','wall_seconds_ms']:
  perf['workloads'][w][m]={'median_of_block_p50':st.median(b[m]['median'] for b in bs),'median_of_block_p95':st.median(b[m]['p95'] for b in bs),'block_p50_range':[min(b[m]['median'] for b in bs),max(b[m]['median'] for b in bs)]}
# Greedy repeated outputs are a reproducibility observation, not semantic quality/admission.
perf['repeated_output_identity']={w:{'requests_with_three_outputs':sum(len(v)==3 for k,v in outputs.items() if k[0]==w),'exact_three_way_matches':sum(len(v)==3 and v[0]==v[1]==v[2] for k,v in outputs.items() if k[0]==w)} for w in by_work}
write('baseline-analysis.json',perf)
# Streaming route extraction: use only audited local (pre-dispatch) real-token counts.
global_counts=defaultdict(lambda:[0]*64);step_tokens=defaultdict(int);cats={}; source_hists=defaultdict(Counter); kernels=defaultdict(Counter); confs=defaultdict(Counter); collective=defaultdict(Counter); step_pad={}; fanout=defaultdict(Counter); route_layout=Counter(); dtype=Counter(); shapes=Counter();files=[]; allsampling={}
for f in sorted((RAW/'routes-attempt01/workers').glob('diag-*-rank*.jsonl.gz')):
 work=f.name.split('-')[1];files.append(f.name)
 with gzip.open(f,'rt') as fp:
  for line in fp:
   x=json.loads(line);kind=x['kind'];step=(work,x['step']);rank=x['rank']
   if kind=='step':
    cats[step]=x['category'];step_tokens[step]+=len(x['tokens']);step_pad[step]=sum(x['padded_sizes']);continue
   cat=cats[step];key=(work,cat)
   if kind=='route':
    route_layout[x['routing_layout']]+=1
    assert x['routing_layout']=='local'
    counts=[a+b for a,b in zip(x['phase_counts']['prefill'],x['phase_counts']['decode'])]
    assert sum(counts)==4*x['token_count']
    g=global_counts[(work,x['step'],x['layer'])]
    for i,v in enumerate(counts):g[i]+=v
    source_hists[key].update(counts)
    for ids in x['topk_ids']:
     dest={int(i)//8 for i in ids}
     fanout[key]['tokens']+=1;fanout[key]['destination_sum']+=len(dest)
     fanout[key]['remote_destination_sum']+=len(dest-{rank})
     fanout[key]['local_destination_tokens']+=rank in dest
   elif kind=='kernel':
    assert x['physical_layout_verified']
    proj=x['projection'];s=kernels[(work,cat,proj)];s['observations']+=1
    s['local_slots']+=x['local_assignment_slots']
    assert sum(a['block_slots'] for a in x['local_expert_assignment'].values())==x['local_assignment_slots']
    bm=x['config']['BLOCK_SIZE_M']
    for a in x['local_expert_assignment'].values():
     s['real_rows']+=a['real_rows'];s['inactive_expert_slots']+=a['block_slots'] if not a['real_rows'] else 0
     s['minimum_rounded_real_slots']+=math.ceil(a['real_rows']/bm)*bm
    confs[(work,cat,proj)][json.dumps(x['config'],sort_keys=True)]+=1
    dtype[(x['input_dtype'],x['weight_dtype'],x['output_dtype'],x['accumulator_dtype'])]+=1
    shapes[(proj,x['weight_shape'][-1],x['weight_shape'][-2])]+=1
   elif kind=='collective':
    s=collective[(work,cat,x['wrapper'],x['name'])];s['calls']+=1;s['logical_input_tensor_bytes']+=sum(t['bytes'] for t in x['tensors'])
 print('Read',f.name,flush=True)
for work in ['S1','S32','L1','L32']:
 ss=[read(f) for f in sorted((RAW/'routes-attempt01/workers').glob(f'diag-{work}-sampling-rank*.json'))]
 assert len(ss)==8 and all(s['step_counts']==ss[0]['step_counts'] and s['token_counts']==ss[0]['token_counts'] for s in ss)
 allsampling[work]=ss[0]
global_hists=defaultdict(Counter);rank_skew=defaultdict(list);expert_skew=defaultdict(list)
for (w,s,l),c in global_counts.items():
 key=(w,cats[(w,s)]);global_hists[key].update(c);rs=[sum(c[i:i+8]) for i in range(0,64,8)]
 rank_skew[key].append(max(rs)/st.mean(rs));expert_skew[key].append(max(c)/st.mean(c))
routes={'sampling':allsampling,'route_layout_counts':dict(route_layout),'logical_projection_K_N':[{'projection':p,'K':k,'N':n,'kernel_observations':count} for (p,k,n),count in sorted(shapes.items())],'dtypes':[{'input_weight_output_accum':k,'count':v} for k,v in dtype.items()],'groups':[],'collectives':[]}
for (w,c),h in sorted(global_hists.items()):
 steps=[s for s in cats if s[0]==w and cats[s]==c]
 group={'workload':w,'category':c,'sampled_steps':len(steps),'total_observed_steps':allsampling[w]['step_counts'][c],'sampled_real_token_observations':sum(step_tokens[s] for s in steps),'padded_input_rows':sum(step_pad[s] for s in steps),'source_expert_M':histdesc(source_hists[(w,c)]),'global_expert_M':histdesc(h),'rank_assignment_max_over_mean':desc(rank_skew[(w,c)]),'expert_assignment_max_over_mean':desc(expert_skew[(w,c)]),'projections':{}}
 for p in ['gate_up','down']:
  s=dict(kernels[(w,c,p)]);s['slot_to_real_row_ratio']=s['local_slots']/s['real_rows'];s['real_row_fill_fraction']=s['real_rows']/s['local_slots'];s['configurations']=[{'config':json.loads(k),'observations':v} for k,v in confs[(w,c,p)].items()];group['projections'][p]=s
 assert group['projections']['gate_up']['real_rows']==sum(m*v for m,v in h.items())
 assert group['projections']['down']['real_rows']==group['projections']['gate_up']['real_rows']
 group['logical_token_destination_fanout']={**fanout[(w,c)], 'mean_destinations':fanout[(w,c)]['destination_sum']/fanout[(w,c)]['tokens'], 'mean_remote_destinations':fanout[(w,c)]['remote_destination_sum']/fanout[(w,c)]['tokens']}
 routes['groups'].append(group)
for (w,c,wrapper,name),v in sorted(collective.items()):routes['collectives'].append({'workload':w,'category':c,'wrapper':wrapper,'name':name,**v})
# Independent cross-check against previously accepted route analyzer.
accepted=read(RUN/'routes-analysis-final.json')
for w,a in accepted.items():
 gs=[g for g in routes['groups'] if g['workload']==w]
 assert sum(g['sampled_steps'] for g in gs)==a['complete_steps']
 assert sum(g['global_expert_M']['assignment_rows'] for g in gs)==a['sampled_routed_rows']
 assert sum(g['sampled_real_token_observations'] for g in gs)==a['captured_token_observations']
write('route-analysis.json',routes)
write('integrity.json',{'protection':protection,'recovery':recovery,'analysis_source_sha256':sha(Path(__file__)),'inputs':{'performance_final':sha(RUN/'performance-final.json'),'routes_analysis_final':sha(RUN/'routes-analysis-final.json')}})
print('Written baseline-analysis.json, route-analysis.json, integrity.json',flush=True)
