"""Seal evidence bindings, preserve original run, and fit inspectable development models."""
from pathlib import Path
import sys,json,time,hashlib,importlib.metadata,platform
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import precision_candidate_selection as model
from edmir.compiler import path_aware_selection as old
import torch,numpy,scipy,dion
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=json.loads((OUT/'run-01/datasets.json').read_text())
for ds in data.values():
 for row in ds['rows']:row['validity']['contract_sha256']=model.evidence_contract(row)
(OUT/'datasets-with-evidence-contracts.json').write_text(json.dumps(data,indent=2)+'\n')
installed=Path(dion.__file__).parent;source=Path('/tmp/edmir-dion-7692479/dion');checked={}
for path in source.glob('*.py'):
 dest=installed/path.name;assert dest.exists() and sha(path)==sha(dest);checked[path.name]=sha(path)
(OUT/'dependency-provenance.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'dion_distribution':importlib.metadata.version('dion'),'dion_commit':model.DION_REVISION,'installed_dion_matches_git_files':checked,'execution':'CPU only; force_eager, no CUDA/MPS; official optimizer implementation unchanged'},indent=2)+'\n')
prior=json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/development-model.json').read_text())['model']['prior']
# No more search: full-data artifacts for inspection only, never validation scores.
recipe={'configs':[['p4','dion3'],['p8','adamw'],['p8','dion3']],'ce_weight':1.,'settings':model.SETTINGS,
        'purpose':'development artifacts selected for inspection after disclosed retrospective comparison; no independent performance claim',
        'new_source_sha256':sha(ROOT/'EDM-IR/edmir/compiler/precision_candidate_selection.py'),
        'binding_addendum':'Run-01 source preserved. Only evidence binding checks and explicit FP32 construction added; features/objective/settings unchanged.',
        'original_run_dataset_sha256':sha(OUT/'run-01/datasets.json'),'bound_dataset_sha256':sha(OUT/'datasets-with-evidence-contracts.json')}
(OUT/'development-fit-plan.json').write_text(json.dumps(recipe,indent=2)+'\n')
for ds,opt in recipe['configs']:
 fitted=model.fit(data[ds]['rows'],optimizer=opt,ce_weight=1.,prior=prior if ds=='p4' else None)
 (OUT/f'development-{ds}-{opt}-ce1.json').write_text(json.dumps({'purpose':'training-data inspection only, not validated deployment model','model':fitted},indent=2)+'\n')
 print('development model saved',ds,opt,flush=True)
# Validate that attaching evidence bindings did not change any saved fold predictions.
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('edmir.compiler._precision_run01',OUT/'run-01/source-snapshot/EDM-IR/edmir/compiler/precision_candidate_selection.py');v1=module_from_spec(spec);spec.loader.exec_module(v1)
comparisons=0;max_delta=0.
for file in (OUT/'run-01').glob('*-fold*.json'):
 item=json.loads(file.read_text());fitted=item['model'];ds=item['dataset'];fold=item['fold'];folds={tuple(x['shape']):x['fold'] for x in data[ds]['folds']}
 surfaces=model.valid_rows([{**r,'split':'evaluation'} for r in data[ds]['rows'] if folds[tuple(r['shape'])]==fold],'evaluation')
 for surface in surfaces:
  before=v1.predict(fitted,surface);after=model.predict(fitted,surface)
  assert before==after;comparisons+=1
(OUT/'binding-regression.json').write_text(json.dumps({'all_saved_fold_predictions_unchanged':True,'shape_comparisons':comparisons,'max_score_delta':max_delta},indent=2)+'\n')
# Bounded portable inference on the larger P4 development table.
fitted=json.loads((OUT/'development-p4-dion3-ce1.json').read_text())['model'];surfaces=model.valid_rows(data['p4']['rows'],'train');start=time.perf_counter()
for s in surfaces:model.predict(fitted,s)
cold=time.perf_counter()-start;times=[]
for _ in range(20):
 start=time.perf_counter()
 for s in surfaces:model.predict(fitted,s)
 times.append(time.perf_counter()-start)
(OUT/'inference-timing.json').write_text(json.dumps({'shape_count':len(surfaces),'candidate_rows':len(data['p4']['rows']),'cold_ms':cold*1000,'warm_batch_median_ms':float(numpy.median(times))*1000,'warm_batch_max_ms':max(times)*1000,'iterations':20,'scope':'CPU metadata validation + NumPy scoring + cached hull/prior; not GPU latency'},indent=2)+'\n')
print('binding regression and inference bound complete',comparisons,flush=True)
