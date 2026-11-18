#!/usr/bin/env python3
"""Explicit allow-list export; never moves originals or reads credentials."""
from pathlib import Path
import json,hashlib,zipfile,sys
HERE=Path(__file__).resolve().parent; CODE=HERE.parents[1]; PAPER=CODE.parent/'llm-serving-mlsys-latex'; DEST=CODE/'output';DEST.mkdir(exist_ok=True)
files={}
def add(p,root):
 if p.is_file() and not any(x in {'__pycache__','.git','node_modules','build'} for x in p.relative_to(root).parts):files[root.name+'/'+p.relative_to(root).as_posix()]=p
for part in ['EDM-IR/edmir','EDM-IR/ir','EDM-IR/runtime','EDM-IR/tests']:
 for p in (CODE/part).rglob('*'):
  if p.suffix in {'.py','.cpp','.hpp','.cu','.cuh','.h','.cmake','.json','.toml','.txt'}:add(p,CODE)
for name in ['EDM-IR/pyproject.toml','EDM-IR/README.md','README.md','SCOPE.md','papers/REFERENCES_2026-10-30.txt']:
 add(CODE/name,CODE)
for p in (CODE/'EDM-IR/tools').glob('*.py'):add(p,CODE)
for p in (CODE/'EDM-IR/examples').rglob('*.json'):add(p,CODE)
for name in ['prepare_w1_gpu.py','prepare_w1_precision_gpu.py','run_w1_gpu.py','run_precision_followup.py','analyze_precision_window.py','replay_w1_precision.py','prepare_real_moe.py','run_real_moe.py','analyze_real_moe.py','build_real_moe_corpus.py','prepare_native_inputs.py','audit_offline_model.py','run_native_moe_smoke.py']:
 add(CODE/'EDM-IR/tools'/name,CODE)
for folder in ['2026-11-17-evidence-synthesis','2026-11-17-real-moe-analysis','2026-11-16-precision-analysis/attempt-01']:
 for p in (CODE/'experiments'/folder).glob('*'):
  if p.suffix in {'.py','.json','.txt','.csv'}:add(p,CODE)
for folder in ['2026-11-11-shadeform-w1','2026-11-16-shadeform-precision','2026-11-17-final-operator','2026-11-17-real-moe-gpu']:
 for n in ['REPORT.txt','RESULTS.txt','analysis.json','stage-summary.json','performance-final.json','routes-analysis-final.json','application-shape-summary.json','recovery-verification.json','recovered/recovery-manifest.json']:
  add(CODE/'experiments'/folder/n,CODE)
for p in (CODE/'experiments/2026-11-17-final-operator/frozen-predictions').glob('*'):add(p,CODE)
for name in ['README.txt','NEXT_STEPS_ZH.txt','CODE_INDEX.txt','DATA_INDEX.json','protection-before.json','validation.json','export_review.py']:
 add(HERE/name,CODE)
for part in ['sections','figures','data','scripts']:
 for p in (PAPER/'latex'/part).rglob('*'):
  if p.suffix in {'.tex','.bib','.pdf','.py','.sh','.json','.csv','.txt','.md','.sty','.bst'}:add(p,PAPER)
for name in ['main.tex','references.bib','mlsys2025.sty','mlsys2025.bst','algorithm.sty','algorithmic.sty','fancyhdr.sty','Makefile','latexmkrc','requirements-figures.txt','README.md']:
 add(PAPER/'latex'/name,PAPER)
for name in ['SUBMISSION_WRITING_POLICY_ZH.txt','MOE_NARRATIVE_REVISION_ZH.txt','moe-narrative-validation.json','real-moe-figure-audit.json']:
 add(PAPER/'latex/review'/name,PAPER)
add(PAPER/'output/pdf/edmir-moe-narrative-2026-11-17.pdf',PAPER)
# Use the newly compiled bibliography, not a stale root-level bbl.
files['llm-serving-mlsys-latex/latex/main.bbl']=PAPER/'latex/build/main.bbl'
manifest={n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(files.items())}
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
verify='''from pathlib import Path\nimport hashlib,json\nr=Path(__file__).resolve().parent\nm=json.loads((r/'SOURCE_MANIFEST.json').read_text())\nfor n,x in m.items():\n p=r/n\n assert p.stat().st_size==x['bytes'], n\n assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'], n\nprint('Verified',len(m),'files')\n'''
path=DEST/'edmir-paper-review-2026-11-17.zip'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n');z.writestr('VERIFY_PACKAGE.py',verify);z.writestr('START_HERE.txt',(HERE/'README.txt').read_text())
print(json.dumps({'archive':str(path),'files':len(files),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
