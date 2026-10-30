#!/usr/bin/env python3
"""Inspect a development scorer on explicitly supplied measured candidate records."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'EDM-IR'),str(Path(__file__).parent)]
from edmir.compiler import precision_candidate_selection as scorer
from review_precision_scoring import precision


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
 p.add_argument('--shape',nargs=4,type=int,required=True,metavar=('P','M','K','N'));a=p.parse_args()
 envelope=json.loads(a.model.read_text());model=envelope.get('model',envelope);data=json.loads(a.data.read_text())
 ds='p'+str(a.shape[0]);dataset=data.get(ds,{})
 rows=[r for r in dataset.get('rows',[]) if r['shape']==a.shape]
 if not rows:p.error('no observations for this shape in supplied dataset; no labels or correctness evidence will be invented')
 # Preserve failed candidates visibly without inventing timing or geometry labels.
 source=hashlib.sha256(a.data.read_bytes()).hexdigest()
 for r in dataset.get('excluded',[]):
  if r['shape']!=a.shape:continue
  failed={'shape':a.shape,'candidate_id':r['candidate_id'],'execution_context':rows[0]['execution_context'],
          'precision':precision(r['candidate_id'].split('@')[0]),'source_id':source,
          'validity':{'status':'failed','source_id':source,'scope':'failure summary in supplied development dataset'}}
  failed['validity']['contract_sha256']=scorer.evidence_contract(failed);rows.append(failed)
 try:result=scorer.predict(model,rows)
 except (ValueError,KeyError) as error:p.error(str(error))
 print(json.dumps({**result,'predicted_us':{k:math.exp(v) for k,v in result['scores'].items()},
       'training_shape':a.shape in model['training_shapes'],'compiler_admission':False,
       'note':'Scores are conditional estimates; selection weights are not calibrated optimality probabilities. Failed candidates have no score.'},indent=2))
if __name__=='__main__':main()
