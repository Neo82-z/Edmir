import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import lsq_linear

import candidate as module
from candidate import (Candidate, ROOT, VARIANTS, HOST_GRID, FUSED_GRID, component_system,
                       fit_component, validate_data, grouped_support, eligible, fused_features,
                       host_features, parameter_key, joint_scores)
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from predict_traffic_schedule import static_candidates

HERE=Path(__file__).resolve().parent


class JointDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
        cls.models={name:Candidate.fit(cls.data,method=name) for name in VARIANTS}

    def test_selected_absolute_fits_match_independent_bvls(self):
        surfaces,_=validate_data(self.data)
        for m in self.models.values():
            for host,parameters,fit in ((True,m.host_parameters,m.host),(False,m.fused_parameters,m.fused)):
                raw,targets,weights,names,ridge=component_system(surfaces,host,parameters,m.basis)
                x=np.array(raw);scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-9)
                penalty=np.eye(len(names))*math.sqrt(ridge);penalty[0,0]=0
                a=np.vstack((x/scale*np.array(weights)[:,None],penalty))
                b=np.r_[np.array(targets)*weights,np.zeros(len(names))]
                alternate=lsq_linear(a,b,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=1000)
                self.assertTrue(alternate.success)
                np.testing.assert_allclose(a@alternate.x,a@(np.array(fit.coefficients)*scale),atol=1e-8,rtol=1e-8)
                self.assertLessEqual(fit.kkt_max_violation,1e-7)

    def test_one_inner_fold_score_matches_explicit_candidate_enumeration(self):
        surfaces,groups=validate_data(self.data);held=groups[0]
        training=[]
        original=module.fit_component
        def recording(ss,*args):
            training.append({s['host'].group_id for s in ss})
            return original(ss,*args)
        with patch.object(module,'fit_component',side_effect=recording):
            scores,folds=joint_scores(surfaces,[held],'resource',time.monotonic()+30)
        self.assertEqual(len(training),60)
        self.assertTrue(all(held not in g for g in training))
        train=[s for s in surfaces if s['host'].group_id!=held]
        test=[s for s in surfaces if s['host'].group_id==held]
        support=grouped_support(train)
        for record in (scores[0],min(scores,key=lambda r:parameter_key(r,'mean_regret')),scores[-1]):
            hp,fp=record['host_parameters'],record['fused_parameters']
            h=fit_component(train,True,hp,'resource');f=fit_component(train,False,fp,'resource')
            regrets=[];errors=[]
            for s in test:
                pred={n:(h.predict(host_features(r,*hp[:2])) if n=='host' else f.predict(fused_features(r,fp[0],'resource')))
                      for n,r in s.items() if eligible(r,support)}
                choice=min(pred,key=lambda n:(pred[n],n!='host',n))
                regrets.append(s[choice].measured_ms/min(r.measured_ms for r in s.values())-1)
                errors.append(sum(((pred[n]-pred['host']-s[n].measured_ms+s['host'].measured_ms)/s['host'].measured_ms)**2
                                  for n in pred if n!='host')/(len(pred)-1))
            self.assertAlmostEqual(record['mean_regret'],sum(regrets)/len(regrets))
            self.assertAlmostEqual(record['max_regret'],max(regrets))
            self.assertAlmostEqual(record['worst_group_mean_regret'],sum(regrets)/len(regrets))
            self.assertAlmostEqual(record['paired_difference_mse'],sum(errors)/len(errors))

    def test_full_grid_and_training_only_parameter_choice(self):
        for m in self.models.values():
            self.assertEqual(len(m.cv_scores),576)
            self.assertEqual({(r['host_parameters'],r['fused_parameters']) for r in m.cv_scores},
                             {(h,f) for h in HOST_GRID for f in FUSED_GRID})
            chosen=min(m.cv_scores,key=lambda r:parameter_key(r,m.objective))
            self.assertEqual((m.host_parameters,m.fused_parameters),(chosen['host_parameters'],chosen['fused_parameters']))
            for fold in m.inner_folds:
                train=set(fold['training_shapes']);test={r['shape'] for r in fold['test']}
                self.assertFalse(train&test)
                self.assertEqual(train|test,set(m.training_shapes))

    def test_model_order_cache_and_label_changes_do_not_leak(self):
        name='resource_joint_mean';m=self.models[name]
        uncached=Candidate.fit(replace(self.data,rows=tuple(reversed(self.data.rows))),method=name,use_cache=False)
        self.assertEqual(m.to_dict(),uncached.to_dict())
        previous=module._bank.cache_info().misses
        changed=replace(self.data,rows=(replace(self.data.rows[0],measured_ms=self.data.rows[0].measured_ms*1.01),*self.data.rows[1:]))
        alternate=Candidate.fit(changed,method=name)
        self.assertEqual(module._bank.cache_info().misses,previous+1)
        self.assertNotEqual(m.cv_scores,alternate.cv_scores)
        self.assertEqual(module._bank.cache_info().maxsize,128)

    def test_static_only_symmetric_choice_and_domain_guards(self):
        for m in self.models.values():
            for surface in self.data.validate().values():
                r=surface['host'];static=static_candidates(r.rows_per_peer,r.k,r.n,self.data.environment_id)
                selected=m.select(static,baseline=self.data.baseline)
                expected=min(selected['predicted_ms'],key=lambda n:(selected['predicted_ms'][n],n!='host',n))
                self.assertEqual(selected['choice'],expected)
                self.assertFalse(selected['eligible_for_compiler_admission'])
            static=static_candidates(64,4096,512,self.data.environment_id)
            with self.assertRaises(ValueError):m.select(static,baseline=PER_SOURCE_CUBLAS)
            with self.assertRaises(ValueError):m.select(static_candidates(192,3072,1536,self.data.environment_id),baseline=self.data.baseline)
            with self.assertRaises(TypeError):m.select(next(iter(self.data.validate().values())),baseline=self.data.baseline)
            renamed=dict(static);renamed['wrong']=renamed.pop('direct')
            with self.assertRaises(ValueError):m.select(renamed,baseline=self.data.baseline)
            mixed=dict(static);mixed['direct']=replace(mixed['direct'],environment_id='other')
            with self.assertRaises(ValueError):m.select(mixed,baseline=self.data.baseline)
        with self.assertRaises(TimeoutError):Candidate.fit(self.data,timeout_seconds=1e-12)
        for timeout in (0,float('inf'),float('nan'),True):
            with self.assertRaises(ValueError):Candidate.fit(self.data,timeout_seconds=timeout)

    def test_grouped_support_not_expanded_and_outer_labels_not_in_fit(self):
        data=replace(self.data,rows=tuple(r for r in self.data.rows if r.n<2048))
        m=Candidate.fit(data)
        static=static_candidates(64,4096,2048,self.data.environment_id)
        with self.assertRaises(ValueError):m.select(static,baseline=data.baseline)
        result=m.select(static,baseline=data.baseline,allow_extrapolation=True)
        self.assertFalse(any('grouped' in n for n in result['predicted_ms']))
        self.assertTrue(all(cell.m==64 for cell in m.grouped_support))

    def test_all_outer_gates_and_review_coverage_preserved(self):
        path=Path(os.environ.get('JOINT_REVIEW_RESULTS',str(HERE/'initial/results.json')))
        result=json.loads(path.read_text())
        self.assertEqual([v['method'] for v in result['variants']],list(VARIANTS))
        for v in result['variants']:
            for protocol in ('primary','stress'):
                self.assertEqual(len(v[protocol]['decisions']),41)
                for fold in v[protocol]['fits']:
                    outer_test={tuple(r['shape']) for r in v[protocol]['decisions'] if r['heldout']==fold['heldout']}
                    model=fold['model'];training={tuple(s) for s in model['training_shapes']}
                    self.assertFalse(outer_test&training)
                    self.assertEqual(len(model['cv_scores']),576)
                    for inner in model['inner_folds']:
                        inside={tuple(s) for s in inner['training_shapes']}|{tuple(r['shape']) for r in inner['test']}
                        self.assertEqual(inside,training)
            self.assertEqual(sum(len(r['deletions']) for r in v['review']['deletion']),328)
            self.assertEqual(len(v['review']['boundary_stress']['records']),34)
            self.assertEqual(len(v['assessment']['checks']),5)
            self.assertEqual(v['assessment']['candidate_accepted'],all(v['assessment']['checks'].values()))

    def test_old_evidence_and_execution_package_unchanged(self):
        for name in ('2026-10-21-additive-selector-repair','2026-10-21-host-boundary-repair','2026-10-21-structural-identifiability'):
            p=ROOT/'experiments'/name/'verified/decision.json';old=json.loads(p.read_text())
            for relative,value in old['source_sha256'].items():self.assertEqual(sha256(ROOT/relative),value)
            for relative,value in old['evidence_sha256'].items():self.assertEqual(sha256(p.parent/relative),value)
        folder=ROOT/'experiments/2026-10-21-boundary-gpu-package'
        delivery=json.loads((folder/'delivery.json').read_text())
        self.assertEqual(sha256(folder/delivery['archive']),delivery['archive_sha256'])
        from prepare_boundary_gpu import verify_bundle
        verify_bundle(folder/'bundle')


if __name__=='__main__':unittest.main()
