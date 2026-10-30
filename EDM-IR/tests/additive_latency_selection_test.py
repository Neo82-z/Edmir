import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'EDM-IR'), str(ROOT/'EDM-IR/tools')]
from edmir.compiler.additive_latency_selection import (
    AdditiveLatencySelector, host_features, fused_features, MAX_ROWS,
)
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.relative_selection import ScheduleCandidate
from independent_selector_validation import candidate


class AdditiveLatencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, *_ = load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
        cls.model = AdditiveLatencySelector.fit(cls.data)
        cls.surface = next(s for s in cls.data.validate().values() if s['host'].shape == (4,64,8192,512))
        cls.candidates = cls.static(cls.surface)

    @classmethod
    def static(cls, surface):
        return {n:ScheduleCandidate.from_measurement(r,cls.data.environment_id) for n,r in surface.items()}

    def test_group_K_q_separate_and_partial_tile_host_has_no_negative_extrapolation(self):
        a = candidate((4,64,1024,1024),'direct_groupedn2',self.data.environment_id)
        b = candidate((4,64,8192,1024),'splitk8_groupedn2',self.data.environment_id)
        self.assertEqual(a.k/a.split_k_slices,b.k/b.split_k_slices)
        self.assertNotEqual(fused_features(a,32),fused_features(b,32))
        a = candidate((4,32,8192,512),'host',self.data.environment_id)
        b = candidate((4,64,8192,512),'host',self.data.environment_id)
        self.assertEqual(host_features(a),host_features(b))
        self.assertTrue(all(x>=0 for x in host_features(a)))

    def test_saved_model_roundtrip_and_standalone_inference_without_numpy_or_scipy(self):
        document = json.loads(json.dumps(self.model.to_dict()))
        saved = AdditiveLatencySelector.from_dict(document)
        expected = self.model.select(self.candidates,baseline=self.data.baseline)
        self.assertEqual(saved.select(self.candidates,baseline=self.data.baseline),expected)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'model.json';p.write_text(json.dumps(document))
            code = '''import json,sys
from edmir.compiler.additive_latency_selection import AdditiveLatencySelector
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
from edmir.compiler.relative_selection import ScheduleCandidate
m=AdditiveLatencySelector.from_dict(json.load(open(sys.argv[1])))
c={n:ScheduleCandidate(**r) for n,r in json.loads(sys.argv[2]).items()}
print(m.select(c,baseline=PER_SOURCE_CUTLASS)["choice"])
assert "numpy" not in sys.modules and "scipy" not in sys.modules
'''
            from dataclasses import asdict
            import os
            r = subprocess.run([sys.executable,'-S','-c',code,str(p),json.dumps({n:asdict(r) for n,r in self.candidates.items()})],
                               env={**os.environ,'PYTHONPATH':str(ROOT/'EDM-IR')},capture_output=True,text=True,timeout=10)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(r.stdout.strip(),expected['choice'])

    def test_training_order_and_common_unit_scale_invariance(self):
        changed = replace(self.data,rows=tuple(replace(r,measured_ms=7*r.measured_ms) for r in reversed(self.data.rows)))
        other = AdditiveLatencySelector.fit(changed)
        self.assertEqual((self.model.capacity,self.model.ridge),(other.capacity,other.ridge))
        for a,b in ((self.model.host,other.host),(self.model.fused,other.fused)):
            np.testing.assert_allclose(np.array(a.coefficients)*7,b.coefficients,atol=1e-10,rtol=1e-9)
        self.assertEqual(other.select(self.candidates,baseline=self.data.baseline)['choice'],
                         self.model.select(self.candidates,baseline=self.data.baseline)['choice'])

    def test_labels_never_enter_inference_and_both_host_directions_use_same_minimum(self):
        with self.assertRaises(TypeError):self.model.select(self.surface,baseline=self.data.baseline)
        poisoned = {n:replace(r,measured_ms=r.measured_ms*(1e4 if n=='host' else 1e-4)) for n,r in self.surface.items()}
        self.assertEqual(self.static(poisoned),self.candidates)
        for s in self.data.validate().values():
            r=self.model.select(self.static(s),baseline=self.data.baseline)
            self.assertEqual(r['predicted_ms'][r['choice']],min(r['predicted_ms'].values()))
            self.assertFalse(r['eligible_for_compiler_admission'])

    def test_wrong_identity_mixed_surface_and_unmeasured_cross_axis_rejected(self):
        with self.assertRaises(ValueError):self.model.select(self.candidates,baseline=PER_SOURCE_CUBLAS)
        with self.assertRaises(ValueError):self.model.select({n:replace(r,environment_id='new-server') for n,r in self.candidates.items()},baseline=self.data.baseline)
        with self.assertRaises(ValueError):self.model.select({**self.candidates,'host':candidate((4,64,1024,512),'host',self.data.environment_id)},baseline=self.data.baseline)
        rows={n:candidate((4,128,4096,2048),n,self.data.environment_id) for n in ('host','splitk4')}
        with self.assertRaises(ValueError):self.model.select(rows,baseline=self.data.baseline)

    def test_group_M_guard_and_extrapolation_is_explicit(self):
        rows={n:candidate((4,128,4096,512),n,self.data.environment_id) for n in ('host','splitk4','splitk4_groupedn2')}
        r=self.model.select(rows,baseline=self.data.baseline)
        self.assertFalse(r['predictions']['splitk4_groupedn2']['eligible'])
        train=replace(self.data,rows=tuple(r for r in self.data.rows if r.n!=2048))
        model=AdditiveLatencySelector.fit(train)
        rows={n:candidate((4,64,4096,2048),n,self.data.environment_id) for n in ('host','splitk4')}
        with self.assertRaises(ValueError):model.select(rows,baseline=self.data.baseline)
        self.assertTrue(model.select(rows,baseline=self.data.baseline,allow_extrapolation=True)['predictions']['host']['training_range_extrapolation'])

    def test_corrupt_model_rejected(self):
        base=json.loads(json.dumps(self.model.to_dict()))
        examples=[]
        for field,value in [('capacity',999),('environment_id',None),('baseline_id','cublas'),('eligible_for_compiler_admission',True),('cv_scores',[])]:
            a=copy.deepcopy(base);a[field]=value;examples.append(a)
        for field,value in [('coefficients',[-1.]*7),('scales',[0.]*7),('kkt_max_violation',.1),('coefficients',[float('nan')]*7)]:
            a=copy.deepcopy(base);a['host'][field]=value;examples.append(a)
        a=copy.deepcopy(base);a['training_bounds'][0][1]=999;examples.append(a)
        a=copy.deepcopy(base);a['injected']='bad';examples.append(a)
        a=copy.deepcopy(base);a['host']['injected']='bad';examples.append(a)
        for obj in examples:
            with self.assertRaises(ValueError):AdditiveLatencySelector.from_dict(obj)

    def test_bounded_fit_and_invalid_inner_folds(self):
        with self.assertRaises(TimeoutError):AdditiveLatencySelector.fit(self.data,timeout_seconds=1e-12)
        with self.assertRaises(ValueError):AdditiveLatencySelector.fit(replace(self.data,rows=self.data.rows*(MAX_ROWS//len(self.data.rows)+1)))
        with self.assertRaises(ValueError):AdditiveLatencySelector.fit(self.data,timeout_seconds=float('inf'))
        # Relabeling individual K values into new groups would leak a sweep.
        row=self.data.rows[0]
        altered=replace(self.data,rows=(replace(row,group_id='wrong-inner-fold'),*self.data.rows[1:]))
        with self.assertRaises(ValueError):AdditiveLatencySelector.fit(altered)

    def test_outer_training_catalog_does_not_include_heldout_shapes(self):
        path=ROOT/'experiments/2026-10-21-additive-selector-repair/results.json'
        result=json.loads(path.read_text())
        self.assertTrue(result['gate']['passed'])
        for protocol in ('primary','stress'):
            for fit in result[protocol]['fits']:
                tests={tuple(r['shape']) for r in result[protocol]['decisions'] if r['heldout']==fit['heldout']}
                self.assertFalse(tests & {tuple(s) for s in fit['model']['training_shapes']})
                self.assertEqual(len(fit['model']['cv_scores']),12)
        self.assertEqual(result['new_independent_labels'],0)

    def test_selection_performance_bounded(self):
        start=time.monotonic()
        for _ in range(1000):self.model.select(self.candidates,baseline=self.data.baseline)
        # Generous smoke bound; actual latency and throughput recorded separately.
        self.assertLess(time.monotonic()-start,10.)


if __name__=='__main__':unittest.main()
