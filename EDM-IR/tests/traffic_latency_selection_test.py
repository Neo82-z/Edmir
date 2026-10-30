import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.traffic_latency_selection import (
    TrafficLatencySelector,host_features,fused_features,training_rows_weights,
)
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.relative_selection import ScheduleCandidate
from predict_traffic_schedule import static_candidates

class TrafficLatencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
        cls.model=TrafficLatencySelector.fit(cls.data)
        cls.rows=static_candidates(64,8192,512,cls.data.environment_id)

    def test_coupled_input_size_and_separate_depth_N(self):
        a=static_candidates(512,2048,512,self.data.environment_id)['host']
        b=static_candidates(128,8192,512,self.data.environment_id)['host']
        self.assertEqual(a.rows_per_peer*a.k,b.rows_per_peer*b.k)
        self.assertEqual(host_features(a,2,1)[4:],host_features(b,2,1)[4:])
        a=static_candidates(64,4096,128,self.data.environment_id)['host']
        b=static_candidates(64,4096,2048,self.data.environment_id)['host']
        self.assertEqual(host_features(a,2,1),host_features(b,2,1))
        b=static_candidates(64,8192,2048,self.data.environment_id)['host']
        self.assertGreater(host_features(b,2,1)[3],0)

    def test_extra_partition_work_vanishes_at_q1(self):
        row=static_candidates(64,4096,512,self.data.environment_id)['direct']
        x=fused_features(row,32)
        self.assertEqual(x[3],0)
        self.assertEqual(x[5],0)
        self.assertGreater(x[-1],0)

    def test_training_weight_definition_and_unit_invariance(self):
        ss=list(self.data.validate().values());rows,w=training_rows_weights(ss,False)
        counts={s['host'].shape:len(s)-1 for s in ss}
        # All weighted residuals are proportional to relative errors after
        # candidate-count correction; the common scale cancels in this ratio.
        values=[v*r.measured_ms*counts[r.shape]**.5 for r,v in zip(rows,w)]
        np.testing.assert_allclose(values,values[0],atol=1e-12)
        altered=replace(self.data,rows=tuple(replace(r,measured_ms=7*r.measured_ms) for r in reversed(self.data.rows)))
        model=TrafficLatencySelector.fit(altered)
        self.assertEqual((self.model.capacity,self.model.ridge,self.model.input_threshold,self.model.depth_threshold),
                         (model.capacity,model.ridge,model.input_threshold,model.depth_threshold))
        np.testing.assert_allclose(np.array(self.model.host.coefficients)*7,model.host.coefficients,atol=1e-9)
        np.testing.assert_allclose(np.array(self.model.fused.coefficients)*7,model.fused.coefficients,atol=1e-9)

    def test_roundtrip_and_no_labels_at_inference(self):
        model=TrafficLatencySelector.from_dict(json.loads(json.dumps(self.model.to_dict())))
        self.assertEqual(model.select(self.rows,baseline=self.data.baseline),self.model.select(self.rows,baseline=self.data.baseline))
        surface=next(iter(self.data.validate().values()))
        with self.assertRaises(TypeError):model.select(surface,baseline=self.data.baseline)
        self.assertFalse(model.select(self.rows,baseline=self.data.baseline)['eligible_for_compiler_admission'])

    def test_baseline_scope_and_grouped_support_rejected(self):
        with self.assertRaises(ValueError):self.model.select(self.rows,baseline=PER_SOURCE_CUBLAS)
        with self.assertRaises(ValueError):self.model.select(static_candidates(64,8192,512,'new-server'),baseline=self.data.baseline)
        with self.assertRaises(ValueError):self.model.select(static_candidates(128,4096,2048,self.data.environment_id),baseline=self.data.baseline)
        result=self.model.select(static_candidates(512,4096,512,self.data.environment_id),baseline=self.data.baseline)
        self.assertFalse(result['predictions']['splitk4_groupedn2']['eligible'])

    def test_corrupt_and_incomplete_inner_search_rejected(self):
        base=json.loads(json.dumps(self.model.to_dict()))
        for key,value in [('host_cv_scores',[]),('host_ridge',0),('depth_threshold',True),('input_threshold',16),('cv_scores',[]),('eligible_for_compiler_admission',True)]:
            obj=copy.deepcopy(base);obj[key]=value
            with self.assertRaises(ValueError):TrafficLatencySelector.from_dict(obj)

    def test_deadline_and_outer_isolation(self):
        with self.assertRaises(TimeoutError):TrafficLatencySelector.fit(self.data,timeout_seconds=1e-12)
        result=json.loads((ROOT/'experiments/2026-10-21-host-boundary-repair/results.json').read_text())
        for protocol in ('primary','stress'):
            for fold in result[protocol]['fits']:
                tests={tuple(r['shape']) for r in result[protocol]['decisions'] if r['heldout']==fold['heldout']}
                self.assertFalse(tests&{tuple(s) for s in fold['model']['training_shapes']})
                self.assertEqual(len(fold['model']['host_cv_scores']),48)
                self.assertEqual(len(fold['model']['cv_scores']),12)

if __name__=='__main__':unittest.main()
