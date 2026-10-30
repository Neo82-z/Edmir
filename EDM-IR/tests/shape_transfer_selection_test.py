from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import shape_transfer_selection as model


class ShapeTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.folds,cls.bases=diagnose.load_inputs()
        cls.train=[r for r in cls.rows if cls.folds[tuple(r['shape'])]!=1]
        cls.query=[r for r in cls.rows if r['shape']==[4,64,512,1024]]
        cls.state=model.assemble(cls.bases[1],cls.train,'work_equivalent_with_M')

    def row(self,shape,candidate='splitk2'):
        return next(r for r in self.rows if r['shape']==list(shape) and r['candidate_id']==candidate)

    def test_equal_work_does_not_erase_reuse_direction(self):
        a=self.row((4,64,512,1024));b=self.row((4,128,512,512))
        self.assertEqual(model.work_key(a),model.work_key(b))
        x,y=model.geometry(a),model.geometry(b)
        for key in ('unique_A_bytes','unique_B_bytes','B_row_stride_bytes','A_reuse_across_N_tiles','B_reuse_across_M_tiles'):
            self.assertNotEqual(x[key],y[key])
        for key in ('ctas','padded_flops','logical_A_load_bytes','logical_B_load_bytes','output_write_bytes'):
            self.assertEqual(x[key],y[key])

    def test_tail_cannot_enter_full_tile_equivalence(self):
        a=self.row((4,32,512,512))
        self.assertIsNone(model.work_key(a))
        self.assertFalse(model.geometry(a)['full_tiles'])

    def test_shared_weight_not_replicated_per_source(self):
        g=model.geometry(self.row((4,64,8192,1024)))
        self.assertEqual(g['per_destination_unique_weight_bytes'],2*8192*1024)
        self.assertEqual(g['per_destination_logical_B_load_bytes'],4*g['logical_B_load_bytes'])
        self.assertEqual(g['per_destination_addressed_input_bytes'],4*g['unique_A_bytes'])

    def test_query_labels_and_candidate_order_independent(self):
        expected=model.predict(self.state,self.query)
        query=deepcopy(self.query)
        for r in query:r.pop('latency_us')
        self.assertEqual(expected,model.predict(self.state,list(reversed(query))))

    def test_subset_never_changes_surviving_candidate_scores(self):
        p=model.predict(self.state,self.query)
        short=model.predict(self.state,self.query[:-1])
        for k,v in short['predicted_log_us'].items():self.assertEqual(p['predicted_log_us'][k],v)

    def test_serialization_order_and_no_mutation(self):
        before=json.dumps([self.state,self.query,self.train],sort_keys=True)
        a=model.predict(self.state,self.query)
        reverse=model.assemble(self.bases[1],list(reversed(self.train)),'work_equivalent_with_M')
        self.assertEqual(a,model.predict(json.loads(json.dumps(reverse)),self.query))
        self.assertEqual(before,json.dumps([self.state,self.query,self.train],sort_keys=True))

    def test_no_fit_and_exact_training_fold_required(self):
        with diagnose.forbid_training() as calls:
            model.assemble(self.bases[1],self.train,'work_equivalent')
        self.assertEqual(calls,[])
        with self.assertRaises(ValueError):model.assemble(self.bases[1],self.train+self.query,'work_equivalent')
        with self.assertRaises(ValueError):model.assemble(self.bases[1],self.train[1:],'work_equivalent')

    def test_identity_and_precision_checked(self):
        for field,value in (('execution_context','other-GPU'),('implementation_id','other-kernel')):
            query=deepcopy(self.query)
            for r in query:r[field]=value
            with self.assertRaises(ValueError):model.predict(self.state,query)
        r=deepcopy(self.query[0]);r['precision']['input']='bf16'
        with self.assertRaises(ValueError):model.geometry(r)

    def test_score_explanation_and_no_admission(self):
        p=model.predict(self.state,self.query)
        self.assertIsNone(p['chosen']);self.assertFalse(p['compiler_admission'])
        for n,v in p['predicted_log_us'].items():
            self.assertAlmostEqual(math.exp(v),p['curve_details'][n]['prediction_us'])
            if p['shape_transfer_details'][n].get('applied'):
                self.assertFalse(p['shape_transfer_details'][n]['cache_equivalence_verified'])
                self.assertFalse(p['support'][n]['supported'])

    def test_tail_guard_restores_original_score_without_removing_candidate(self):
        train=[r for r in self.rows if self.folds[tuple(r['shape'])]!=2]
        query=[r for r in self.rows if r['shape']==[4,128,512,512]]
        state=model.assemble(self.bases[2],train,'tail_guarded_M')
        p=model.predict(state,query);old=model.axis.base.predict(self.bases[2],query)
        for r in query:
            n=r['candidate_id']
            if not r['is_host']:
                self.assertEqual(p['predicted_log_us'][n],old['predicted_log_us'][n])
                self.assertEqual(p['shape_transfer_details'][n]['reason'],'M_tail_regime_changed')
        self.assertEqual(set(p['predicted_log_us']),set(old['predicted_log_us']))

    def test_existing_K_anchors_remain_authoritative(self):
        for shape,fold in self.folds.items():
            if fold!=1:continue
            query=[r for r in self.rows if tuple(r['shape'])==shape]
            old=model.axis.base.predict(self.bases[1],query);p=model.predict(self.state,query)
            for n,d in old['curve_details'].items():
                if d['curve_applied'] or d['host_interval_applied']:
                    self.assertEqual(old['predicted_log_us'][n],p['predicted_log_us'][n])


if __name__=='__main__':unittest.main()
