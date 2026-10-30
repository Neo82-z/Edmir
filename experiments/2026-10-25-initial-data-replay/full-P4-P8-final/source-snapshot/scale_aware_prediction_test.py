"""Archive-derived counts, strict evaluation separation and stable decisions."""
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.measurement_dataset import load_selector_dataset, P4_ENV, P8_ENV
from edmir.compiler.scale_aware_prediction import ScaleAwareModel, geometry, legacy_graph, FAMILIES
from edmir.compiler.mechanism_prediction import geometry as previous_geometry
from replay_p8_model import audit_archived_graph, forecast


def enumerate_gemm(shape, q, g):
    """Independent useful-element accounting including partial M/N tiles."""
    _,m,k,n=shape; ctas=a=b=output=boundaries=0
    for lo_m in range(0,m,64):
        rows=min(64,m-lo_m)
        for lo_n in range(0,n,128*g):
            cols=min(128*g,n-lo_n)
            for partition in range(q):
                ctas+=1; a+=2*rows*(k//q); b+=2*cols*(k//q)
                if partition:
                    boundaries+=1; output+=4*rows*cols
    return ctas,a,b,output,boundaries


class ScaleAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p4,cls.cal,cls.test,_=load_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
        cls.model=ScaleAwareModel.fit(cls.cal,environment_id=P8_ENV)

    def test_graph_against_archived_implementation(self):
        audit=audit_archived_graph()
        self.assertEqual(audit['graphs']['8'],{'ops':64,'queue_edges':157,'unit_duration_chain':33})
        self.assertEqual(legacy_graph(4),(16,31,11))

    def test_counts_against_independent_partial_tile_enumeration(self):
        for p in (4,8):
            for m,n in ((32,128),(65,264),(512,2048)):
                for f in FAMILIES[1:]:
                    x=geometry((p,m,8192,n),f)
                    c,a,b,o,e=enumerate_gemm((p,m,8192,n),x['q'],x['g'])
                    self.assertEqual((x['ctas_per_gemm'],x['logical_A_per_gemm_bytes'],
                        x['logical_B_per_gemm_bytes'],x['extra_output_per_gemm_bytes']), (c,a,b,o))
                    self.assertEqual(x['fused_logical_remote_A_per_rank_bytes'],a*(p-1))
                    self.assertEqual(x['fused_logical_local_A_per_rank_bytes'],a)
                    self.assertEqual(e,x['m_tiles']*x['n_tiles']*(x['q']-1))
                    self.assertIsNone(x['NVLink_bytes'])

    def test_fixed_peer_and_fixed_destination_rows_scaling(self):
        a=geometry((4,64,4096,512),'splitk4')
        b=geometry((8,64,4096,512),'splitk4')
        c=geometry((8,32,4096,512),'splitk4')
        self.assertEqual(b['unique_remote_A_per_rank_bytes']/a['unique_remote_A_per_rank_bytes'],7/3)
        self.assertEqual(c['unique_remote_A_per_rank_bytes']/a['unique_remote_A_per_rank_bytes'],7/6)
        self.assertEqual(b['program_gemms']/a['program_gemms'],4)
        self.assertEqual(b['rows_per_destination']/a['rows_per_destination'],2)
        self.assertEqual(c['rows_per_destination'],a['rows_per_destination'])

    def test_split_k_preserves_unique_bytes_and_enlarges_output_cost(self):
        a=geometry((8,64,8192,512),'direct');b=geometry((8,64,8192,512),'splitk16')
        self.assertEqual(a['unique_remote_A_per_rank_bytes'],b['unique_remote_A_per_rank_bytes'])
        self.assertEqual(a['fused_logical_remote_A_per_rank_bytes'],b['fused_logical_remote_A_per_rank_bytes'])
        self.assertEqual(a['extra_output_per_gemm_bytes'],0)
        self.assertEqual(b['extra_output_per_gemm_bytes'],4*64*512*15)
        self.assertFalse(a['serial_split_K_enabled']);self.assertTrue(b['serial_split_K_enabled'])
        self.assertIsNone(geometry((8,64,8192,512),'host')['fused_logical_remote_A_per_rank_bytes'])

    def test_candidate_permutation_and_training_order(self):
        shape=self.test[0].shape
        expected=self.model.rank(shape,FAMILIES,environment_id=P8_ENV)
        for order in (FAMILIES[::-1], FAMILIES[4:]+FAMILIES[:4]):
            self.assertEqual(expected,self.model.rank(shape,order,environment_id=P8_ENV))
        reordered=ScaleAwareModel.fit(self.cal[::-1],environment_id=P8_ENV)
        self.assertEqual(self.model.to_dict(),reordered.to_dict())

    def test_target_labels_never_enter_prediction(self):
        changed=[replace(r,measured_ms=r.measured_ms*17) for r in self.test]
        before=forecast(self.model,self.test,P8_ENV);after=forecast(self.model,changed,P8_ENV)
        self.assertEqual([r['predicted_us'] for r in before],[r['predicted_us'] for r in after])
        self.assertTrue(set(self.model.training_shapes).isdisjoint(r.shape for r in self.test))
        self.assertEqual(len(self.model.training_shapes),4)

    def test_cross_environment_requires_explicit_transfer_and_identity(self):
        with self.assertRaises(ValueError):self.model.predict((4,64,4096,512),'splitk2',environment_id=P4_ENV)
        with self.assertRaises(ValueError):self.model.predict((8,64,4096,512),'splitk2',environment_id=P4_ENV)
        with self.assertRaises(ValueError):ScaleAwareModel.fit(self.cal,environment_id=P4_ENV)
        with self.assertRaises(ValueError):self.model.predict((8,64,4096,512),'host_merged_cublaslt',environment_id=P8_ENV)
        result=self.model.predict((4,64,4096,512),'splitk2',environment_id=P4_ENV,allow_transfer=True)
        self.assertTrue(result['environment_transfer']);self.assertFalse(result['compiler_admission'])

    def test_bounds_invalid_shapes_and_duplicate_rows(self):
        for s,f in (((16,64,512,512),'splitk2'),((8,64,32,512),'splitk16'),((8,64,512,512),'splitk32'),((),'host')):
            with self.assertRaises(ValueError):geometry(s,f)
        with self.assertRaises(ValueError):ScaleAwareModel.fit(self.cal+self.cal[:1],environment_id=P8_ENV)
        with self.assertRaises(ValueError):ScaleAwareModel.fit(self.cal,environment_id=P8_ENV,retained_fraction=False)
        with self.assertRaises(ValueError):self.model.rank(self.test[0].shape,['host','host'],environment_id=P8_ENV)
        with self.assertRaises(ValueError):previous_geometry((8,64,4096,512),'splitk2')

    def test_p8_full_coverage_and_original_split_remain_intact(self):
        self.assertEqual((len(self.p4),len(self.cal),len(self.test)),(346,44,209))
        for shape in {r.shape for r in self.cal+self.test}:
            self.assertEqual({r.family for r in self.cal+self.test if r.shape==shape},set(FAMILIES))


if __name__=='__main__':unittest.main()
