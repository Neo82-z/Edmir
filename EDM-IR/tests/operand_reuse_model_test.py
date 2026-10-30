from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.dependency_wave_model import KernelCase
from edmir.compiler import operand_reuse_model as model


def case(**kw):return replace(KernelCase('synthetic-test',4,64,512,256,2,1),**kw)
def scenario(**kw):return replace(model.ReuseScenario(40*1024**2,'m_fast',True,True,2),**kw)


class OperandReuseTests(unittest.TestCase):
    def test_cold_compulsory_and_warm_reuse_with_all_panels_resident(self):
        c=case();r=model.replay(c,scenario());a,b=[p['operands'] for p in r['passes']]
        self.assertEqual(a['B']['read_miss_bytes'],2*c.k*c.n)
        self.assertEqual(a['A_local']['read_miss_bytes'],2*c.m*c.k)
        self.assertEqual(a['A_remote']['read_miss_bytes'],(c.world_size-1)*2*c.m*c.k)
        self.assertTrue(all(b[x]['read_miss_bytes']==0 for x in b))
        self.assertEqual(a['C']['read_miss_bytes'],0)  # Written before accumulating read.
        self.assertEqual(r['passes'][-1]['dirty_bytes_after'],2*c.world_size*c.m*c.n)

    def test_capacity_zero_is_all_bypass_not_infinite_cache(self):
        r=model.replay(case(),scenario(capacity_bytes=0))
        for p in r['passes']:
            for x in p['operands'].values():self.assertEqual(x['read_miss_bytes'],x['read_bytes'])
            self.assertEqual(p['resident_bytes_after'],0)
            self.assertEqual(p['operands']['C']['bypass_write_bytes'],p['operands']['C']['write_bytes'])

    def test_shared_weights_have_one_compulsory_copy(self):
        c=case();shared=model.replay(c,scenario());private=model.replay(c,scenario(share_weights=False))
        self.assertEqual(private['passes'][0]['operands']['B']['read_miss_bytes'],
                         c.world_size*shared['passes'][0]['operands']['B']['read_miss_bytes'])
        self.assertEqual(private['footprint']['unique_weight_bytes'],c.world_size*shared['footprint']['unique_weight_bytes'])

    def test_peer_placement_is_explicit_and_changes_only_stated_path(self):
        r=model.replay(case(),scenario(remote_A_cached=False))
        for p in r['passes']:
            x=p['operands']['A_remote'];self.assertEqual(x['read_bytes'],x['read_miss_bytes'])
        self.assertEqual(r['passes'][1]['operands']['A_local']['read_miss_bytes'],0)
        self.assertEqual(r['passes'][1]['operands']['B']['read_miss_bytes'],0)

    def test_work_is_invariant_to_capacity_and_order_with_tail(self):
        c=case(m=65,n=136,q=4,group_n=2);counts=c.counts()
        for capacity in (0,8192,1048576):
            for order in ('m_fast','n_fast'):
                r=model.replay(c,scenario(capacity_bytes=capacity,tile_order=order))
                for p in r['passes']:
                    self.assertEqual(p['operands']['B']['read_bytes'],c.world_size*counts['logical_B_load_bytes_before_cache_reuse'])
                    self.assertEqual(p['operands']['C']['write_bytes'],c.world_size*counts['logical_output_write_bytes'])
                self.assertLessEqual(r['peak_resident_bytes'],capacity)

    def test_split_does_not_multiply_operand_work_but_does_multiply_writeback(self):
        values=[model.replay(case(q=q),scenario())['passes'][0]['operands'] for q in (1,2,4)]
        for x in ('A_local','A_remote','B'):
            self.assertEqual(len({v[x]['read_bytes'] for v in values}),1)
        self.assertEqual(values[2]['C']['write_bytes'],4*values[0]['C']['write_bytes'])

    def test_grouping_changes_A_replay_and_preserves_unique_operands(self):
        a=model.replay(case(n=512),scenario());b=model.replay(case(n=512,group_n=2),scenario())
        self.assertEqual(a['footprint'],b['footprint'])
        self.assertEqual(a['passes'][0]['operands']['A_remote']['read_bytes'],2*b['passes'][0]['operands']['A_remote']['read_bytes'])
        self.assertEqual(a['passes'][0]['operands']['B']['read_bytes'],b['passes'][0]['operands']['B']['read_bytes'])

    def test_LRU_variable_sizes_and_dirty_eviction(self):
        c=model.PanelCache(10)
        c.access('a',6,'A_local');c.access('b',4,'B');c.access('a',6,'A_local')
        c.access('c',4,'C',write=True)
        self.assertEqual(list(c.items),['a','c'])
        self.assertEqual(c.stats['A_local']['read_hit_bytes'],6)
        c.access('b',4,'B');c.access('a',6,'A_local')
        self.assertEqual(c.stats['C']['dirty_eviction_bytes'],4)
        self.assertEqual(c.used,10)

    def test_LRU_matches_independent_fixed_line_reference(self):
        sequence=['a','b','c','a','d','a','c','b','b','e']
        reference=[];misses=0;c=model.PanelCache(3)
        for key in sequence:
            if key in reference:reference.remove(key)
            else:misses+=1
            reference.append(key);reference=reference[-3:]
            c.access(key,1,'B')
        self.assertEqual(list(c.items),reference)
        self.assertEqual(c.stats['B']['read_miss_bytes'],misses)

    def test_bad_precision_capacity_and_reference_budget_rejected(self):
        for c in (case(precision='bf16'),case(k=513),case(m=4096,n=4096,k=8192,q=1)):
            with self.assertRaises(ValueError):model.replay(c,scenario())
        for s in (scenario(capacity_bytes=-1),scenario(tile_order='actual_cuda'),scenario(iterations=5),scenario(remote_A_cached=1)):
            with self.assertRaises(ValueError):model.replay(case(),s)
        c=model.PanelCache(20);c.access('a',4,'B')
        with self.assertRaises(ValueError):c.access('a',8,'B')

    def test_CPU_timeout_and_reference_count(self):
        c=case(n=2048,k=8192)
        with patch.object(model.time,'monotonic',side_effect=[0.,3.]):
            with self.assertRaises(TimeoutError):model.replay(c,scenario())
        r=model.replay(case(),scenario());self.assertEqual(r['references'],model.reference_count(case(),2))

    def test_no_microseconds_or_training_labels(self):
        r=model.replay(case(),scenario());self.assertIsNone(r['prediction_us']);self.assertIsNone(r['chosen'])
        self.assertFalse(r['training_eligible']);self.assertFalse(r['physical_traffic_verified'])
        self.assertIsNone(r['footprint']['unconditional_physical_cache_footprint_bytes'])


if __name__=='__main__':unittest.main()
