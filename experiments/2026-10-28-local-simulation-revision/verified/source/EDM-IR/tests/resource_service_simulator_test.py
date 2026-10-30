from dataclasses import replace
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.dependency_wave_model import KernelCase, ServiceScenario, simulate as constant_simulate
from edmir.compiler.resource_service_simulator import ResourceScenario, simulate


def case(**kw):
    return replace(KernelCase('synthetic-sm80',4,64,512,512,2,1),**kw)


def scenario(**kw):
    return replace(ResourceScenario(1.,1.,3.,2.,1e12,1e12,0.,0.,'synthetic_units',
                                    'test fixture; no hardware calibration'),**kw)


class ResourceServiceTests(unittest.TestCase):
    def test_compute_limited_limit_matches_independent_constant_model(self):
        for q in (1,2,4,8):
            c=case(q=q)
            for slots in (1,3,108):
                for order in ('partition_major','tile_major'):
                    s=scenario()
                    fluid=simulate(c,s,slots=slots,order=order,remote_source=True)
                    old=constant_simulate(c,ServiceScenario(1+c.depth,3,2,'u','constant'),slots=slots,order=order)
                    self.assertAlmostEqual(fluid['scenario_makespan'],old['scenario_makespan'],places=7)

    def test_single_slot_remote_services_overlap_by_max_not_sum(self):
        c=case(n=128,q=1)
        s=scenario(compute_fixed=5,compute_per_iteration=0,first_epilogue=0,
                   memory_bytes_per_unit=4096,link_bytes_per_unit=4096,
                   a_delivery_fraction=1,b_delivery_fraction=1)
        # Mainloop max(5, 128*512*2/4096, 64*512*2/4096)=32.
        # Output 64*128*2/4096=4; total 36.
        r=simulate(c,s,slots=1,remote_source=True)
        self.assertEqual(r['scenario_makespan'],36.)

    def test_memory_only_time_equals_total_bytes_over_bandwidth(self):
        s=scenario(compute_fixed=0,compute_per_iteration=0,first_epilogue=0,later_epilogue=0,
                   a_delivery_fraction=1,b_delivery_fraction=1,memory_bytes_per_unit=4096)
        for q in (1,2,4):
            for slots in (1,3,108):
                r=simulate(case(q=q),s,slots=slots,remote_source=False)
                self.assertAlmostEqual(r['scenario_makespan'],r['memory_service_bytes']/4096)

    def test_more_shared_traffic_changes_service_even_when_all_ctas_fit(self):
        s=scenario(compute_fixed=0,compute_per_iteration=0,first_epilogue=0,later_epilogue=0,
                   a_delivery_fraction=1,b_delivery_fraction=1,memory_bytes_per_unit=4096)
        a=simulate(case(n=128,q=1),s,slots=108,remote_source=False)
        b=simulate(case(n=256,q=1),s,slots=108,remote_source=False)
        self.assertAlmostEqual(b['scenario_makespan'],2*a['scenario_makespan'])

    def test_work_conservation_tail_and_split_writeback(self):
        c=case(m=65,n=136,q=4,group_n=2)
        s=scenario(a_delivery_fraction=.5,b_delivery_fraction=.25)
        r=simulate(c,s,slots=3,remote_source=True)
        counts=c.counts()
        output=counts['logical_output_write_bytes']+counts['logical_extra_output_read_bytes']
        self.assertEqual(r['memory_service_bytes'],output+.25*counts['logical_B_load_bytes_before_cache_reuse'])
        self.assertEqual(r['link_service_bytes'],.5*counts['logical_A_load_bytes_before_cache_reuse'])
        self.assertEqual(r['completed_ctas'],c.ctas)
        self.assertEqual(r['stage_transitions'],4*c.ctas)
        self.assertLessEqual(r['events'],6*c.ctas)

    def test_units_scale_time_and_preserve_work(self):
        c=case(q=4);a=scenario(a_delivery_fraction=1,b_delivery_fraction=1,memory_bytes_per_unit=8192,link_bytes_per_unit=4096)
        b=replace(a,compute_fixed=a.compute_fixed*10,compute_per_iteration=a.compute_per_iteration*10,
                  first_epilogue=a.first_epilogue*10,later_epilogue=a.later_epilogue*10,
                  memory_bytes_per_unit=a.memory_bytes_per_unit/10,link_bytes_per_unit=a.link_bytes_per_unit/10)
        x=simulate(c,a,slots=5,remote_source=True);y=simulate(c,b,slots=5,remote_source=True)
        self.assertAlmostEqual(y['scenario_makespan'],10*x['scenario_makespan'])
        self.assertEqual(x['memory_service_bytes'],y['memory_service_bytes'])

    def test_wider_template_changes_replay_not_only_cta_count(self):
        s=scenario(a_delivery_fraction=1,b_delivery_fraction=1)
        x=simulate(case(n=1024,q=4),s,slots=108,remote_source=True)
        y=simulate(case(n=1024,q=8,group_n=2),s,slots=108,remote_source=True)
        self.assertEqual(x['completed_ctas'],y['completed_ctas'])
        self.assertEqual(x['link_service_bytes'],2*y['link_service_bytes'])
        self.assertNotEqual(x['memory_service_bytes'],y['memory_service_bytes'])

    def test_one_slot_has_no_resident_dependency_wait(self):
        for order in ('partition_major','tile_major'):
            r=simulate(case(q=8),scenario(),slots=1,remote_source=True,order=order)
            self.assertEqual(r['resident_wait_slot_time'],0.)

    def test_reject_unbounded_bad_inputs_and_precision(self):
        for c in (case(precision='bf16'),case(m=8192,q=16),case(k=513)):
            with self.assertRaises(ValueError):simulate(c,scenario(),slots=108,remote_source=True)
        for s in (scenario(memory_bytes_per_unit=0),scenario(a_delivery_fraction=-.1),
                  scenario(compute_fixed=float('nan')),scenario(unit=''),scenario(b_delivery_fraction=True)):
            with self.assertRaises(ValueError):simulate(case(),s,slots=108,remote_source=True)
        with self.assertRaises(ValueError):simulate(case(),scenario(),slots=108,remote_source='yes')
        with self.assertRaises(ValueError):simulate(case(),scenario(),slots=108,remote_source=True,max_seconds=100)

    def test_synthetic_scenario_never_claims_measured_latency(self):
        r=simulate(case(),scenario(),slots=108,remote_source=True)
        self.assertIsNone(r['prediction_us'])
        self.assertFalse(r['training_eligible'])
        self.assertFalse(r['compiler_admission'])


if __name__=='__main__': unittest.main()
