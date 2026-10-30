from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler import pipeline_dependency_model as model
from edmir.compiler.dependency_wave_model import KernelCase, ServiceScenario, simulate


def case(**kwargs):
    return replace(KernelCase('synthetic', 4, 64, 512, 256, 2, 1), **kwargs)


def service(**kwargs):
    return replace(model.PipelineService(2., 4., 13., 7., 1., .5, 1., 'synthetic_units', 'test only'), **kwargs)


def independent_queue_reference(c, s):
    """Explicit pending-group list, without graph nodes or prefix recurrence."""
    now = 0.; groups = []
    # g1 A copies split 1/2 + 1/2; g2 copies only in the first subgroup.
    main_chunks = ((.5, .5), (.5, .5)) if c.group_n == 1 else ((1., .5), (0., .5))
    def issue(index, chunks):
        nonlocal now
        completions = []; valid = index < c.depth
        for a, b in chunks:
            if a:
                now += a * s.issue_A * (1 if valid else s.masked_issue_fraction)
                completions.append(now + (s.latency_A if valid else 0))
            if b:
                now += b * s.issue_B * (1 if valid else s.masked_issue_fraction)
                completions.append(now + (s.latency_B if valid else 0))
        groups.append(max(completions))
    issue(0, ((1., 1.),)); issue(1, ((1., 1.),))
    now = max(now, max(groups[:-1])) + s.barrier
    for i in range(c.depth):
        now += s.mma_half; issue(i + 2, main_chunks)
        now = max(now, max(groups[:-1])) + s.barrier
        now += s.mma_half
    return max(now, max(groups)) + s.barrier


class PipelineDependencyTests(unittest.TestCase):
    def test_zero_latency_has_closed_form_no_extra_memory_reads(self):
        for d in (1, 2, 8, 16, 64):
            c = case(k=32*d, q=1)
            s = service(latency_A=0, latency_B=0)
            r = model.simulate_cta(c, s)
            expected = (d+2)*(s.issue_A+s.issue_B+s.barrier) + 2*d*s.mma_half
            self.assertEqual(r['scenario_cta_time'], expected)
            self.assertEqual(r['initial_exposed_wait'] + r['loop_exposed_wait'] + r['final_exposed_wait'], 0)

    def test_graph_agrees_with_independent_pending_group_queue(self):
        for d in (1, 2, 3, 8, 32):
            for g in (1, 2):
                for latency in (0, 5, 37):
                    c = case(k=32*d, q=1, group_n=g)
                    s = service(latency_A=latency, latency_B=latency/2, masked_issue_fraction=.25)
                    self.assertAlmostEqual(model.simulate_cta(c, s)['scenario_cta_time'], independent_queue_reference(c, s))

    def test_source_loop_counts_including_shorter_than_prologue(self):
        for d in (1, 2, 3, 8, 16):
            # Translate the source's counter control, separately from profile().
            remaining = d; issued_valid = 0
            for _ in range(2):
                issued_valid += remaining > 0; remaining -= 1
            loops = 0
            while remaining > -2:
                issued_valid += remaining > 0; remaining -= 1; loops += 1
            p = model.profile(case(k=32*d, q=1))
            self.assertEqual(p['valid_K_groups'], issued_valid)
            self.assertEqual(p['mainloop_iterations'], loops)
            self.assertEqual(p['masked_K_groups'], 2 + loops - issued_valid)

    def test_thread_map_counts_and_grouped_width_are_distinct(self):
        # Independent WarpRakedThreadMap integer derivation from source policy.
        def iterations(contiguous, strided, warp_contiguous, warp_strided, threads):
            wc, ws = contiguous//8//warp_contiguous, strided//warp_strided
            warps = threads//32; partition_s = min(ws, warps)
            partition_c = warps//partition_s if warps > ws else 1
            return wc//partition_c * (ws//partition_s)
        for g in (1, 2):
            p = model.profile(case(group_n=g))
            self.assertEqual(p['A_copy_loop_iterations_per_thread_per_stage'], iterations(32, 64, 4, 8, 128*g))
            self.assertEqual(p['B_copy_loop_iterations_per_thread_per_stage'], iterations(128*g, 32, 8, 4, 128*g))
        narrow, wide = model.profile(case()), model.profile(case(group_n=2))
        self.assertEqual(narrow['A_copy_loop_iterations_per_thread_per_stage'], 2)
        self.assertEqual(wide['A_copy_loop_iterations_per_thread_per_stage'], 1)
        self.assertEqual(narrow['B_copy_loop_iterations_per_thread_per_stage'], wide['B_copy_loop_iterations_per_thread_per_stage'])

    def test_wait_is_before_second_mma_and_drain_before_completion(self):
        r = model.simulate_cta(case(), service(), trace=True)
        nodes = r['nodes']; by_name = {n['name']: i for i,n in enumerate(nodes)}
        for i in range(case().depth):
            wait = by_name[f'wait_group_1_{i}']
            self.assertIn(wait, nodes[by_name[f'mma_second_{i}']]['parents'])
            self.assertIn(by_name[f'prefix_ready_{i+1}'], nodes[wait]['parents'])
        self.assertEqual(nodes[-1]['name'], 'final_wait_all')
        self.assertTrue(all(all(p < i for p in n['parents']) for i,n in enumerate(nodes)))

    def test_mainloop_preserves_AB_subgroup_issue_order(self):
        for g, expected in ((1, ['A_issue_2_0','B_issue_2_0','A_issue_2_1','B_issue_2_1']),
                            (2, ['A_issue_2_0','B_issue_2_0','B_issue_2_1'])):
            nodes = model.simulate_cta(case(group_n=g), service(), trace=True)['nodes']
            self.assertEqual([n['name'] for n in nodes if n['kind']=='issue' and '_2_' in n['name']], expected)

    def test_nonnegative_service_increase_cannot_reduce_completion(self):
        c = case(); s = service(); before = model.simulate_cta(c,s)['scenario_cta_time']
        for key in ('issue_A','issue_B','latency_A','latency_B','mma_half','barrier'):
            self.assertGreaterEqual(model.simulate_cta(c,replace(s,**{key:getattr(s,key)+11}))['scenario_cta_time'], before)

    def test_scaling_units_scales_completion(self):
        s = service(); scaled = replace(s, **{k:7*getattr(s,k) for k in ('issue_A','issue_B','latency_A','latency_B','mma_half','barrier')})
        self.assertAlmostEqual(model.simulate_cta(case(),scaled)['scenario_cta_time'], 7*model.simulate_cta(case(),s)['scenario_cta_time'])

    def test_q_changes_masked_and_barrier_work_not_total_valid_K_work(self):
        profiles = [(c,model.profile(c)) for c in (case(q=1),case(q=2),case(q=4))]
        self.assertEqual(len({c.ctas*p['valid_K_groups'] for c,p in profiles}), 1)
        self.assertEqual([p['gemm_masked_group_sites'] for c,p in profiles], [4,8,16])

    def test_all_fit_short_K_shape_invariance(self):
        gaps = []
        for n in (128,256,512,1024,2048):
            times = []
            for q in (1,2):
                c = case(n=n,q=q)
                a = model.simulate_cta(c,service())['scenario_cta_time']
                times.append(simulate(c,ServiceScenario(a,1,2,'synthetic_units','test'),slots=108)['scenario_makespan'])
            gaps.append(times[1]-times[0])
        self.assertEqual(len(set(gaps)),1)

    def test_rejects_bad_services_precision_and_depth_before_graph(self):
        for c in (case(k=32*(model.MAX_DEPTH+1),q=1),case(precision='bf16')):
            with self.assertRaises(ValueError): model.simulate_cta(c,service())
        for s in (service(issue_A=float('nan')),service(latency_B=-1),service(masked_issue_fraction=2),service(unit='')):
            with self.assertRaises(ValueError):model.simulate_cta(case(),s)
        g = model.DependencyGraph()
        with self.assertRaises(ValueError):g.add('bad','test',1,(0,))

    def test_CPU_deadline_and_maximum_depth_bound(self):
        with patch.object(model.time,'monotonic',side_effect=[0.,3.]):
            with self.assertRaises(TimeoutError):model.simulate_cta(case(),service())
        r = model.simulate_cta(case(k=32*model.MAX_DEPTH,q=1),service())
        self.assertLess(r['node_count'], model.MAX_NODES)
        self.assertLess(r['edge_count'], 2*r['node_count'])

    def test_no_hardware_prediction_or_candidate_choice(self):
        r = model.simulate_cta(case(),service())
        self.assertIsNone(r['prediction_us']);self.assertFalse(r['compiler_admission'])
        self.assertFalse(r['training_eligible']);self.assertFalse(r['hardware_scheduling_verified'])


if __name__ == '__main__':
    unittest.main()
