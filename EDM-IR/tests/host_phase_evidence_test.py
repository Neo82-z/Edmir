from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from summarize_host_phases import summarize


def fixture():
    return {'kind':'host_phase_diagnostic','training_eligible':False,'correctness_passed':True,
            'precision':'fp16_fp32acc_fp16out','dependency_policy':'resource_queues_v1','metadata_mode':'prepared',
            'shape':[2,2,32,8],'family':'direct','device_service_us':None,
            'samples_ns':[{'submit':a,'completion_wait':b,'wall':a+b} for a,b in ((1,100),(100,1),(100,100))]}


class HostPhaseEvidenceTests(unittest.TestCase):
    def test_means_add_but_phase_medians_do_not(self):
        result=summarize(fixture()); p=result['phases']
        self.assertAlmostEqual(p['submit']['mean_us']+p['completion_wait']['mean_us'],p['wall']['mean_us'])
        self.assertNotEqual(p['submit']['median_us']+p['completion_wait']['median_us'],p['wall']['median_us'])
        self.assertEqual(p['wall']['median_us'],.101)
        self.assertIsNone(result['device_service_us'])
        self.assertFalse(result['training_eligible'])

    def test_bad_sample_identity_rejected(self):
        for key,value in (('wall',0),('submit',-1),('submit',True),('completion_wait',float('nan'))):
            r=fixture();r['samples_ns'][0][key]=value
            with self.assertRaises(ValueError):summarize(r)

    def test_standard_timing_rows_not_accepted(self):
        for patch in ({'kind':'timing_label'},{'training_eligible':True},{'correctness_passed':False},
                      {'precision':'bf16'},{'device_service_us':1.}):
            r=fixture();r.update(patch)
            with self.assertRaises(ValueError):summarize(r)

    def test_bound_and_identity(self):
        for patch in ({'samples_ns':[]},{'samples_ns':fixture()['samples_ns']*22},{'shape':[2,2,33,8]},
                      {'family':'host_merged_cublaslt'},{'metadata_mode':'merged'},{'dependency_policy':'other'}):
            r=fixture();r.update(patch)
            with self.assertRaises(ValueError):summarize(r)

    def test_samples_not_mutated_or_sorted_in_place(self):
        r=fixture();before=deepcopy(r);a=summarize(r)
        self.assertEqual(r,before)
        r['samples_ns'].reverse();self.assertEqual(a,summarize(r))


if __name__=='__main__':unittest.main()
