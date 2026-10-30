"""Independent optimization, metric enumeration and evaluation boundaries."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"EDM-IR"),str(ROOT/"EDM-IR/tools")]
from edmir.compiler.mechanism_prediction import HOSTS, RIDGE, nonnegative_fit
from edmir.compiler.regime_prediction import RegimeModel, max_branch_fit
from edmir.compiler.retained_regime_prediction import RetainedRegimeModel, RETAINED_FRACTIONS
from edmir.compiler.metric_evidence import enumerate_tiles, metric_evidence
from edmir.compiler.prediction_support import linear_target_support
from post_gpu_evidence import load_evidence, coverage
from evaluate_post_gpu_revision import evaluate_fold, choices
from evaluate_host_coverage_revision import unique_extension, host_fold


class NumericalTests(unittest.TestCase):
    def test_known_coefficients_across_all_declared_fractions(self):
        x=[[1.,l,w] for l in (1.,2.,3.) for w in (.5,1.,2.,4.,8.)]
        for rho in RETAINED_FRACTIONS:
            y=[7+max(5*l,2*w)+rho*min(5*l,2*w) for _,l,w in x]
            f=max_branch_fit(x,y,[1.]*len(y),retained_fraction=rho)
            for actual,want in zip(f['coefficients_us'],(7,5,2)):
                self.assertAlmostEqual(actual,want,places=4)
            self.assertLessEqual(f['kkt_relative_violation'],1e-9)

    def test_sum_endpoint_matches_independent_nnls(self):
        x=[[1.,l,w] for l in (1.,2.,5.) for w in (1.,3.,8.)]
        y=[8+3*l+2*w+(i%3)*.1 for i,(_,l,w) in enumerate(x)]
        a=max_branch_fit(x,y,[1.]*len(y),retained_fraction=1.)
        b=nonnegative_fit(x,y,[1.]*len(y))
        for u,v in zip(a['coefficients_us'],b['coefficients_us']):self.assertAlmostEqual(u,v,places=6)

    def test_dense_profile_bounds_the_convex_partition_solution(self):
        x=[[1.,l,w] for l in (1.,2.,4.) for w in (.5,1.,3.,9.)]
        rho=.5;y=[4+max(3.1*l,1.7*w)+rho*min(3.1*l,1.7*w)+(i%3-.8)*.5 for i,(_,l,w) in enumerate(x)]
        weights=[1.+i%2 for i in range(len(x))]
        f=max_branch_fit(x,y,weights,retained_fraction=rho)
        sl2=sum(r[1]**2 for r in x)/len(x);sw2=sum(r[2]**2 for r in x)/len(x)
        best=math.inf
        for j in range(1001):
            t=j/100;v=[max(t*l,w)+rho*min(t*l,w) for _,l,w in x]
            aa=sum(weights)+RIDGE;ab=sum(w*z for w,z in zip(weights,v))
            bb=sum(w*z*z for w,z in zip(weights,v))+RIDGE*(t*t*sl2+sw2)
            ra=sum(w*z for w,z in zip(weights,y));rb=sum(w*z*s for w,z,s in zip(weights,v,y))
            det=aa*bb-ab*ab
            for c,b in ((max(ra/aa,0),0),(0,max(rb/bb,0)),((ra*bb-rb*ab)/det,(rb*aa-ra*ab)/det)):
                if min(c,b)<0:continue
                objective=sum(w*(c+b*z-vy)**2 for w,z,vy in zip(weights,v,y))+RIDGE*(c*c+b*b*(t*t*sl2+sw2))
                best=min(best,objective)
        self.assertLessEqual(f['objective_us2'],best+1e-7)
        self.assertLess(best-f['objective_us2'],.02)

    def test_units_and_order(self):
        x=[[1.,l,w] for l in (1.,2.,3.) for w in (1.,2.,4.,8.)]
        y=[11+max(5*l,2*w)+.75*min(5*l,2*w) for _,l,w in x]
        a=max_branch_fit(x,y,[1.]*len(x),retained_fraction=.75)
        b=max_branch_fit([[r[0],r[1]*1024,r[2]/1024] for r in x[::-1]],y[::-1],[1.]*len(x),retained_fraction=.75)
        for u,v in zip(a['coefficients_us'],(b['coefficients_us'][0],b['coefficients_us'][1]*1024,b['coefficients_us'][2]/1024)):
            self.assertAlmostEqual(u,v,places=6)

    def test_invalid_fraction_and_time_budget(self):
        for rho in (-1,1.1,math.nan,math.inf,True):
            with self.assertRaises(ValueError):max_branch_fit([[1.,2.]],[3.],[1.],retained_fraction=rho)
        with patch('edmir.compiler.regime_prediction.time.monotonic',side_effect=(0.,2.)):
            with self.assertRaises(TimeoutError):max_branch_fit([[1.,2.]],[3.],[1.],timeout_seconds=1.,retained_fraction=.5)

    def test_rank_deficiency_does_not_always_invalidate_target_support(self):
        train=[[1.,k,2*k] for k in (1.,2.,3.)]
        inside=linear_target_support(train,[1.,4.,8.])
        outside=linear_target_support(train,[1.,4.,7.])
        self.assertEqual(inside['training_rank'],2)
        self.assertTrue(inside['target_in_training_row_span'])
        self.assertFalse(outside['target_in_training_row_span'])
        self.assertEqual(outside['rank_with_target'],3)
        self.assertIsNone(outside['confidence_probability'])


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data,cls.provenance=load_evidence()

    def test_partial_tiles_count_valid_elements_not_allocated_sectors(self):
        self.assertEqual(enumerate_tiles((4,65,256,264),'splitk2_groupedn2'),{
            'ctas':8,'logical_A_replay_bytes':66560,'logical_B_replay_bytes':270336,
            'extra_serial_output_bytes':68640,'partition_iterations':4,'ordered_tile_boundaries':4})
        result=metric_evidence((4,65,256,264),'splitk2_groupedn2')
        self.assertTrue(result['enumeration_matches'])
        self.assertTrue(all(v is None for v in result['unidentified_hardware_metrics'].values()))

    def test_enumeration_is_bounded(self):
        with self.assertRaises(ValueError):enumerate_tiles((4,2**20,256,2**20),'splitk8')
        with self.assertRaises(ValueError):enumerate_tiles((4,64,96,512),'splitk2')

    def test_data_remains_separate_and_unmeasured_stays_missing(self):
        self.assertEqual({k:len(v) for k,v in self.data.items()},{'old':69,'primary':57,'repeat':57,'extension':12})
        self.assertFalse(self.provenance['repeat_or_extension_used_for_fit'])
        self.assertNotEqual(self.provenance['contexts']['old'],self.provenance['contexts']['primary'])
        self.assertTrue(any(v['status']=='unmeasured' for row in coverage(self.data) for v in row['candidates'].values()))

    def test_endpoints_hosts_and_contribution_sum(self):
        rows=self.data['primary'];base=RegimeModel.fit(rows)
        zero=RetainedRegimeModel.fit(rows,retained_fraction=0.)
        mid=RetainedRegimeModel.fit(rows,retained_fraction=.5)
        for row in rows:
            args=(row['shape'],row['family']);kw={'context_id':base.context_id}
            a=base.predict(*args,**kw);z=zero.predict(*args,**kw);b=mid.predict(*args,**kw)
            self.assertAlmostEqual(a['predicted_us'],z['predicted_us'],places=8)
            self.assertAlmostEqual(b['predicted_us'],sum(b['contributions_us'].values()),places=8)
            if row['family'] in HOSTS:self.assertEqual(a,b)
            else:self.assertFalse(b['regime']['physical_overlap_fraction_identified'])

    def test_target_labels_do_not_change_heldout_fit_or_repeat_predictions(self):
        target=(4,256,8192,512)
        before=evaluate_fold(self.data['primary'],'retained_0.5','shape',target,self.data['repeat'])
        changed=copy.deepcopy(self.data['primary'])
        for r in changed:
            if tuple(r['shape'])==target:r['observed_us']*=7
        after=evaluate_fold(changed,'retained_0.5','shape',target,self.data['repeat'])
        self.assertEqual(before['model'],after['model'])
        self.assertEqual(before['repeat_predictions'],after['repeat_predictions'])

    def test_axis_holdout_excludes_entire_axis(self):
        for axis,value,index in (('M',256,1),('N',2048,3)):
            f=evaluate_fold(self.data['primary'],'retained_0.5',axis,value)
            self.assertTrue(all(s[index]!=value for s in f['model']['training_shapes']))
            self.assertTrue(all(r['family'] not in HOSTS for r in f['predictions']))

    def test_two_hosts_cost_two_candidates_and_use_same_support(self):
        rows=[{'shape':[4,64,768,512],'candidate_id':cid,'predicted_us':pred,'observed_us':obs,
               'quality_flags':[]} for cid,pred,obs in (
                   ('host_merged_cutlass@merged',300,234),('host_merged_cublaslt@merged',260,251),('splitk2@prepared',200,400))]
        r=choices(rows)[0]
        self.assertTrue(r['matched_two_hosts'])
        self.assertEqual(r['policies']['two_merged_hosts']['evaluated_candidates'],2)
        self.assertEqual(r['policies']['two_merged_hosts']['choice_loss_fraction'],0)
        self.assertGreater(r['policies']['model_top2']['choice_loss_fraction'],0)

    def test_extension_only_adds_missing_labels(self):
        added=unique_extension(self.data['primary'],self.data['extension'])
        self.assertEqual(len(added),8)
        existing={(tuple(r['shape']),r['candidate_id']) for r in self.data['primary']}
        self.assertTrue(all((tuple(r['shape']),r['candidate_id']) not in existing for r in added))
        self.assertTrue(all(r['family'].startswith('host_merged_') for r in added))

    def test_host_coverage_excludes_whole_target_shape_in_both_sources(self):
        rows=self.data['primary']+unique_extension(self.data['primary'],self.data['extension'])
        target=[4,64,4096,2048]
        a,train=host_fold(rows,target)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['shape']==target:r['observed_us']*=9
        b,_=host_fold(changed,target)
        self.assertEqual(a.to_dict(),b.to_dict())
        self.assertTrue(all(r['shape']!=target for r in train))

    def test_existing_extension_fills_M512_linear_support(self):
        from edmir.compiler.mechanism_prediction import basis
        rows=self.data['primary']+unique_extension(self.data['primary'],self.data['extension'])
        target=[4,512,4096,512];family='host_merged_cutlass'
        _,train=host_fold(rows,target)
        vectors=[basis(r['shape'],family,'cutlass_depth')[1] for r in train if r['family']==family]
        support=linear_target_support(vectors,basis(target,family,'cutlass_depth')[1])
        self.assertEqual(support['training_rank'],4)
        self.assertTrue(support['target_in_training_row_span'])


if __name__=='__main__':unittest.main()
