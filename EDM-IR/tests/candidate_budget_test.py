import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.candidate_budget import rank_predictions, shortlist, score, training_only_order, uniform_random_reference


def rows():
    return [dict(candidate_id=c, observed_us=t, predicted_us=p, shape=[4,64,1024,512])
            for c,t,p in [('host',10.,12.),('q2',12.,10.),('q4',25.,20.),('q8',40.,30.)]]


class CandidateBudgetTest(unittest.TestCase):
    def test_ranking_does_not_read_test_labels(self):
        data=rows();order=rank_predictions(data)
        self.assertEqual(order,rank_predictions([{**r,'observed_us':1000-r['observed_us']} for r in data]))
        self.assertEqual(order,rank_predictions([{k:v for k,v in r.items() if k!='observed_us'} for r in data]))

    def test_all_permutations_and_exact_ties(self):
        data=[{**r,'predicted_us':10.} for r in rows()]
        expected=rank_predictions(data)
        for order in itertools.permutations(data):self.assertEqual(rank_predictions(order),expected)

    def test_random_reference_matches_independent_exhaustive_subsets(self):
        for data in (rows(),[{**r,'observed_us':10.} for r in rows()]):
            for k in range(1,5):
                scores=[score(data,list(s)) for s in itertools.combinations([r['candidate_id'] for r in data],k)]
                ref=uniform_random_reference(data,k)
                self.assertAlmostEqual(ref['expected_choice_loss_fraction'],sum(s['choice_loss_fraction'] for s in scores)/len(scores))
                self.assertAlmostEqual(ref['near_optimal_retention_probability'],sum(s['near_optimal_retained'] for s in scores)/len(scores))

    def test_protected_host_consumes_budget(self):
        order=rank_predictions(rows())
        self.assertEqual(shortlist(order,1,protected='host'),['host'])
        self.assertEqual(shortlist(order,2,protected='host'),['host','q2'])
        with self.assertRaises(ValueError):shortlist(order,2,protected='unmeasured')

    def test_no_test_leakage_or_sparse_candidate_advantage(self):
        train=rows()+[{**r,'shape':[4,256,8192,512]} for r in rows() if r['candidate_id']!='host']
        result=training_only_order(train,[4,512,4096,512])
        self.assertNotIn('host',result['order'])
        with self.assertRaises(ValueError):training_only_order(train,[4,64,1024,512])

    def test_missing_duplicate_nonfinite_and_flagged_labels_rejected(self):
        for data in [rows()+[rows()[0]],[{**r,'observed_us':math.nan} for r in rows()],
                     [{**r,'quality_flags':['drift']} for r in rows()]]:
            with self.assertRaises(ValueError):score(data,['host'])
        with self.assertRaises(ValueError):score(rows(),['unmeasured'])

    def test_full_budget_and_monotonic_loss(self):
        order=rank_predictions(rows())
        losses=[score(rows(),shortlist(order,k))['choice_loss_fraction'] for k in (1,2,3,4)]
        self.assertEqual(losses,sorted(losses,reverse=True))
        self.assertEqual(losses[-1],0.)
        self.assertEqual(score(rows(),shortlist(order,9))['candidate_count_reduction'],0.)


if __name__=='__main__':unittest.main()
