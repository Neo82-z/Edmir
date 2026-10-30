import copy,json,unittest
from pathlib import Path
from assessment import assess
HERE=Path(__file__).resolve().parent

class AssessmentTest(unittest.TestCase):
    def test_good_average_cannot_hide_a_boundary_failure(self):
        result=json.loads((HERE/'results.json').read_text());review=json.loads((HERE/'review_results.json').read_text())
        self.assertTrue(result['gate']['passed'])
        outcome=assess(result,review)
        self.assertFalse(outcome['candidate_accepted'])
        self.assertFalse(outcome['checks']['N_one_percent_stress_all_correct'])
        self.assertFalse(outcome['checks']['M512_all_training_deletions_preserve_best'])
    def test_missing_stress_cases_are_not_a_pass(self):
        result=json.loads((HERE/'results.json').read_text());review=json.loads((HERE/'review_results.json').read_text())
        review['boundary_stress']['records']=[]
        with self.assertRaises(ValueError):assess(result,review)

if __name__=='__main__':unittest.main()
