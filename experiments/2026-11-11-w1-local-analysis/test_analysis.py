"""Aggregation and eligibility must not turn invalid or dependent data into wins."""
import unittest
from analyze import summarize, allowed_families, paired


def row(repetition, samples):
    return dict(repetition=repetition,samples_ms=samples,job_index=repetition,
                order_position=0,implementation_id="same",gemm_algorithm_id="same")


class AnalysisTests(unittest.TestCase):
    def test_process_median_not_pooled_samples(self):
        # Three equal-size processes: median-of-medians=10, pooled median=1.
        rows=[row(0,[1.]*100),row(1,[1.]*49+[10.]*51),row(2,[1.]*49+[20.]*51)]
        self.assertEqual(summarize(rows)["median_ms"],10.)

    def test_reordered_process_input_has_identical_result(self):
        rows=[row(i,[float(i+1)]*100) for i in range(3)]
        self.assertEqual(summarize(rows),summarize(list(reversed(rows))))

    def test_missing_or_duplicate_process_is_not_complete(self):
        for rows in ([row(0,[1.]*100)]*3,[row(0,[1.]*100)]):
            with self.assertRaises(ValueError):summarize(rows)

    def test_failure_on_any_seed_excludes_all_timings(self):
        rows=[dict(family=f,input_seed=s,correctness_passed=(f=="host" or s!=2))
              for f in ("host","splitk8") for s in (1,2,3)]
        self.assertEqual(allowed_families(rows,[1,2,3]),{"host"})

    def test_missing_correctness_seed_rejected(self):
        with self.assertRaises(ValueError):
            allowed_families([dict(family="host",input_seed=1,correctness_passed=True)],[1,2,3])

    def test_paired_delta_preserves_reversal(self):
        a=dict(family="a",median_ms=2,repeat_medians_ms=[1,2,4])
        b=dict(family="b",median_ms=3,repeat_medians_ms=[3,3,3])
        result=paired(a,b)
        self.assertEqual(result["delta_a_minus_b_ms"],[-2,-1,1])
        self.assertLess(result["reduction_a_vs_b_pct"][2],0)


if __name__=="__main__":unittest.main()
