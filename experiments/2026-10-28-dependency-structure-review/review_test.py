from copy import deepcopy
import math
import unittest
import review


class FrozenControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.folds, cls.models = review.diagnosis.load_inputs()
        cls.model = cls.models[0]
        cls.train = [r for r in cls.rows if cls.folds[tuple(r['shape'])] != 0]
        cls.test = [{**r, 'split':'evaluation'} for r in cls.rows if cls.folds[tuple(r['shape'])] == 0]

    def test_control_updates_no_coefficients_and_does_not_mutate_source(self):
        before = deepcopy(self.model)
        with review.no_fitting() as calls:
            changed = review.capacity_control(self.model, self.train)
        self.assertEqual(calls, [])
        self.assertEqual(before, self.model)
        self.assertEqual(changed['stage_prior']['fit'], self.model['stage_prior']['fit'])
        self.assertEqual(changed['stage_prior']['curvature_transfer'], self.model['stage_prior']['curvature_transfer'])
        self.assertEqual(changed['host_interval'], self.model['host_interval'])

    def test_static_anchor_transformation_ignores_labels(self):
        altered = deepcopy(self.train)
        for row in altered: row['latency_us'] = 1e99
        with review.no_fitting():
            a = review.capacity_control(self.model, self.train)
            b = review.capacity_control(self.model, altered)
        self.assertEqual(a, b)

    def test_training_anchors_remain_exact_after_coordinate_change(self):
        with review.no_fitting():
            changed = review.capacity_control(self.model, self.train)
            for shape in {tuple(r['shape']) for r in self.train}:
                rr = [r for r in self.train if tuple(r['shape']) == shape]
                prediction = review.scorer.predict(changed, rr)
                for row in rr:
                    if row['is_host']: continue
                    self.assertAlmostEqual(math.exp(prediction['predicted_log_us'][row['candidate_id']]), row['latency_us'], places=8)

    def test_whole_shape_leakage_and_duplicate_anchor_rejected(self):
        with self.assertRaises(ValueError): review.capacity_control(self.model, self.train+self.test)
        fused = next(r for r in self.train if not r['is_host'])
        with self.assertRaises(ValueError): review.capacity_control(self.model, self.train+[fused])

    def test_query_labels_order_and_serialization_do_not_change_scores(self):
        import json
        with review.no_fitting():
            changed = review.capacity_control(self.model, self.train)
            shape = self.test[0]['shape']; rr = [r for r in self.test if r['shape']==shape]
            prediction = review.scorer.predict(changed, rr)
            altered = deepcopy(rr)
            for row in altered: row['latency_us'] = 1e99
            self.assertEqual(prediction, review.scorer.predict(changed, altered))
            self.assertEqual(prediction, review.scorer.predict(changed, list(reversed(rr))))
            self.assertEqual(prediction, review.scorer.predict(json.loads(json.dumps(changed)), rr))

    def test_short_k_raw_difference_unchanged_for_every_fold(self):
        # Both geometries are below both capacity choices. Any claimed repair
        # by only changing capacity would violate this exact feature identity.
        with review.no_fitting():
            for fold, original in self.models.items():
                train = [r for r in self.rows if self.folds[tuple(r['shape'])]!=fold]
                changed = review.capacity_control(original, train)
                for row in self.rows:
                    if row['shape'][2]==512 and row['candidate_id'] in ('direct','splitk2'):
                        self.assertEqual(review.scorer.transferred_value(original['stage_prior'],row),
                                         review.scorer.transferred_value(changed['stage_prior'],row))


if __name__ == '__main__': unittest.main()
