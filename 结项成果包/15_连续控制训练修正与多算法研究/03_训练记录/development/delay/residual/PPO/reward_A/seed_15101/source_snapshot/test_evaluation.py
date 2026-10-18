"""Acceptance and inference checks using synthetic scenes, never research tests."""
from __future__ import annotations
import unittest
import numpy as np
from .baselines import PIDController
from .evaluation import evaluate, paired_cluster_bootstrap, summary
from .protocol import DEADLINE, DURATION, DT, LIMIT


def scene(angle=0., rate=0., sample_id=0, parameter=0.):
    return dict(kind='lag', split='synthetic', region='synthetic', sample_id=sample_id,
                scenario_id=f'synthetic_{sample_id}_{parameter}', parameter_s=parameter,
                angle_deg=angle, omega_deg_s=rate)


class ConstantController:
    def __init__(self, command):
        self.command = command
    def reset(self, n, initial_observation=None):
        self.n = n
    def act(self, observation, elapsed=None):
        return np.full(self.n, self.command)


class EvaluationTests(unittest.TestCase):
    def test_zero_scene_is_successful_for_the_entire_horizon(self):
        data = evaluate('lag', [scene()], PIDController({}), keep_traces=True)
        row, traces = data['rows'][0], data['traces']
        self.assertTrue(row['success'])
        self.assertTrue(row['complete'])
        self.assertEqual(row['actual_cost'], 0.)
        self.assertEqual(row['penalized_selection_cost'], 0.)
        self.assertEqual(row['settling_time_s'], 0.)
        self.assertEqual(traces['theta'].shape, (round(DURATION/DT)+1, 1))
        self.assertEqual(traces['time'][-1], DURATION)
        self.assertTrue(summary(data['rows'])['cost_comparison_eligible'])

    def test_failure_is_charged_remaining_maximum_cost_without_changing_physical_cost(self):
        data = evaluate('lag', [scene(angle=169., rate=20.)], ConstantController(LIMIT), keep_traces=True)
        row, traces = data['rows'][0], data['traces']
        self.assertTrue(row['failed'])
        self.assertFalse(row['success'])
        self.assertTrue(row['complete'])
        self.assertAlmostEqual(row['actual_cost'], LIMIT**2*row['failure_time_s'])
        self.assertAlmostEqual(row['penalized_selection_cost'], LIMIT**2*DURATION)
        self.assertAlmostEqual(row['actual_cost'], np.sum(traces['actual_cost_interval']))
        failed_index = np.flatnonzero(traces['failed'][:, 0])[0]
        np.testing.assert_array_equal(traces['theta'][failed_index:, 0],
                                      np.full(len(traces['time'])-failed_index, traces['theta'][failed_index, 0]))
        self.assertFalse(summary(data['rows'])['cost_comparison_eligible'])

    def test_final_position_alone_cannot_pass_deadline(self):
        # Independent double integrator: omega=-angle/22 reaches zero only at22.
        data = evaluate('lag', [scene(angle=2., rate=-2./DURATION)], ConstantController(0.))
        row = data['rows'][0]
        self.assertLess(row['final_error_deg'], 1e-10)
        self.assertFalse(row['success'])
        self.assertIsNone(row['settling_time_s'])
        self.assertGreater(row['sustained_settling_time_s'], DEADLINE)

    def test_out_of_limit_controller_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate('lag', [scene()], ConstantController(LIMIT+1.))


class ClusterInferenceTests(unittest.TestCase):
    @staticmethod
    def rows():
        before, after = [], []
        for i, cost in enumerate((1., 2., 4., 8.)):
            for parameter in (.1, .2, .3):
                row = dict(scene(sample_id=i, parameter=parameter), success=True, actual_cost=cost)
                before.append(row)
                after.append(dict(row, actual_cost=cost*.75))
        return before, after

    def test_cluster_resampling_preserves_parameter_repeats(self):
        before, after = self.rows()
        result = paired_cluster_bootstrap(before, after, resamples=1000)
        self.assertEqual(result['initial_state_clusters'], 4)
        self.assertEqual(result['paired_scenes'], 12)
        self.assertEqual(result['parameter_repeats_per_cluster'], 3)
        np.testing.assert_allclose(result['relative_reduction_interval'], [.25, .25])
        self.assertGreater(result['difference_interval'][0], 0.)

    def test_failure_does_not_become_low_cost_superiority(self):
        before, after = self.rows()
        after[0] = dict(after[0], success=False, actual_cost=0.)
        result = paired_cluster_bootstrap(before, after, resamples=1000)
        self.assertFalse(result['cost_comparison_eligible'])
        self.assertIsNone(result['difference_interval'])

    def test_mismatched_scenes_or_unequal_parameter_coverage_are_rejected(self):
        before, after = self.rows()
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(before, after[:-1], resamples=1000)
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(before[:-1], after[:-1], resamples=1000)
        after[0] = dict(after[0], angle_deg=1.)
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(before, after, resamples=1000)


if __name__ == '__main__':
    unittest.main()
