"""Acceptance end points and paired statistics, independent of optimizer code."""
import unittest
import numpy as np
from actuator_study.evaluation import evaluate
from actuator_study.analysis import paired_interval


class Constant:
    def __init__(self, last=0.):
        self.last = last

    def reset(self, n, initial_observation):
        self.n = n

    def act(self, observation, elapsed):
        return np.where(elapsed >= 21.95-1e-9, self.last, 0.)


class EvaluationTests(unittest.TestCase):
    def scene(self, angle):
        return [dict(parameter_s=0., angle_deg=angle, omega_deg_s=0., sample_id=0)]

    def test_final_sample_is_not_excluded(self):
        result = evaluate('lag', self.scene(.99999), Constant(.5), True)
        self.assertFalse(result['rows'][0]['success'])
        self.assertFalse(result['rows'][0]['extension_pass'])
        self.assertAlmostEqual(result['rows'][0]['actual_cost'], .5**2*.05)
        self.assertEqual(result['traces']['theta'].shape, (441, 1))
        self.assertEqual(result['traces']['command'].shape, (440, 1))

    def test_quiescent_inside_and_outside(self):
        inside = evaluate('delay', self.scene(.5), Constant())['rows'][0]
        outside = evaluate('delay', self.scene(2.), Constant())['rows'][0]
        self.assertTrue(inside['success'])
        self.assertEqual(inside['settling_time_s'], 0.)
        self.assertAlmostEqual(inside['rms_error_deg'], .5)
        self.assertFalse(outside['success'])
        self.assertIsNone(outside['settling_time_s'])

    def test_bootstrap_resamples_initials_not_parameter_cells(self):
        # Perfect within-initial correlation must not acquire artificial precision.
        a = paired_interval(np.array([[-1.], [1.]]), 123, 2000, .95)
        b = paired_interval(np.repeat(np.array([[-1.], [1.]]), 6, axis=1), 123, 2000, .95)
        self.assertEqual(a['low'], b['low'])
        self.assertEqual(a['high'], b['high'])
        self.assertEqual(b['independent_initial_states'], 2)


if __name__ == '__main__':
    unittest.main()
