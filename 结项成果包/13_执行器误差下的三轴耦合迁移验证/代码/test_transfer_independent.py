"""Independent complete representative maneuvers and overall acceptance checks.

These are numerical verification cases, not held-out research responses and not
three-axis policy training. State accuracy is checked against a separate ODE;
acceptance logic is isolated with deliberately scripted, nonphysical states.
"""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

import transfer_study as s


_INERTIA = np.array([1.2, 1., .8])


def _reference_derivative(time, state, command, parameter, kind):
    scalar, vector = state[0], state[1:4]
    omega, actual = state[4:7], state[7:10]
    torque = actual if kind == 'lag' and parameter > 0. else command
    qdot = .5*np.r_[-np.dot(vector, omega), scalar*omega+np.cross(vector, omega)]
    acceleration = (torque-np.cross(omega, _INERTIA*omega))/_INERTIA
    actuator = (command-actual)/parameter if kind == 'lag' and parameter > 0. else np.zeros(3)
    return np.r_[qdot, acceleration, actuator, np.dot(torque, torque)]


class RepresentativeManeuvers(unittest.TestCase):
    def test_full_reversing_maneuvers_against_independent_continuous_ode(self):
        initial_angles = [25., -20., 30.]
        initial_rate = [.10, -.08, .06]
        scalar_last = Rotation.from_euler('xyz', initial_angles, degrees=True).as_quat()
        quaternion = scalar_last[[3, 0, 1, 2]]
        count = 80
        commands = np.array([[.32*(-1)**(k//8), -.27*(-1)**(k//5), .21*(-1)**(k//11)]
                             for k in range(count)])
        cases = [('lag', 0.), ('lag', .05), ('lag', .3), ('lag', .5),
                 ('delay', 0.), ('delay', .05), ('delay', .3), ('delay', .5)]
        for kind, parameter in cases:
            with self.subTest(kind=kind, parameter=parameter):
                plant = s.Plant(kind, [dict(euler_deg=initial_angles,
                                           omega_deg_s=np.rad2deg(initial_rate).tolist(),
                                           parameter_s=parameter)])
                reference = np.r_[quaternion, initial_rate, np.zeros(3), 0.]
                independent_cost, recorded_cost = 0., 0.
                max_q_error, max_rate_error = 0., 0.
                delay = round(parameter/s.DT) if kind == 'delay' else 0
                for k, command in enumerate(commands):
                    applied = (commands[k-delay] if k >= delay else np.zeros(3)) if kind == 'delay' else command
                    reference[-1] = 0.
                    solution = solve_ivp(_reference_derivative, (0., s.DT), reference,
                                         args=(applied, parameter, kind), method='DOP853',
                                         rtol=2e-12, atol=2e-14)
                    self.assertTrue(solution.success)
                    reference = solution.y[:, -1]
                    reference[:4] /= np.linalg.norm(reference[:4])
                    response = plant.step(command[None, :])
                    self.assertFalse(plant.failed[0])
                    # Compare both representatives of the same orientation.
                    q_error = min(np.max(np.abs(plant.q[0]-reference[:4])),
                                  np.max(np.abs(plant.q[0]+reference[:4])))
                    max_q_error = max(max_q_error, q_error)
                    max_rate_error = max(max_rate_error, np.max(np.abs(plant.omega[0]-reference[4:7])))
                    expected_torque = reference[7:10] if kind == 'lag' and parameter > 0. else applied
                    np.testing.assert_allclose(response['tau_end'][0], expected_torque, atol=3e-12, rtol=0.)
                    recorded_cost += np.sum(response['actual_cost_axis'])
                    independent_cost += reference[-1]
                # Global integration budgets remain far below 1deg/.5deg/s.
                self.assertLess(max_q_error, 1e-5)
                self.assertLess(max_rate_error, 1e-5)
                self.assertAlmostEqual(recorded_cost, independent_cost, places=10)
                self.assertEqual(plant.k, count)


class _ZeroController:
    def reset(self, n, observation):
        self.n = n

    def act(self, observation, elapsed):
        return np.zeros((self.n, 3))


def _scripted_plant(angles, rates, failures=None, error_axis=None):
    """Build a metrics-only fixture; intentionally does not claim valid physics."""
    angles, rates = np.asarray(angles), np.asarray(rates)
    if failures is None:
        failures = np.zeros(len(angles), dtype=bool)
    if error_axis is None:
        error_axis = np.array([1., 0., 0.])
    error_axis = np.asarray(error_axis)/np.linalg.norm(error_axis)

    class ScriptedPlant:
        def __init__(self, kind, scenes):
            self.k = 0
            self.n = 1
            self._load()

        def _load(self):
            angle = angles[self.k]
            self.q = np.r_[np.cos(angle/2.), error_axis*np.sin(angle/2.)][None, :]
            self.omega = np.array([rates[self.k]])
            self.failed = np.array([failures[self.k]])

        def observation(self):
            return np.c_[-error_axis[None, :]*angles[self.k], self.omega]

        def step(self, command):
            self.k += 1
            self._load()
            return dict(command=np.zeros((1, 3)), tau_start=np.zeros((1, 3)),
                        tau_end=np.zeros((1, 3)), actual_cost_axis=np.array([[.001, .002, .003]]),
                        command_cost_axis=np.array([[.002, .003, .004]]))
    return ScriptedPlant


class OverallAcceptanceTests(unittest.TestCase):
    def _evaluate(self, angle_deg, rate_deg_s, failures=None, axis=None):
        if np.ndim(angle_deg) == 0:
            angle_deg = np.full(441, angle_deg)
        rates = np.broadcast_to(np.asarray(rate_deg_s), (441, 3))
        fixture = _scripted_plant(np.deg2rad(angle_deg), np.deg2rad(rates), failures, axis)
        with patch.object(s, 'Plant', fixture):
            return s.evaluate('lag', [dict(sample_id=0, parameter_s=.1)], _ZeroController(), True)

    def test_acceptance_uses_total_attitude_and_rate_norms(self):
        # All axis components can pass separately while the overall norm fails.
        result = self._evaluate(np.sqrt(3)*.8, [0., 0., 0.], axis=[1., 1., 1.])
        self.assertFalse(result['rows'][0]['success'])
        self.assertAlmostEqual(result['rows'][0]['final_error_deg'], np.sqrt(3)*.8)
        result = self._evaluate(0., [.4, .4, .4])
        self.assertFalse(result['rows'][0]['success'])
        self.assertAlmostEqual(result['rows'][0]['final_rate_norm_deg_s'], np.sqrt(3)*.4)
        result = self._evaluate(.99, [.49, 0., 0.])
        self.assertTrue(result['rows'][0]['success'])
        self.assertEqual(result['rows'][0]['settling_time_s'], 0.)

    def test_deadline_and_final_samples_are_both_included(self):
        angles = np.full(441, .2)
        angles[:160] = 2.
        result = self._evaluate(angles, [0., 0., 0.])
        self.assertTrue(result['rows'][0]['success'])
        self.assertEqual(result['rows'][0]['settling_time_s'], 8.)
        angles[160] = 2.
        self.assertFalse(self._evaluate(angles, [0., 0., 0.])['rows'][0]['success'])
        angles[:] = .2
        angles[-1] = 2.
        self.assertFalse(self._evaluate(angles, [0., 0., 0.])['rows'][0]['success'])

    def test_failure_overrides_small_errors_and_costs_sum_all_axes(self):
        failures = np.zeros(441, dtype=bool)
        failures[100:] = True
        result = self._evaluate(.1, [0., 0., 0.], failures)
        row = result['rows'][0]
        self.assertFalse(row['success'])
        self.assertTrue(row['failed'])
        self.assertIsNone(row['settling_time_s'])
        self.assertAlmostEqual(row['actual_cost'], 440*(.001+.002+.003))
        self.assertAlmostEqual(row['command_cost'], 440*(.002+.003+.004))
        self.assertEqual(result['traces']['time'][-1], 22.)
        self.assertEqual(result['traces']['q'].shape, (441, 1, 4))


if __name__ == '__main__':
    unittest.main()
