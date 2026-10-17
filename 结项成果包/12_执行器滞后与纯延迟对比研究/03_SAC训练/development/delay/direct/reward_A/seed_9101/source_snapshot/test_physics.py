"""Independent physics checks; run with ``python -m unittest test_physics -v``."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad, solve_ivp

from actuator_study.physics import BatchPlant, ideal_interval, lag_interval
from actuator_study.protocol import DT, DURATION, FAIL_ANGLE, FAIL_RATE, LIMIT


class LagPhysicsTests(unittest.TestCase):
    def test_analytic_step_and_reversal(self):
        plant = BatchPlant('lag', 1)
        plant.reset(.2, -.1, .17)
        expected_theta, expected_omega, expected_tau = .2, -.1, 0.
        for command in [.4]*5+[-.3]*7+[0.]*3:
            old_tau = expected_tau
            exponent = np.exp(-DT/.17)
            expected_tau = command+(old_tau-command)*exponent
            expected_theta += (expected_omega*DT+command*DT**2/2
                               +(old_tau-command)*(.17*DT-.17**2*(1-exponent)))
            expected_omega += command*DT+(old_tau-command)*.17*(1-exponent)
            result = plant.step(np.array([command]))
            assert_allclose(plant.theta, expected_theta, rtol=0., atol=2e-15)
            assert_allclose(plant.omega, expected_omega, rtol=0., atol=2e-15)
            assert_allclose(result['tau_start'], old_tau, rtol=0., atol=2e-15)
            assert_allclose(result['tau_end'], expected_tau, rtol=0., atol=2e-15)
        self.assertLess(float(plant.actual_torque[0]), 0.)

    def test_joint_state_and_cost_against_independent_ode_solver(self):
        rng = np.random.default_rng(20261017)
        parameters = np.array([.003, .05, .11, .3, .5, 4.])
        initial_theta = rng.uniform(-.8, .8, len(parameters))
        initial_omega = rng.uniform(-.2, .2, len(parameters))
        plant = BatchPlant('lag', len(parameters), inertia=1.7)
        plant.reset(initial_theta, initial_omega, parameters)
        reference = np.column_stack((initial_theta, initial_omega,
                                     np.zeros((len(parameters), 2))))
        for _ in range(12):
            requested = rng.uniform(-1., 1., len(parameters))
            clipped = np.clip(requested, -LIMIT, LIMIT)
            result = plant.step(requested)
            costs = np.zeros(len(parameters))
            for i, (command, parameter) in enumerate(zip(clipped, parameters)):
                def rhs(time, state):
                    return [state[1], state[2]/1.7,
                            (command-state[2])/parameter, state[2]**2]
                reference[i, 3] = 0.
                solution = solve_ivp(rhs, (0., DT), reference[i],
                                     method='DOP853', rtol=2e-12, atol=2e-14)
                self.assertTrue(solution.success)
                reference[i] = solution.y[:, -1]
                costs[i] = reference[i, 3]
            assert_allclose(plant.theta, reference[:, 0], rtol=2e-12, atol=5e-13)
            assert_allclose(plant.omega, reference[:, 1], rtol=2e-12, atol=5e-13)
            assert_allclose(plant.actual_torque, reference[:, 2], rtol=3e-11, atol=5e-13)
            assert_allclose(result['actual_cost'], costs, rtol=5e-11, atol=5e-14)

    def test_torque_square_integral_against_quadrature_across_time_scales(self):
        parameters = np.array([1e-4, 1e-3, .03, .05, .17, .5, 20., 1e5, 1e9])
        for old_torque, command in [(0., .5), (.5, -.5), (-.4, .2), (.2, .2)]:
            _, _, _, costs = lag_interval(0., 0., old_torque, command,
                                          parameters, DT)
            expected = [quad(lambda time: (old_torque+(command-old_torque)
                                           *(-np.expm1(-time/parameter)))**2,
                             0., DT, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
                        for parameter in parameters]
            assert_allclose(costs, expected, rtol=3e-10, atol=2e-15)
            self.assertTrue(np.all(costs >= 0.))

    def test_zero_time_constant_matches_ideal_double_integrator(self):
        rng = np.random.default_rng(4921)
        lag = BatchPlant('lag', 5, inertia=2.3)
        ideal = BatchPlant('delay', 5, inertia=2.3)
        theta, omega = rng.uniform(-.2, .2, (2, 5))
        lag.reset(theta, omega, 0.)
        ideal.reset(theta, omega, 0.)
        expected_theta, expected_omega = theta.copy(), omega.copy()
        for _ in range(60):
            command = rng.uniform(-.9, .9, 5)
            clipped = np.clip(command, -LIMIT, LIMIT)
            expected_theta += expected_omega*DT+.5*clipped*DT**2/2.3
            expected_omega += clipped*DT/2.3
            lag_result, ideal_result = lag.step(command), ideal.step(command)
            assert_allclose(lag.theta, expected_theta, rtol=0., atol=2e-15)
            assert_allclose(lag.omega, expected_omega, rtol=0., atol=2e-15)
            for key in lag_result:
                assert_array_equal(lag_result[key], ideal_result[key])
            assert_array_equal(lag.theta, ideal.theta)
            assert_array_equal(lag.omega, ideal.omega)

    def test_helper_scalar_zero_interval_and_extreme_positive_constants(self):
        theta, omega, torque, cost = lag_interval(.3, -.1, .2, -.4, .1, 0.)
        assert_allclose([theta, omega, torque, cost], [.3, -.1, .2, 0.])
        theta, omega, torque, cost = lag_interval(.3, -.1, 0., -.4, 0., DT)
        ideal_theta, ideal_omega, ideal_cost = ideal_interval(.3, -.1, -.4, DT)
        assert_allclose([theta, omega, torque, cost],
                        [ideal_theta, ideal_omega, -.4, ideal_cost])
        theta, omega, torque, cost = lag_interval(0., 0., .5, 0.,
                                                  np.array([1e-320, 1e20]), DT)
        self.assertTrue(np.all(np.isfinite([theta, omega, torque, cost])))
        assert_allclose([theta[0], omega[0], torque[0], cost[0]], 0., atol=1e-300)
        assert_allclose([theta[1], omega[1], torque[1], cost[1]],
                        [.5*.5*DT**2, .5*DT, .5, .25*DT], rtol=1e-14)


class DelayPhysicsTests(unittest.TestCase):
    def test_zero_one_and_two_step_delay_without_off_by_one(self):
        plant = BatchPlant('delay', 3)
        plant.reset(0., 0., np.array([0., DT, 2*DT]))
        commands = np.array([[.2, .3, -.4], [-.4, .2, .1], [.1, -.3, .5],
                             [.5, .4, -.2], [-.2, -.5, .3]])
        theta, omega = np.zeros(3), np.zeros(3)
        for k, command in enumerate(commands):
            applied = np.array([command[0], commands[k-1, 1] if k >= 1 else 0.,
                                commands[k-2, 2] if k >= 2 else 0.])
            theta += omega*DT+.5*applied*DT**2
            omega += applied*DT
            result = plant.step(command)
            assert_array_equal(result['tau_start'], applied)
            assert_array_equal(result['tau_end'], applied)
            assert_allclose(result['actual_cost'], applied**2*DT, atol=0.)
            assert_allclose(plant.theta, theta, rtol=0., atol=2e-17)
            assert_allclose(plant.omega, omega, rtol=0., atol=2e-17)

    def test_long_delay_and_growing_history_preserve_other_episodes(self):
        plant = BatchPlant('delay', 2, duration=2.)
        plant.reset(0., 0., np.array([2*DT, DT]))
        plant.step(np.array([.1, .2]))
        plant.step(np.array([.3, -.4]))
        plant.reset(.1, -.05, 13*DT, mask=np.array([True, False]))
        for k in range(14):
            result = plant.step(np.array([.5, .1]))
            self.assertEqual(result['tau_end'][0], 0. if k < 13 else .5)
            if k == 0:
                self.assertEqual(result['tau_end'][1], -.4)
        self.assertEqual(plant.step_count[0], 14)
        self.assertEqual(plant.step_count[1], 16)

    def test_masked_delay_reset_clears_prehistory(self):
        plant = BatchPlant('delay', 2)
        plant.reset(0., 0., DT)
        plant.step(np.array([.3, -.2]))
        plant.reset(0., 0., DT, mask=np.array([True, False]))
        result = plant.step(np.array([.4, .5]))
        assert_array_equal(result['tau_end'], [0., -.2])
        result = plant.step(np.array([.1, .2]))
        assert_array_equal(result['tau_end'], [.4, .5])


class BatchLifecycleTests(unittest.TestCase):
    def test_masked_reset_and_constant_episode_parameters(self):
        plant = BatchPlant('lag', 3)
        plant.reset([.1, .2, .3], [.01, .02, .03], [0., .1, .2])
        for _ in range(3):
            plant.step(np.array([.1, -.2, .3]))
        middle = (plant.theta[1], plant.omega[1], plant.actual_torque[1],
                  plant.step_count[1], plant.elapsed[1])
        observation = plant.reset([-.1, -.3], [.2, -.2], [.4, .5],
                                  mask=np.array([True, False, True]))
        assert_array_equal([plant.theta[1], plant.omega[1], plant.actual_torque[1],
                            plant.step_count[1], plant.elapsed[1]], middle)
        assert_array_equal(plant.actual_torque[[0, 2]], 0.)
        assert_array_equal(plant.step_count, [0, 3, 0])
        assert_array_equal(plant.elapsed[[0, 2]], 0.)
        assert_array_equal(plant.parameter_s, [.4, .1, .5])
        assert_array_equal(observation, np.column_stack((-plant.theta, plant.omega)))
        with self.assertRaises(ValueError):
            plant.parameter_s[1] = .8
        for _ in range(50):
            plant.step(np.array([.03, -.01, .02]))
        assert_array_equal(plant.parameter_s, [.4, .1, .5])

    def test_physical_clipping_precedes_actuator_and_costs(self):
        for kind in ['lag', 'delay']:
            plant = BatchPlant(kind, 2)
            plant.reset(0., 0., 0.)
            result = plant.step(np.array([50., -90.]))
            assert_array_equal(result['command'], [LIMIT, -LIMIT])
            assert_array_equal(result['tau_start'], [LIMIT, -LIMIT])
            assert_array_equal(result['tau_end'], [LIMIT, -LIMIT])
            assert_allclose(result['command_cost'], LIMIT**2*DT)
            assert_allclose(result['actual_cost'], LIMIT**2*DT)
            assert_allclose(plant.theta, [.5*LIMIT*DT**2, -.5*LIMIT*DT**2])

    def test_failure_cost_is_charged_then_physics_freezes_while_clock_advances(self):
        plant = BatchPlant('lag', 2, duration=.2)
        plant.reset([np.deg2rad(169.999), 0.], [.1, np.deg2rad(299.99)], 0.)
        result = plant.step(np.full(2, .5))
        assert_array_equal(result['new_failure'], True)
        assert_array_equal(result['done'], True)
        assert_allclose(result['actual_cost'], .25*DT)
        assert_allclose(result['command_cost'], .25*DT)
        assert_array_equal(plant.actual_torque, .5)
        failed_theta, failed_omega = plant.theta.copy(), plant.omega.copy()
        for k in range(2, 5):
            result = plant.step(np.full(2, -.5))
            assert_array_equal(plant.theta, failed_theta)
            assert_array_equal(plant.omega, failed_omega)
            for key in ['command', 'actual_cost', 'command_cost', 'tau_start', 'tau_end']:
                assert_array_equal(result[key], 0.)
            assert_array_equal(result['new_failure'], False)
            assert_array_equal(result['done'], True)
            assert_allclose(plant.elapsed, k*DT, rtol=0., atol=1e-16)
        assert_array_equal(plant.step_count, 4)
        self.assertLess(DT, plant.elapsed[0])
        plant.reset(0., 0., .1, mask=np.array([True, False]))
        assert_array_equal(plant.failed, [False, True])
        assert_array_equal(plant.elapsed, [0., .2])

    def test_failure_thresholds_are_strict(self):
        plant = BatchPlant('lag', 2, duration=DT)
        plant.reset([np.deg2rad(FAIL_ANGLE), -np.deg2rad(FAIL_ANGLE)], 0., 0.)
        assert_array_equal(plant.step(np.zeros(2))['new_failure'], False)
        plant.reset(0., [np.deg2rad(FAIL_RATE), -np.deg2rad(FAIL_RATE)], 0.)
        assert_array_equal(plant.step(np.zeros(2))['new_failure'], False)

    def test_final_time_is_exact_and_finished_rows_remain_done(self):
        for kind in ['lag', 'delay']:
            plant = BatchPlant(kind, 2)
            plant.reset(0., 0., 0.)
            count = round(DURATION/DT)
            for k in range(count):
                result = plant.step(np.zeros(2))
                assert_array_equal(result['done'], k == count-1)
            assert_array_equal(plant.elapsed, DURATION)
            assert_array_equal(plant.step_count, count)
            result = plant.step(np.full(2, .5))
            assert_array_equal(result['done'], True)
            assert_array_equal(result['command'], 0.)
            assert_array_equal(result['actual_cost'], 0.)
            assert_array_equal(result['command_cost'], 0.)
            assert_array_equal(plant.step_count, count)
            assert_array_equal(plant.theta, 0.)

    def test_partial_final_interval_matches_physical_duration(self):
        plant = BatchPlant('delay', 1, dt=.05, duration=.12)
        plant.reset(0., 0., 0.)
        total_cost = 0.
        for _ in range(3):
            result = plant.step(np.array([.1]))
            total_cost += result['actual_cost'][0]
        self.assertTrue(result['done'][0])
        self.assertEqual(plant.elapsed[0], .12)
        assert_allclose(plant.theta, .5*.1*.12**2, rtol=0., atol=1e-18)
        assert_allclose(plant.omega, .1*.12, rtol=0., atol=1e-17)
        self.assertAlmostEqual(total_cost, .1**2*.12, places=16)

    def test_nonfinite_inputs_invalid_delay_and_invalid_shapes_are_rejected(self):
        for field in ['dt', 'inertia', 'limit', 'duration']:
            for value in [float('nan'), float('inf'), 0., -1.]:
                with self.assertRaises(ValueError):
                    BatchPlant('lag', 2, **{field: value})
        for n in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                BatchPlant('lag', n)
        with self.assertRaises(ValueError):
            BatchPlant('unknown', 1)
        for kind in ['lag', 'delay']:
            plant = BatchPlant(kind, 2)
            plant.reset([.1, .2], 0., 0.)
            before = plant.observation()
            for state in [float('nan'), float('inf'), -float('inf')]:
                for values in [(state, 0., 0.), (0., state, 0.), (0., 0., state)]:
                    with self.assertRaises(ValueError):
                        plant.reset(*values)
                    assert_array_equal(plant.observation(), before)
                with self.assertRaises(ValueError):
                    plant.step(np.array([0., state]))
                assert_array_equal(plant.step_count, 0)
            for commands in [0., [0.], [0., 0., 0.]]:
                with self.assertRaises(ValueError):
                    plant.step(commands)
            with self.assertRaises(ValueError):
                plant.reset(0., 0., -.1)
            with self.assertRaises(ValueError):
                plant.reset(0., 0., 0., mask=np.array([1, 0]))
        with self.assertRaises(ValueError):
            BatchPlant('delay', 1).reset(0., 0., .07)
        with self.assertRaises(ValueError):
            lag_interval(0., 0., 0., np.inf, .1, DT)


if __name__ == '__main__':
    unittest.main()
