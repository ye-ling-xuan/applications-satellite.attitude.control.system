"""Independent ODE, power balance, delay-index and absorbing-state checks.

Run from this directory with ``python -m pytest test_continuous_physics.py -q``.
These tests do not import the training code or reuse its integration formulas.
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad, solve_ivp

from continuous_study.physics import BatchPlant, ideal_interval, lag_interval
from continuous_study.protocol import DT, DURATION, FAIL_ANGLE, FAIL_RATE, LIMIT


class IndependentLagTests(unittest.TestCase):
    def test_joint_ode_and_mechanical_power_balance(self):
        rng = np.random.default_rng(20261018001)
        constants = np.array([.003, .05, .17, .3, .5, 5.])
        inertia = 1.7
        theta = rng.uniform(-.4, .4, constants.size)
        omega = rng.uniform(-.2, .2, constants.size)
        reference = np.column_stack((theta, omega, np.zeros((constants.size, 3))))
        plant = BatchPlant('lag', constants.size, inertia=inertia)
        plant.reset(theta, omega, constants)
        initial_kinetic = .5*inertia*omega**2
        for _ in range(15):
            requested = rng.uniform(-.9, .9, constants.size)
            result = plant.step(requested)
            for i, command in enumerate(np.clip(requested, -LIMIT, LIMIT)):
                reference[i, 3] = 0.  # per-interval torque-square integral

                def rhs(time, state):
                    del time
                    angle, rate, torque, cost, work = state
                    return [rate, torque/inertia, (command-torque)/constants[i],
                            torque*torque, torque*rate]

                solution = solve_ivp(rhs, (0., DT), reference[i], method='DOP853',
                                     rtol=2e-12, atol=2e-14)
                self.assertTrue(solution.success)
                reference[i] = solution.y[:, -1]
            assert_allclose(plant.theta, reference[:, 0], rtol=2e-12, atol=5e-13)
            assert_allclose(plant.omega, reference[:, 1], rtol=2e-12, atol=5e-13)
            assert_allclose(plant.actual_torque, reference[:, 2], rtol=3e-11, atol=5e-13)
            assert_allclose(result['actual_cost'], reference[:, 3], rtol=6e-11, atol=5e-14)
            # Torque-square effort is not energy. Actual mechanical work has
            # units of joules and must equal the kinetic-energy increment.
            assert_allclose(.5*inertia*plant.omega**2-initial_kinetic,
                            reference[:, 4], rtol=2e-11, atol=5e-14)

    def test_cost_quadrature_command_reversals_and_extreme_time_scales(self):
        constants = np.array([1e-4, .003, .05, .3, 20., 1e5, 1e9])
        for previous, command in [(0., .5), (.5, -.5), (-.4, .2), (.2, .2)]:
            _, _, _, cost = lag_interval(.2, -.1, previous, command, constants, DT)
            expected = [quad(lambda t: (previous+(command-previous)
                                        *(-np.expm1(-t/constant)))**2,
                             0., DT, epsabs=1e-14, epsrel=1e-12, limit=200)[0]
                        for constant in constants]
            assert_allclose(cost, expected, rtol=3e-10, atol=2e-15)
            self.assertTrue(np.all(cost >= 0.))

    def test_ideal_limit_and_zero_duration(self):
        theta, omega, command, inertia = .3, -.1, -.4, 2.3
        expected_theta = theta+omega*DT+.5*command*DT**2/inertia
        expected_omega = omega+command*DT/inertia
        values = lag_interval(theta, omega, .2, command, 0., DT, inertia)
        assert_allclose(values, [expected_theta, expected_omega, command,
                                 command**2*DT], rtol=0., atol=1e-16)
        values = lag_interval(theta, omega, .2, command, .1, 0., inertia)
        assert_allclose(values, [theta, omega, .2, 0.], rtol=0., atol=0.)
        very_fast = lag_interval(0., 0., .5, 0., np.array([1e-320]), DT)
        self.assertTrue(np.all(np.isfinite(very_fast)))

    def test_named_ideal_plant_is_identical_to_zero_lag(self):
        ideal = BatchPlant('ideal', 2, inertia=1.8)
        lag = BatchPlant('lag', 2, inertia=1.8)
        ideal.reset([.3, -.1], [.03, -.02], 0.)
        lag.reset([.3, -.1], [.03, -.02], 0.)
        for command in ([.4, -.5], [-.5, .3], [.2, 0.], [3., -5.]):
            first, second = ideal.step(command), lag.step(command)
            for key in first:
                assert_array_equal(first[key], second[key])
            assert_array_equal(ideal.observation(), lag.observation())
        with self.assertRaises(ValueError):
            ideal.reset(0., 0., .1)


class IndependentDelayTests(unittest.TestCase):
    def test_absolute_command_index_reconstructs_mixed_delay_batch(self):
        rng = np.random.default_rng(20261018002)
        offsets = np.array([0, 1, 2, 6, 11])
        inertia = 1.3
        commands = rng.uniform(-1., 1., (35, offsets.size))
        clipped = np.clip(commands, -LIMIT, LIMIT)
        plant = BatchPlant('delay', offsets.size, inertia=inertia)
        plant.reset(0., 0., offsets*DT)
        expected_theta = np.zeros(offsets.size)
        expected_omega = np.zeros(offsets.size)
        for k, requested in enumerate(commands):
            # This reference uses the entire immutable command log, not the
            # implementation's shifting queue, avoiding shared index logic.
            applied = np.array([clipped[k-m, j] if k >= m else 0.
                                for j, m in enumerate(offsets)])
            expected_theta += expected_omega*DT+.5*applied*DT**2/inertia
            expected_omega += applied*DT/inertia
            result = plant.step(requested)
            assert_array_equal(result['tau_start'], applied)
            assert_array_equal(result['tau_end'], applied)
            assert_allclose(plant.theta, expected_theta, rtol=0., atol=3e-16)
            assert_allclose(plant.omega, expected_omega, rtol=0., atol=3e-16)
            assert_allclose(result['actual_cost'], applied**2*DT, rtol=0., atol=0.)

    def test_masked_reset_discards_only_selected_history_and_can_expand_queue(self):
        plant = BatchPlant('delay', 2, duration=2.)
        plant.reset(0., 0., DT)
        plant.step(np.array([.2, -.3]))
        plant.step(np.array([.4, -.1]))
        old_other = (plant.theta[1], plant.omega[1], plant.elapsed[1])
        plant.reset(.1, -.05, 13*DT, mask=np.array([True, False]))
        assert_array_equal([plant.theta[1], plant.omega[1], plant.elapsed[1]], old_other)
        for k in range(14):
            result = plant.step(np.array([.5, .3]))
            self.assertEqual(result['tau_end'][0], 0. if k < 13 else .5)
            if k == 0:
                self.assertEqual(result['tau_end'][1], -.1)


class AbsorbingLifecycleTests(unittest.TestCase):
    def test_failure_is_absorbing_but_done_only_at_fixed_horizon(self):
        plant = BatchPlant('lag', 2, duration=.2)
        plant.reset([np.deg2rad(FAIL_ANGLE)-1e-5, 0.],
                    [.1, np.deg2rad(FAIL_RATE)-1e-5], 0.)
        result = plant.step(np.full(2, .5))
        assert_array_equal(result['failed'], True)
        assert_array_equal(result['new_failure'], True)
        assert_array_equal(result['done'], False)
        assert_allclose(result['actual_cost'], .25*DT)
        frozen_theta, frozen_omega = plant.theta.copy(), plant.omega.copy()
        # Returned flags must be snapshots, so consumers cannot change physics.
        result['failed'][:] = False
        for k in range(2, 5):
            result = plant.step(np.full(2, -.5))
            assert_array_equal(result['failed'], True)
            assert_array_equal(result['new_failure'], False)
            assert_array_equal(result['done'], k == 4)
            assert_array_equal(plant.theta, frozen_theta)
            assert_array_equal(plant.omega, frozen_omega)
            for name in ('command', 'tau_start', 'tau_end', 'actual_cost', 'command_cost'):
                assert_array_equal(result[name], 0.)
            assert_allclose(plant.elapsed, k*DT, rtol=0., atol=0.)
        plant.step(np.full(2, .5))
        assert_array_equal(plant.step_count, 4)
        plant.reset(0., 0., .1, mask=np.array([True, False]))
        assert_array_equal(plant.failed, [False, True])
        assert_array_equal(plant.elapsed, [0., .2])

    def test_fractional_final_interval_and_default_horizon(self):
        plant = BatchPlant('delay', 1, dt=.05, duration=.12)
        plant.reset(0., 0., 0.)
        cost = 0.
        for k in range(3):
            result = plant.step(np.array([.1]))
            cost += result['actual_cost'][0]
            self.assertEqual(bool(result['done'][0]), k == 2)
        self.assertEqual(plant.elapsed[0], .12)
        assert_allclose(plant.theta, .5*.1*.12**2, rtol=0., atol=1e-18)
        assert_allclose(plant.omega, .1*.12, rtol=0., atol=1e-17)
        self.assertAlmostEqual(cost, .1**2*.12, places=16)
        plant = BatchPlant('lag', 1)
        plant.reset(0., 0., 0.)
        for k in range(round(DURATION/DT)):
            result = plant.step(np.zeros(1))
            self.assertEqual(bool(result['done'][0]), k == round(DURATION/DT)-1)
        self.assertEqual(plant.elapsed[0], DURATION)

    def test_observation_excludes_actuator_state_and_invalid_inputs_are_atomic(self):
        plant = BatchPlant('lag', 2)
        plant.reset([.1, -.2], [.03, -.04], [.05, .3])
        assert_array_equal(plant.observation(), [[-.1, .03], [.2, -.04]])
        self.assertEqual(plant.observation().shape, (2, 2))
        for args in [(np.nan, 0., .1), (0., np.inf, .1), (0., 0., -.1)]:
            before = plant.observation().copy()
            with self.assertRaises(ValueError):
                plant.reset(*args)
            assert_array_equal(plant.observation(), before)
        with self.assertRaises(ValueError):
            plant.step(np.array([np.nan, .1]))
        assert_array_equal(plant.step_count, 0)
        with self.assertRaises(ValueError):
            BatchPlant('delay', 1).reset(0., 0., .07)


if __name__ == '__main__':
    unittest.main()
