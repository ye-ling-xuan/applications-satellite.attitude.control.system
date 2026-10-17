"""Independent controller checks; these tests never read experiment splits."""
from __future__ import annotations

import unittest

import numpy as np

from .baselines import (CandidateBatchCompensated, CandidateBatchPID,
                        ModelCompensatedController, PIDController,
                        broad_candidates, evaluate_candidates, rank)
from .protocol import DT, INERTIA, LIMIT


def _rk4_lag(state, command, nominal, substeps=100):
    """Independent numerical ODE, with no production plant implementation."""
    def derivative(x):
        return np.array([x[1], x[2]/INERTIA, (command-x[2])/nominal])
    h = DT/substeps
    x = np.asarray(state, float).copy()
    for _ in range(substeps):
        a = derivative(x)
        b = derivative(x+.5*h*a)
        c = derivative(x+.5*h*b)
        d = derivative(x+h*c)
        x += h*(a+2*b+2*c+d)/6
    return x


def _constant_torque_step(state, command):
    theta, omega = state
    return np.array([theta+omega*DT+.5*command*DT**2/INERTIA,
                     omega+command*DT/INERTIA])


class PIDTests(unittest.TestCase):
    def test_feedback_sign_and_zero_gains(self):
        controller = PIDController(dict(kp=2., kd=3., ki=0.))
        obs = np.array([[-.1, .01], [.1, -.01]])
        np.testing.assert_allclose(controller.act(obs), [-.23, .23])
        np.testing.assert_array_equal(PIDController({}).act(obs), [0., 0.])

    def test_propose_is_pure_and_integral_updates_once(self):
        controller = PIDController(dict(ki=1.))
        obs = np.array([[.2, 0.]])
        first = controller.propose(obs)
        np.testing.assert_array_equal(controller.integral, [0.])
        np.testing.assert_array_equal(controller.propose(obs), first)
        controller.commit(obs, first, first)
        np.testing.assert_allclose(controller.integral, [.2*DT])

    def test_saturation_and_residual_antiwindup(self):
        controller = PIDController(dict(kp=.1, ki=1.))
        obs = np.array([[1., 0.]])
        pid_raw = controller.propose(obs)
        combined_raw = pid_raw+.6
        controller.commit(obs, np.clip(combined_raw, -LIMIT, LIMIT), combined_raw)
        np.testing.assert_array_equal(controller.integral, [0.])
        # Saturation caused by a residual of the opposite sign allows unwinding.
        combined_raw = controller.propose(obs)-.8
        controller.commit(obs, np.clip(combined_raw, -LIMIT, LIMIT), combined_raw)
        np.testing.assert_allclose(controller.integral, [DT])

    def test_clipping_integral_bound_and_reset(self):
        controller = PIDController(dict(kp=10., ki=1., integral_limit=.06))
        for _ in range(5):
            np.testing.assert_array_equal(controller.act(np.array([[1., 0.]])), [LIMIT])
        np.testing.assert_array_equal(controller.integral, [0.])
        controller = PIDController(dict(ki=.1, integral_limit=.06))
        for _ in range(5):
            controller.act(np.array([[1., 0.]]))
        np.testing.assert_allclose(controller.integral, [.06])
        controller.reset(2, np.zeros((2, 2)))
        np.testing.assert_array_equal(controller.integral, [0., 0.])

    def test_reject_invalid_parameters_and_measurements(self):
        with self.assertRaises(ValueError):
            PIDController(dict(kp=-1.)).reset(1)
        with self.assertRaises(ValueError):
            PIDController({}).act(np.array([[np.nan, 0.]]))
        with self.assertRaises(ValueError):
            PIDController({}).act(np.zeros(2))

    def test_candidate_batch_matches_individual_controllers(self):
        candidates = [dict(kp=.8, kd=1.2, ki=.02), dict(kp=2., kd=1.5, ki=0.)]
        controllers = [PIDController(p) for p in candidates]
        batch = CandidateBatchPID(candidates)
        rng = np.random.default_rng(145)
        for _ in range(25):
            obs = rng.normal(0., .2, (3, 2))
            expected = np.concatenate([c.act(obs) for c in controllers])
            np.testing.assert_allclose(batch.act(np.tile(obs, (2, 1))), expected, atol=1e-14)
        with self.assertRaises(ValueError):
            batch.reset(3)


class CompensationTests(unittest.TestCase):
    def test_lag_observer_recovers_nominal_torque_from_measured_rate(self):
        nominal = .17
        controller = ModelCompensatedController("lag", dict(
            kp=1.4, kd=1.1, nominal_parameter_s=nominal,
            actuator_feedback=.5, observer_gain=1.))
        state = np.array([.3, -.02, 0.])
        for _ in range(35):
            command = controller.act(np.array([[-state[0], state[1]]]))[0]
            self.assertAlmostEqual(controller.torque_estimate[0], state[2], places=10)
            state = _rk4_lag(state, command, nominal)
        controller.reset(1, np.zeros((1, 2)))
        np.testing.assert_array_equal(controller.torque_estimate, [0.])

    def test_lag_augmented_feedback_nominal_closed_loop(self):
        nominal, wn, damping, gamma = .2, 1.2, 1., 1.
        controller = ModelCompensatedController("lag", dict(
            kp=INERTIA*gamma*wn**2,
            kd=INERTIA*(nominal*wn**2+2*damping*gamma*wn),
            actuator_feedback=2*damping*wn*nominal+gamma-1,
            observer_gain=1., nominal_parameter_s=nominal))
        state = np.array([.25, 0., 0.])
        for _ in range(round(10./DT)):
            command = controller.act(np.array([[-state[0], state[1]]]))[0]
            state = _rk4_lag(state, command, nominal, substeps=20)
        self.assertLess(np.max(np.abs(state)), .001)

    def test_delay_prediction_matches_independent_queue_rollout(self):
        steps = 4
        controller = ModelCompensatedController("delay", dict(
            kp=1.1, kd=1.7, nominal_parameter_s=steps*DT))
        state = np.array([.3, -.01])
        queue = [0.]*steps
        for _ in range(60):
            obs = np.array([[-state[0], state[1]]])
            command = controller.act(obs)[0]
            expected = state.copy()
            for pending in queue:
                expected = _constant_torque_step(expected, pending)
            np.testing.assert_allclose(controller.last_prediction[0], [-expected[0], expected[1]], atol=1e-13)
            actual = queue.pop(0)
            queue.append(command)
            state = _constant_torque_step(state, actual)

    def test_zero_delay_is_exactly_the_same_pid(self):
        parameters = dict(kp=1.2, kd=1.4, ki=.04, nominal_parameter_s=0.)
        ordinary = PIDController(parameters)
        predictor = ModelCompensatedController("delay", parameters)
        rng = np.random.default_rng(25)
        for _ in range(20):
            obs = rng.normal(0., .1, (5, 2))
            np.testing.assert_allclose(predictor.act(obs), ordinary.act(obs), atol=1e-14)

    def test_nominal_delay_closed_loop(self):
        steps = 3
        controller = ModelCompensatedController("delay", dict(
            kp=1.2, kd=1.8, nominal_parameter_s=steps*DT))
        state = np.array([.3, .01])
        queue = [0.]*steps
        for _ in range(round(10./DT)):
            command = controller.act(np.array([[-state[0], state[1]]]))[0]
            actual = queue.pop(0)
            queue.append(command)
            state = _constant_torque_step(state, actual)
        self.assertLess(np.max(np.abs(state)), .001)

    def test_all_zero_gains_and_zero_initial_state(self):
        for kind in ("lag", "delay"):
            controller = ModelCompensatedController(kind, dict(
                nominal_parameter_s=.2, actuator_feedback=.5, observer_gain=.5))
            for _ in range(10):
                np.testing.assert_array_equal(controller.act(np.zeros((2, 2))), [0., 0.])

    def test_compensated_batch_matches_separate_nominal_models(self):
        rng = np.random.default_rng(451)
        for kind in ("lag", "delay"):
            candidates = [dict(kp=1., kd=1.5, ki=.01, nominal_parameter_s=.05,
                               actuator_feedback=.7, observer_gain=1.),
                          dict(kp=.8, kd=1.2, ki=0., nominal_parameter_s=.2,
                               actuator_feedback=.4, observer_gain=.5)]
            separate = [ModelCompensatedController(kind, p) for p in candidates]
            batch = CandidateBatchCompensated(kind, candidates)
            for _ in range(25):
                obs = rng.normal(0., .15, (3, 2))
                expected = np.concatenate([c.act(obs) for c in separate])
                np.testing.assert_allclose(batch.act(np.tile(obs, (2, 1))), expected, atol=1e-13)

    def test_invalid_nominal_models(self):
        for kind, nominal in (("lag", 0.), ("delay", .075), ("delay", -.1)):
            with self.assertRaises(ValueError):
                ModelCompensatedController(kind, dict(nominal_parameter_s=nominal)).reset(1)


class SearchDesignTests(unittest.TestCase):
    def test_rank_prefers_reliability_then_actual_effort(self):
        self.assertLess(rank(dict(passes=10, mean_actual_cost=100.)),
                        rank(dict(passes=9, mean_actual_cost=.1)))
        self.assertLess(rank(dict(passes=10, mean_actual_cost=1.)),
                        rank(dict(passes=10, mean_actual_cost=2.)))

    def test_search_cannot_accept_test_split(self):
        # Refusal occurs before importing evaluation or loading any scenes.
        with self.assertRaises(ValueError):
            evaluate_candidates("lag", "pid", [dict(kp=1.)], "test", "forbidden")

    def test_broad_search_covers_small_large_and_integral_gains(self):
        candidates = broad_candidates("lag", "pid")
        self.assertAlmostEqual(min(c["kp"] for c in candidates), .05)
        self.assertAlmostEqual(max(c["kp"] for c in candidates), 12.)
        self.assertAlmostEqual(min(c["kd"] for c in candidates), .2)
        self.assertAlmostEqual(max(c["kd"] for c in candidates), 12.)
        self.assertTrue(any(c["ki"] > 0. for c in candidates))
        self.assertTrue(any(c["ki"] == 0. for c in candidates))


if __name__ == "__main__":
    unittest.main()
