"""Independent geometric and numerical checks before any DQN research run."""
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from route_study import protocol as p
from route_study.physics import interval_intersects, simulate, analytic_actions, features, rewards


class RoutePhysicsTests(unittest.TestCase):
    def test_safe_direction_and_complementary_wrong_direction(self):
        population = np.deg2rad(np.array([[-140., 140.], [140., -140.], [-30., 40.], [40., -30.]]))
        action = analytic_actions(population)
        np.testing.assert_array_equal(action, [1, 0, 1, 0])
        for i in range(4):
            a, b = population[i]
            end = a+((b-a)%(2*np.pi) if action[i] else -((a-b)%(2*np.pi)))
            grid = np.linspace(a, end, 10001)
            forbidden = abs((grid-p.CENTER+np.pi)%(2*np.pi)-np.pi) <= p.HALF_WIDTH
            self.assertFalse(np.any(forbidden))

    def test_endpoint_skip_and_internal_turn_are_detected(self):
        # Both ends outside a forbidden band, but the trajectory crosses it.
        self.assertTrue(interval_intersects(np.deg2rad(150.), np.deg2rad(800.), 0., .1))
        # An internal extremum touches the lower edge and returns outside.
        edge = p.CENTER-p.HALF_WIDTH
        theta, omega, torque, interval = edge-.01, .4, -8., .1
        self.assertTrue(interval_intersects(theta, omega, torque, interval))
        self.assertFalse(interval_intersects(theta-.0001, omega, torque, interval))
        self.assertTrue(interval_intersects(edge, 0., 0.))

    def test_periodic_geometry_matches_dense_independent_check(self):
        rng = np.random.default_rng(416)
        theta = rng.uniform(-8*np.pi, 8*np.pi, 200)
        omega = rng.uniform(-10., 10., 200)
        torque = rng.uniform(-20., 20., 200)
        result = interval_intersects(theta, omega, torque)
        t = np.linspace(0., p.DT, 4001)
        path = theta[:, None]+omega[:, None]*t+.5*torque[:, None]/p.INERTIA*t*t
        distance = abs((path-p.CENTER+np.pi)%(2*np.pi)-np.pi)
        dense = np.any(distance <= p.HALF_WIDTH, axis=1)
        np.testing.assert_array_equal(result, dense)

    def test_exact_dynamics_cost_and_saturation_against_solve_ivp(self):
        population = np.deg2rad(np.array([[-100., 100.], [-10., 20.]]))
        result = simulate(population, [1, 1], dict(kp=2., kd=3., ki=.01), keep_traces=True)
        trace = result['traces']
        self.assertLessEqual(abs(trace['command']).max(), p.LIMIT)
        for i in range(2):
            state = np.array([population[i, 0], 0.])
            for k in range(30):
                torque = trace['command'][k, i]
                solution = solve_ivp(lambda t, y: [y[1], torque/p.INERTIA], [0., p.DT], state,
                                     rtol=1e-11, atol=1e-13)
                state = solution.y[:, -1]
                np.testing.assert_allclose(state, [trace['theta'][k+1, i], trace['omega'][k+1, i]], atol=2e-12)
            self.assertAlmostEqual(result['rows'][i]['torque_cost'], np.sum(trace['command'][:, i]**2)*p.DT)

    def test_collision_freezes_and_zeroes_subsequent_cost(self):
        population = np.deg2rad(np.array([[-140., 140.], [-30., 40.]]))
        output = simulate(population, [0, 1], dict(kp=1., kd=2., ki=0.), keep_traces=True)
        self.assertTrue(output['rows'][0]['collision'])
        first = output['rows'][0]['collision_interval']
        trace = output['traces']
        np.testing.assert_array_equal(trace['theta'][first+1:, 0], trace['theta'][first+1, 0])
        np.testing.assert_array_equal(trace['command'][first+1:, 0], 0.)
        self.assertFalse(output['rows'][0]['success'])

    def test_feature_information_and_reward_order(self):
        population = np.array([[.1, -.2]])
        feature = features(population)
        self.assertEqual(feature.shape, (1, 4))
        self.assertEqual(feature.dtype, np.float32)
        base = dict(torque_cost=1., settling_time_s=5.)
        values = rewards([dict(base, collision=True, success=False),
                          dict(base, collision=False, success=False),
                          dict(base, collision=False, success=True)])
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_independent_split_sampling_and_conditions(self):
        a = p.sample(np.random.default_rng(p.SPLIT_SEEDS['development']), 64)
        b = p.sample(np.random.default_rng(p.SPLIT_SEEDS['validation']), 128)
        self.assertTrue(np.all(abs(a) <= p.BOUND))
        self.assertTrue(np.all(abs(a[:, 0]-a[:, 1]) >= p.MIN_SEPARATION))
        self.assertFalse(any(np.array_equal(x, y) for x in a for y in b))
        pressure = p.sample(np.random.default_rng(12), 128, 'detour')
        self.assertTrue(np.all(abs(pressure[:, 0]-pressure[:, 1]) > np.pi))


if __name__ == '__main__':
    unittest.main(verbosity=2)
