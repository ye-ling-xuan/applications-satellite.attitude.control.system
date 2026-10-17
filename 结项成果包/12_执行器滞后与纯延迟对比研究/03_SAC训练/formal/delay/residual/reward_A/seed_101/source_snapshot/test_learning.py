"""Interface/terminal/reproducibility tests, never research training."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from actuator_study import protocol as p
from actuator_study.learning import (
    FeatureHistory, SACController, SACVecEnv, _PolicyInterface, configuration,
    reward_values, train_run, predeclare, _rank,
)


class RecordingPolicy:
    def __init__(self, action):
        self.action = action
        self.seen = []

    def predict(self, features, deterministic=True):
        if not deterministic:
            raise AssertionError('evaluation must be deterministic')
        self.seen.append(features.copy())
        return np.full((features.shape[0], 1), self.action, dtype=np.float32), None


class LearningTests(unittest.TestCase):
    def test_history_reset_order_and_real_sent_commands(self):
        history = FeatureHistory('direct')
        initial = np.array([[np.pi/3, np.pi/6], [-np.pi/6, 0.]])
        history.reset(2, initial)
        features = history.features(np.zeros(2))
        self.assertEqual(features.shape, (2, 94))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[0, :62], np.ones(62))
        np.testing.assert_array_equal(features[:, 62:], 0.)
        command = np.array([.5, -.25])
        next_measurement = initial+np.array([[-.01, -.02], [.03, .04]])
        history.record_command(command)
        history.observe(next_measurement)
        once = history.features(np.full(2, .05))
        history.observe(next_measurement)
        np.testing.assert_array_equal(once, history.features(np.full(2, .05)))
        np.testing.assert_allclose(once[:, 91], [1., -.5])
        np.testing.assert_allclose(once[:, 93], 1/30)
        np.testing.assert_allclose(history.measurements[:, -1], next_measurement)
        np.testing.assert_array_equal(history.measurements[:, :-1], np.repeat(initial[:, None], 30, axis=1))

    def test_history_full_window_and_partial_reset(self):
        history = FeatureHistory('direct')
        history.reset(2, np.zeros((2, 2)))
        for k in range(35):
            history.record_command(np.array([k/100, -k/100]))
            history.observe(np.full((2, 2), k+1.))
        np.testing.assert_array_equal(history.valid, 30)
        np.testing.assert_array_equal(history.measurements[0, :, 0], np.arange(5., 36.))
        np.testing.assert_allclose(history.commands[0], np.arange(5., 35.)/100)
        before = history.features(np.full(2, 1.75))[1].copy()
        history.reset_rows(np.array([True, False]), np.array([[9., 8.], [35., 35.]]))
        after = history.features(np.array([0., 1.75]))
        np.testing.assert_array_equal(after[1], before)
        self.assertEqual(after[0, 93], 0.)
        np.testing.assert_array_equal(history.measurements[0], np.tile([9., 8.], (31, 1)))

    def test_residual_exact_formula_and_total_command_antiwindup(self):
        from actuator_study.baselines import PIDController
        parameters = dict(kp=1., kd=.2, ki=.3)
        obs = np.array([[1., 0.], [.1, -.1], [-1., 0.]])
        interface = _PolicyInterface('residual', parameters)
        interface.reset(3, obs)
        pid = PIDController(parameters)
        pid.reset(3, obs)
        raw = pid.propose(obs)
        feature = interface.features(obs, np.zeros(3))
        np.testing.assert_allclose(feature[:, 94], raw/.5)
        np.testing.assert_array_equal(feature[:, 95], 0.)
        action = np.array([-1., .5, 1.])
        expected = np.clip(raw+.25*action, -.5, .5)
        pid.commit(obs, expected, raw=raw+.25*action)
        actual = interface.command(obs, action)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(interface.pid.integral, pid.integral)
        np.testing.assert_array_equal(interface.history.commands[:, -1], expected)

    def test_residual_integration_uses_total_saturation_only(self):
        # First two rows are unsaturated despite a nonzero residual. Last two
        # increments would deepen positive total saturation and are rejected.
        interface = _PolicyInterface('residual', dict(kp=1., kd=1., ki=.5))
        obs = np.array([[.1, 0.], [.1, 0.], [.4, 0.], [.1, -.8]])
        interface.reset(4, obs)
        interface.pid.integral[3] = -1.
        before = interface.pid.integral.copy()
        command = interface.command(obs, np.array([-1., 1., 1., 1.]))
        np.testing.assert_allclose(command[:2], [-.1475, .3525])
        np.testing.assert_allclose(interface.pid.integral[:2], [.005, .005])
        self.assertEqual(interface.pid.integral[2], before[2])
        # Positive increment would deepen positive total saturation here too.
        self.assertEqual(interface.pid.integral[3], before[3])

        unwinding = _PolicyInterface('residual', dict(kp=1., kd=1., ki=.5))
        negative_error = np.array([[-.1, -1.]])
        unwinding.reset(1, negative_error)
        unwinding.command(negative_error, np.array([1.]))
        self.assertAlmostEqual(unwinding.pid.integral[0], -.005)

    def test_zero_residual_restores_identical_pid(self):
        from actuator_study.baselines import PIDController
        parameters = dict(kp=1., kd=.2, ki=.3)
        interface = _PolicyInterface('residual', parameters)
        pid = PIDController(parameters)
        initial = np.array([[.8, .1], [-.2, .3]])
        interface.reset(2, initial)
        pid.reset(2, initial)
        for k in range(5):
            obs = initial/(k+1)
            np.testing.assert_array_equal(interface.command(obs, np.zeros(2)), pid.act(obs))
            np.testing.assert_array_equal(interface.pid.integral, pid.integral)
            # A new synthetic measured sample is consumed after each command.
            interface.history.observe(initial/(k+2))

    def test_training_deployment_features_and_commands_identical(self):
        for kind in p.KINDS:
            for method in p.METHODS:
                with self.subTest(kind=kind, method=method):
                    parameters = dict(kp=.9, kd=1.6, ki=.02)
                    env = SACVecEnv(kind, method, 'A', 9101, parameters)
                    try:
                        features = env.reset()
                        policy = RecordingPolicy(.3)
                        controller = SACController(policy, method, parameters)
                        controller.reset(8, env.plant.observation())
                        for _ in range(12):
                            physical_obs = env.plant.observation()
                            command = controller.act(physical_obs, env.plant.elapsed)
                            np.testing.assert_array_equal(policy.seen[-1], features)
                            features, _, dones, _ = env.step(np.full((8, 1), .3, dtype=np.float32))
                            self.assertFalse(np.any(dones))
                            np.testing.assert_array_equal(command, env.interface.history.commands[:, -1])
                            np.testing.assert_array_equal(controller.interface.history.commands, env.interface.history.commands)
                            if method == 'residual':
                                np.testing.assert_array_equal(controller.interface.pid.integral, env.interface.pid.integral)
                    finally:
                        env.close()

    def test_true_finite_terminal_observation_then_clean_auto_reset(self):
        env = SACVecEnv('delay', 'residual', 'A', 9101, dict(kp=1., kd=2., ki=.1))
        try:
            env.reset()
            # Move just the clock to its last transition: no learned model is
            # trained, and the dynamics still integrate a full 0.05 s interval.
            env.plant.step_count[:] = 439
            env.plant.elapsed[:] = 21.95
            features, _, dones, infos = env.step(np.zeros((8, 1)))
            self.assertTrue(np.all(dones))
            for info in infos:
                self.assertFalse(info['TimeLimit.truncated'])
                self.assertEqual(info['termination_reason'], 'finite_horizon_22s')
                self.assertEqual(info['terminal_observation'].shape, (96,))
                self.assertEqual(info['terminal_observation'][92], 1.)
                self.assertGreater(info['terminal_observation'][93], 0.)
                self.assertEqual(info['episode']['l'], 440)
            np.testing.assert_array_equal(features[:, 92:94], 0.)
            np.testing.assert_array_equal(features[:, 62:92], 0.)
            np.testing.assert_array_equal(env.interface.pid.integral, 0.)
            self.assertEqual(env.total_steps, 8)
        finally:
            env.close()

    def test_physical_failure_terminal_has_failure_penalty(self):
        env = SACVecEnv('lag', 'direct', 'A', 9101)
        try:
            env.reset()
            env.plant.reset(np.deg2rad(np.full(8, 169.99)), np.deg2rad(np.full(8, 20.)), np.full(8, .1))
            env.interface.reset(8, env.plant.observation())
            _, rewards, dones, infos = env.step(np.zeros((8, 1)))
            np.testing.assert_array_equal(rewards, -1000.)
            self.assertTrue(np.all(dones))
            self.assertTrue(all(i['episode']['failed'] for i in infos))
            self.assertTrue(all(i['termination_reason'] == 'physical_failure' for i in infos))
            self.assertTrue(all(not i['TimeLimit.truncated'] for i in infos))
        finally:
            env.close()

    def test_rng_streams_reproducible_independent_and_reset_isolated(self):
        first = SACVecEnv('delay', 'direct', 'A', 9101)
        second = SACVecEnv('delay', 'direct', 'A', 9101)
        other = SACVecEnv('lag', 'direct', 'A', 9101)
        try:
            np.testing.assert_array_equal(first.reset(), second.reset())
            other.reset()
            self.assertNotEqual(first.reset_infos[0]['initial_angle_deg'], other.reset_infos[0]['initial_angle_deg'])
            self.assertEqual(len({row['initial_angle_deg'] for row in first.reset_infos}), 8)
            mask = np.zeros(8, dtype=bool)
            mask[0] = True
            first._reset_rows(mask)
            # Rank 1 did not consume any new random values during rank 0 reset.
            first._reset_rows(np.array([False, True]+[False]*6))
            second._reset_rows(np.array([False, True]+[False]*6))
            self.assertEqual(first.reset_infos[1], second.reset_infos[1])
        finally:
            first.close()
            second.close()
            other.close()

    def test_reward_exact_torque_and_inclusive_deadline(self):
        obs = np.array([[np.deg2rad(2.), 0.], [0., 0.]])
        costs = np.array([0., p.DT*p.LIMIT**2])
        before = reward_values(obs, np.array([7.95, 7.95]), costs, np.array([False, False]), 'A')
        deadline = reward_values(obs, np.array([8., 8.]), costs, np.array([False, False]), 'A')
        self.assertAlmostEqual(float(before[0]), -.05*(2/5)**2, places=7)
        self.assertAlmostEqual(float(before[1]), -.005, places=7)
        self.assertAlmostEqual(float(deadline[0]-before[0]), -.05*8*2, places=7)
        self.assertEqual(deadline[1], before[1])
        for reward_name in ('A', 'B'):
            extreme = reward_values(np.array([[10000., -10000.]]), [22.], [.0125], [False], reward_name)
            self.assertTrue(np.isfinite(extreme[0]))
            self.assertGreater(extreme[0], -40.)

    def test_hidden_simulator_quantities_do_not_change_policy_features(self):
        env = SACVecEnv('lag', 'direct', 'A', 9101)
        try:
            features = env.reset()
            env.plant.actual_torque[:] = .49
            env.plant._parameter_s[:] = .29
            again = env.interface.features(env.plant.observation(), env.plant.elapsed)
            np.testing.assert_array_equal(features, again)
        finally:
            env.close()

    def test_declared_reward_and_optimizer_configuration(self):
        settings = configuration()
        self.assertEqual(set(settings['rewards']), {'A', 'B'})
        self.assertEqual(settings['sac']['ent_coef'], 'auto_0.01')
        self.assertEqual(settings['sac']['gradient_steps'], 1)
        self.assertEqual(settings['sac']['n_envs'], 8)
        with self.assertRaises(ValueError):
            SACVecEnv('lag', 'direct', 'A', 9101, n_envs=1)

    def test_sb3_training_pipeline_optimizer_metadata_and_roundtrip_smoke_64_steps(self):
        from stable_baselines3 import SAC
        import torch
        torch.set_num_threads(1)
        from actuator_study.learning import SAC_SETTINGS
        captured = []

        class CapturingSAC(SAC):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)

        stats = dict(count=576, passes=560, success_rate=560/576,
                     physical_failures=0, mean_actual_cost=.1, mean_command_cost=.11)
        # Budgets and validation are isolated test fixtures. Only 64 actual
        # SB3 interactions are executed; no study attempt or scenario is used.
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            shutil.copytree(Path(__file__).parent, study/'代码', ignore=shutil.ignore_patterns('__pycache__'))
            p.write_json(study/'02_传统控制'/'selected_lag.json', dict(pid=dict(parameters=dict(kp=1., kd=2., ki=0.))))
            with (patch.object(p, 'STUDY', study), patch.object(p, 'DEVELOPMENT_STEPS', 64),
                  patch.object(p, 'VALIDATE_EVERY', 32), patch.dict(SAC_SETTINGS, learning_starts=16),
                  patch('stable_baselines3.SAC', CapturingSAC),
                  patch.object(p, 'scenes', return_value=[{}]*576),
                  patch('actuator_study.evaluation.evaluate', return_value=dict(rows=[dict(success=True)], traces=None)),
                  patch('actuator_study.evaluation.summary', return_value=stats)):
                predeclare()
                metadata = train_run('lag', 'direct', 'A', 9101, 64, 'development')
                model = captured[0]
                self.assertEqual(model.num_timesteps, 64)
                self.assertEqual(model.env.total_steps, 64)
                self.assertEqual(model._n_updates, 6)
                self.assertEqual(model.replay_buffer.observations.dtype, np.float32)
                self.assertTrue(np.all(model.replay_buffer.timeouts == 0.))
                self.assertAlmostEqual(model.target_entropy, -1.)
                self.assertNotEqual(float(model.log_ent_coef.exp().item()), .01)
                self.assertTrue(model.actor.optimizer.state)
                self.assertTrue(model.critic.optimizer.state)
                self.assertTrue(model.ent_coef_optimizer.state)
                self.assertEqual(metadata['status'], 'complete')
                self.assertEqual(metadata['actual_steps'], 64)
                self.assertEqual(metadata['optimizer_updates'], 6)
                self.assertEqual(metadata['source_sha256'], metadata['source_sha256_after'])
                self.assertTrue(metadata['source_sha256'])
                self.assertEqual(metadata['best']['steps'], 32)  # Earlier equal checkpoint wins.
                self.assertGreater(metadata['training_steps_per_second'], 0.)
                self.assertEqual([row['steps'] for row in metadata['checkpoints']], [32, 64])
                self.assertEqual([row['optimizer_updates'] for row in metadata['checkpoints']], [2, 6])
                directory = study/metadata['run_path']
                saved = p.read_json(directory/'training_metadata.json')
                self.assertEqual(saved, metadata)
                self.assertEqual(metadata['best']['model_sha256'], p.sha256(study/metadata['best']['model_path']))
                self.assertEqual(metadata['final']['model_sha256'], p.sha256(study/metadata['final']['model_path']))
                for relative, digest in metadata['source_sha256'].items():
                    self.assertEqual(p.sha256(directory/'source_snapshot'/Path(relative).relative_to('代码')), digest)
                self.assertEqual(len((directory/'optimizer_steps.jsonl').read_text(encoding='utf-8').splitlines()), 6)
                initials = [json.loads(line) for line in (directory/'training_episode_initials.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(initials), 8)
                self.assertTrue(all(.05 <= row['parameter_s'] <= .3 for row in initials))
                features = model.env.interface.features(model.env.plant.observation(), model.env.plant.elapsed)
                loaded = SAC.load(study/metadata['final']['model_path'], device='cpu')
                np.testing.assert_array_equal(model.predict(features, deterministic=True)[0], loaded.predict(features, deterministic=True)[0])
                self.assertTrue(loaded.actor.optimizer.state)
                self.assertTrue(loaded.critic.optimizer.state)
                self.assertTrue(loaded.ent_coef_optimizer.state)
                with self.assertRaises(FileExistsError):
                    train_run('lag', 'direct', 'A', 9101, 64, 'development')

    def test_frozen_heldout_manifest_blocks_both_training_stages_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            p.write_json(study/'04_独立测试'/'freeze.json', dict(frozen=True))
            with patch.object(p, 'STUDY', study):
                for stage, seed, steps in [('development', 9101, 150000), ('formal', 101, 500000)]:
                    with self.assertRaisesRegex(RuntimeError, 'further research training is prohibited'):
                        train_run('lag', 'direct', 'A', seed, steps, stage)
            self.assertFalse((study/'03_SAC训练').exists())

    def test_checkpoint_selection_uses_passes_then_cost_then_earlier(self):
        def item(passes, cost, steps):
            return dict(steps=steps, validation=dict(passes=passes, mean_actual_cost=cost))
        self.assertGreater(_rank(item(570, .2, 100000)), _rank(item(569, .01, 50000)))
        self.assertGreater(_rank(item(570, .1, 100000)), _rank(item(570, .2, 50000)))
        self.assertGreater(_rank(item(570, .1, 50000)), _rank(item(570, .1, 100000)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
