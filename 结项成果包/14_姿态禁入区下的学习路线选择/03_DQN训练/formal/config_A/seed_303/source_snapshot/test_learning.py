"""Isolated DQN optimizer/cache tests, never retained as study evidence."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from route_study import protocol as p
from route_study.learning import SETTINGS, predeclare, prepare_database, train_run, _vec_class


class LearningTests(unittest.TestCase):
    def test_one_step_environment_only_returns_chosen_physical_reward(self):
        population = np.array([[.1, .2], [-.5, .5]])
        data = dict(population=population, features=np.zeros((2, 4), np.float32),
                    rewards=np.array([[10., -10.], [-2., 10.]], np.float32),
                    final_theta=np.zeros((2, 2)), success=np.array([[True, False], [False, True]]),
                    collision=np.array([[False, True], [False, False]]),
                    physical_intervals=np.full((2, 2), 440))
        env = _vec_class()(data, 1401)
        try:
            env.reset()
            env.row_ids[:] = np.arange(16)%2
            actions = np.arange(16)%2
            before = env.row_ids.copy()
            _, reward, done, infos = env.step(actions)
            np.testing.assert_array_equal(reward, data['rewards'][before, actions])
            self.assertTrue(np.all(done))
            self.assertTrue(all(not item['TimeLimit.truncated'] for item in infos))
            self.assertEqual(env.decisions, 16)
            self.assertEqual(len(env.seen_pairs), 2)
        finally:
            env.close()

    def test_real_dqn_64_decisions_optimizer_checkpoints_and_checksums(self):
        from stable_baselines3 import DQN
        captured = []

        class CapturingDQN(DQN):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)

        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            shutil.copytree(Path(__file__).parent, study/'代码', ignore=shutil.ignore_patterns('__pycache__'))
            with (patch.object(p, 'STUDY', study), patch.object(p, 'DEVELOPMENT_DECISIONS', 64),
                  patch.object(p, 'CHECK_EVERY', 32), patch.dict(p.COUNTS, training=32),
                  patch.dict(SETTINGS, learning_starts=16), patch('stable_baselines3.DQN', CapturingDQN)):
                p.design()
                predeclare()
                p.write_json(study/'02_传统对照'/'selected_pid.json', dict(parameters=dict(kp=1., kd=2., ki=0.), analytic_feasibility_passed=True))
                database = prepare_database()
                self.assertEqual(database['simulated_trajectories'], 64)
                metadata = train_run('development', 'A', 1401)
                self.assertEqual(metadata['status'], 'complete')
                self.assertEqual(metadata['actual_decisions'], 64)
                self.assertEqual(metadata['optimizer_updates'], 6)
                model = captured[0]
                self.assertTrue(model.policy.optimizer.state)
                self.assertEqual(model.gamma, 0.)
                self.assertAlmostEqual(model.exploration_rate, .05)
                self.assertTrue(np.any(model.replay_buffer.dones))
                self.assertTrue(np.all(model.replay_buffer.timeouts == 0.))
                self.assertEqual(metadata['source_sha256'], metadata['source_sha256_after'])
                path = study/metadata['best']['model_path']
                self.assertEqual(p.sha256(path), metadata['best']['model_sha256'])
                model_observations = np.zeros((4, 4), np.float32)
                restored = DQN.load(study/metadata['final']['model_path'], device='cpu')
                np.testing.assert_array_equal(model.predict(model_observations, deterministic=True)[0], restored.predict(model_observations, deterministic=True)[0])
                self.assertTrue(restored.policy.optimizer.state)
                directory = study/'03_DQN训练'/'development'/'config_A'/'seed_1401'
                self.assertEqual(len((directory/'optimizer_steps.jsonl').read_text(encoding='utf-8').splitlines()), 3)
                with (directory/'selected_action_experience.csv').open(encoding='utf-8') as stream:
                    self.assertEqual(sum(1 for _ in stream), 65)
                self.assertLessEqual(metadata['unique_state_action_pairs_seen'], 64)
                self.assertEqual(metadata['repeat_state_action_replays'], 64-metadata['unique_state_action_pairs_seen'])
                self.assertEqual(json.loads((directory/'training_metadata.json').read_text(encoding='utf-8')), metadata)
                p.write_json(study/'04_独立测试'/'freeze.json', dict(frozen=True))
                with self.assertRaisesRegex(RuntimeError, 'prohibits subsequent training'):
                    train_run('development', 'B', 1402)

    def test_baseline_response_independent_verifier_roundtrip(self):
        from route_study.delivery import freeze, test_responses
        from route_study.verify import verify
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            shutil.copytree(Path(__file__).parent, study/'代码', ignore=shutil.ignore_patterns('__pycache__'))
            with patch.object(p, 'STUDY', study), patch.dict(p.COUNTS, training=4, test=4, detour=4, edge=4):
                p.design()
                predeclare()
                p.write_json(study/'02_传统对照'/'selected_pid.json', dict(parameters=dict(kp=1., kd=2., ki=0.), analytic_feasibility_passed=True))
                prepare_database()
                p.write_json(study/'03_DQN训练'/'development_decision.json', dict(passed_gate=False))
                freeze()
                outputs = test_responses()
                self.assertEqual(len(outputs), 6)
                result = verify()
                self.assertTrue(result['passed'])
                self.assertEqual(len(result['checks']), 6)
                self.assertTrue(all(item['count'] == 4 for item in result['checks']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
