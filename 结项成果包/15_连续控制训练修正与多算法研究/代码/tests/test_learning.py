"""Behavioral tests for corrected reward, shared interfaces and real SB3 runs."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from continuous_study import protocol as p
from continuous_study import learning as l


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.last_features = None

    def predict(self, features, deterministic=True):
        assert deterministic
        self.last_features = features.copy()
        return np.full((len(features), 1), self.action, np.float32), None


def test_history_dimensions_alignment_idempotence_and_partial_reset():
    h = l.FeatureHistory('direct')
    initial = np.array([[l.ERROR_SCALE, l.RATE_SCALE], [-l.ERROR_SCALE, 0.]])
    h.reset(2, initial)
    f = h.features(np.zeros(2))
    assert f.shape == (2, 40) and f.dtype == np.float32
    np.testing.assert_array_equal(f[0, :26], 1.)
    np.testing.assert_array_equal(f[:, 26:], 0.)
    h.record_command(np.array([.25, -.5]))
    next_obs = initial*.9
    h.observe(next_obs)
    once = h.features([.05, .05])
    h.observe(next_obs)
    np.testing.assert_array_equal(once, h.features([.05, .05]))
    np.testing.assert_allclose(once[:, 37], [.5, -1.])
    np.testing.assert_allclose(once[:, 39], 1/12)
    h.reset_rows([True, False], next_obs)
    np.testing.assert_array_equal(h.features([0., .05])[1], once[1])
    assert h.features([0., .05])[0, 39] == 0.


def test_zero_residual_is_exact_same_pid_over_clipping_and_unwinding():
    from continuous_study.baselines import PIDController
    pars = dict(kp=1., kd=.3, ki=.2)
    interface = l.PolicyInterface('residual', pars)
    pid = PIDController(pars)
    obs = np.array([[1., -.5], [-1., .5], [.1, 0.]])
    interface.reset(3, obs)
    pid.reset(3, obs)
    for k in range(20):
        current = obs/(k+1)
        np.testing.assert_array_equal(interface.command(current, np.zeros(3)), pid.act(current))
        np.testing.assert_array_equal(interface.pid.integral, pid.integral)
        interface.history.observe(obs/(k+2))


@pytest.mark.parametrize('kind', ['lag', 'delay', 'ideal'])
@pytest.mark.parametrize('mode', ['direct', 'residual'])
def test_training_deployment_observations_actions_and_pid_states_identical(kind, mode):
    pars = dict(kp=.5, kd=1.2, ki=.01) if mode == 'residual' else None
    env = l.StudyVecEnv(kind, mode, 'A', 19101, pars)
    policy = FixedPolicy(.17)
    controller = l.Controller(policy, mode, pars)
    try:
        features = env.reset()
        controller.reset(8, env.plant.observation())
        for _ in range(24):
            command = controller.act(env.plant.observation(), env.plant.elapsed)
            np.testing.assert_array_equal(policy.last_features, features)
            features, _, done, _ = env.step(np.full((8, 1), .17, np.float32))
            assert not np.any(done)
            np.testing.assert_array_equal(command, env.interface.history.commands[:, -1])
            np.testing.assert_array_equal(controller.interface.history.commands, env.interface.history.commands)
            if mode == 'residual':
                np.testing.assert_array_equal(controller.interface.pid.integral, env.interface.pid.integral)
    finally:
        env.close()


def test_failure_is_absorbing_and_never_early_reset():
    env = l.StudyVecEnv('ideal', 'direct', 'A', 19101)
    try:
        env.reset()
        env.plant.reset(np.deg2rad(np.full(8, 169.99)), np.deg2rad(np.full(8, 20.)), np.zeros(8))
        env.interface.reset(8, env.plant.observation())
        initial_ids = env.episode_ids.copy()
        _, reward, done, _ = env.step(np.zeros((8, 1)))
        assert np.all(env.plant.failed) and not np.any(done)
        assert np.all(reward < -p.DT*l.maximum_rate_cost('A'))
        frozen = env.plant.observation().copy()
        for k in range(1, 440):
            features, _, done, infos = env.step(np.zeros((8, 1)))
            if k < 439:
                np.testing.assert_array_equal(env.plant.observation(), frozen)
                np.testing.assert_array_equal(env.episode_ids, initial_ids)
                assert not np.any(done)
        assert np.all(done)
        for info in infos:
            assert info['episode']['failed']
            assert info['episode']['l'] == 440
            assert not info['TimeLimit.truncated']
            assert info['terminal_observation'][38] == 1.
        np.testing.assert_array_equal(features[:, 38:40], 0.)
        np.testing.assert_array_equal(env.episode_ids, initial_ids+1)
    finally:
        env.close()


@pytest.mark.parametrize('reward_name', ['A', 'B'])
def test_absorbing_failure_penalty_strictly_exceeds_every_ordinary_interval(reward_name):
    rng = np.random.default_rng(77)
    obs = rng.uniform(-1e5, 1e5, (1000, 2))
    costs = rng.uniform(0., p.DT*p.LIMIT**2, len(obs))
    ordinary = l.reward_values(obs, np.full(len(obs), 22.), costs, np.zeros(len(obs), bool), reward_name)
    failed = l.reward_values(obs, np.full(len(obs), 22.), costs, np.ones(len(obs), bool), reward_name)
    assert np.all(failed < ordinary)
    np.testing.assert_allclose(failed, -p.DT*(l.maximum_rate_cost(reward_name)+1.), rtol=1e-6)


@pytest.mark.parametrize('reward_name', ['A', 'B'])
def test_potential_shaping_discounted_sum_has_no_final_state_incentive(reward_name):
    rng = np.random.default_rng(717)
    path = rng.normal(0., .2, (441, 2))
    shape = 0.
    base = 0.
    for k in range(440):
        current = path[k+1:k+2]
        unshaped = l.reward_values(current, [(k+1)*p.DT], [0.], [False], reward_name)
        shaped = l.reward_values(current, [(k+1)*p.DT], [0.], [False], reward_name, path[k:k+1], [k == 439])
        shape += p.GAMMA**k*float(shaped[0])
        base += p.GAMMA**k*float(unshaped[0])
    np.testing.assert_allclose(shape-base, -l.potential(path[:1])[0], atol=1e-5)


@pytest.mark.parametrize('reward_name', ['A', 'B'])
def test_successful_pid_discounted_return_beats_deliberate_failure(reward_name):
    from continuous_study.physics import BatchPlant
    from continuous_study.baselines import PIDController
    def roll(failure):
        plant = BatchPlant('lag', 1)
        plant.reset(np.deg2rad([60.]), [0.], [.05])
        controller = PIDController(dict(kp=.6, kd=1.5, ki=0.))
        controller.reset(1, plant.observation())
        result, good = 0., True
        for k in range(440):
            previous = plant.observation()
            step = plant.step(np.array([.5]) if failure else controller.act(previous))
            obs = plant.observation()
            result += p.GAMMA**k*float(l.reward_values(obs, plant.elapsed, step['actual_cost'], plant.failed, reward_name, previous, step['done'])[0])
            if plant.elapsed[0] >= 8.-1e-10:
                good &= abs(obs[0, 0]) <= np.deg2rad(1.) and abs(obs[0, 1]) <= np.deg2rad(.5) and not plant.failed[0]
        return result, good, bool(plant.failed[0])
    success = roll(False)
    failed = roll(True)
    assert success[1] and not success[2]
    assert failed[2] and success[0] > failed[0]


@pytest.mark.parametrize('algorithm', ['SAC', 'TD3', 'PPO'])
def test_real_algorithm_smoke_zero_mean_optimizer_and_archive_restore(algorithm):
    import stable_baselines3 as sb3
    from stable_baselines3.common.logger import configure
    import torch
    settings = dict(l.ALGORITHM_SETTINGS[algorithm])
    if algorithm == 'PPO':
        settings.update(n_steps=8, batch_size=32, n_epochs=2)
    else:
        settings.update(learning_starts=16, batch_size=16)
    env = l.StudyVecEnv('ideal', 'direct', 'A', 19101)
    try:
        with patch.dict(l.ALGORITHM_SETTINGS, {algorithm: settings}):
            model = l.build_model(algorithm, env, 19101, 64)
        model.set_logger(configure(None, []))
        features = env.reset()
        action, _ = model.predict(features, deterministic=True)
        np.testing.assert_array_equal(action, 0.)
        model.learn(64)
        assert model.num_timesteps == 64
        assert model._n_updates > 0
        assert all(v > 0 for v in model._continuous_optimizer_counts.values())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.zip'
            model.save(path)
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                assert any('optimizer' in name for name in names)
            restored = getattr(sb3, algorithm).load(path, device='cpu')
            expected, _ = model.predict(features, deterministic=True)
            actual, _ = restored.predict(features, deterministic=True)
            np.testing.assert_array_equal(actual, expected)
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, restored.policy.state_dict()[key], rtol=0., atol=0.)
            assert restored._n_updates == model._n_updates
            if algorithm == 'PPO':
                assert restored.policy.optimizer.state_dict()['state']
            else:
                assert restored.actor.optimizer.state_dict()['state']
                assert restored.critic.optimizer.state_dict()['state']
    finally:
        env.close()


def test_full_training_entrypoint_retains_initial_checkpoint_and_actual_optimization():
    settings = dict(l.ALGORITHM_SETTINGS['PPO'], n_steps=8, batch_size=32, n_epochs=2)
    scenes = [dict(kind='ideal', split='validation', region='ideal', parameter_s=0., sample_id=i,
                   scenario_id=f'ideal_validation_{i}', angle_deg=angle, omega_deg_s=0.)
              for i, angle in enumerate((-5., 5.))]
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(p, 'STUDY', Path(tmp)), patch.object(p, 'SANITY_STEPS', 64), patch.object(p, 'VALIDATE_EVERY', 64), patch.dict(l.ALGORITHM_SETTINGS, {'PPO': settings}):
            l.predeclare()
            metadata = l.train_run('ideal', 'PPO', p.SANITY_SEEDS['PPO'], 64, 'sanity', mode='direct', validation_scenes=scenes)
            assert metadata['status'] == 'complete'
            assert metadata['actual_steps'] == 64
            assert metadata['optimizer_updates'] == 2
            assert metadata['optimizer_step_counts']['policy_value'] == 4
            assert metadata['checkpoints'][0]['steps'] == 0
            assert metadata['checkpoints'][0]['actor_statistics']['deterministic_action_abs_max'] == 0.
            assert metadata['checkpoints'][1]['steps'] == 64
            assert metadata['source_sha256'] == metadata['source_sha256_after']
            model_path = Path(tmp)/metadata['best']['model_path']
            assert model_path.exists() and p.sha256(model_path) == metadata['best']['model_sha256']
