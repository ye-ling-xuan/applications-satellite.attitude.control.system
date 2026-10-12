"""Scientific/API checks for fast batch RK4 training, not final benchmarks."""
import csv
import json

import numpy as np
import pytest

from multi_algorithm_benchmark.envs import ThreeAxisEnv
from multi_algorithm_benchmark.vector_training import FastThreeVecEnv


@pytest.mark.parametrize("n_envs", [1, 5])
@pytest.mark.parametrize("randomized", [False, True])
def test_fast_matches_gym_rk4_distribution_noise_and_actions(n_envs, randomized):
    """Same rank seed matches the full reset distribution AND noise stream."""
    vector = FastThreeVecEnv(n_envs=n_envs, seed=51, randomized=randomized)
    gyms = [ThreeAxisEnv(randomized=randomized) for _ in range(n_envs)]
    gym_obs = np.stack([env.reset(seed=51 + rank)[0] for rank, env in enumerate(gyms)])
    np.testing.assert_array_equal(vector.reset(), gym_obs)
    for rank, env in enumerate(gyms):
        np.testing.assert_allclose(vector.inertia[rank], env.plant.inertia[0], atol=0)
        np.testing.assert_allclose(vector.bias[rank], env.plant.bias[0], atol=0)
        np.testing.assert_allclose(vector.noise_angle[rank], env.plant.noise_angle[0], atol=0)
    rng = np.random.default_rng(18)
    for _ in range(45):
        actions = rng.uniform(-1.5, 1.5, (n_envs, 3))
        obs, rewards, dones, infos = vector.step(actions)
        for rank, env in enumerate(gyms):
            gym_obs, gym_reward, terminated, truncated, info = env.step(actions[rank])
            np.testing.assert_array_equal(obs[rank], gym_obs)
            assert rewards[rank] == pytest.approx(gym_reward, rel=1e-6)
            assert dones[rank] == (terminated or truncated)
            np.testing.assert_allclose(vector.q[rank], env.plant.q[0], atol=1e-14)
            np.testing.assert_allclose(vector.omega[rank], env.plant.omega[0], atol=1e-14)
            np.testing.assert_array_equal(infos[rank]["torque"], info["torque"])
    vector.close()


def test_async_row_termination_autoreset_and_monitor_actual_return(tmp_path):
    vector = FastThreeVecEnv(2, seed=7, monitor_dir=tmp_path, randomized=False,
                             config={"dt": .02, "duration": .055})
    first = {"euler_deg": [10., 0., 0.], "omega_deg_s": [301., 0., 0.], "noise_seed": 1}
    second = {"euler_deg": [0., 0., 0.], "omega_deg_s": [0., 0., 0.], "noise_seed": 2}
    vector.set_options([first, second]); vector.reset()
    returns = np.zeros(2)
    obs, reward, dones, info = vector.step(np.zeros((2, 3)))
    returns += reward
    assert dones.tolist() == [True, False]
    assert info[0]["failed"] and not info[0]["TimeLimit.truncated"]
    assert info[0]["episode"]["l"] == 1
    assert info[0]["episode"]["r"] == pytest.approx(float(reward[0]), rel=1e-6)
    assert "terminal_observation" in info[0] and "terminal_observation" not in info[1]
    assert vector.time.tolist() == [0., .02]
    assert vector.step_count.tolist() == [0, 1]
    assert abs(info[0]["terminal_observation"][3]) > np.radians(300.)
    assert not np.array_equal(obs[0], info[0]["terminal_observation"])
    for _ in range(2):
        _, reward, dones, info = vector.step(np.zeros((2, 3)))
        returns += reward
    assert dones.tolist() == [False, True]
    assert info[1]["TimeLimit.truncated"] and not info[1]["failed"]
    assert info[1]["t"] == pytest.approx(.055)
    assert info[1]["episode"]["l"] == 3
    assert info[1]["episode"]["r"] == pytest.approx(float(returns[1]), abs=1e-7)
    assert vector.time.tolist() == [.04, 0.]
    vector.close()
    with (tmp_path / "monitor_1.monitor.csv").open(encoding="utf-8") as stream:
        header = stream.readline()
        assert "t_start" in json.loads(header[1:])
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and int(rows[0]["l"]) == 3
    assert float(rows[0]["r"]) == pytest.approx(returns[1], abs=1e-7)


def test_seed_pending_options_api_and_repeated_reset_reproducibility():
    vector = FastThreeVecEnv(3, seed=11)
    a = vector.reset()
    vector.seed(11)
    np.testing.assert_array_equal(a, vector.reset())
    vector.set_attr("note", "rank one", indices=1)
    assert vector.get_attr("note", indices=1) == ["rank one"]
    assert len(vector.env_method("config_dict")) == 3
    assert vector.get_attr("render_mode") == [None] * 3
    assert vector.env_is_wrapped(object) == [False] * 3
    assert vector.get_images() == [None] * 3
    opts = {"euler_deg": [0., 0., 0.], "omega_deg_s": [0., 0., 0.],
            "disturbance_bias": 0., "disturbance_amplitude": 0.,
            "noise_theta_deg": 0., "noise_omega_deg": 0.}
    returned = vector.env_method("reset", seed=91, options=opts, indices=[1])
    np.testing.assert_array_equal(returned[0][0], np.zeros(6))
    assert vector.get_attr("step_count", indices=1) == [0]
    vector.step_async(np.zeros((3, 3)))
    with pytest.raises(RuntimeError):
        vector.step_async(np.zeros((3, 3)))
    vector.reset()
    with pytest.raises(RuntimeError):
        vector.step_wait()
    vector.close()


def test_many_auto_resets_match_dummy_gym_and_terminal_noise_streams():
    from stable_baselines3.common.vec_env import DummyVecEnv
    # A fractional final interval verifies exact horizon handling on every
    # episode, while each rank continues its independently seeded reset RNG.
    config = {"duration": .095}
    vector = FastThreeVecEnv(3, seed=110, config=config)
    dummy = DummyVecEnv([lambda: ThreeAxisEnv(True, config=config) for _ in range(3)])
    dummy.seed(110)
    np.testing.assert_array_equal(vector.reset(), dummy.reset())
    rng = np.random.default_rng(701)
    terminal_count = 0
    for _ in range(30):
        actions = rng.uniform(-1.4, 1.4, (3, 3))
        obs, reward, dones, infos = vector.step(actions)
        reference_obs, reference_reward, reference_dones, reference_infos = dummy.step(actions)
        np.testing.assert_array_equal(obs, reference_obs)
        np.testing.assert_array_equal(reward, reference_reward)
        np.testing.assert_array_equal(dones, reference_dones)
        for rank in np.flatnonzero(dones):
            np.testing.assert_array_equal(infos[rank]["terminal_observation"],
                                          reference_infos[rank]["terminal_observation"])
            terminal_count += 1
    assert terminal_count == 18
    vector.close(); dummy.close()


@pytest.mark.parametrize("algorithm", ["PPO", "SAC", "TD3"])
def test_sb3_rollout_smoke_with_fast_vector_env(algorithm):
    """A short developer rollout checks all algorithm-specific VecEnv paths."""
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    torch.set_num_threads(1)
    vector = FastThreeVecEnv(2, seed=8, config={"duration": .06})
    common = dict(policy="MlpPolicy", env=vector, seed=8, device="cpu", verbose=0,
                  policy_kwargs={"net_arch": [16, 16]})
    if algorithm == "PPO":
        model = PPO(**common, n_steps=8, batch_size=8, n_epochs=1)
    else:
        cls = SAC if algorithm == "SAC" else TD3
        model = cls(**common, buffer_size=128, learning_starts=8, batch_size=8,
                    train_freq=(1, "step"), gradient_steps=1)
    model.learn(total_timesteps=32)
    assert model.num_timesteps == 32
    assert model.ep_info_buffer and all(item["l"] == 3 for item in model.ep_info_buffer)
    action, _ = model.predict(vector.reset(), deterministic=True)
    assert action.shape == (2, 3) and np.all(np.isfinite(action))
    vector.close()
