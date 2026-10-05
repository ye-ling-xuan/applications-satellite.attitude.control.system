from dataclasses import replace
import numpy as np
import pytest
gymnasium = pytest.importorskip("gymnasium")
from attitude_control.config import Config
from attitude_control.core import Episode, Scenario
from attitude_control.controllers import action_command
from attitude_control.env import AttitudeEnv
from attitude_control.experiments import contract, check_contract, create_output


def test_seed_replays_initial_state_and_trajectory():
    env = AttitudeEnv()
    def trace():
        first, _ = env.reset(seed=42)
        values = [first]
        for _ in range(50):
            values.append(env.step([0])[0])
        return np.asarray(values)
    np.testing.assert_array_equal(trace(), trace())


@pytest.mark.parametrize("mode", ["pure", "residual"])
def test_training_and_evaluation_share_exact_transition(mode):
    scenario = Scenario(angle_deg=30, rate_deg_s=10, noise_std=0.01, seed=5)
    env = AttitudeEnv(mode=mode)
    env.reset(options={"scenario": scenario})
    reference = Episode(Config(), scenario)
    for action in [[0.5], [-1], [0], [1], [0.3]]:
        expected = reference.step(action_command(action, reference, mode))
        actual = env.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert env.observation_space.contains(actual[0])


def test_time_limit_is_truncation_and_failure_is_termination():
    env = AttitudeEnv(Config(dt=1, duration=1), mode="pure")
    env.reset(options={"scenario": Scenario(angle_deg=0)})
    _, _, terminated, truncated, _ = env.step([0])
    assert truncated and not terminated
    env.reset(options={"scenario": Scenario(angle_deg=89, rate_deg_s=10)})
    _, _, terminated, truncated, _ = env.step([0])
    assert terminated and not truncated


def test_model_config_mismatch_is_rejected():
    metadata = {"mode": "residual", "contract": contract(Config())}
    check_contract(metadata, Config(), "residual")
    with pytest.raises(ValueError):
        check_contract(metadata, replace(Config(), kp=6), "residual")
    with pytest.raises(ValueError):
        check_contract(metadata, Config(), "pure")


def test_existing_results_cannot_be_overwritten(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        create_output(tmp_path)


def test_residual_pid_integral_is_not_silently_ignored():
    with pytest.raises(ValueError, match="ki must be zero"):
        AttitudeEnv(replace(Config(), ki=0.1), mode="residual")


def test_randomization_is_centered_on_configured_target_and_inertia():
    env = AttitudeEnv(replace(Config(), target_deg=100, inertia=10))
    env.reset(seed=42)
    assert abs(np.degrees(env.episode.plant.error)) <= 60
    assert 7 <= env.episode.plant.inertia <= 13
