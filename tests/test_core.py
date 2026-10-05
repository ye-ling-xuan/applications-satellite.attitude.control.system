import math
import numpy as np
import pytest
from attitude_control.config import Config
from attitude_control.core import Episode, Plant, Scenario, observation
from attitude_control.controllers import PID, action_command
from attitude_control.simulation import rollout


def test_exact_constant_torque_and_common_saturation():
    p = Plant(Config(dt=0.1), Scenario(angle_deg=0, inertia=2))
    assert p.step(100) == 2
    assert p.omega == pytest.approx(0.1)
    assert p.theta == pytest.approx(0.005)


def test_zero_torque_conserves_angular_velocity():
    p = Plant(Config(), Scenario(rate_deg_s=10))
    old_rate = p.omega
    old_angle = p.theta
    for _ in range(500):
        p.step(0)
    assert p.omega == old_rate
    assert p.theta == pytest.approx(old_angle + old_rate * 10)


def test_pid_has_no_initial_derivative_kick():
    episode = Episode(Config(), Scenario(angle_deg=30))
    assert PID(3, 1, 0).command(episode) == pytest.approx(-3 * math.pi / 6)


def test_pid_integral_does_not_wind_up():
    episode = Episode(Config(), Scenario(angle_deg=60))
    pid = PID(10, 1, 1)
    for _ in range(100):
        pid.command(episode)
    assert pid.integral == 0


def test_zero_residual_is_pd():
    episode = Episode(Config(), Scenario(angle_deg=30, rate_deg_s=10))
    assert action_command([0], episode, "residual") == pytest.approx(PID(3, 3).command(episode))


def test_disturbance_is_paired_and_reproducible():
    scenario = Scenario(seed=42, noise_std=0.1, pulse_torque=0.3)
    assert np.array_equal(scenario.disturbances(Config()), scenario.disturbances(Config()))


def test_rollout_aligns_state_and_torque_samples():
    result, trace = rollout(Config(), Scenario(angle_deg=0), lambda ep, obs: 0)
    assert result["success"]
    assert result["settling_time_s"] == 0
    assert len(trace["angle_deg"]) == len(trace["torque_nm"]) + 1
    assert trace["time_s"][-1] == 10


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_action_is_rejected(value):
    with pytest.raises(ValueError):
        action_command([value], Episode(Config(), Scenario()), "pure")


def test_cannot_step_after_horizon():
    episode = Episode(Config(dt=1, duration=1), Scenario(angle_deg=0))
    episode.step(0)
    assert episode.truncated and not episode.terminated
    with pytest.raises(RuntimeError):
        episode.step(0)


def test_observation_uses_error_and_last_applied_torque():
    episode = Episode(Config(target_deg=10), Scenario(angle_deg=30))
    assert observation(episode.plant, 2) == pytest.approx([-20 / 180, 0, 1])


@pytest.mark.parametrize("kwargs", [{"inertia": 0}, {"dt": 0}, {"kp": -1}, {"duration": 1.03}, {"residual_limit": 3}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
