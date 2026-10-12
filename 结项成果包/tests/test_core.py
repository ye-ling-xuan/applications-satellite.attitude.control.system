"""Checks for physically meaningful dynamics, fairness and metric definitions."""

import numpy as np
import pytest

from satellite_benchmark import EnvConfig, PIDConfig, PIDController, SatelliteEnv, compute_metrics


def make_trace(t, theta, omega=None, failed=None, torque=None):
    n = len(t)
    return {"t": np.asarray(t), "theta_deg": np.asarray(theta),
            "omega_deg_s": np.zeros(n) if omega is None else np.asarray(omega),
            "torque": np.zeros(n) if torque is None else np.asarray(torque),
            "failed": np.zeros(n, dtype=bool) if failed is None else np.asarray(failed)}


def test_exact_constant_acceleration_and_action_mapping():
    cfg = EnvConfig(dt=0.1, inertia=2.0, disturbance_bias=0.1)
    env = SatelliteEnv(cfg)
    env.reset(seed=1, options={"theta_deg": 0.0, "omega_deg_s": 0.0})
    _, _, terminated, truncated, info = env.step([1.0])
    acceleration = (cfg.max_torque + cfg.disturbance_bias) / cfg.inertia
    assert info["theta_deg"] == pytest.approx(np.degrees(0.5 * acceleration * cfg.dt**2))
    assert info["omega_deg_s"] == pytest.approx(np.degrees(acceleration * cfg.dt))
    assert info["commanded_torque"] == cfg.max_torque
    assert not terminated and not truncated


def test_deadzone_affects_applied_torque_and_success_does_not_end_episode():
    env = SatelliteEnv(EnvConfig(deadzone=0.1))
    env.reset(seed=1, options={"theta_deg": 0.0, "omega_deg_s": 0.0})
    _, _, terminated, truncated, info = env.step([0.1])
    assert info["torque"] == 0
    assert info["commanded_torque"] == pytest.approx(0.05)
    assert not terminated and not truncated


def test_noise_reproducibility_and_noise_draws_do_not_depend_on_action():
    cfg = EnvConfig(noise_theta_deg=0.3, noise_omega_deg=0.2)
    envs = [SatelliteEnv(cfg), SatelliteEnv(cfg)]
    observations = [env.reset(seed=81, options={"theta_deg": 20, "omega_deg_s": 0})[0] for env in envs]
    np.testing.assert_array_equal(*observations)
    for _ in range(20):
        observations, infos = [], []
        for env, action in zip(envs, ([-0.3], [0.7])):
            obs, _, _, _, info = env.step(action)
            observations.append(obs)
            infos.append(info)
        errors = [np.asarray(obs, dtype=float) - np.radians([info["theta_deg"], info["omega_deg_s"]])
                  for obs, info in zip(observations, infos)]
        np.testing.assert_allclose(errors[0], errors[1], atol=5e-8)


def test_domain_randomization_repeats_with_seed_and_honors_inertia_override():
    cfg = EnvConfig(randomized=True, inertia_range=(0.7, 1.5),
                    disturbance_range=(-0.02, 0.02), noise_range_deg=(0.1, 0.5))
    env = SatelliteEnv(cfg)
    obs1, info1 = env.reset(seed=21)
    obs2, info2 = env.reset(seed=21)
    np.testing.assert_array_equal(obs1, obs2)
    assert info1 == info2
    assert 0.7 <= info1["inertia"] <= 1.5
    _, info3 = env.reset(seed=21, options={"inertia": 1.8})
    assert info3["inertia"] == 1.8


def test_divergence_ends_episode_and_duration_can_have_fractional_final_step():
    env = SatelliteEnv()
    env.reset(seed=1, options={"theta_deg": 169.9, "omega_deg_s": 10})
    _, _, terminated, _, info = env.step([0])
    assert terminated and info["failed"]
    with pytest.raises(RuntimeError):
        env.step([0])
    cfg = EnvConfig(dt=0.07, duration=0.2, hold_seconds=0.1)
    env = SatelliteEnv(cfg)
    env.reset(seed=1, options={"theta_deg": 0, "omega_deg_s": 0})
    for _ in range(3):
        _, _, terminated, truncated, info = env.step([0])
    assert not terminated and truncated
    assert info["t"] == pytest.approx(0.2)


def test_pulse_onset_is_not_delayed_by_accumulated_float_time_error():
    cfg = EnvConfig(pulse_torque=0.1, pulse_start=8, pulse_duration=0.5)
    env = SatelliteEnv(cfg)
    env.reset(seed=1, options={"theta_deg": 0, "omega_deg_s": 0})
    infos = []
    for _ in range(171):
        _, _, _, _, info = env.step([0])
        infos.append(info)
    assert infos[159]["t"] == 8
    assert infos[159]["disturbance"] == 0  # Applied to (7.95, 8.00].
    assert infos[160]["disturbance"] == 0.1  # Applied to (8.00, 8.05].
    assert sum(info["disturbance"] != 0 for info in infos) == 10


def test_pid_integral_limit_and_conditional_anti_windup():
    saturated = PIDController(PIDConfig(kp=2, ki=1, kd=0, integral_limit=0.2))
    for _ in range(200):
        action = saturated.act([1.0, 0.0])
    assert action[0] == -1
    assert saturated.integral == 0  # Actuator saturation does not accumulate hidden demand.
    integral_only = PIDController(PIDConfig(kp=0, ki=0.1, kd=0, integral_limit=0.2))
    for _ in range(200):
        integral_only.act([0.1, 0.0])
    assert integral_only.integral == pytest.approx(-0.2)
    integral_only.reset()
    assert integral_only.integral == 0


def test_settling_uses_final_continuous_suffix_and_not_first_crossing():
    cfg = EnvConfig(duration=10, hold_seconds=3)
    # First arrival at t=1 is followed by exit at t=2; only t=4 onwards settles.
    trace = make_trace(np.arange(11), [30, 0.5, 2, 1.2, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0])
    metrics = compute_metrics(trace, cfg)
    assert metrics["success"]
    assert metrics["settling_time_s"] == 4
    assert metrics["overshoot_deg"] == 0  # Initial 30 degrees is not overshoot.


def test_success_requires_rate_bound_full_duration_and_no_failure():
    cfg = EnvConfig(duration=10, hold_seconds=3)
    trace = make_trace(np.arange(11), np.zeros(11), omega=np.ones(11))
    assert not compute_metrics(trace, cfg)["success"]
    short = make_trace(np.arange(8), np.zeros(8))
    assert not compute_metrics(short, cfg)["success"]
    failed = make_trace(np.arange(11), np.zeros(11), failed=[False]*10+[True])
    assert not compute_metrics(failed, cfg)["success"]
    # Being in tolerance for fewer than the required seconds is insufficient.
    late = make_trace(np.arange(11), [2]*8+[0]*3)
    metrics = compute_metrics(late, cfg)
    assert not metrics["success"] and metrics["settling_time_s"] is None


def test_overshoot_crossing_and_torque_cost_use_correct_direction_and_intervals():
    cfg = EnvConfig(duration=1, hold_seconds=0.5)
    trace = make_trace([0, 0.5, 1], [30, 5, -2], torque=[0, 0.5, 0.2])
    metrics = compute_metrics(trace, cfg)
    assert metrics["overshoot_deg"] == 2
    assert metrics["control_cost"] == pytest.approx(0.5*0.5**2 + 0.5*0.2**2)
    negative = make_trace([0, 0.5, 1], [-30, -5, 3])
    assert compute_metrics(negative, cfg)["overshoot_deg"] == 3
    at_target = make_trace([0, 0.5, 1], [0, 0, 0])
    assert compute_metrics(at_target, cfg)["overshoot_deg"] is None
