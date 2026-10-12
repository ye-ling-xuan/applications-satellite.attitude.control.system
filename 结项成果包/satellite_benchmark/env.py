"""Gymnasium environment shared by training, PID and policy evaluation."""

from dataclasses import replace
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi) without wrapping the physical state."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class SatelliteEnv(gym.Env):
    """One rigid axis, commanded actuator torque and external disturbance.

    Observation is [measured attitude error in radians, measured angular rate
    in radians/second]. Action is a normalized scalar in [-1, 1]. Reward is a
    dense quadratic cost; an episode always runs for the configured duration
    unless physical divergence is detected. There is no early-success bonus.

    Two independent noise draws are consumed at every observation, even when
    their standard deviations are zero. Identical reset seeds and parameters
    therefore give identical exogenous noise across controllers.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict | None = None):
        super().__init__()
        self.config = EnvConfig(**config) if isinstance(config, dict) else replace(config or EnvConfig())
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.theta = 0.0
        self.omega = 0.0
        self.t = 0.0
        self._steps = 0
        self.inertia = self.config.inertia
        self.disturbance_bias = self.config.disturbance_bias
        self.noise_theta_deg = self.config.noise_theta_deg
        self.noise_omega_deg = self.config.noise_omega_deg
        self._done = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.config
        opts = options or {}
        # Consume initial-state draws consistently even when options override them.
        sampled_theta = self.np_random.uniform(*cfg.init_theta_range_deg)
        sampled_omega = self.np_random.uniform(*cfg.init_omega_range_deg_s)
        theta_deg = float(opts.get("theta_deg", sampled_theta))
        omega_deg_s = float(opts.get("omega_deg_s", sampled_omega))
        if cfg.randomized:
            self.inertia = float(self.np_random.uniform(*cfg.inertia_range))
            self.disturbance_bias = float(self.np_random.uniform(*cfg.disturbance_range))
            self.noise_theta_deg = float(self.np_random.uniform(*cfg.noise_range_deg))
            self.noise_omega_deg = float(self.np_random.uniform(*cfg.noise_range_deg))
        else:
            self.inertia = cfg.inertia
            self.disturbance_bias = cfg.disturbance_bias
            self.noise_theta_deg = cfg.noise_theta_deg
            self.noise_omega_deg = cfg.noise_omega_deg
        self.inertia = float(opts.get("inertia", self.inertia))
        self.disturbance_bias = float(opts.get("disturbance_bias", self.disturbance_bias))
        self.noise_theta_deg = float(opts.get("noise_theta_deg", self.noise_theta_deg))
        self.noise_omega_deg = float(opts.get("noise_omega_deg", self.noise_omega_deg))
        for name, value in (("theta_deg", theta_deg), ("omega_deg_s", omega_deg_s),
                            ("inertia", self.inertia), ("disturbance_bias", self.disturbance_bias),
                            ("noise_theta_deg", self.noise_theta_deg),
                            ("noise_omega_deg", self.noise_omega_deg)):
            if not math.isfinite(value):
                raise ValueError(f"reset option {name} must be finite")
        if self.inertia <= 0 or self.noise_theta_deg < 0 or self.noise_omega_deg < 0:
            raise ValueError("inertia must be positive and noise deviations nonnegative")
        self.theta = math.radians(theta_deg)
        self.omega = math.radians(omega_deg_s)
        self.t = 0.0
        self._steps = 0
        self._done = False
        return self._observation(), self._info(0.0, 0.0, self._disturbance(0.0), False)

    def _observation(self):
        theta_draw, omega_draw = self.np_random.normal(size=2)
        measured_error = self.theta - math.radians(self.config.target_deg)
        measured_error += math.radians(self.noise_theta_deg) * theta_draw
        measured_omega = self.omega + math.radians(self.noise_omega_deg) * omega_draw
        return np.array([wrap_angle(measured_error), measured_omega], dtype=np.float32)

    def _disturbance(self, time: float) -> float:
        cfg = self.config
        disturbance = self.disturbance_bias
        disturbance += cfg.disturbance_amplitude * math.sin(2.0 * math.pi * cfg.disturbance_frequency * time)
        if cfg.pulse_start <= time < cfg.pulse_start + cfg.pulse_duration:
            disturbance += cfg.pulse_torque
        return float(disturbance)

    def _info(self, torque, commanded_torque, disturbance, failed):
        return {"t": float(self.t), "theta_deg": float(math.degrees(self.theta)),
                "omega_deg_s": float(math.degrees(self.omega)), "torque": float(torque),
                "commanded_torque": float(commanded_torque), "disturbance": float(disturbance),
                "inertia": float(self.inertia), "failed": bool(failed)}

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode is finished; call reset() before step().")
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise ValueError("action must contain one finite normalized torque")
        cfg = self.config
        commanded_torque = float(np.clip(values[0], -1.0, 1.0) * cfg.max_torque)
        torque = commanded_torque if abs(commanded_torque) >= cfg.deadzone else 0.0
        disturbance = self._disturbance(self.t)
        next_time = min(cfg.duration, (self._steps + 1) * cfg.dt)
        dt = next_time - self.t
        acceleration = (torque + disturbance) / self.inertia
        self.theta += self.omega * dt + 0.5 * acceleration * dt * dt
        self.omega += acceleration * dt
        self._steps += 1
        self.t = next_time
        true_error = self.theta - math.radians(cfg.target_deg)
        failed = abs(math.degrees(true_error)) > cfg.divergence_angle_deg or abs(math.degrees(self.omega)) > cfg.divergence_omega_deg_s
        terminated = bool(failed)
        truncated = bool(self.t >= cfg.duration - 1e-10)
        self._done = terminated or truncated
        reward = -(cfg.reward_angle_weight * true_error**2
                   + cfg.reward_omega_weight * self.omega**2
                   + cfg.reward_torque_weight * torque**2)
        if failed:
            # Prevent an agent from avoiding future quadratic costs by ending
            # an episode through deliberate divergence.
            reward -= cfg.divergence_penalty
        return self._observation(), float(reward), terminated, truncated, self._info(torque, commanded_torque, disturbance, failed)
