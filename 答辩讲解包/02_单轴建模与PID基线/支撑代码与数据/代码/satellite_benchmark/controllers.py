"""PID reference controller with output saturation and integral anti-windup."""

from dataclasses import replace
import numpy as np

from .config import PIDConfig


class PIDController:
    """Torque = -Kp*error - Kd*measured_rate + Ki*integral(-error).

    Integration is bounded and conditionally accepted: when the proposed output
    saturates, integration is allowed only if it moves torque toward the allowed
    range. Derivative uses measured rate, avoiding derivative kick and giving
    PID the same observations as a learned policy.
    """

    def __init__(self, config: PIDConfig | dict | None = None, max_torque: float = 0.5, dt: float = 0.05):
        self.config = PIDConfig(**config) if isinstance(config, dict) else replace(config or PIDConfig())
        if not np.isfinite(max_torque) or max_torque <= 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("max_torque and dt must be finite and positive")
        self.max_torque = float(max_torque)
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.integral = 0.0

    def act(self, observation):
        values = np.asarray(observation, dtype=float).reshape(-1)
        if values.size != 2 or not np.all(np.isfinite(values)):
            raise ValueError("PID observation must contain finite error and angular rate")
        error, rate = values
        cfg = self.config
        candidate = float(np.clip(self.integral - error * self.dt, -cfg.integral_limit, cfg.integral_limit))
        base = -cfg.kp * error - cfg.kd * rate
        candidate_torque = base + cfg.ki * candidate
        integral_delta_torque = cfg.ki * (candidate - self.integral)
        inside_range = abs(candidate_torque) <= self.max_torque
        drives_inward = (candidate_torque > self.max_torque and integral_delta_torque < 0) or (candidate_torque < -self.max_torque and integral_delta_torque > 0)
        if inside_range or drives_inward:
            self.integral = candidate
        torque = np.clip(base + cfg.ki * self.integral, -self.max_torque, self.max_torque)
        return np.array([torque / self.max_torque], dtype=np.float32)
