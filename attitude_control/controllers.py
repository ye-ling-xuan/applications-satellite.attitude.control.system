import math
import numpy as np


class PID:
    """Derivative on measured rate; conditional integration prevents windup."""
    def __init__(self, kp, kd, ki=0.0):
        self.kp, self.kd, self.ki = kp, kd, ki
        self.integral = 0.0

    def command(self, episode):
        p, c = episode.plant, episode.config
        trial_integral = self.integral + p.error * c.dt
        raw = self.kp * p.error - self.kd * p.omega + self.ki * trial_integral
        if abs(raw) <= c.max_torque or raw * p.error <= 0:
            self.integral = trial_integral
        return self.kp * p.error - self.kd * p.omega + self.ki * self.integral


def action_command(action, episode, mode):
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size != 1 or not math.isfinite(float(a[0])):
        raise ValueError("action must contain exactly one finite scalar")
    a = float(np.clip(a[0], -1.0, 1.0))
    c, p = episode.config, episode.plant
    if mode == "pure":
        return c.max_torque * a
    if mode == "residual":
        # Fixed PD baseline avoids hidden PID integral state in the RL observation.
        return c.kp * p.error - c.kd * p.omega + c.residual_limit * a
    raise ValueError(f"unknown control mode: {mode}")
