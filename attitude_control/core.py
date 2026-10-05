"""Shared dynamics, scenarios, observations, and stage cost (no Gym dependency)."""
from dataclasses import dataclass
import math
import numpy as np
from .config import Config

OBSERVATION_VERSION = "error_pi-rate_scale-previous_torque-v1"
DYNAMICS_VERSION = "single-axis-zoh-exact-v1"
REWARD_VERSION = "normalized-quadratic-slew-times-dt-v1"


@dataclass(frozen=True)
class Scenario:
    name: str = "nominal"
    angle_deg: float = 30.0
    rate_deg_s: float = 0.0
    inertia: float = 1.0
    bias_torque: float = 0.0
    pulse_torque: float = 0.0
    noise_std: float = 0.0  # external torque noise, not measurement noise
    seed: int = 0

    def __post_init__(self):
        numbers = (self.angle_deg, self.rate_deg_s, self.inertia, self.bias_torque,
                   self.pulse_torque, self.noise_std)
        if not all(math.isfinite(v) for v in numbers) or self.inertia <= 0 or self.noise_std < 0:
            raise ValueError("invalid scenario parameters")

    def disturbances(self, config):
        t = np.arange(config.steps) * config.dt
        pulse = self.pulse_torque * ((t >= 3.0) & (t < 3.5))
        noise = np.random.default_rng(self.seed).normal(0, self.noise_std, config.steps)
        return self.bias_torque + pulse + noise


class Plant:
    def __init__(self, config: Config, scenario: Scenario):
        self.config = config
        self.inertia = scenario.inertia
        self.theta = math.radians(scenario.angle_deg)
        self.omega = math.radians(scenario.rate_deg_s)

    @property
    def error(self):
        # Local single-axis maneuver, deliberately no angle wrapping.
        return math.radians(self.config.target_deg) - self.theta

    def step(self, command, disturbance=0.0):
        if not math.isfinite(command) or not math.isfinite(disturbance):
            raise ValueError("torques must be finite")
        limit, dt = self.config.max_torque, self.config.dt
        torque = float(min(limit, max(-limit, command)))
        acceleration = (torque + disturbance) / self.inertia
        # Exact update for constant torque over the sample interval.
        self.theta += self.omega * dt + 0.5 * acceleration * dt * dt
        self.omega += acceleration * dt
        return torque


def observation(plant, previous_torque):
    return np.array([plant.error / math.pi,
                     plant.omega / math.radians(plant.config.rate_scale_deg_s),
                     previous_torque / plant.config.max_torque], dtype=np.float32)


def stage_cost(plant, torque, previous_torque):
    c = plant.config
    return ((plant.error / math.radians(c.angle_scale_deg)) ** 2
            + c.rate_weight * (plant.omega / math.radians(c.rate_scale_deg_s)) ** 2
            + c.torque_weight * (torque / c.max_torque) ** 2
            + c.slew_weight * ((torque - previous_torque) / c.max_torque) ** 2)


class Episode:
    def __init__(self, config, scenario):
        self.config = config
        if abs(scenario.angle_deg - config.target_deg) > config.failure_angle_deg:
            raise ValueError("initial angle must be inside the failure boundary")
        self.plant = Plant(config, scenario)
        self.disturbances = scenario.disturbances(config)
        self.previous_torque = 0.0
        self.index = 0
        self.terminated = False
        self.truncated = False

    def step(self, command):
        if self.terminated or self.truncated:
            raise RuntimeError("episode has ended; reset before stepping")
        disturbance = float(self.disturbances[self.index])
        torque = self.plant.step(command, disturbance)
        cost = stage_cost(self.plant, torque, self.previous_torque) * self.config.dt
        self.previous_torque = torque
        self.index += 1
        self.terminated = abs(self.plant.error) > math.radians(self.config.failure_angle_deg)
        self.truncated = self.index >= self.config.steps and not self.terminated
        reward = -cost - (100.0 if self.terminated else 0.0)
        return observation(self.plant, torque), reward, self.terminated, self.truncated, {
            "torque": torque, "disturbance": disturbance, "stage_cost_integral": cost,
        }
