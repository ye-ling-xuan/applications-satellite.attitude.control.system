"""Explicit, versioned PPO training tasks; physics and evaluation metrics stay shared."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .core import Scenario, observation


@dataclass(frozen=True)
class Task:
    profile: str = "v1"
    torque_penalty: float = 0.5

    def __post_init__(self):
        if self.profile not in ("v1", "precision_nominal", "precision_robust"):
            raise ValueError("unknown task profile")
        if not math.isfinite(self.torque_penalty) or self.torque_penalty < 0:
            raise ValueError("torque penalty must be finite and nonnegative")
        if self.profile == "v1" and self.torque_penalty != 0.5:
            raise ValueError("v1 reward does not use this torque penalty parameter")

    def to_dict(self):
        return {**asdict(self), "version": "explicit-ppo-task-v2"}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") not in ("explicit-ppo-task-v1", "explicit-ppo-task-v2"):
            raise ValueError("unknown saved task version")
        if data["version"] == "explicit-ppo-task-v1":
            return cls(profile=data["profile"])
        return cls(profile=data["profile"], torque_penalty=data["torque_penalty"])

    @property
    def observation_angle_scale_deg(self):
        return 180.0 if self.profile == "v1" else 40.0

    def obs(self, episode):
        obs = observation(episode.plant, episode.previous_torque)
        obs[0] *= 180.0 / self.observation_angle_scale_deg
        return obs

    def sample(self, config, rng, randomize):
        precision = self.profile != "v1"
        robust = randomize and self.profile != "precision_nominal"
        bound = min(40.0 if precision else 60.0, config.failure_angle_deg)
        return Scenario(
            angle_deg=config.target_deg + float(rng.uniform(-bound, bound)),
            rate_deg_s=0.0 if self.profile == "precision_nominal" else float(rng.uniform(-10, 10)),
            inertia=config.inertia * float(rng.uniform(0.7, 1.3)) if robust else config.inertia,
            bias_torque=float(rng.uniform(-0.03, 0.03)) if robust else 0.0,
            seed=int(rng.integers(0, 2**31)))

    def reward(self, episode, applied_torque, base_reward):
        if self.profile == "v1":
            return base_reward
        p, c = episode.plant, episode.config
        # Matches archived quadratic coefficients; error = target - theta.
        reward = (-5.0 * p.error**2 - p.omega**2 - self.torque_penalty * applied_torque**2
                  + 2.0 * p.error * p.omega)
        settled = (abs(p.error) <= math.radians(c.angle_tolerance_deg)
                   and abs(p.omega) <= math.radians(c.rate_tolerance_deg_s))
        if settled:
            reward += 2.0
        if episode.terminated:
            reward -= 50.0
        return float(reward)


class PolicyAdapter:
    """Evaluation supplies canonical observations; transform using the saved task."""
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def predict(self, observation, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32).copy()
        if obs.shape[-1] != 3:
            raise ValueError("canonical observation must have three entries")
        obs[..., 0] *= 180.0 / self.task.observation_angle_scale_deg
        return self.model.predict(obs, deterministic=deterministic)
