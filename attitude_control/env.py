import gymnasium as gym
import numpy as np
from .config import Config
from .core import Episode, Scenario, observation
from .controllers import action_command
from .tasks import Task


class AttitudeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, mode="residual", randomize=True, task=None):
        super().__init__()
        self.config = config or Config()
        if mode not in ("pure", "residual"):
            raise ValueError("mode must be pure or residual")
        if mode == "residual" and self.config.ki != 0:
            raise ValueError("residual mode uses PD; ki must be zero")
        self.mode = mode
        self.randomize = randomize
        self.task = task or Task()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        # Rates/errors can exceed normalization scales, so no false [-1, 1] bound.
        self.observation_space = gym.spaces.Box(
            np.array([-np.inf, -np.inf, -1.0], np.float32),
            np.array([np.inf, np.inf, 1.0], np.float32), dtype=np.float32)
        self.episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "scenario" in options:
            scenario = options["scenario"]
            if isinstance(scenario, dict):
                scenario = Scenario(**scenario)
        else:
            scenario = self.task.sample(self.config, self.np_random, self.randomize)
        self.episode = Episode(self.config, scenario)
        return self.task.obs(self.episode), {}

    def step(self, action):
        if self.episode is None:
            raise RuntimeError("reset before stepping")
        _, reward, terminated, truncated, info = self.episode.step(action_command(action, self.episode, self.mode))
        return self.task.obs(self.episode), self.task.reward(self.episode, info["torque"], reward), terminated, truncated, info
