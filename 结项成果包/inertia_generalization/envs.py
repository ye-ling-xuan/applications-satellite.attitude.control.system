"""Pure-inertia V3 interfaces over the unchanged, verified V2 batch plant.

Initial-state, inertia and sensor-seed generators are separate streams. Changing
the training mode therefore cannot shift the sequence of sampled initial states.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from multi_algorithm_benchmark.envs import BatchSatelliteEnv as _BatchSatelliteEnv, axis_name


MODES = ("nominal", "randomized")
ZERO_NUISANCES = ("disturbance_bias", "disturbance_amplitude", "pulse_torque",
                  "noise_theta_deg", "noise_omega_deg")


def mode_name(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return mode


def check_physical_inertia(axis, scale):
    """Positive moments, and nonnegative triangle inequalities for rigid bodies."""
    axis = axis_name(axis)
    dimensions = 1 if axis == "single" else 3
    value = np.asarray(scale, dtype=float)
    if value.ndim == 0:
        value = np.full(dimensions, float(value))
    if value.shape != (dimensions,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("inertia_scale must be a positive finite scalar or axis-sized vector")
    inertia = value * (np.array([1.]) if dimensions == 1 else np.array([1.2, 1., .8]))
    if dimensions == 3 and np.any(2 * inertia > inertia.sum() + 1e-12):
        raise ValueError("principal inertias violate rigid-body triangle inequalities")
    return inertia


def check_pure_scene(axis, scene):
    check_physical_inertia(axis, scene.get("inertia_scale", 1.))
    for key in ZERO_NUISANCES:
        value = np.asarray(scene.get(key, 0.), dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value != 0):
            raise ValueError(f"V3 pure-inertia experiments require {key}=0")
    return scene


class BatchSatelliteEnv(_BatchSatelliteEnv):
    """V2 exact/RK4 propagation with V3 physical-inertia and nuisance checks."""
    def __init__(self, axis, scenarios, config=None):
        scenes = list(scenarios)
        for scene in scenes:
            check_pure_scene(axis, scene)
        super().__init__(axis, scenes, config)


class EpisodeSampler:
    """Matched initial-state sequence, independent of mode and inertia draws."""
    def __init__(self, axis, mode="nominal", seed=None):
        self.axis = axis_name(axis)
        self.mode = mode_name(mode)
        self.reseed(seed)

    def reseed(self, seed=None):
        sequence = np.random.SeedSequence(seed)
        initial, inertia, noise = sequence.spawn(3)
        self.initial_rng = np.random.default_rng(initial)
        self.inertia_rng = np.random.default_rng(inertia)
        self.noise_rng = np.random.default_rng(noise)
        self.seed = int(sequence.entropy)
        self.episode_index = 0

    def sample(self, options=None):
        dimensions = 1 if self.axis == "single" else 3
        angle = self.initial_rng.uniform(-60., 60., dimensions)
        rate = self.initial_rng.uniform(-5., 5., dimensions)
        # Both modes consume this separate stream once, so its state is matched.
        rho_draw = float(self.inertia_rng.uniform(.7, 1.3))
        scene = {"inertia_scale": rho_draw if self.mode == "randomized" else 1.,
                 "noise_seed": int(self.noise_rng.integers(0, 2**31)),
                 "omega_deg_s": float(rate[0]) if dimensions == 1 else rate.tolist(),
                 **{key: 0. for key in ZERO_NUISANCES}}
        scene["theta_deg" if dimensions == 1 else "euler_deg"] = float(angle[0]) if dimensions == 1 else angle.tolist()
        scene.update(deepcopy(options or {}))
        check_pure_scene(self.axis, scene)
        record = {"episode_index": self.episode_index, "stream_seed": self.seed,
                  "mode": self.mode, **deepcopy(scene)}
        self.episode_index += 1
        return scene, record


class InertiaSatelliteEnv(gym.Env):
    """Both axis interfaces use exactly the batch plant used for deployment."""
    metadata = {"render_modes": []}

    def __init__(self, axis="single", mode="nominal", config=None):
        super().__init__()
        self.axis = axis_name(axis)
        self.mode = mode_name(mode)
        self.dimensions = 1 if self.axis == "single" else 3
        self.config = dict(config or {})
        self.dt = float(self.config.get("dt", .05 if self.axis == "single" else .02))
        self.duration = float(self.config.get("duration", 20.))
        self.max_torque = float(self.config.get("max_torque", .5 if self.axis == "single" else 2.))
        self.action_space = spaces.Box(-1., 1., shape=(self.dimensions,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2*self.dimensions,), dtype=np.float32)
        self.sampler = EpisodeSampler(self.axis, self.mode)
        self.plant = None
        self.scene = None
        self.episode_record = None

    def config_dict(self):
        return {"axis": self.axis, "mode": self.mode, "dt": self.dt,
                "duration": self.duration, "max_torque": self.max_torque,
                "inertia": [1.] if self.axis == "single" else [1.2, 1., .8],
                "inertia_scale_distribution": "fixed 1" if self.mode == "nominal" else "Uniform[0.7,1.3] per episode",
                "initial_angle_range_deg": [-60., 60.], "initial_rate_range_deg_s": [-5., 5.],
                "independent_streams": ["initial_state", "inertia", "noise_seed"],
                "integrator": "exact_constant_acceleration" if self.axis == "single" else "RK4",
                "reward_angle_weight": 6., "reward_omega_weight": .6,
                "reward_torque_weight": .03, "divergence_penalty": 1000.,
                **{key: 0. for key in ZERO_NUISANCES}, **self.config}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sampler.reseed(seed)
        self.scene, self.episode_record = self.sampler.sample(options)
        self.plant = BatchSatelliteEnv(self.axis, [self.scene], self.config)
        return self.plant.observation()[0], {"t": 0., "failed": False,
                                           "inertia_scale": deepcopy(self.scene["inertia_scale"]),
                                           "episode_initial_state": deepcopy(self.episode_record)}

    def step(self, action):
        if self.plant is None or self.plant.failed[0] or self.plant.time >= self.duration-1e-10:
            raise RuntimeError("call reset before stepping a completed episode")
        obs, reward, terminated, truncated, info = self.plant.step(np.asarray(action).reshape(1, self.dimensions))
        values = {key: (value[0].copy() if isinstance(value[0], np.ndarray) else value[0])
                  if isinstance(value, np.ndarray) else value for key, value in info.items()}
        values["inertia_scale"] = deepcopy(self.scene["inertia_scale"])
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), values


def make_env(axis="single", mode="nominal", config=None):
    return InertiaSatelliteEnv(axis, mode, config)
