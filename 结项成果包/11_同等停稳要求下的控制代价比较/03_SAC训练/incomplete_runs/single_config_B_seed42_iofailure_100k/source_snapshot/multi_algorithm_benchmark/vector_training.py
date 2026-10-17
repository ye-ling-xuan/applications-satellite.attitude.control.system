"""Batch RK4 training plant, equivalent to the ThreeAxisEnv deployment plant.

Each row owns its reset RNG and a separately seeded sensor-noise stream.  The
training distribution, quaternion convention and normalized actuator mapping
are identical to ThreeAxisEnv.  This avoids eight Python/Gym RK4 calls per
vector step without changing the dynamics or early termination rules.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .envs import euler_to_quaternion, normalize_quaternion, quat_multiply, quaternion_error


class FastThreeVecEnv(VecEnv):
    """SB3 VecEnv with independent episode lifetimes and automatic row resets.

    ``seed`` and ``set_options`` follow SB3 semantics: pending values apply on
    the next explicit vector reset, while automatic resets continue each row's
    RNG.  Monitor-compatible CSV files record actual episode return and length.
    Structural physics (dt, duration and limit) are common across rows.
    """

    def __init__(self, n_envs=8, seed=0, monitor_dir=None, randomized=True, config=None):
        if not isinstance(n_envs, (int, np.integer)) or n_envs < 1:
            raise ValueError("n_envs must be a positive integer")
        self.config = dict(config or {})
        self.dt = float(self.config.get("dt", .02))
        self.duration = float(self.config.get("duration", 20.))
        self.max_torque = float(self.config.get("max_torque", 2.))
        if not all(np.isfinite(x) and x > 0 for x in (self.dt, self.duration, self.max_torque)):
            raise ValueError("dt, duration and torque limit must be positive and finite")
        self.steps = math.ceil(self.duration / self.dt)
        self.randomized = bool(randomized)
        self.render_mode = None
        self._closed = False
        self._ready = False
        self._actions = None
        self._rngs = [np.random.default_rng(int(seed) + rank) for rank in range(n_envs)]
        self._row_attrs = [{} for _ in range(n_envs)]
        self._scenes = [{} for _ in range(n_envs)]
        self.q = np.zeros((n_envs, 4)); self.q[:, 0] = 1.
        self.omega = np.zeros((n_envs, 3))
        self.inertia = np.zeros((n_envs, 3))
        self.bias = np.zeros((n_envs, 3))
        self.amplitude = np.zeros((n_envs, 3))
        self.pulse = np.zeros((n_envs, 3))
        self.noise_angle = np.zeros((n_envs, 3))
        self.noise_rate = np.zeros((n_envs, 3))
        self.pulse_start = np.full(n_envs, 8.)
        self.pulse_duration = np.full(n_envs, .5)
        self.time = np.zeros(n_envs)
        self.step_count = np.zeros(n_envs, dtype=np.int64)
        self.torque = np.zeros((n_envs, 3))
        self.disturbance = np.zeros((n_envs, 3))
        self.failed = np.zeros(n_envs, dtype=bool)
        self._noise = np.empty((self.steps + 1, n_envs, 6))
        self._episode_returns = np.zeros(n_envs)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int64)
        self._angle_weight = float(self.config.get("reward_angle_weight", 6.))
        self._rate_weight = float(self.config.get("reward_omega_weight", .6))
        self._torque_weight = float(self.config.get("reward_torque_weight", .03))
        self._divergence_penalty = float(self.config.get("divergence_penalty", 1000.))
        self._frequency = np.array([.2, .3, .25])
        self._sine_factors = np.array([1., .6, .8])
        self._nominal_inertia = np.array([1.2, 1., .8])
        self._started = time.time()
        self._monitor_streams = []
        self._monitor_writers = []
        super().__init__(n_envs,
                         spaces.Box(np.full(6, -np.inf, np.float32), np.full(6, np.inf, np.float32)),
                         spaces.Box(-1., 1., shape=(3,), dtype=np.float32))
        self.seed(int(seed))
        if monitor_dir is not None:
            directory = Path(monitor_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for rank in range(n_envs):
                stream = (directory / f"monitor_{rank}.monitor.csv").open("w", newline="", encoding="utf-8")
                stream.write("#" + json.dumps({"t_start": self._started, "env_id": "FastThreeVecEnv"}) + "\n")
                writer = csv.DictWriter(stream, fieldnames=["r", "l", "t"])
                writer.writeheader(); stream.flush()
                self._monitor_streams.append(stream); self._monitor_writers.append(writer)

    @staticmethod
    def _vector(value, size=3):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(size, float(array))
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"expected a finite scalar or {size}-vector")
        return array

    def config_dict(self):
        return {"axis": "three", "dt": self.dt, "duration": self.duration,
                "max_torque": self.max_torque, "inertia": [1.2, 1., .8],
                "randomized": self.randomized, "integrator": "RK4",
                "reward_angle_weight": 6., "reward_omega_weight": .6,
                "reward_torque_weight": .03, "divergence_penalty": 1000.,
                "target_quaternion": [1., 0., 0., 0.], **self.config}

    def _reset_row(self, rank, seed=None, options=None):
        if seed is not None:
            self._rngs[rank] = np.random.default_rng(int(seed))
        rng = self._rngs[rank]
        scene = {"euler_deg": rng.uniform(-60., 60., 3),
                 "omega_deg_s": rng.uniform(-5., 5., 3),
                 "noise_seed": int(rng.integers(0, 2**31))}
        if self._row_attrs[rank].get("randomized", self.randomized):
            scene.update(inertia_scale=float(rng.uniform(.8, 1.2)),
                         disturbance_bias=rng.uniform(-.02, .02, 3),
                         disturbance_amplitude=.02,
                         noise_theta_deg=float(rng.uniform(0, .15)),
                         noise_omega_deg=float(rng.uniform(0, .15)))
        scene.update(options or {})
        self._scenes[rank] = dict(scene)
        self.q[rank] = (normalize_quaternion(scene["quaternion"]) if "quaternion" in scene else
                        euler_to_quaternion(np.radians(self._vector(scene["euler_deg"]))))
        self.omega[rank] = np.radians(self._vector(scene["omega_deg_s"]))
        self.inertia[rank] = self._nominal_inertia * self._vector(scene.get("inertia_scale", 1.))
        if np.any(self.inertia[rank] <= 0):
            raise ValueError("positive inertias required")
        for name, key in (("bias", "disturbance_bias"), ("amplitude", "disturbance_amplitude"),
                          ("pulse", "pulse_torque"), ("noise_angle", "noise_theta_deg"),
                          ("noise_rate", "noise_omega_deg")):
            getattr(self, name)[rank] = self._vector(scene.get(key, 0.))
        if np.any(self.noise_angle[rank] < 0) or np.any(self.noise_rate[rank] < 0):
            raise ValueError("sensor deviations cannot be negative")
        self.pulse_start[rank] = float(scene.get("pulse_start", 8.))
        self.pulse_duration[rank] = float(scene.get("pulse_duration", .5))
        self._noise[:, rank] = np.random.default_rng(int(scene["noise_seed"])).normal(size=(self.steps + 1, 6))
        self.time[rank] = 0.; self.step_count[rank] = 0
        self.failed[rank] = False
        self.torque[rank] = 0.; self.disturbance[rank] = 0.
        self._episode_returns[rank] = 0.; self._episode_lengths[rank] = 0
        self.reset_infos[rank] = {"t": 0., "failed": False}

    @property
    def error_vector(self):
        return quaternion_error(self.q)

    def _observation(self, error=None):
        error = self.error_vector if error is None else error
        draws = self._noise[self.step_count, np.arange(self.num_envs)]
        return np.concatenate((error + np.radians(self.noise_angle) * draws[:, :3],
                               self.omega + np.radians(self.noise_rate) * draws[:, 3:]), axis=1).astype(np.float32)

    def reset(self):
        if self._closed:
            raise RuntimeError("environment is closed")
        for rank in range(self.num_envs):
            self._reset_row(rank, self._seeds[rank], self._options[rank])
        self._reset_seeds(); self._reset_options()
        self._actions = None; self._ready = True
        return self._observation().copy()

    def step_async(self, actions):
        if self._closed or not self._ready:
            raise RuntimeError("call reset before stepping an open environment")
        if self._actions is not None:
            raise RuntimeError("a step is already pending")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, 3) or not np.all(np.isfinite(actions)):
            raise ValueError(f"actions must have finite shape ({self.num_envs}, 3)")
        self._actions = actions.copy()

    def _derivative(self, q, w, torque):
        w_dot = (torque - np.cross(w, self.inertia * w)) / self.inertia
        w_quat = np.concatenate((np.zeros((self.num_envs, 1)), w), axis=1)
        return .5 * quat_multiply(q, w_quat), w_dot

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("call step_async before step_wait")
        actions = self._actions; self._actions = None
        self.torque = np.clip(actions, -1, 1) * self.max_torque
        pulse_on = (self.time >= self.pulse_start) & (self.time < self.pulse_start + self.pulse_duration)
        self.disturbance = (self.bias + self.amplitude * self._sine_factors *
                            np.sin(2 * np.pi * self._frequency * self.time[:, None]) +
                            self.pulse * pulse_on[:, None])
        force = self.torque + self.disturbance
        next_time = np.minimum(self.duration, (self.step_count + 1) * self.dt)
        h = (next_time - self.time)[:, None]
        q, w = self.q, self.omega
        a1, b1 = self._derivative(q, w, force)
        a2, b2 = self._derivative(q + .5*h*a1, w + .5*h*b1, force)
        a3, b3 = self._derivative(q + .5*h*a2, w + .5*h*b2, force)
        a4, b4 = self._derivative(q + h*a3, w + h*b3, force)
        self.q = normalize_quaternion(q + h*(a1 + 2*a2 + 2*a3 + a4)/6)
        self.omega = w + h*(b1 + 2*b2 + 2*b3 + b4)/6
        self.step_count += 1; self.time = next_time
        error = self.error_vector
        error_deg = np.degrees(np.linalg.norm(error, axis=1))
        rate_deg_s = np.degrees(np.linalg.norm(self.omega, axis=1))
        terminated = rate_deg_s > 300.
        truncated = self.time >= self.duration - 1e-10
        self.failed = terminated.copy()
        rewards = -(self._angle_weight*np.sum(error**2, axis=1) +
                    self._rate_weight*np.sum(self.omega**2, axis=1) +
                    self._torque_weight*np.sum(self.torque**2, axis=1))
        rewards -= terminated * self._divergence_penalty
        dones = terminated | truncated
        self._episode_returns += rewards; self._episode_lengths += 1
        observations = self._observation(error)
        infos = [{"t": float(self.time[rank]), "error_deg": float(error_deg[rank]),
                  "rate_deg_s": float(rate_deg_s[rank]), "torque": self.torque[rank].copy(),
                  "disturbance": self.disturbance[rank].copy(), "failed": bool(terminated[rank]),
                  "TimeLimit.truncated": bool(truncated[rank] and not terminated[rank])}
                 for rank in range(self.num_envs)]
        for rank in np.flatnonzero(dones):
            rank = int(rank)
            episode = {"r": float(self._episode_returns[rank]),
                       "l": int(self._episode_lengths[rank]),
                       "t": round(time.time() - self._started, 6)}
            infos[rank]["episode"] = episode
            infos[rank]["terminal_observation"] = observations[rank].copy()
            if self._monitor_writers:
                self._monitor_writers[rank].writerow(episode)
                self._monitor_streams[rank].flush()
            self._reset_row(rank)
        if np.any(dones):
            # Only terminal rows change; continuing rows retain their sampled
            # observation, including the correct episode noise index.
            observations[dones] = self._observation()[dones]
        return observations.copy(), rewards.astype(np.float32), dones.copy(), infos

    def get_attr(self, attr_name, indices=None):
        values = []
        row_fields = {"q", "omega", "inertia", "bias", "amplitude", "pulse", "noise_angle",
                      "noise_rate", "pulse_start", "pulse_duration", "time", "step_count", "torque",
                      "disturbance", "failed", "error_vector"}
        for rank in self._get_indices(indices):
            if attr_name in self._row_attrs[rank]:
                value = self._row_attrs[rank][attr_name]
            elif attr_name == "scene":
                value = dict(self._scenes[rank])
            elif attr_name == "np_random":
                value = self._rngs[rank]
            elif attr_name in row_fields:
                value = getattr(self, attr_name)[rank]
            else:
                value = getattr(self, attr_name)
            values.append(value.copy() if isinstance(value, np.ndarray) else value)
        return values

    def set_attr(self, attr_name, value, indices=None):
        row_fields = {"q", "omega", "inertia", "bias", "amplitude", "pulse", "noise_angle",
                      "noise_rate", "pulse_start", "pulse_duration", "time", "step_count", "torque",
                      "disturbance", "failed"}
        # Changing the horizon or actuator on selected rows would invalidate
        # the declared shared plant and sensor-noise array dimensions.
        if attr_name in ("dt", "duration", "max_torque", "steps", "config"):
            raise AttributeError(f"{attr_name} is structural; configure it at construction")
        for rank in self._get_indices(indices):
            if attr_name in row_fields:
                getattr(self, attr_name)[rank] = value
            else:
                self._row_attrs[rank][attr_name] = value

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        ranks = list(self._get_indices(indices))
        if method_name == "config_dict":
            return [{**self.config_dict(), "randomized": self._row_attrs[rank].get("randomized", self.randomized)}
                    for rank in ranks]
        if method_name == "reset":
            if not self._ready and set(ranks) != set(range(self.num_envs)):
                raise RuntimeError("initialize every row with vector reset before resetting selected rows")
            seed = method_kwargs.get("seed")
            options = method_kwargs.get("options")
            for rank in ranks:
                self._reset_row(rank, seed, options)
            self._ready = True
            obs = self._observation()
            return [(obs[rank].copy(), dict(self.reset_infos[rank])) for rank in ranks]
        if method_name in ("render", "close"):
            return [None for _ in ranks]
        raise NotImplementedError(f"row method {method_name!r} is not exposed")

    def env_is_wrapped(self, wrapper_class, indices=None):
        # Monitor output is implemented here, rather than by Gym wrappers.
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return [None for _ in range(self.num_envs)]

    def close(self):
        if not self._closed:
            for stream in self._monitor_streams:
                stream.close()
            self._closed = True
            self._actions = None
