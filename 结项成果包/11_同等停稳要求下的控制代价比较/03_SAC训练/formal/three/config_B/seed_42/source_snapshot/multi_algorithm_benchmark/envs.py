"""Unified plants for deployment comparisons; observations never expose true state.

Single axis preserves the original package's exact constant-acceleration plant.
Three axes use RK4, diagonal inertia and scalar-first body-to-inertial quaternions.
All actions are normalized; torque limits are .5 and 2 N m, respectively.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from satellite_benchmark.config import EnvConfig
from satellite_benchmark.env import SatelliteEnv


def axis_name(axis):
    aliases = {"single": "single", "single_axis": "single", 1: "single",
               "three": "three", "three_axis": "three", 3: "three"}
    if axis not in aliases:
        raise ValueError("axis must be single or three")
    return aliases[axis]


def quat_multiply(q, r):
    q, r = np.asarray(q, float), np.asarray(r, float)
    scalar = q[..., :1]*r[..., :1] - np.sum(q[..., 1:]*r[..., 1:], axis=-1, keepdims=True)
    vector = q[..., :1]*r[..., 1:] + r[..., :1]*q[..., 1:] + np.cross(q[..., 1:], r[..., 1:])
    return np.concatenate([scalar, vector], axis=-1)


def normalize_quaternion(q):
    q = np.asarray(q, float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(q)) or np.any(norm <= 1e-15):
        raise ValueError("quaternion must be finite and nonzero")
    return q / norm


def euler_to_quaternion(euler_rad):
    e = np.asarray(euler_rad, float)
    cr, cp, cy = np.moveaxis(np.cos(e/2), -1, 0)
    sr, sp, sy = np.moveaxis(np.sin(e/2), -1, 0)
    return np.stack([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy], axis=-1)


def quaternion_error(q, target=None):
    """Shortest target correction in BODY coordinates; q and -q agree.

    At exactly pi, choose a deterministic sign from the first nonzero axis
    component. This prevents quaternion representation from changing the action.
    """
    q = normalize_quaternion(q)
    target = np.array([1., 0., 0., 0.]) if target is None else normalize_quaternion(target)
    inverse = q.copy(); inverse[..., 1:] *= -1
    error = normalize_quaternion(quat_multiply(inverse, target))
    signs = np.where(error[..., 0] < -1e-12, -1., 1.)
    tie = np.abs(error[..., 0]) <= 1e-12
    first = np.argmax(np.abs(error[..., 1:]) > 1e-12, axis=-1)
    value = np.take_along_axis(error[..., 1:], first[..., None], axis=-1)[..., 0]
    signs = np.where(tie & (value < 0), -1., signs)
    error = error * signs[..., None]
    v = error[..., 1:]
    length = np.linalg.norm(v, axis=-1)
    angle = 2*np.arctan2(length, np.clip(error[..., 0], 0, 1))
    ratio = np.divide(angle, length, out=np.full_like(length, 2.), where=length > 1e-12)
    return v * ratio[..., None]


def _vector(value, dimensions):
    a = np.asarray(value, float)
    if a.ndim == 0:
        return np.full(dimensions, float(a))
    if a.shape != (dimensions,):
        raise ValueError(f"expected scalar or {dimensions} vector")
    return a


class BatchSatelliteEnv:
    """Vectorized fixed-horizon plants with identical exogenous noise per scene.

    Scene dictionaries use degrees for initial attitude/rate and sensor noise,
    N m for disturbance, and an inertia multiplier relative to the nominal body.
    Failed rows are frozen; failed=True remains sticky and excludes success.
    """
    def __init__(self, axis, scenarios, config=None):
        self.axis = axis_name(axis)
        self.dimensions = 1 if self.axis == "single" else 3
        self.scenarios = list(scenarios)
        if not self.scenarios:
            raise ValueError("at least one scene is required")
        self.n = len(self.scenarios)
        self.config = dict(config or {})
        self.dt = float(self.config.get("dt", .05 if self.axis == "single" else .02))
        self.duration = float(self.config.get("duration", 20.))
        self.max_torque = float(self.config.get("max_torque", .5 if self.axis == "single" else 2.))
        if not all(np.isfinite(v) and v > 0 for v in (self.dt, self.duration, self.max_torque)):
            raise ValueError("dt, duration and torque limit must be positive and finite")
        self.steps = math.ceil(self.duration/self.dt)
        self.angle_weight = float(self.config.get("reward_angle_weight", 6.))
        self.rate_weight = float(self.config.get("reward_omega_weight", .6))
        self.torque_weight = float(self.config.get("reward_torque_weight", .03))
        nominal = np.array([1.]) if self.dimensions == 1 else np.array([1.2, 1., .8])
        self.inertia = np.stack([nominal*_vector(s.get("inertia_scale", 1.), self.dimensions) for s in self.scenarios])
        if np.any(~np.isfinite(self.inertia)) or np.any(self.inertia <= 0):
            raise ValueError("positive finite inertias required")
        for attr, key, default in (("bias", "disturbance_bias", 0.),
                                   ("amplitude", "disturbance_amplitude", 0.),
                                   ("pulse", "pulse_torque", 0.),
                                   ("noise_angle", "noise_theta_deg", 0.),
                                   ("noise_rate", "noise_omega_deg", 0.)):
            setattr(self, attr, np.stack([_vector(s.get(key, default), self.dimensions) for s in self.scenarios]))
        if np.any(self.noise_angle < 0) or np.any(self.noise_rate < 0):
            raise ValueError("sensor deviations cannot be negative")
        self.pulse_start = np.array([s.get("pulse_start", 8.) for s in self.scenarios])
        self.pulse_duration = np.array([s.get("pulse_duration", .5) for s in self.scenarios])
        self.frequencies = np.array([.2]) if self.dimensions == 1 else np.array([.2, .3, .25])
        self.sine_factors = np.array([1.]) if self.dimensions == 1 else np.array([1., .6, .8])
        self.reset()

    def reset(self):
        self.time, self.step_count = 0., 0
        self.failed = np.zeros(self.n, dtype=bool)
        self.omega = np.stack([np.radians(_vector(s.get("omega_deg_s", 0.), self.dimensions)) for s in self.scenarios])
        if self.axis == "single":
            self.theta = np.array([np.radians(s.get("theta_deg", 0.)) for s in self.scenarios])[:, None]
        else:
            self.q = np.stack([normalize_quaternion(s["quaternion"]) if "quaternion" in s else
                               euler_to_quaternion(np.radians(_vector(s.get("euler_deg", 0.), 3)))
                               for s in self.scenarios])
        self._noise = np.stack([np.random.default_rng(int(s.get("noise_seed", 0))).normal(
            size=(self.steps+1, self.dimensions*2)) for s in self.scenarios], axis=1)
        self.torque = np.zeros_like(self.omega)
        self.disturbance = np.zeros_like(self.omega)
        return self.observation()

    @property
    def error_vector(self):
        if self.axis == "single":
            return (self.theta + np.pi) % (2*np.pi)-np.pi
        return quaternion_error(self.q)

    @property
    def true_error_deg(self):
        return np.degrees(np.linalg.norm(self.error_vector, axis=1))

    @property
    def rate_deg_s(self):
        return np.degrees(np.linalg.norm(self.omega, axis=1))

    def observation(self):
        draws = self._noise[min(self.step_count, self.steps)]
        e = self.error_vector + np.radians(self.noise_angle)*draws[:, :self.dimensions]
        w = self.omega + np.radians(self.noise_rate)*draws[:, self.dimensions:]
        if self.axis == "single":
            e = (e+np.pi) % (2*np.pi)-np.pi
        return np.concatenate([e, w], axis=1).astype(np.float32)

    def disturbance_at(self, time):
        sine = self.amplitude*self.sine_factors*np.sin(2*np.pi*self.frequencies*time)
        pulse_on = (time >= self.pulse_start) & (time < self.pulse_start+self.pulse_duration)
        return self.bias+sine+self.pulse*pulse_on[:, None]

    def _derivative(self, q, w, torque):
        w_dot = (torque-np.cross(w, self.inertia*w))/self.inertia
        w_quat = np.concatenate([np.zeros((self.n, 1)), w], axis=1)
        return .5*quat_multiply(q, w_quat), w_dot

    def step(self, actions):
        if self.step_count >= self.steps:
            raise RuntimeError("fixed horizon finished")
        actions = np.asarray(actions, float).reshape(self.n, self.dimensions)
        if np.any(~np.isfinite(actions)):
            raise ValueError("actions must be finite")
        self.torque = np.clip(actions, -1, 1)*self.max_torque
        self.disturbance = self.disturbance_at(self.time)
        # A failed row is an artificial padding state, not a physical satellite
        # completing the horizon. Charge the failure interval, then no further
        # actuator effort because the frozen state is no longer integrated.
        self.torque[self.failed] = 0.
        self.disturbance[self.failed] = 0.
        force = self.torque+self.disturbance
        next_time = min(self.duration, (self.step_count+1)*self.dt)
        h = next_time-self.time
        if self.axis == "single":
            acceleration = force/self.inertia
            theta = self.theta+self.omega*h+.5*acceleration*h*h
            w = self.omega+acceleration*h
            self.theta = np.where(self.failed[:, None], self.theta, theta)
        else:
            q, w0 = self.q, self.omega
            a1,b1 = self._derivative(q,w0,force)
            a2,b2 = self._derivative(q+.5*h*a1,w0+.5*h*b1,force)
            a3,b3 = self._derivative(q+.5*h*a2,w0+.5*h*b2,force)
            a4,b4 = self._derivative(q+h*a3,w0+h*b3,force)
            q_new = normalize_quaternion(q+h*(a1+2*a2+2*a3+a4)/6)
            w = w0+h*(b1+2*b2+2*b3+b4)/6
            self.q = np.where(self.failed[:, None], self.q, q_new)
        self.omega = np.where(self.failed[:, None], self.omega, w)
        previous_failure = self.failed.copy()
        self.failed |= self.rate_deg_s > 300.
        if self.axis == "single":
            self.failed |= np.abs(np.degrees(self.theta[:, 0])) > 170.
        self.step_count += 1
        self.time = next_time
        reward = -(self.angle_weight*np.sum(self.error_vector**2,axis=1)
                   +self.rate_weight*np.sum(self.omega**2,axis=1)
                   +self.torque_weight*np.sum(self.torque**2,axis=1))
        reward -= (self.failed & ~previous_failure)*float(self.config.get("divergence_penalty", 1000.))
        info = {"t": self.time, "error_deg": self.true_error_deg.copy(),
                "rate_deg_s": self.rate_deg_s.copy(), "torque": self.torque.copy(),
                "disturbance": self.disturbance.copy(), "failed": self.failed.copy()}
        return self.observation(), reward, self.failed.copy(), np.full(self.n,self.time>=self.duration-1e-10), info


class ThreeAxisEnv(gym.Env):
    """Gym facade over exactly the batch plant used in final evaluation."""
    metadata = {"render_modes": []}
    def __init__(self, randomized=False, config=None):
        super().__init__()
        self.config = dict(config or {})
        self.randomized = bool(randomized)
        self.dt = float(self.config.get("dt", .02))
        self.duration = float(self.config.get("duration", 20.))
        self.max_torque = float(self.config.get("max_torque", 2.))
        self.action_space = spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-np.inf]*6,np.float32),np.array([np.inf]*6,np.float32))
        self.plant = None

    def config_dict(self):
        return {"axis": "three", "dt": self.dt, "duration": self.duration,
                "max_torque": self.max_torque, "inertia": [1.2,1.,.8],
                "randomized": self.randomized, "integrator": "RK4",
                "reward_angle_weight": 6., "reward_omega_weight": .6,
                "reward_torque_weight": .03, "divergence_penalty": 1000.,
                "target_quaternion": [1.,0.,0.,0.], **self.config}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = dict(options or {})
        scene = {"euler_deg": self.np_random.uniform(-60.,60.,3).tolist(),
                 "omega_deg_s": self.np_random.uniform(-5.,5.,3).tolist(),
                 "noise_seed": int(self.np_random.integers(0,2**31))}
        if self.randomized:
            scene.update(inertia_scale=float(self.np_random.uniform(.8,1.2)),
                         disturbance_bias=self.np_random.uniform(-.02,.02,3).tolist(),
                         disturbance_amplitude=.02,
                         noise_theta_deg=float(self.np_random.uniform(0,.15)),
                         noise_omega_deg=float(self.np_random.uniform(0,.15)))
        scene.update(opts)
        self.plant = BatchSatelliteEnv("three", [scene], self.config)
        return self.plant.observation()[0], {"t": 0., "failed": False}

    def step(self, action):
        if self.plant is None or self.plant.failed[0] or self.plant.time >= self.duration-1e-10:
            raise RuntimeError("call reset before stepping a completed episode")
        obs,reward,terminated,truncated,info = self.plant.step(np.asarray(action).reshape(1,3))
        values = {k:(v[0] if isinstance(v,np.ndarray) else v) for k,v in info.items()}
        return obs[0],float(reward[0]),bool(terminated[0]),bool(truncated[0]),values


def make_env(axis="single", randomized=False, config=None):
    axis = axis_name(axis)
    if axis == "three":
        return ThreeAxisEnv(randomized,config)
    overrides = asdict(config) if isinstance(config,EnvConfig) else dict(config or {})
    if randomized:
        overrides.update(randomized=True,inertia_range=(.8,1.2),
                         disturbance_range=(-.02,.02),noise_range_deg=(0.,.15),
                         disturbance_amplitude=.02)
    env = SatelliteEnv(EnvConfig(**overrides))
    env.config_dict = lambda: {"axis":"single", "integrator":"exact_constant_acceleration", **asdict(env.config)}
    return env
