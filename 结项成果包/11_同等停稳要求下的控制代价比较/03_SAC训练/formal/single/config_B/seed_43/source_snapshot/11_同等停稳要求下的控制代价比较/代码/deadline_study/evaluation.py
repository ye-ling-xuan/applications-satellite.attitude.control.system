"""Shared full-horizon propagation and deadline metrics, no test-time selection."""
from __future__ import annotations
import numpy as np
from multi_algorithm_benchmark.envs import BatchSatelliteEnv, euler_to_quaternion, normalize_quaternion
from .protocol import DT, LIMIT, DURATION, DEADLINES, ANGLE_TOL, RATE_TOL


class Plant(BatchSatelliteEnv):
    """Unchanged verified equations with zero-noise arrays stored as a view."""
    def reset(self):
        # This study has exactly zero disturbances/noise; reject accidental use.
        if any(np.any(getattr(self, key) != 0.) for key in ("bias", "amplitude", "pulse", "noise_angle", "noise_rate")):
            raise ValueError("this experiment requires zero nuisance inputs")
        self.time, self.step_count = 0., 0
        self.failed = np.zeros(self.n, bool)
        d = self.dimensions
        self.omega = np.array([np.broadcast_to(np.radians(s.get("omega_deg_s", 0.)), (d,)) for s in self.scenarios]).copy()
        if self.axis == "single":
            self.theta = np.radians([s.get("theta_deg", 0.) for s in self.scenarios])[:, None]
        else:
            self.q = np.array([normalize_quaternion(s["quaternion"]) if "quaternion" in s else
                               euler_to_quaternion(np.radians(s.get("euler_deg", [0., 0., 0.]))) for s in self.scenarios])
        self._noise = np.broadcast_to(np.zeros((1, 1, 2*d)), (self.steps+1, self.n, 2*d))
        self.torque = np.zeros_like(self.omega)
        self.disturbance = np.zeros_like(self.omega)
        return self.observation()


def metric_key(deadline):
    return f"success_{int(deadline)}s"


def metrics_from_arrays(time, error, rate, torque, failed):
    """Arrays have shape [samples, scenes, ...]; applied torque starts at row 1."""
    t, e, w = np.asarray(time), np.asarray(error), np.asarray(rate)
    tau, f = np.asarray(torque), np.asarray(failed, bool)
    if e.ndim == 1:
        e, w, f = e[:, None], w[:, None], f[:, None]
        tau = tau[:, None, :] if tau.ndim == 2 else tau[:, None, None]
    complete = len(t) > 1 and t[0] == 0. and abs(t[-1]-DURATION) < 1e-8
    inside = (e <= ANGLE_TOL) & (w <= RATE_TOL) & ~f
    cost = np.sum(np.sum(tau[1:]**2, axis=2)*np.diff(t)[:, None], axis=0)
    rms = np.sqrt(np.trapz(e**2, t, axis=0)/max(t[-1]-t[0], 1e-12))
    out = []
    for i in range(e.shape[1]):
        bad = np.flatnonzero(~inside[:, i])
        start = int(bad[-1]+1) if bad.size else 0
        settled = float(t[start]) if complete and not f[:, i].any() and start < len(t) else None
        row = {"control_cost": float(cost[i]), "settling_time_s": settled,
               "rms_error_deg": float(rms[i]), "failed": bool(f[:, i].any()),
               "complete": bool(complete), "final_error_deg": float(e[-1, i]),
               "final_rate_deg_s": float(w[-1, i])}
        for deadline in DEADLINES:
            tail = t >= deadline-1e-9
            row[metric_key(deadline)] = bool(complete and not f[:, i].any() and tail.any() and inside[tail, i].all())
            row[f"max_angle_after_{int(deadline)}s"] = float(e[tail, i].max()) if tail.any() else None
            row[f"max_rate_after_{int(deadline)}s"] = float(w[tail, i].max()) if tail.any() else None
        out.append(row)
    return out


def evaluate(axis, scenes, controller, keep_traces=False):
    """Controller.act(observation, time) returns normalized physical-axis commands."""
    plant = Plant(axis, scenes, {"duration": DURATION, "dt": DT[axis], "max_torque": LIMIT[axis]})
    if hasattr(controller, "reset"):
        controller.reset(plant.n)
    n, d, samples = plant.n, plant.dimensions, plant.steps+1
    times = np.empty(samples)
    errors, rates = np.empty((samples, n)), np.empty((samples, n))
    failed = np.empty((samples, n), bool)
    torque = np.zeros((samples, n, d))
    extra = {}
    if keep_traces:
        extra = {"observation": np.empty((samples, n, d*2), np.float32),
                 "omega_rad_s": np.empty((samples, n, d)),
                 "attitude": np.empty((samples, n, 1 if d == 1 else 4))}
    obs = plant.observation()
    for k in range(samples):
        times[k], errors[k], rates[k], failed[k], torque[k] = plant.time, plant.true_error_deg, plant.rate_deg_s, plant.failed, plant.torque
        if keep_traces:
            extra["observation"][k], extra["omega_rad_s"][k] = obs, plant.omega
            extra["attitude"][k] = plant.theta if d == 1 else plant.q
        if k < samples-1:
            obs, _, _, _, _ = plant.step(controller.act(obs, plant.time))
    rows = metrics_from_arrays(times, errors, rates, torque, failed)
    result = {"metrics": rows}
    if keep_traces:
        result["traces"] = {"time": times, "error_deg": errors, "rate_deg_s": rates,
                            "torque": torque, "failed": failed, **extra}
    return result


def rank(rows, deadline):
    return (sum(r[metric_key(deadline)] for r in rows), -float(np.mean([r["control_cost"] for r in rows])))
