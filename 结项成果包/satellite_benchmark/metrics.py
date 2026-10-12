"""Metrics calculated from true state, with explicit terminal-hold criteria."""

import numpy as np

from .config import EnvConfig


def compute_metrics(trace: dict, config: EnvConfig | dict) -> dict:
    """Evaluate a trace containing reset info followed by info from every step.

    Success requires the complete episode, no failure, and both attitude and
    angular-rate bounds at every recorded point in the last hold_seconds.
    Settling time is the start of the final uninterrupted in-tolerance suffix,
    reported only when that suffix is long enough to satisfy the hold criterion.
    These are sampled-data criteria; samples are spaced by the simulator dt.

    A torque stored at sample i was applied over (t[i-1], t[i]], so integral
    control cost uses torque[1:] rather than trapezoidal endpoint averaging.
    """
    cfg = EnvConfig(**config) if isinstance(config, dict) else config
    keys = ("t", "theta_deg", "omega_deg_s", "torque", "failed")
    arrays = {key: np.asarray(trace[key]).reshape(-1) for key in keys}
    n = len(arrays["t"])
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError("trace arrays must be nonempty and have the same length")
    t = arrays["t"].astype(float)
    theta = arrays["theta_deg"].astype(float)
    omega = arrays["omega_deg_s"].astype(float)
    torque = arrays["torque"].astype(float)
    if not all(np.all(np.isfinite(x)) for x in (t, theta, omega, torque)):
        raise ValueError("trace contains nonfinite numerical values")
    if t[0] < -1e-10 or np.any(np.diff(t) <= 0):
        raise ValueError("trace time must be nonnegative and strictly increasing")
    errors = theta - cfg.target_deg
    abs_errors = np.abs(errors)
    failed = bool(np.any(arrays["failed"].astype(bool)))
    elapsed = float(t[-1] - t[0])
    complete = t[0] <= 1e-8 and t[-1] >= cfg.duration - 1e-8
    in_bounds = (abs_errors <= cfg.theta_tol_deg) & (np.abs(omega) <= cfg.omega_tol_deg_s)
    outside = np.flatnonzero(~in_bounds)
    suffix_start = int(outside[-1] + 1) if outside.size else 0
    hold_observed = suffix_start < n and t[-1] - t[suffix_start] >= cfg.hold_seconds - 1e-8
    success = bool(complete and not failed and hold_observed)
    settling = float(t[suffix_start]) if success else None
    tail = t >= max(t[0], cfg.duration - cfg.hold_seconds) - 1e-8
    if not np.any(tail):
        # Early failures still get an interpretable final error statistic.
        tail = t >= max(t[0], t[-1] - cfg.hold_seconds) - 1e-8
    intervals = np.diff(t)
    control_cost = float(np.sum(torque[1:]**2 * intervals))
    if elapsed > 0:
        rms_error = float(np.sqrt(np.sum(0.5 * (errors[:-1]**2 + errors[1:]**2) * intervals) / elapsed))
        integrated_abs_error = float(np.sum(0.5 * (abs_errors[:-1] + abs_errors[1:]) * intervals))
    else:
        rms_error = float(abs_errors[0])
        integrated_abs_error = 0.0
    if abs(errors[0]) <= 1e-12:
        overshoot = None  # No initial displacement defines a crossing direction.
    else:
        overshoot = float(max(0.0, np.max(-np.sign(errors[0]) * errors)))
    return {"success": success, "settling_time_s": settling,
            "steady_error_deg": float(np.mean(abs_errors[tail])),
            "steady_signed_error_deg": float(np.mean(errors[tail])),
            "rms_error_deg": rms_error, "integrated_abs_error_deg_s": integrated_abs_error,
            "final_error_deg": float(errors[-1]), "final_omega_deg_s": float(omega[-1]),
            "control_cost": control_cost, "overshoot_deg": overshoot,
            "peak_torque": float(np.max(np.abs(torque))),
            "peak_omega_deg_s": float(np.max(np.abs(omega))),
            "failed": failed, "duration_s": elapsed, "completed": bool(complete),
            "terminal_hold_observed_s": float(t[-1] - t[suffix_start]) if suffix_start < n else 0.0}
