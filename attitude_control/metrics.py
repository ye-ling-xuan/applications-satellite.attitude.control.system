"""Metrics use N+1 state samples and N applied torques."""
import numpy as np


def metrics(time, angle_deg, rate_deg_s, torque, config, failed=False):
    t, angle, rate, u = map(lambda x: np.asarray(x, dtype=float),
                            (time, angle_deg, rate_deg_s, torque))
    if t.ndim != 1 or any(a.shape != t.shape for a in (angle, rate)) or u.shape != (len(t) - 1,) or len(t) < 2:
        raise ValueError("expected N+1 state samples and N torque samples")
    if not all(np.all(np.isfinite(a)) for a in (t, angle, rate, u)) or not np.all(np.diff(t) > 0):
        raise ValueError("samples must be finite, with strictly increasing time")
    if not np.allclose(np.diff(t), config.dt):
        raise ValueError("sample interval must match config.dt")
    error = angle - config.target_deg
    good = ((np.abs(error) <= config.angle_tolerance_deg)
            & (np.abs(rate) <= config.rate_tolerance_deg_s))
    bad = np.flatnonzero(~good)
    start = int(bad[-1] + 1) if len(bad) else 0
    held = start < len(t) and t[-1] - t[start] >= config.hold_seconds - 1e-9
    completed = t[-1] >= config.duration - 1e-9
    success = bool(held and completed and not failed)
    direction = np.sign(error[0])
    overshoot = float(max(0.0, np.max(-direction * error))) if direction else float(np.max(np.abs(error)))
    tail = t >= t[-1] - config.hold_seconds
    return {
        "success": success,
        "failed": bool(failed),
        "settling_time_s": float(t[start]) if success else None,
        "overshoot_deg": overshoot,
        "tail_mean_abs_error_deg": float(np.mean(np.abs(error[tail]))),
        "final_error_deg": float(error[-1]),
        "final_rate_deg_s": float(rate[-1]),
        "rmse_deg": float(np.sqrt(np.mean(error ** 2))),
        "torque_effort_nm2_s": float(np.sum(u ** 2) * config.dt),
        "peak_torque_nm": float(np.max(np.abs(u))),
        "saturation_fraction": float(np.mean(np.abs(u) >= config.max_torque - 1e-7)),
        "torque_total_variation_nm": float(np.sum(np.abs(np.diff(np.r_[0.0, u])))),
        "observed_duration_s": float(t[-1]),
    }
