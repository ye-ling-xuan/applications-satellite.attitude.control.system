from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    dt: float = 0.02
    duration: float = 10.0
    inertia: float = 1.0
    max_torque: float = 2.0
    residual_limit: float = 0.5
    target_deg: float = 0.0
    failure_angle_deg: float = 90.0
    angle_tolerance_deg: float = 1.0
    rate_tolerance_deg_s: float = 0.2
    hold_seconds: float = 1.0
    kp: float = 3.0
    kd: float = 3.0
    ki: float = 0.0
    # Dimensionless cost scales, identical for every controller.
    angle_scale_deg: float = 40.0
    rate_scale_deg_s: float = 50.0
    rate_weight: float = 0.1
    torque_weight: float = 0.02
    slew_weight: float = 0.005

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        positive = ("dt", "duration", "inertia", "max_torque", "residual_limit",
                    "failure_angle_deg", "angle_tolerance_deg", "rate_tolerance_deg_s",
                    "hold_seconds", "angle_scale_deg", "rate_scale_deg_s")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("time, scales, inertia, and limits must be positive")
        if self.residual_limit > self.max_torque:
            raise ValueError("residual_limit must not exceed max_torque")
        if self.duration < self.hold_seconds:
            raise ValueError("duration must allow the success hold interval")
        if not math.isclose(self.duration / self.dt, round(self.duration / self.dt), abs_tol=1e-8):
            raise ValueError("duration must be an integer multiple of dt")
        if any(getattr(self, name) < 0 for name in ("kp", "kd", "ki", "rate_weight", "torque_weight", "slew_weight")):
            raise ValueError("gains and cost weights must be nonnegative")

    @property
    def steps(self):
        return round(self.duration / self.dt)

    def to_dict(self):
        return asdict(self)
