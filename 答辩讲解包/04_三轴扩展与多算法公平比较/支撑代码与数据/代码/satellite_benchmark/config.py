"""Shared physical, experimental and controller settings.

Angles in configuration files are degrees; the dynamics and controller use radians.
"""

from dataclasses import dataclass
import math


@dataclass
class EnvConfig:
    dt: float = 0.05
    duration: float = 20.0
    max_torque: float = 0.5
    inertia: float = 1.0
    deadzone: float = 0.0
    disturbance_bias: float = 0.0
    disturbance_amplitude: float = 0.0
    disturbance_frequency: float = 0.2
    noise_theta_deg: float = 0.0
    noise_omega_deg: float = 0.0
    pulse_torque: float = 0.0
    pulse_start: float = 8.0
    pulse_duration: float = 0.5
    target_deg: float = 0.0
    init_theta_range_deg: tuple[float, float] = (-60.0, 60.0)
    init_omega_range_deg_s: tuple[float, float] = (-5.0, 5.0)
    inertia_range: tuple[float, float] = (1.0, 1.0)
    disturbance_range: tuple[float, float] = (0.0, 0.0)
    noise_range_deg: tuple[float, float] = (0.0, 0.0)
    randomized: bool = False
    theta_tol_deg: float = 1.0
    omega_tol_deg_s: float = 0.5
    hold_seconds: float = 5.0
    reward_angle_weight: float = 6.0
    reward_omega_weight: float = 0.6
    reward_torque_weight: float = 0.03
    divergence_angle_deg: float = 170.0
    divergence_omega_deg_s: float = 300.0
    divergence_penalty: float = 1000.0

    def __post_init__(self):
        positive = ("dt", "duration", "max_torque", "inertia", "theta_tol_deg",
                    "omega_tol_deg_s", "hold_seconds", "divergence_angle_deg",
                    "divergence_omega_deg_s")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.hold_seconds > self.duration:
            raise ValueError("hold_seconds cannot exceed duration")
        nonnegative = ("deadzone", "disturbance_amplitude", "disturbance_frequency",
                       "noise_theta_deg", "noise_omega_deg", "pulse_duration",
                       "reward_angle_weight", "reward_omega_weight", "reward_torque_weight", "divergence_penalty")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("disturbance_bias", "pulse_torque", "pulse_start", "target_deg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("init_theta_range_deg", "init_omega_range_deg_s", "inertia_range",
                     "disturbance_range", "noise_range_deg"):
            values = tuple(getattr(self, name))
            if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] > values[1]:
                raise ValueError(f"{name} must contain finite ordered lower and upper bounds")
            setattr(self, name, values)
        if self.inertia_range[0] <= 0 or self.noise_range_deg[0] < 0:
            raise ValueError("inertia_range must be positive and noise_range_deg nonnegative")


@dataclass
class PIDConfig:
    kp: float = 2.0
    ki: float = 0.1
    kd: float = 2.5
    integral_limit: float = 2.0

    def __post_init__(self):
        for name in ("kp", "ki", "kd", "integral_limit"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
