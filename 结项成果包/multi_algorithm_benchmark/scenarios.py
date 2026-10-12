"""Predeclared paired deployment scenes; validation and test never overlap."""
import numpy as np
from .envs import axis_name


CASES = [
    ("nominal", {}),
    ("inertia_low", {"inertia_scale": .7}),
    ("inertia_high", {"inertia_scale": 1.3}),
    ("disturbance_003", {"disturbance_bias": .01, "disturbance_amplitude": .03}),
    ("disturbance_006", {"disturbance_bias": .02, "disturbance_amplitude": .06}),
    ("sensor_03", {"noise_theta_deg": .3, "noise_omega_deg": .3}),
    ("sensor_06", {"noise_theta_deg": .6, "noise_omega_deg": .6}),
    ("pulse", {"pulse_torque": .15, "pulse_start": 8., "pulse_duration": .5}),
    ("combined", {"inertia_scale": 1.3, "disturbance_bias": .02,
                  "disturbance_amplitude": .03, "noise_theta_deg": .3,
                  "noise_omega_deg": .3, "pulse_torque": .1}),
]


def make_scenarios(axis, split="test", samples_per_case=20):
    axis = axis_name(axis)
    if split not in ("test", "validation"):
        raise ValueError("split must be validation or test")
    if samples_per_case < 1:
        raise ValueError("samples_per_case must be positive")
    # Each split has independent initial-state and noise generators. Axes share
    # identifiers/case conditions, while initial attitudes have native dimension.
    rng = np.random.default_rng(20261011 if split == "test" else 20260917)
    dimensions = 1 if axis == "single" else 3
    states = []
    for sample in range(samples_per_case):
        euler = rng.uniform(-60.,60.,dimensions)
        rate = rng.uniform(-5.,5.,dimensions)
        if sample == 0:
            euler = np.array([53.]) if dimensions == 1 else np.array([51.,-27.,38.])
            rate = np.array([3.5]) if dimensions == 1 else np.array([3.,-2.,1.])
        elif sample == 1:
            euler = np.array([-43.]) if dimensions == 1 else np.array([-39.,22.,-49.])
            rate = np.array([-2.5]) if dimensions == 1 else np.array([-2.,1.,-3.])
        if split == "validation":
            euler *= .83; rate *= .8
        states.append((euler,rate))
    scenes = []
    for case,params in CASES:
        for sample,(euler,rate) in enumerate(states):
            family = ("inertia" if case.startswith("inertia") else "disturbance" if case.startswith("disturbance") else
                      "sensor" if case.startswith("sensor") else case)
            level = params.get("inertia_scale", params.get("disturbance_amplitude",params.get("noise_theta_deg",params.get("pulse_torque",0.))))
            scene = {"scenario_id":f"{split}_{axis}_{case}_{sample:03d}",
                     "case":case,"case_id":case,"family":family,"level":level,
                     "sample_id":sample, "split":split, "sample":sample,
                     "noise_seed":int(rng.integers(1,2**31)), **params}
            if dimensions == 1:
                scene.update(theta_deg=float(euler[0]),omega_deg_s=float(rate[0]))
            else:
                scene.update(euler_deg=euler.tolist(),omega_deg_s=rate.tolist())
            scenes.append(scene)
    return scenes


def validation_scenarios(axis):
    # 8 conditions x 2 independent states; .6-degree noise remains a pressure
    # test, while .3-degree sensor scenes support declared filter selection.
    return [s for s in make_scenarios(axis,"validation",2) if s["case"]!="sensor_06"]
