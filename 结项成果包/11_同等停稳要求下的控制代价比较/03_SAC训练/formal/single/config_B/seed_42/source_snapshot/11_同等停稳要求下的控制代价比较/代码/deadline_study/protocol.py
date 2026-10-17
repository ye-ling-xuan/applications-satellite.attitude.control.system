"""Declared conditions; held-out controller responses are never used for selection."""
from pathlib import Path
import hashlib
import json
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
PACKAGE = STUDY.parent
DEADLINES = (4., 6., 8.)
DURATION = 20.
ANGLE_TOL = 1.
RATE_TOL = .5
SEEDS = (41, 42, 43)
SPLIT_SEEDS = {"development": 2026101701, "validation": 2026101702, "test": 2026101703}
COUNTS = {"development": 32, "validation": 96, "test": 256}
DT = {"single": .05, "three": .02}
LIMIT = {"single": .5, "three": 2.}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenarios(axis, split):
    if axis not in DT or split not in SPLIT_SEEDS:
        raise ValueError("unknown axis or split")
    d = 1 if axis == "single" else 3
    rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SEEDS[split], d]))
    angles = rng.uniform(-60., 60., (COUNTS[split], d))
    rates = rng.uniform(-5., 5., (COUNTS[split], d))
    result = []
    for i in range(COUNTS[split]):
        row = {"scenario_id": f"{split}_{axis}_{i:04d}", "sample_id": i,
               "axis": axis, "split": split, "inertia_scale": 1.,
               "noise_seed": SPLIT_SEEDS[split]+i, "disturbance_bias": 0.,
               "disturbance_amplitude": 0., "pulse_torque": 0.,
               "noise_theta_deg": 0., "noise_omega_deg": 0.}
        if d == 1:
            row.update(theta_deg=float(angles[i, 0]), omega_deg_s=float(rates[i, 0]))
        else:
            row.update(euler_deg=angles[i].tolist(), omega_deg_s=rates[i].tolist())
        result.append(row)
    return result


def freeze_design():
    directory = STUDY/"01_实验方案"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"任务定义.json"
    design = {
        "study": "control_effort_with_identical_settling_constraints",
        "axes": ["single", "three"], "deadlines_s": list(DEADLINES),
        "primary_deadline_s": 8., "duration_s": DURATION,
        "success": "every sample from deadline through 20s: geodesic angle<=1deg AND rate norm<=0.5deg/s; complete; no divergence",
        "angle_tolerance_deg": ANGLE_TOL, "rate_tolerance_deg_s": RATE_TOL,
        "cost": "sum over applied zero-order-held intervals: ||actual torque||^2 * dt",
        "cost_is_measured_electrical_energy": False,
        "initial_angle_component_deg": [-60., 60.], "initial_rate_component_deg_s": [-5., 5.],
        "inertia": {"single": [1.], "three": [1.2, 1., .8]},
        "dt": DT, "torque_limit_each_axis_Nm": LIMIT,
        "noise_and_external_disturbance": 0., "split_seeds": SPLIT_SEEDS, "scene_counts": COUNTS,
        "fresh_training_seeds": list(SEEDS),
        "selection": "validation only; maximize number passing then minimize complete-horizon mean effort",
        "reliability_gate_for_advantage": "PID and every reported SAC seed must pass all independent test scenes at the same deadline",
        "advantage_scope": "finite tested conditions, control effort only; no equal-trajectory or equal-electricity claim",
        "all_deadlines_and_seeds_reported": True,
        "test_controller_responses_read_before_design": False,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != design:
            raise RuntimeError("existing task definition differs; do not overwrite")
    else:
        write_json(path, design)
    for axis in DT:
        for split in SPLIT_SEEDS:
            target = directory/f"{split}_{axis}.json"
            value = scenarios(axis, split)
            if target.exists():
                if json.loads(target.read_text(encoding="utf-8")) != value:
                    raise RuntimeError("existing split changed")
            else:
                write_json(target, value)
    return design
