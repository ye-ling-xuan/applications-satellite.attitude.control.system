"""Independent audit of frozen artifacts, metrics, actions and physical replay.

This module never trains, selects a controller, or calls the new study's
controller/metric implementations.  Run only after the held-out test is closed.
The prior BatchSatelliteEnv is the independently exercised dynamics reference.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .protocol import STUDY, PACKAGE, DT, LIMIT, DURATION, DEADLINES, ANGLE_TOL, RATE_TOL, COUNTS, scenarios


TOLERANCES = {
    "time_s": 1e-10,
    "metrics": 1e-9,
    "applied_action_torque_Nm": 5e-7,
    "replay_attitude": 1e-9,
    "replay_omega_rad_s": 1e-9,
    "replay_error_deg": 1e-7,
    "replay_rate_deg_s": 1e-7,
    "replay_observation": 1e-6,
    "quaternion_norm": 1e-9,
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bounded(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact outside package: " + str(relative))
    return path


class _Audit:
    def __init__(self):
        self.issue_count = 0
        self.issues = []
        self.maxima = {}
        self.check_count = 0

    def require(self, condition, detail):
        self.check_count += 1
        if not bool(condition):
            self.issue_count += 1
            if len(self.issues) < 100:
                self.issues.append(str(detail))

    def compare(self, key, actual, expected, tolerance, detail):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, detail + ": shape mismatch")
        if a.shape != b.shape:
            return
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        self.require(finite, detail + ": nonfinite values")
        if not finite:
            return
        delta = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        self.maxima[key] = max(self.maxima.get(key, 0.), delta)
        self.require(delta <= tolerance, f"{detail}: max absolute difference {delta:.12g} > {tolerance}")


def _boolean(value):
    value = str(value).strip().lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError("not a boolean: " + repr(value))


def _independent_metrics(trace):
    """Scalar interval sums and inclusive deadline checks, without shared code."""
    time = np.asarray(trace["time"], float)
    result = []
    for scene_index in range(trace["error_deg"].shape[1]):
        angle = trace["error_deg"][:, scene_index]
        rate = trace["rate_deg_s"][:, scene_index]
        failure = trace["failed"][:, scene_index].astype(bool)
        torque = trace["torque"][:, scene_index]
        completed = bool(len(time) > 1 and time[0] == 0. and abs(time[-1]-DURATION) < 1e-8)
        physically_failed = bool(failure.any())
        effort = 0.
        squared_angle_integral = 0.
        last_bad = -1
        for k, (angle_k, rate_k, failed_k) in enumerate(zip(angle, rate, failure)):
            if angle_k > ANGLE_TOL or rate_k > RATE_TOL or failed_k:
                last_bad = k
            if k:
                h = float(time[k]-time[k-1])
                # tau[k] was held on (time[k-1], time[k]].  tau[0] is not applied.
                effort += h*float(np.dot(torque[k], torque[k]))
                squared_angle_integral += .5*h*(float(angle[k-1])**2+float(angle_k)**2)
        settling = None
        if completed and not physically_failed and last_bad+1 < len(time):
            settling = float(time[last_bad+1])
        row = {
            "control_cost": effort,
            "settling_time_s": settling,
            "rms_error_deg": float(np.sqrt(squared_angle_integral/max(float(time[-1]-time[0]), 1e-12))),
            "failed": physically_failed,
            "complete": completed,
            "final_error_deg": float(angle[-1]),
            "final_rate_deg_s": float(rate[-1]),
        }
        for deadline in DEADLINES:
            # All production samples are checked; the exact deadline sample is
            # included.  The clock grid is independently validated below.
            indexes = np.flatnonzero(time >= deadline-1e-9)
            passing = completed and not physically_failed and len(indexes) > 0
            for k in indexes:
                if angle[k] > ANGLE_TOL or rate[k] > RATE_TOL or failure[k]:
                    passing = False
            suffix = str(int(deadline)) + "s"
            row["success_"+suffix] = bool(passing)
            row["max_angle_after_"+suffix] = float(max(angle[indexes])) if len(indexes) else None
            row["max_rate_after_"+suffix] = float(max(rate[indexes])) if len(indexes) else None
        result.append(row)
    return result


def _check_csv(audit, descriptor, scenes, expected, rows_by_identity):
    comparisons = 0
    controller_id = descriptor["id"]
    for scene, metrics in zip(scenes, expected):
        identity = (controller_id, scene["scenario_id"])
        audit.require(identity in rows_by_identity, "missing CSV episode " + str(identity))
        if identity not in rows_by_identity:
            continue
        row = rows_by_identity[identity]
        audit.require(row["axis"] == descriptor["axis"], "CSV axis " + str(identity))
        audit.require(int(row["seed"]) == int(descriptor["seed"]), "CSV model seed " + str(identity))
        audit.require(int(row["sample_id"]) == int(scene["sample_id"]), "CSV sample id " + str(identity))
        for key, value in metrics.items():
            comparisons += 1
            audit.require(key in row, "CSV metric missing " + key + " " + str(identity))
            if key not in row:
                continue
            label = f"{controller_id} {scene['sample_id']} {key}"
            if isinstance(value, bool):
                audit.require(_boolean(row[key]) == value, "boolean metric mismatch " + label)
            elif value is None:
                audit.require(str(row[key]).strip().lower() in ("", "none", "null"), "missing-value metric mismatch " + label)
            else:
                audit.compare("metrics_"+key, float(row[key]), value, TOLERANCES["metrics"], label)
    return comparisons


def _check_shapes(audit, axis, trace, count):
    steps = int(round(DURATION/DT[axis]))
    d = 1 if axis == "single" else 3
    shapes = {
        "time": (steps+1,), "error_deg": (steps+1, count), "rate_deg_s": (steps+1, count),
        "torque": (steps+1, count, d), "failed": (steps+1, count),
        "observation": (steps+1, count, 2*d), "omega_rad_s": (steps+1, count, d),
        "attitude": (steps+1, count, 1 if d == 1 else 4),
    }
    all_valid = True
    for key, shape in shapes.items():
        correct = key in trace and trace[key].shape == shape
        audit.require(correct, f"{axis} trace {key} shape must be {shape}")
        if not correct:
            all_valid = False
        elif key != "failed":
            audit.require(np.isfinite(trace[key]).all(), f"{axis} trace {key} must be finite")
            all_valid &= bool(np.isfinite(trace[key]).all())
    if not all_valid:
        return False
    audit.require(np.isin(trace["failed"], [False, True]).all(), axis + " failure flag values")
    audit.require(not trace["failed"][0].any(), axis + " initial failure must be false")
    audit.require(not np.any(np.diff(trace["failed"].astype(int), axis=0) < 0), axis + " failures must remain sticky")
    audit.compare("time_s", trace["time"], np.linspace(0., DURATION, steps+1), TOLERANCES["time_s"], axis + " full clock grid")
    audit.compare("initial_torque_Nm", trace["torque"][0], np.zeros((count, d)), 0., axis + " initial torque is unapplied zero")
    audit.require(np.max(np.abs(trace["torque"])) <= LIMIT[axis]+1e-12, axis + " actuator limit")
    independently_computed_rate = np.sqrt(np.sum(np.asarray(trace["omega_rad_s"], float)**2, axis=2))*180./np.pi
    audit.compare("saved_rate_from_omega_deg_s", trace["rate_deg_s"], independently_computed_rate,
                  TOLERANCES["replay_rate_deg_s"], axis + " angular-rate vector norm")
    attitude = np.asarray(trace["attitude"], float)
    if d == 1:
        independent_angle = np.abs((attitude[:, :, 0]+np.pi) % (2*np.pi)-np.pi)*180./np.pi
    else:
        norms = np.linalg.norm(attitude, axis=2)
        audit.compare("quaternion_norm", norms, np.ones_like(norms), TOLERANCES["quaternion_norm"], axis + " unit quaternions")
        independent_angle = 2*np.arctan2(np.linalg.norm(attitude[:, :, 1:], axis=2), np.abs(attitude[:, :, 0]))*180./np.pi
    audit.compare("saved_angle_from_attitude_deg", trace["error_deg"], independent_angle,
                  TOLERANCES["replay_error_deg"], axis + " physical attitude error")
    return True


def _action_replay(audit, descriptor, trace, package):
    """Reconstruct direct network/PID formulas without importing study classes."""
    axis = descriptor["axis"]
    d = 1 if axis == "single" else 3
    n = trace["observation"].shape[1]
    kind = descriptor["kind"]
    model = zero_action = None
    integral = np.zeros((n, d))
    if kind in ("legacy", "fresh"):
        from stable_baselines3 import SAC
        model_path = _bounded(package, descriptor["model_path"])
        audit.require(_sha(model_path) == descriptor["model_sha256"], descriptor["id"] + " model hash")
        model = SAC.load(model_path, device="cpu")
        if kind == "legacy":
            zero_action = np.asarray(model.predict(np.zeros((1, 2*d), np.float32), deterministic=True)[0], float)[0]
    elif kind == "pid":
        p = descriptor["parameters"]
        kp, ki, kd = [np.broadcast_to(np.asarray(p[name], float), (d,)) for name in ("kp", "ki", "kd")]
        integral_limit = np.broadcast_to(np.asarray(p.get("integral_limit", 2.), float), (d,))
    else:
        raise ValueError("unknown controller kind " + str(kind))
    for k, elapsed in enumerate(trace["time"][:-1]):
        observation = np.asarray(trace["observation"][k], float)
        if kind == "pid":
            correction = -observation[:, :d] if d == 1 else observation[:, :d]
            base = kp*correction-kd*observation[:, d:]
            candidate_integral = np.clip(integral+correction*DT[axis], -integral_limit, integral_limit)
            candidate_torque = base+ki*candidate_integral
            increment = ki*(candidate_integral-integral)
            accepts = ((np.abs(candidate_torque) <= LIMIT[axis])
                       | ((candidate_torque > LIMIT[axis]) & (increment < 0.))
                       | ((candidate_torque < -LIMIT[axis]) & (increment > 0.)))
            integral = np.where(accepts, candidate_integral, integral)
            expected_torque = np.clip(base+ki*integral, -LIMIT[axis], LIMIT[axis])
        elif kind == "legacy":
            raw = np.asarray(model.predict(observation.astype(np.float32), deterministic=True)[0], float)
            correction = np.clip(raw-zero_action, -1., 1.)
            expected_torque = np.clip(correction*float(descriptor.get("scale", 1.)), -1., 1.)*LIMIT[axis]
        else:
            # Same declared physical inputs; build time observation independently.
            physical = observation.astype(np.float32)
            physical[:, d:] *= 2.
            augmented = np.concatenate((physical, np.full((n, 1), (8.-float(elapsed))/8.)), axis=1).astype(np.float32)
            action = np.asarray(model.predict(augmented, deterministic=True)[0], float)
            if descriptor["calibrated"]:
                zero_reference = np.zeros_like(augmented)
                zero_reference[:, -1] = augmented[:, -1]
                action -= np.asarray(model.predict(zero_reference, deterministic=True)[0], float)
            expected_torque = np.clip(action, -1., 1.)*LIMIT[axis]
        expected_torque[trace["failed"][k].astype(bool)] = 0.
        audit.compare("applied_action_torque_Nm", trace["torque"][k+1], expected_torque,
                      TOLERANCES["applied_action_torque_Nm"], f"{descriptor['id']} independent action at t={elapsed:.6g}")
    return (len(trace["time"])-1)*n


def _physical_replay(audit, descriptor, scenes, trace):
    from multi_algorithm_benchmark.envs import BatchSatelliteEnv
    axis = descriptor["axis"]
    reference = BatchSatelliteEnv(axis, scenes, {"duration": DURATION, "dt": DT[axis], "max_torque": LIMIT[axis]})
    for k in range(len(trace["time"])):
        if k:
            reference.step(trace["torque"][k]/LIMIT[axis])
        audit.compare("replay_attitude", trace["attitude"][k], reference.theta if axis == "single" else reference.q,
                      TOLERANCES["replay_attitude"], f"{descriptor['id']} state sample {k}")
        audit.compare("replay_omega_rad_s", trace["omega_rad_s"][k], reference.omega,
                      TOLERANCES["replay_omega_rad_s"], f"{descriptor['id']} angular velocity sample {k}")
        audit.compare("replay_error_deg", trace["error_deg"][k], reference.true_error_deg,
                      TOLERANCES["replay_error_deg"], f"{descriptor['id']} angle sample {k}")
        audit.compare("replay_rate_deg_s", trace["rate_deg_s"][k], reference.rate_deg_s,
                      TOLERANCES["replay_rate_deg_s"], f"{descriptor['id']} rate sample {k}")
        audit.compare("replay_observation", trace["observation"][k], reference.observation(),
                      TOLERANCES["replay_observation"], f"{descriptor['id']} observation sample {k}")
        audit.require(np.array_equal(trace["failed"][k], reference.failed), f"{descriptor['id']} failure flag sample {k}")
    return reference.steps*len(scenes)


def verify(study_root=None):
    """Verify finished held-out outputs and write a factual audit report.

    The optional root is for independently constructed verifier fixtures.  This
    function does not alter deployment, training, original data, or conclusions.
    """
    study = Path(STUDY if study_root is None else study_root).resolve()
    package = PACKAGE.resolve() if study_root is None else study.parent.resolve()
    report_path = study/"06_核验"/"独立核验报告.json"
    deployment_path = study/"01_实验方案"/"部署冻结.json"
    csv_path = study/"04_独立测试"/"episode_metrics.csv"
    audit = _Audit()
    record = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all exported held-out episodes: independent metrics, frozen model/PID actions, original-plant physical replay",
        "uses_new_controller_or_metric_implementation": False,
        "trains_or_changes_parameters": False,
        "tolerances_absolute": TOLERANCES,
        "relative_tolerance": 0.,
        "counts": {"controllers": 0, "episodes": 0, "csv_rows": 0, "metric_values": 0,
                   "physical_state_action_intervals": 0, "independently_recomputed_action_intervals": 0,
                   "frozen_files": 0},
        "controllers": [], "file_sha256": {},
        "statistical_scope": "no hypothesis selection or performance claims; training seeds are not counted as independent initial-state samples",
    }
    try:
        import torch
        torch.set_num_threads(1)
        deployment = _read(deployment_path)
        record["file_sha256"][str(deployment_path.relative_to(study))] = _sha(deployment_path)
        frozen = deployment["frozen_sha256"]
        audit.require(bool(frozen), "deployment must bind pre-test artifacts")
        for relative, expected in frozen.items():
            target = _bounded(package, relative)
            actual = _sha(target)
            audit.require(actual == expected, "frozen artifact changed: " + relative)
            record["file_sha256"]["package/"+relative] = actual
            record["counts"]["frozen_files"] += 1
        for axis in DT:
            saved = _read(study/"01_实验方案"/f"test_{axis}.json")
            audit.require(saved == scenarios(axis, "test"), axis + " test scene definition differs from declared generator")
        descriptors = deployment["controllers"]
        identifiers = [c["id"] for c in descriptors]
        audit.require(len(set(identifiers)) == len(identifiers), "controller ids must be unique")
        audit.require(all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in identifiers), "controller ids must be safe ASCII filenames")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        record["counts"]["csv_rows"] = len(csv_rows)
        record["file_sha256"][str(csv_path.relative_to(study))] = _sha(csv_path)
        rows_by_identity = {}
        for row in csv_rows:
            key = row["controller_id"], row["scenario_id"]
            audit.require(key not in rows_by_identity, "duplicate CSV episode " + str(key))
            rows_by_identity[key] = row
        expected_identities = set()
        for controller_index, descriptor in enumerate(descriptors, start=1):
            controller_id, axis = descriptor["id"], descriptor["axis"]
            before_issues = audit.issue_count
            local = {"id": controller_id, "axis": axis, "kind": descriptor["kind"], "seed": descriptor["seed"]}
            try:
                scenes = scenarios(axis, "test")
                audit.require(len(scenes) == COUNTS["test"], controller_id + " scene count")
                expected_identities.update((controller_id, s["scenario_id"]) for s in scenes)
                trace_path = study/"04_独立测试"/"trajectories"/(controller_id+".npz")
                local["trajectory_sha256"] = _sha(trace_path)
                record["file_sha256"][str(trace_path.relative_to(study))] = local["trajectory_sha256"]
                with np.load(trace_path, allow_pickle=False) as archive:
                    trace = {key: archive[key] for key in archive.files if key in (
                        "time", "error_deg", "rate_deg_s", "torque", "failed", "observation", "omega_rad_s", "attitude")}
                if _check_shapes(audit, axis, trace, len(scenes)):
                    metrics = _independent_metrics(trace)
                    record["counts"]["metric_values"] += _check_csv(audit, descriptor, scenes, metrics, rows_by_identity)
                    record["counts"]["independently_recomputed_action_intervals"] += _action_replay(audit, descriptor, trace, package)
                    record["counts"]["physical_state_action_intervals"] += _physical_replay(audit, descriptor, scenes, trace)
                    record["counts"]["episodes"] += len(metrics)
                    local["episodes"] = len(metrics)
                    local["success_counts"] = {str(int(t)): sum(m["success_"+str(int(t))+"s"] for m in metrics) for t in DEADLINES}
                    local["mean_control_cost"] = float(np.mean([m["control_cost"] for m in metrics]))
                local["status"] = "passed" if audit.issue_count == before_issues else "failed"
            except Exception as exc:
                audit.require(False, controller_id + " audit error: " + repr(exc))
                local.update(status="failed", error=repr(exc))
            record["controllers"].append(local)
            record["counts"]["controllers"] += 1
            print(f"[independent verification] {controller_index}/{len(descriptors)} "
                  f"controller={controller_id} status={local['status']}", flush=True)
        audit.require(set(rows_by_identity) == expected_identities, "CSV episode membership differs from frozen controller/scene Cartesian product")
        expected_trace_files = {c["id"]+".npz" for c in descriptors}
        present_trace_files = {p.name for p in (study/"04_独立测试"/"trajectories").glob("*.npz")}
        audit.require(present_trace_files == expected_trace_files, "trajectory membership differs from frozen controllers")
        record["file_sha256"]["verification_source.py"] = _sha(Path(__file__))
    except Exception as exc:
        audit.require(False, "audit setup error: " + repr(exc))
        record["error"] = repr(exc)
    record.update(status="passed" if audit.issue_count == 0 else "failed",
                  checks=audit.check_count, issue_count=audit.issue_count,
                  issues_first_100=audit.issues, maximum_absolute_errors=audit.maxima)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(report_path)
    return record


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
