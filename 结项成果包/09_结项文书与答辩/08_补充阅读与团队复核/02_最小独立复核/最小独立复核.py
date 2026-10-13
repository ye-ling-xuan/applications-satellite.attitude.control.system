"""Read-only replay of two fixed V3 scenes; not a substitute for full auditing.

No training, selection, report export, formal verification entry point or metric
helper is called. Metric formulas below use only NumPy and saved physical states.
Model inference and optional rollout require the existing research dependencies.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]
FIXTURE_PATH = HERE / "固定复核场景.json"
EXPECTED_PATH = HERE / "预期结果.json"
FAILURE_PATH = HERE / "失败教学样本.json"
CONTROLLERS = ("PID_N", "PID_R", "SAC_N", "SAC_R")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve_input(relative):
    path = (PACKAGE / relative).resolve()
    if not path.is_relative_to(PACKAGE.resolve()) or not path.is_file():
        raise ValueError("Input must be an existing file inside the package: " + relative)
    return path


def compare_arrays(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise ValueError(label + ": shape mismatch or nonfinite data")
    difference = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
    if difference > tolerance:
        raise ValueError(f"{label}: maximum absolute difference {difference:.9g} > {tolerance:g}")
    return difference


def independent_metrics(trace, axis, protocol):
    """Rebuild error from theta/quaternion and rate from omega, then integrate.

    The quaternion is scalar-first and body-to-inertial; target is identity.
    2*atan2(||q_vector||, abs(q_scalar)) gives its shortest geodesic angle.
    Torque[k+1] acts over (time[k], time[k+1]], so cost uses right-end commands.
    """
    t = np.asarray(trace["time"], float)
    omega = np.asarray(trace["omega_rad_s"], float)
    torque = np.asarray(trace["torque"], float)
    failed = np.asarray(trace["failed"], bool)
    dimensions = 1 if axis == "single" else 3
    dt = float(protocol["dt"][axis])
    duration = float(protocol["duration"])
    hold = float(protocol["hold_seconds"])
    n = round(duration / dt) + 1
    if t.shape != (n,) or omega.shape != (n, dimensions) or torque.shape != (n, dimensions):
        raise ValueError("Unexpected fixed-horizon trace dimensions")
    if failed.shape != (n,) or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Invalid failure mask or time grid")
    if abs(t[0]) > 1e-9 or not np.allclose(np.diff(t), dt, atol=1e-9, rtol=0):
        raise ValueError("Saved time grid differs from the fixed protocol")
    if axis == "single":
        theta = np.asarray(trace["theta_rad"], float)
        if theta.shape != (n, 1):
            raise ValueError("Unexpected theta dimensions")
        wrapped = (theta[:, 0] + np.pi) % (2 * np.pi) - np.pi
        error = np.abs(wrapped) * (180 / np.pi)
        quaternion_norm_error = None
    else:
        quaternion = np.asarray(trace["quaternion"], float)
        if quaternion.shape != (n, 4):
            raise ValueError("Unexpected quaternion dimensions")
        norms = np.sqrt(np.sum(quaternion ** 2, axis=1))
        if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
            raise ValueError("Invalid quaternion")
        quaternion_norm_error = float(np.max(np.abs(norms - 1)))
        q = quaternion / norms[:, None]
        error = 2 * np.arctan2(np.sqrt(np.sum(q[:, 1:] ** 2, axis=1)), np.abs(q[:, 0])) * (180 / np.pi)
    rate = np.sqrt(np.sum(omega ** 2, axis=1)) * (180 / np.pi)
    if not all(np.all(np.isfinite(a)) for a in (error, rate, torque)):
        raise ValueError("Nonfinite physical state or command")
    if np.max(np.abs(torque)) > float(protocol["max_torque"][axis]) + 1e-12:
        raise ValueError("Saved command exceeds the actuator limit")
    tail = t >= duration - hold - 1e-9
    in_band = (error <= protocol["angle_tolerance_deg"]) & (rate <= protocol["rate_tolerance_deg_s"]) & ~failed
    complete = bool(t[-1] >= duration - 1e-9)
    success = bool(complete and not np.any(failed) and np.any(tail) and np.all(in_band[tail]))
    outside = np.flatnonzero(~in_band)
    start = int(outside[-1] + 1) if outside.size else 0
    settling = float(t[start]) if success and start < n and t[-1] - t[start] >= hold - 1e-9 else None
    squared_error = error * error
    rms = math.sqrt(float(np.sum((squared_error[:-1] + squared_error[1:]) * np.diff(t) / 2)) / (t[-1] - t[0]))
    control_cost = float(np.sum(np.sum(torque[1:] ** 2, axis=1) * np.diff(t)))
    failures = np.flatnonzero(failed)
    failure_time = float(t[failures[0]]) if failures.size else None
    metrics = {
        "success": success, "settling_time_s": settling, "rms_error_deg": rms,
        "tail_mean_error_deg": float(np.mean(error[tail])),
        "tail_max_error_deg": float(np.max(error[tail])),
        "tail_max_rate_deg_s": float(np.max(rate[tail])), "control_cost": control_cost,
        "final_error_deg": float(error[-1]), "final_rate_deg_s": float(rate[-1]),
        "failed": bool(np.any(failed)), "complete": complete,
        "failure_time_s": failure_time,
        "active_duration_s": failure_time if failure_time is not None else float(t[-1] - t[0]),
        "max_error_deg": float(np.max(error)), "max_rate_deg_s": float(np.max(rate)),
    }
    return metrics, error, rate, quaternion_norm_error


def compare_metrics(actual, expected, tolerances, label):
    differences = {}
    for key, wanted in expected.items():
        got = actual[key]
        if isinstance(wanted, bool) or wanted is None:
            if got != wanted:
                raise ValueError(f"{label}.{key}: {got!r} != {wanted!r}")
            differences[key] = 0.0
        else:
            delta = abs(float(got) - float(wanted))
            limit = tolerances["metric_absolute"] + tolerances["metric_relative"] * abs(float(wanted))
            if delta > limit:
                raise ValueError(f"{label}.{key}: difference {delta:.9g} > {limit:.9g}")
            differences[key] = delta
    return differences


def csv_metrics(row, keys):
    result = {}
    for key in keys:
        value = row[key]
        if key in ("success", "failed", "complete"):
            if value not in ("True", "False"):
                raise ValueError("Unexpected CSV boolean")
            result[key] = value == "True"
        else:
            result[key] = None if value == "" else float(value)
    return result


def policy_check(fixture, entry, trace, tolerances, model_cache):
    """Read frozen weights and replay actions on every saved observation.

    PID's existing runtime is reused only for action replay. No existing metric
    function, training/selection routine or result-export routine is imported.
    """
    axis, name = entry["axis"], entry["controller"]
    observation = np.asarray(trace["observation"], np.float32)
    limit = fixture["protocol"]["max_torque"][axis]
    if name.startswith("PID"):
        from multi_algorithm_benchmark.controllers import PIDController
        controller = PIDController(axis, "improved", gains=entry["pid_gains"])
        controller.reset(1)
        normalized = np.stack([controller.act(row) for row in observation[:-1]])
        bias_difference = None
    else:
        from stable_baselines3 import SAC
        model_path = resolve_input(entry["model_path"])
        if entry["model_path"] not in model_cache:
            model_cache[entry["model_path"]] = SAC.load(model_path, device="cpu")
        model = model_cache[entry["model_path"]]
        if int(model.num_timesteps) != entry["checkpoint_steps"]:
            raise ValueError("Checkpoint step differs from fixture")
        zero, _ = model.predict(np.zeros((1, observation.shape[1]), np.float32), deterministic=True)
        bias = np.clip(np.asarray(zero, float), -1, 1)[0]
        bias_difference = compare_arrays(bias, entry["zero_action_normalized"], tolerances["zero_action_normalized"], "zero action")
        raw, _ = model.predict(observation[:-1], deterministic=True)
        normalized = np.clip(np.clip(np.asarray(raw, float), -1, 1) - bias, -1, 1)
        controller = lambda obs: np.clip(np.clip(np.asarray(model.predict(np.asarray(obs, np.float32), deterministic=True)[0], float), -1, 1) - bias, -1, 1)
    reproduced = normalized * limit
    reproduced[np.asarray(trace["failed"][:-1], bool)] = 0
    maximum = compare_arrays(reproduced, trace["torque"][1:], tolerances["action_torque_Nm"], "replayed actuator commands")
    return {"status": "passed", "commands_checked": len(reproduced), "max_torque_difference_Nm": maximum,
            "torque_absolute_tolerance_Nm": tolerances["action_torque_Nm"], "zero_action_max_difference": bias_difference}, controller


def short_rollout(entry, trace, controller, seconds, tolerances, protocol):
    """Optional smoke replay with the existing plant, not independent physics."""
    from multi_algorithm_benchmark.envs import BatchSatelliteEnv
    env = BatchSatelliteEnv(entry["axis"], [entry["scenario"]])
    if hasattr(controller, "reset"):
        controller.reset(1)
    steps = min(env.steps, round(seconds / env.dt))
    maxima = {"time": 0.0, "state": 0.0, "observation": 0.0, "torque": 0.0}
    state_name = "theta_rad" if entry["axis"] == "single" else "quaternion"
    replay = {name: [] for name in ("time", "omega_rad_s", "torque", "failed", state_name)}
    for index in range(steps + 1):
        state = env.theta[0] if entry["axis"] == "single" else env.q[0]
        replay["time"].append(float(env.time))
        replay["omega_rad_s"].append(env.omega[0].copy())
        replay["torque"].append(env.torque[0].copy())
        replay["failed"].append(bool(env.failed[0]))
        replay[state_name].append(state.copy())
        maxima["state"] = max(maxima["state"], compare_arrays(state, trace[state_name][index], tolerances["rollout_state"], "rollout attitude"))
        maxima["state"] = max(maxima["state"], compare_arrays(env.omega[0], trace["omega_rad_s"][index], tolerances["rollout_state"], "rollout angular rate"))
        obs = env.observation()
        maxima["observation"] = max(maxima["observation"], compare_arrays(obs[0], trace["observation"][index], tolerances["rollout_state"], "rollout observation"))
        maxima["torque"] = max(maxima["torque"], compare_arrays(env.torque[0], trace["torque"][index], tolerances["rollout_torque_Nm"], "rollout torque"))
        maxima["time"] = max(maxima["time"], abs(float(env.time - trace["time"][index])))
        if not np.array_equal(env.failed, np.asarray([trace["failed"][index]])):
            raise ValueError("Rollout physical failure mask mismatch")
        if index < steps:
            action = controller.act(obs) if hasattr(controller, "act") else controller(obs)
            env.step(action)
    result = {"status": "passed", "steps": steps, "duration_s": float(env.time), "max_absolute_differences": maxima,
              "scope": "same frozen controller and existing shared plant; does not independently validate dynamics"}
    if steps == env.steps:
        replay_trace = {name: np.asarray(values) for name, values in replay.items()}
        replay_metrics = independent_metrics(replay_trace, entry["axis"], protocol)[0]
        saved_metrics = independent_metrics(trace, entry["axis"], protocol)[0]
        for key in ("success", "complete", "failed"):
            if replay_metrics[key] != saved_metrics[key]:
                raise ValueError("Full-horizon replay acceptance differs: " + key)
        result["full_horizon_metrics"] = replay_metrics
        result["full_horizon_acceptance_matches_saved"] = True
        result["full_horizon_numeric_metric_differences"] = {
            key: abs(replay_metrics[key] - saved_metrics[key]) for key in saved_metrics
            if saved_metrics[key] is not None and not isinstance(saved_metrics[key], bool)}
    return result


def validate_log_directory(value):
    if value is None:
        return None
    target = Path(value).expanduser().resolve()
    # All published package paths and original project paths stay protected.
    if target.is_relative_to(PACKAGE.parent.resolve()):
        raise ValueError("Choose a new/empty personal log directory outside the entire original project")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Log directory must be new or empty; existing logs are never overwritten")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metrics", "policies"), default="policies",
                        help="metrics needs NumPy only; policies also loads four frozen SAC models and replays PID")
    parser.add_argument("--rollout-seconds", type=float, default=0,
                        help="optional shared-plant smoke replay, 0 to 20 seconds per fixed controller")
    parser.add_argument("--log-dir", type=Path, help="explicit new/empty personal directory; default writes nothing")
    parser.add_argument("--failure-example", action="store_true",
                        help="metrics-only post-result teaching sample; does not replace the two main fixed scenes")
    args = parser.parse_args()
    if not math.isfinite(args.rollout_seconds) or not 0 <= args.rollout_seconds <= 20:
        parser.error("--rollout-seconds must be between 0 and 20")
    if args.mode == "metrics" and args.rollout_seconds:
        parser.error("rollout requires --mode policies")
    if args.failure_example and args.mode != "metrics":
        parser.error("--failure-example requires --mode metrics")
    log_dir = validate_log_directory(args.log_dir)
    started = time.perf_counter()
    selected_fixture = FAILURE_PATH if args.failure_example else FIXTURE_PATH
    selected_expected = FAILURE_PATH if args.failure_example else EXPECTED_PATH
    fixture = read_json(selected_fixture)
    expected = fixture["expected_results"] if args.failure_example else read_json(EXPECTED_PATH)
    if args.failure_example:
        if fixture["selection"] != {"post_result_teaching_sample": True, "axis": "three", "controller": "SAC_R", "model_seed": 33,
                                     "rule": "first scenario_id in lexicographic order with success=False and failed=False in the frozen formal CSV"}:
            raise ValueError("The post-result teaching selection rule changed")
        if len(fixture["cases"]) != 1:
            raise ValueError("Failure teaching fixture must contain one trace")
    else:
        if fixture["selection"] != {"axes": ["single", "three"], "rho": 1.4, "sample_id": 0, "sac_seed": 11,
                                   "controllers": list(CONTROLLERS)}:
            raise ValueError("The two fixed scenes may not be changed or selected using performance")
        identities = {(entry["axis"], entry["controller"]) for entry in fixture["cases"]}
        if len(fixture["cases"]) != 8 or identities != {(axis, name) for axis in ("single", "three") for name in CONTROLLERS}:
            raise ValueError("Fixture must contain exactly two scenes and their eight fixed control instances")
        for entry in fixture["cases"]:
            scene = entry["scenario"]
            expected_seed = 0 if entry["controller"].startswith("PID") else 11
            if entry["model_seed"] != expected_seed or scene["axis"] != entry["axis"] or scene["rho"] != 1.4 or scene["sample_id"] != 0 or scene["experiment"] != "scalar":
                raise ValueError("A case changed the fixed axis, rho, initial state or controller seed")
    before = {}
    for relative, wanted in fixture["input_sha256"].items():
        before[relative] = digest(resolve_input(relative))
        if before[relative] != wanted:
            raise ValueError("Frozen input hash mismatch: " + relative)
    frozen_protocol = read_json(resolve_input(fixture["protocol_path"]))
    if any(fixture["protocol"][key] != frozen_protocol[key] for key in fixture["protocol"]):
        raise ValueError("Fixture acceptance constants differ from the frozen research protocol")
    if not args.failure_example and expected["formal_csv_sha256"] != digest(resolve_input(fixture["formal_csv"])):
        raise ValueError("Expected-result provenance does not match the frozen formal CSV")
    rows = list(csv.DictReader(resolve_input(fixture["formal_csv"]).open(encoding="utf-8-sig", newline="")))
    if args.failure_example:
        candidates = sorted([r for r in rows if r["axis"] == "three" and r["controller"] == "SAC_R" and
                             r["model_seed"] == "33" and r["success"] == "False" and r["failed"] == "False"],
                            key=lambda row: row["scenario_id"])
        entry = fixture["cases"][0]
        if not candidates or entry["axis"] != "three" or entry["controller"] != "SAC_R" or entry["model_seed"] != 33 or entry["csv_trace_file"] != candidates[0]["trace_file"]:
            raise ValueError("Teaching sample no longer matches the declared post-result CSV selection rule")
    if args.mode == "policies":
        sys.path.insert(0, str(PACKAGE))
        import torch
        torch.set_num_threads(1)
    result = {"status": "passed", "mode": args.mode, "fixed_selection": fixture["selection"],
              "execution_identity": "person executing this command; no student contribution is inferred",
              "python_version": platform.python_version(), "python_executable": sys.executable,
              "packages": {"numpy": np.__version__}, "tolerances": fixture["tolerances"],
              "fixture_file": selected_fixture.name, "fixture_sha256": digest(selected_fixture),
              "expected_results_file": selected_expected.name, "expected_results_sha256": digest(selected_expected),
              "writes_by_default": False, "records": []}
    if args.mode == "policies":
        result["packages"].update({name: importlib.metadata.version(name) for name in ("torch", "stable-baselines3", "gymnasium")})
    cache = {}
    for entry in fixture["cases"]:
        key = entry["key"]
        axis, controller = entry["axis"], entry["controller"]
        matching = [r for r in rows if r["axis"] == axis and r["scenario_id"] == entry["scenario"]["scenario_id"]
                    and r["controller"] == controller and int(r["model_seed"]) == entry["model_seed"]]
        if len(matching) != 1 or matching[0]["trace_file"] != entry["csv_trace_file"]:
            raise ValueError("Fixed fixture does not match exactly one formal CSV row: " + key)
        with np.load(resolve_input(entry["trace_path"]), allow_pickle=False) as archive:
            trace = {name: archive[name].copy() for name in archive.files}
        metadata = json.loads(trace.pop("metadata_json").item())
        if (metadata["scenario"] != entry["scenario"] or metadata["axis"] != axis or
                metadata["controller"]["controller"] != controller or metadata["controller"]["model_seed"] != entry["model_seed"]):
            raise ValueError("Saved trace metadata does not match the fixed scene")
        metrics, physical_error, physical_rate, norm_error = independent_metrics(trace, axis, fixture["protocol"])
        tolerances = fixture["tolerances"]
        state_consistency = {
            "max_error_array_difference_deg": compare_arrays(physical_error, trace["error_deg"], tolerances["state_metric_deg"], "saved error vs physical state"),
            "max_rate_array_difference_deg_s": compare_arrays(physical_rate, trace["rate_deg_s"], tolerances["state_metric_deg"], "saved rate vs physical state"),
            "quaternion_max_norm_error": norm_error,
        }
        if norm_error is not None and norm_error > tolerances["quaternion_norm"]:
            raise ValueError("Quaternion norm exceeds tolerance")
        wanted = expected["cases"][key]
        csv_difference = compare_metrics(metrics, csv_metrics(matching[0], wanted.keys()), tolerances, "formal CSV")
        expected_difference = compare_metrics(metrics, wanted, tolerances, "fixed expected result")
        metadata_difference = compare_metrics(metrics, metadata["metrics"], tolerances, "NPZ metadata")
        item = {"key": key, "status": "passed", "metrics_recomputed_from_physical_states": metrics,
                "trace_state_consistency": state_consistency,
                "csv_metric_absolute_differences": csv_difference,
                "expected_metric_absolute_differences": expected_difference,
                "npz_metadata_metric_absolute_differences": metadata_difference,
                "action_replay": {"status": "not_requested"}, "short_rollout": {"status": "not_requested"}}
        if args.failure_example:
            if metrics["success"] or metrics["failed"] or metrics["tail_max_error_deg"] > fixture["protocol"]["angle_tolerance_deg"] or metrics["tail_max_rate_deg_s"] <= fixture["protocol"]["rate_tolerance_deg_s"]:
                raise ValueError("Teaching trace does not demonstrate angular-rate holding failure without physical divergence")
            item["teaching_classification"] = "angular-rate threshold exceeded in the final five seconds; angle threshold held and no physical divergence"
        if args.mode == "policies":
            item["action_replay"], loaded = policy_check(fixture, entry, trace, tolerances, cache)
            if args.rollout_seconds:
                item["short_rollout"] = short_rollout(entry, trace, loaded, args.rollout_seconds, tolerances, fixture["protocol"])
        result["records"].append(item)
    after = {relative: digest(resolve_input(relative)) for relative in before}
    if before != after:
        raise ValueError("An input changed while the read-only review was running")
    result.update(frozen_inputs_unchanged=True, checked_input_count=len(before),
                  checked_input_sha256=after, elapsed_s=round(time.perf_counter() - started, 3),
                  limitation=("One failure selected after results for teaching only. It cannot establish the frequency or cause of failure and does not replace the full independent audit."
                              if args.failure_example else "Two fixed scenes, eight traces and SAC seed 11 only. Shared PID/plant runtime is reused. This does not replace the full independent audit or establish aggregate success or weak seed 33 behavior."))
    if log_dir is not None:
        # Recheck after computation and use exclusive creation for race-safe logs.
        validate_log_directory(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "minimal_review_output.json"
        with log_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
