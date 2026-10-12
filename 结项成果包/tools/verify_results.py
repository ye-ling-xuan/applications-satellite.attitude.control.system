"""Read-only audit of every formal episode, trace, aggregate and selected model.

Only the JSON audit report is written. Models, test data and criteria are never
changed. The model hashes are captured at verification time; they are not
misrepresented as hashes recorded before evaluation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from satellite_benchmark.config import EnvConfig
from satellite_benchmark.metrics import compute_metrics

EXPECTED_CONTROLLERS = {("PID", 0), *( (f"PPO_{mode}", seed)
                        for mode in ("nominal", "randomized")
                        for seed in (11, 22, 33))}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _same(expected, actual, label):
    if expected is None:
        _require(actual is None or actual == "", f"{label}: expected null, got {actual!r}")
    elif isinstance(expected, (bool, np.bool_)):
        if isinstance(actual, str):
            _require(actual in ("True", "False"), f"{label}: invalid Boolean {actual!r}")
            actual = actual == "True"
        _require(isinstance(actual, (bool, np.bool_)) and bool(actual) == bool(expected),
                 f"{label}: Boolean differs")
    elif isinstance(expected, (float, int, np.number)):
        number = float(actual)
        _require(math.isfinite(number) and math.isclose(float(expected), number,
                 rel_tol=1e-10, abs_tol=1e-9), f"{label}: {expected!r} != {actual!r}")
    else:
        _require(expected == actual, f"{label}: {expected!r} != {actual!r}")


def _arrays_same(expected, actual, label, atol=1e-10):
    _require(np.shape(expected) == np.shape(actual) and
             np.allclose(expected, actual, rtol=1e-10, atol=atol), label)


def _independent_summary(rows):
    """Aggregate recomputed metrics without calling evaluation.summarize."""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("controller", "family", "case_id", "level"))
        groups.setdefault(key, []).append(row)
    result = []
    for (controller, family, case_id, level), group in groups.items():
        by_seed = {}
        for row in group:
            by_seed.setdefault(row["model_seed"], []).append(row)
        seed_success = [statistics.mean(r["success"] for r in g) for g in by_seed.values()]
        seed_rms = [statistics.mean(r["rms_error_deg"] for r in g) for g in by_seed.values()]
        settling = [r["settling_time_s"] for r in group if r["settling_time_s"] is not None]
        result.append({
            "controller": controller, "family": family, "case_id": case_id,
            "level": level, "n_episodes": len(group), "n_model_seeds": len(by_seed),
            "success_rate": statistics.mean(r["success"] for r in group),
            "success_seed_std": statistics.stdev(seed_success) if len(seed_success) > 1 else None,
            "rms_error_deg_mean": statistics.mean(r["rms_error_deg"] for r in group),
            "rms_error_seed_std": statistics.stdev(seed_rms) if len(seed_rms) > 1 else None,
            "steady_error_deg_mean": statistics.mean(r["steady_error_deg"] for r in group),
            "control_cost_mean": statistics.mean(r["control_cost"] for r in group),
            "settling_time_s_mean_successful": statistics.mean(settling) if settling else None,
            "settled_episodes": len(settling),
            "failed_rate": statistics.mean(r["failed"] for r in group),
            "outside_training_range": group[0]["outside_training_range"],
        })
    return result


def verify_results(results_dir=None, models_dir=None, report_path=None):
    results_dir = Path(results_dir) if results_dir else ROOT / "04_实验数据与图表"
    models_dir = Path(models_dir) if models_dir else ROOT / "03_训练模型与日志" / "正式训练_100万步"
    report_path = Path(report_path) if report_path else ROOT / "06_结项说明与报告" / "result_verification.json"
    started = time.perf_counter()
    errors, checks = [], {}
    model_records = []
    checked_episodes, sample_count = 0, 0
    current_sources = {p.name: _sha(p) for p in sorted((ROOT / "satellite_benchmark").glob("*.py"))}
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed", "read_only_input_audit": True,
        "hash_capture_time": "This verification, after formal evaluation; training source hashes retain their original capture time.",
        "expected_scenarios": 800, "expected_episodes": 5600,
        "expected_controller_instances_per_scenario": 7,
        "tolerance": {"numeric_relative": 1e-10, "numeric_absolute": 1e-9},
        "current_source_sha256": current_sources,
        "checks": checks, "models": model_records, "errors": errors,
    }
    try:
        scenarios = _json(results_dir / "test_scenarios.json")
        rows = _csv(results_dir / "episode_metrics.csv")
        summary = _csv(results_dir / "summary.csv")
        manifest = _json(results_dir / "evaluation_manifest.json")
        scene_map = {s["scenario_id"]: s for s in scenarios}
        _require(len(scenarios) == len(scene_map) == 800, "Expected 800 unique test scenarios")
        _require(len(rows) == 5600, "Expected 5600 episode rows")
        _same(800, manifest["scenario_count"], "manifest scenario count")
        _same(5600, manifest["episode_count"], "manifest episode count")
        _same(50, manifest["samples_per_case"], "manifest samples per case")
        _same("test", manifest["split"], "manifest split")
        _require(not manifest["missing_models"], "Formal evaluation is missing models")
        descriptors = {(d["controller"], int(d["model_seed"])) for d in manifest["controllers"]}
        _require(descriptors == EXPECTED_CONTROLLERS and len(manifest["controllers"]) == 7,
                 "Manifest must describe PID and the six selected PPO models")
        by_case = {}
        for scene in scenarios:
            _require(scene["split"] == "test", "A scenario is not held-out test")
            by_case.setdefault(scene["case_id"], []).append(scene)
        _require(len(by_case) == 16 and all(len(g) == 50 for g in by_case.values()),
                 "Expected 16 cases with 50 scenarios each")
        by_scene, requested_traces = {}, set()
        for row in rows:
            descriptor = (row["controller"], int(row["model_seed"]))
            _require(row["scenario_id"] in scene_map, "Episode refers to unknown scenario")
            by_scene.setdefault(row["scenario_id"], []).append(descriptor)
            relative = Path(row["trace_file"])
            _require(not relative.is_absolute() and ".." not in relative.parts and
                     relative.parts[0] == "raw_traces" and relative.suffix == ".npz",
                     f"Unsafe or invalid trace path: {relative}")
            _require(relative.as_posix() not in requested_traces, "Multiple CSV rows refer to the same trace")
            requested_traces.add(relative.as_posix())
        _require(set(by_scene) == set(scene_map) and all(len(g) == 7 and set(g) == EXPECTED_CONTROLLERS
                 for g in by_scene.values()), "Every scenario must have exactly seven distinct controller instances")
        existing_traces = {p.relative_to(results_dir).as_posix() for p in (results_dir / "raw_traces").glob("*.npz")}
        _require(existing_traces == requested_traces, "CSV references and 5600 NPZ files do not correspond one-to-one")
        checks["scenario_episode_controller_counts"] = True
        checks["csv_npz_one_to_one"] = True

        references, recomputed_rows = {}, []
        trace_digests = []
        vector_keys = ("t", "time_s", "theta_deg", "omega_deg_s", "torque",
                       "commanded_torque", "disturbance", "inertia", "failed", "error_deg")
        for index, row in enumerate(rows):
            label = f"{row['controller']} seed {row['model_seed']} {row['scenario_id']}"
            try:
                scene = scene_map[row["scenario_id"]]
                cfg = EnvConfig(**scene["env_config"])
                for csv_key, scene_key in (("scenario_id", "scenario_id"), ("case_id", "case_id"),
                    ("family", "family"), ("level", "level"), ("sample_id", "sample_id"),
                    ("scenario_seed", "seed"), ("theta_deg", "theta_deg"),
                    ("omega_deg_s", "omega_deg_s"), ("outside_training_range", "outside_training_range")):
                    _same(scene[scene_key], row[csv_key], f"{label} {csv_key}")
                blob = (results_dir / row["trace_file"]).read_bytes()
                trace_digests.append((row["trace_file"], hashlib.sha256(blob).hexdigest()))
                with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                    _require(set(vector_keys) | {"metadata_json"} == set(archive.files), f"{label}: unexpected NPZ schema")
                    trace = {key: archive[key] for key in vector_keys}
                    metadata = json.loads(str(archive["metadata_json"].item()))
                _same(row["controller"], metadata["controller"], f"{label} metadata controller")
                _same(int(row["model_seed"]), metadata["model_seed"], f"{label} metadata model seed")
                _require(metadata["scenario"] == scene, f"{label}: NPZ scenario metadata differs")
                n = len(trace["t"])
                _require(n >= 2 and all(a.ndim == 1 and len(a) == n for a in trace.values()), f"{label}: vector lengths")
                _require(all(np.all(np.isfinite(a)) for a in trace.values()), f"{label}: nonfinite state/input")
                t = trace["t"]
                _arrays_same(np.minimum(np.arange(n) * cfg.dt, cfg.duration), t, f"{label}: sample times")
                _require(t[0] == 0 and np.all(np.diff(t) > 0) and t[-1] <= cfg.duration + 1e-9,
                         f"{label}: time range")
                _arrays_same(t, trace["time_s"], f"{label}: time alias")
                _arrays_same(trace["theta_deg"] - cfg.target_deg, trace["error_deg"], f"{label}: error alias")
                _same(scene["theta_deg"], float(trace["theta_deg"][0]), f"{label}: initial attitude")
                _same(scene["omega_deg_s"], float(trace["omega_deg_s"][0]), f"{label}: initial rate")
                _arrays_same(np.full(n, cfg.inertia), trace["inertia"], f"{label}: inertia")
                _require(np.max(np.abs(trace["torque"])) <= cfg.max_torque + 1e-12 and
                         np.max(np.abs(trace["commanded_torque"])) <= cfg.max_torque + 1e-12,
                         f"{label}: actuator torque exceeds bound")
                _arrays_same(np.where(np.abs(trace["commanded_torque"]) >= cfg.deadzone,
                     trace["commanded_torque"], 0), trace["torque"], f"{label}: applied actuator torque")
                starts = np.r_[0, t[:-1]]
                disturbance = cfg.disturbance_bias + cfg.disturbance_amplitude * np.sin(
                    2 * np.pi * cfg.disturbance_frequency * starts)
                disturbance += cfg.pulse_torque * ((starts >= cfg.pulse_start) &
                                  (starts < cfg.pulse_start + cfg.pulse_duration))
                _arrays_same(disturbance, trace["disturbance"], f"{label}: exogenous disturbance formula")
                dt = np.diff(t)
                acceleration = np.degrees((trace["torque"][1:] + trace["disturbance"][1:]) / cfg.inertia)
                _arrays_same(trace["omega_deg_s"][:-1] + acceleration * dt,
                             trace["omega_deg_s"][1:], f"{label}: angular-rate dynamics")
                _arrays_same(trace["theta_deg"][:-1] + trace["omega_deg_s"][:-1] * dt + 0.5 * acceleration * dt**2,
                             trace["theta_deg"][1:], f"{label}: attitude dynamics")
                expected_fail = ((np.abs(trace["error_deg"]) > cfg.divergence_angle_deg) |
                                 (np.abs(trace["omega_deg_s"]) > cfg.divergence_omega_deg_s))
                _require(np.array_equal(expected_fail, trace["failed"].astype(bool)) and
                         not np.any(trace["failed"][:-1]), f"{label}: failure flags")
                _require(t[-1] >= cfg.duration - 1e-8 or bool(trace["failed"][-1]), f"{label}: unexplained early end")
                reference = references.setdefault(scene["scenario_id"], {
                    key: trace[key].copy() for key in ("t", "disturbance", "inertia")})
                common = min(n, len(reference["t"]))
                for key in reference:
                    _arrays_same(reference[key][:common], trace[key][:common], f"{label}: paired {key}")
                metrics = compute_metrics(trace, cfg)
                for key, value in metrics.items():
                    _same(value, row[key], f"{label} CSV metric {key}")
                    _same(value, metadata["metrics"][key], f"{label} NPZ metric {key}")
                recomputed_rows.append({"controller": row["controller"], "model_seed": int(row["model_seed"]),
                    "family": scene["family"], "case_id": scene["case_id"], "level": scene["level"],
                    "outside_training_range": scene["outside_training_range"], **metrics})
                checked_episodes += 1
                sample_count += n
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
            if (index + 1) % 800 == 0:
                print(f"Verified {index + 1}/{len(rows)} formal episode files", flush=True)
        checks["all_vectors_finite_and_sample_times_valid"] = checked_episodes == 5600
        checks["actuator_bounds_and_discrete_dynamics"] = checked_episodes == 5600
        checks["paired_initial_states_and_disturbance_sequences"] = checked_episodes == 5600
        checks["all_episode_metrics_recomputed"] = checked_episodes == 5600
        if checked_episodes == 5600:
            recomputed = _independent_summary(recomputed_rows)
            summary_by_key = {(r["controller"], r["case_id"]): r for r in summary}
            _require(len(summary) == len(summary_by_key) == len(recomputed) == 48, "Expected 48 unique summary groups")
            for record in recomputed:
                stored = summary_by_key[(record["controller"], record["case_id"])]
                _require(set(stored) == set(record), "Summary schema differs")
                for key, value in record.items():
                    _same(value, stored[key], f"summary {record['controller']} {record['case_id']} {key}")
            checks["summary_independently_recomputed"] = True
        report["trace_collection_sha256"] = hashlib.sha256("\n".join(
            f"{name}\t{digest}" for name, digest in sorted(trace_digests)).encode("utf-8")).hexdigest()

        for mode in ("nominal", "randomized"):
            for seed in (11, 22, 33):
                run_dir = models_dir / mode / f"seed_{seed}"
                selected = run_dir / "best_model.zip"
                training_metadata = _json(run_dir / "training_metadata.json")
                training_manifest = _json(run_dir / "training_manifest.json")
                best = _json(run_dir / "best_checkpoint.json")
                history = _csv(run_dir / "validation_history.csv")
                _require(training_metadata["mode"] == mode and training_metadata["seed"] == seed, "Model training identity differs")
                _require(training_metadata["selection_split"] == best["selection_split"] == "validation" and
                    training_metadata["test_data_used_in_training_or_selection"] is False,
                    "Model selection provenance is not validation only")
                _require(training_metadata["source_sha256"] == training_manifest["source_sha256"],
                         "Formal metadata and initial manifest source hashes differ")
                with zipfile.ZipFile(selected) as archive:
                    model_data = json.loads(archive.read("data"))
                _same(best["timesteps"], model_data["num_timesteps"], f"{mode} {seed}: checkpoint steps")
                _same(best["timesteps"], training_metadata["best_checkpoint_timesteps"], f"{mode} {seed}: metadata best steps")
                _same(best["score"], training_metadata["best_validation_score"], f"{mode} {seed}: best score")
                _same(max(float(r["score"]) for r in history), best["score"], f"{mode} {seed}: history maximum")
                source = training_metadata["source_sha256"]
                differences = [{"file": name, "training_sha256": original,
                    "current_sha256": current_sources.get(name)} for name, original in source.items()
                    if current_sources.get(name) != original]
                model_records.append({
                    "mode": mode, "seed": seed,
                    "selected_model": selected.relative_to(ROOT).as_posix(),
                    "selected_model_sha256": _sha(selected),
                    "best_checkpoint_timesteps": best["timesteps"],
                    "actual_training_timesteps": training_metadata["actual_timesteps"],
                    "training_source_sha256": source,
                    "source_files_changed_since_training": differences,
                    "training_manifest_sha256": _sha(run_dir / "training_manifest.json"),
                    "training_metadata_sha256": _sha(run_dir / "training_metadata.json"),
                    "best_checkpoint_sha256": _sha(run_dir / "best_checkpoint.json"),
                })
        checks["six_selected_model_hashes_and_validation_provenance"] = len(model_records) == 6
        report["input_sha256"] = {name: _sha(results_dir / name) for name in (
            "test_scenarios.json", "episode_metrics.csv", "summary.csv", "evaluation_manifest.json")}
        report["paired_noise_scope"] = "Raw traces contain true states and external disturbance, not sensor-noise draws. Matching scenario noise settings/seeds are checked; noise-draw sharing is tested separately by the environment regression suite."
        report["source_difference_policy"] = "Original training-source hashes are never rewritten. Differences record post-training continuation/logging, plotting or audit improvements and must be read with the project work explanation."
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report.update({"checked_episode_count": checked_episodes, "checked_sample_count": sample_count,
                   "wall_time_s": time.perf_counter() - started})
    report["status"] = "passed" if not errors and checks and all(checks.values()) and len(model_records) == 6 else "failed"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Result verification: {report['status']}; {checked_episodes} episodes; {len(errors)} errors; "
          f"{report['wall_time_s']:.2f} s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_results(args.results, args.models, args.output)
    if report["errors"]:
        for error in report["errors"][:10]:
            print(error, file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
