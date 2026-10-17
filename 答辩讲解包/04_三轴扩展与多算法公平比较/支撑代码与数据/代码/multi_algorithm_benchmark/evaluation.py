"""Held-out, paired comparisons of PID/PPO/SAC/TD3 on both attitude plants."""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .controllers import ModelController, PIDController
from .metrics import evaluate_batch
from .scenarios import make_scenarios


PID_NAMES = {"legacy": "baseline_PID", "tuned": "tuned_PID", "improved": "improved_PID"}
ALGORITHMS = ("PPO", "SAC", "TD3")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot export empty result table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    """Per-case statistics retain failed trajectories and separate seed variation."""
    groups = {}
    for row in rows:
        key = (row["axis"], row["controller"], row["family"], row["case_id"], row["level"])
        groups.setdefault(key, []).append(row)
    output = []
    for (axis, controller, family, case_id, level), group in groups.items():
        seed_groups = {}
        for row in group:
            seed_groups.setdefault(row["model_seed"], []).append(row)
        seed_success = [np.mean([r["success"] for r in g]) for g in seed_groups.values()]
        seed_error = [np.mean([r["tail_mean_error_deg"] for r in g]) for g in seed_groups.values()]
        settled = [r["settling_time_s"] for r in group if r["settling_time_s"] is not None]
        record = {"axis": axis, "controller": controller, "family": family, "case_id": case_id,
                  "level": level, "n_episodes": len(group), "n_model_seeds": len(seed_groups),
                  "success_rate": float(np.mean([r["success"] for r in group])),
                  "success_seed_std": float(np.std(seed_success, ddof=1)) if len(seed_success) > 1 else None,
                  "tail_error_seed_std": float(np.std(seed_error, ddof=1)) if len(seed_error) > 1 else None,
                  "settling_time_s_mean_successful": float(np.mean(settled)) if settled else None,
                  "settled_episodes": len(settled)}
        for field in ("rms_error_deg", "tail_mean_error_deg", "tail_max_error_deg",
                      "tail_max_rate_deg_s", "control_cost", "final_error_deg", "final_rate_deg_s"):
            record[field + "_mean"] = float(np.mean([r[field] for r in group]))
        record["failed_rate"] = float(np.mean([r.get("failed", False) for r in group]))
        output.append(record)
    return output


def paired_tables(rows):
    """Compare each scene after averaging learning seeds, never treating seeds as scenes."""
    groups = {}
    for row in rows:
        groups.setdefault((row["axis"], row["scenario_id"], row["controller"]), []).append(row)
    by_scene = {}
    fields = ("rms_error_deg", "tail_mean_error_deg", "control_cost")
    for (axis, scene_id, controller), group in groups.items():
        means = {field: float(np.mean([r[field] for r in group])) for field in fields}
        means["success_rate"] = float(np.mean([r["success"] for r in group]))
        by_scene.setdefault((axis, scene_id), {})[controller] = {
            **means, "family": group[0]["family"], "case_id": group[0]["case_id"]}
    pairwise, improvements = [], []
    for (axis, scene_id), controllers in by_scene.items():
        for left, right in combinations(sorted(controllers), 2):
            a, b = controllers[left], controllers[right]
            record = {"axis": axis, "scenario_id": scene_id, "family": a["family"],
                      "case_id": a["case_id"], "controller_a": left, "controller_b": right}
            for field in (*fields, "success_rate"):
                record[field + "_a"] = a[field]
                record[field + "_b"] = b[field]
                record[field + "_b_minus_a"] = b[field] - a[field]
            pairwise.append(record)
        if "improved_PID" in controllers:
            for baseline in ("baseline_PID", "tuned_PID"):
                if baseline not in controllers:
                    continue
                a, b = controllers[baseline], controllers["improved_PID"]
                record = {"axis": axis, "scenario_id": scene_id, "family": a["family"],
                          "case_id": a["case_id"], "baseline": baseline,
                          "improved": "improved_PID"}
                for field in (*fields, "success_rate"):
                    record[field + "_baseline"] = a[field]
                    record[field + "_improved"] = b[field]
                    record[field + "_change"] = b[field] - a[field]
                improvements.append(record)
    return pairwise, improvements


def policy_calibration_pairs(rows):
    """Pair each calibrated policy with its identical checkpoint seed and scene."""
    keyed = {}
    for row in rows:
        key = (row["axis"], row["controller"], row["model_seed"], row["scenario_id"])
        if key in keyed:
            raise ValueError(f"Duplicate controller/seed/scenario result: {key}")
        keyed[key] = row
    pairs = []
    fields = ("rms_error_deg", "tail_mean_error_deg", "tail_max_error_deg",
              "tail_max_rate_deg_s", "control_cost", "final_error_deg", "final_rate_deg_s")
    for key, calibrated in keyed.items():
        axis, controller, seed, scene = key
        if not controller.endswith("_calibrated"):
            continue
        algorithm = controller.removesuffix("_calibrated")
        raw_key = (axis, algorithm, seed, scene)
        if raw_key not in keyed:
            raise ValueError(f"Calibrated result has no raw checkpoint/scene partner: {key}")
        raw = keyed[raw_key]
        pair = {"axis": axis, "algorithm": algorithm, "model_seed": seed,
                "scenario_id": scene, "case_id": raw["case_id"], "family": raw["family"],
                "sample_id": raw.get("sample_id", 0),
                "success_raw": raw["success"], "success_calibrated": calibrated["success"],
                "success_change": int(calibrated["success"]) - int(raw["success"]),
                "success_recovered": bool(calibrated["success"] and not raw["success"]),
                "success_lost": bool(raw["success"] and not calibrated["success"]),
                "settling_time_s_raw": raw["settling_time_s"],
                "settling_time_s_calibrated": calibrated["settling_time_s"],
                "trace_file_raw": raw["trace_file"], "trace_file_calibrated": calibrated["trace_file"]}
        for field in fields:
            pair[field + "_raw"] = raw[field]
            pair[field + "_calibrated"] = calibrated[field]
            pair[field + "_change"] = calibrated[field] - raw[field]
        pairs.append(pair)
    return pairs


def _export_results(output_dir, axis, scenes, descriptors):
    rows, descriptor_records = [], []
    for descriptor in descriptors:
        controller = descriptor.pop("object")
        descriptor_records.append(dict(descriptor))
        results = evaluate_batch(axis, scenes, controller)
        trace_dir = Path(output_dir) / "raw_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            scene, metrics = result["scenario"], result["metrics"]
            name = f"{descriptor['controller']}_seed_{descriptor['model_seed']}_{scene['scenario_id']}.npz"
            relative = Path("raw_traces") / name
            metadata = {"axis": axis, "controller": descriptor, "scenario": scene, "metrics": metrics}
            np.savez_compressed(Path(output_dir) / relative, **result["trace"],
                                metadata_json=json.dumps(metadata, ensure_ascii=False))
            rows.append({"axis": axis, "scenario_id": scene["scenario_id"],
                         "case_id": scene.get("case_id", scene.get("family", "unknown")),
                         "family": scene.get("family", "unknown"), "level": scene.get("level", 0),
                         "sample_id": scene.get("sample_id", 0),
                         "scenario_seed": scene.get("noise_seed", 0),
                         "controller": descriptor["controller"], "model_seed": descriptor["model_seed"],
                         **metrics, "trace_file": relative.as_posix()})
        print(f"Evaluated {axis} {descriptor['controller']} seed={descriptor['model_seed']}: {len(results)} full-horizon scenes", flush=True)
    write_json(Path(output_dir) / "test_scenarios.json", scenes)
    write_json(Path(output_dir) / "controllers.json", descriptor_records)
    write_csv(Path(output_dir) / "episode_metrics.csv", rows)
    write_csv(Path(output_dir) / "summary.csv", summarize(rows))
    return rows, descriptor_records


def evaluate_models(models_dir, output_dir, samples_per_case=20, seeds=(11, 22, 33),
                    include_historical=True):
    """Evaluate every requested fresh model; absent runs fail before reading test results."""
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    from .improvements import CalibratedModelController

    torch.set_num_threads(1)
    models_dir, output_dir = Path(models_dir), Path(output_dir)
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if samples_per_case < 1:
        raise ValueError("samples_per_case must be positive")
    pid_file = models_dir / "pid_configs.json"
    gains = json.loads(pid_file.read_text(encoding="utf-8"))
    classes = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    paths = {(axis, algorithm, seed): models_dir / axis / algorithm / f"seed_{seed}" / "best_model.zip"
             for axis in ("single", "three") for algorithm in ALGORITHMS for seed in seeds}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Requested fresh models are incomplete: " + ", ".join(missing))
    incomplete = [str(p.parent / "training_metadata.json") for p in paths.values()
                  if not (p.parent / "training_metadata.json").exists()]
    if incomplete:
        raise FileNotFoundError("Only completed fresh training runs may enter formal testing: " + ", ".join(incomplete))
    training_records = {key: json.loads((path.parent / "training_metadata.json").read_text(encoding="utf-8"))
                        for key, path in paths.items()}
    for axis in ("single", "three"):
        budgets = {record["requested_steps"] for key, record in training_records.items() if key[0] == axis}
        if len(budgets) != 1:
            raise ValueError(f"Unequal requested interaction budgets within {axis}: {budgets}")
    for key, path in paths.items():
        if sha256(path) != training_records[key]["model_sha256"]:
            raise ValueError(f"Best model differs from completed-training hash: {path}")
    calibration_manifest_file = models_dir / "policy_calibration_manifest.json"
    calibration_validation = json.loads(calibration_manifest_file.read_text(encoding="utf-8"))
    validated = {(record["axis"], record["algorithm"], int(record["seed"]))
                 for record in calibration_validation["runs"]}
    if not set(paths).issubset(validated):
        raise ValueError("Every requested calibrated checkpoint must first have validation evidence")
    if calibration_validation["split"] != "validation" or calibration_validation["test_used_for_design_or_selection"]:
        raise ValueError("Calibration evidence must use validation only")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, controller_records, scene_counts = [], {}, {}
    for axis in ("single", "three"):
        scenes = make_scenarios(axis, split="test", samples_per_case=samples_per_case)
        scene_counts[axis] = len(scenes)
        descriptors = []
        axis_gains = gains[axis]
        for kind, label in PID_NAMES.items():
            selected = axis_gains.get(kind)
            if isinstance(selected, dict) and "gains" in selected:
                selected = selected["gains"]
            descriptors.append({"controller": label, "model_seed": 0, "source": "validation_pid_configs",
                                "pid_gains": selected, "object": PIDController(axis, kind=kind, gains=selected)})
        for algorithm in ALGORITHMS:
            for seed in seeds:
                path = paths[axis, algorithm, seed]
                model = classes[algorithm].load(path, device="cpu")
                training = training_records[axis, algorithm, seed]
                common = {"model_seed": seed, "model_path": str(path), "model_sha256": sha256(path),
                          "checkpoint_steps": int(model.num_timesteps),
                          "actual_training_steps": training["actual_steps"],
                          "requested_training_steps": training["requested_steps"],
                          "gradient_updates": training["gradient_updates"]}
                descriptors.append({**common, "controller": algorithm,
                                    "source": "fresh_matched_budget_training", "object": ModelController(model, axis)})
                calibrated_controller = CalibratedModelController(model, axis)
                zero_action = np.asarray(calibrated_controller.zero_action).reshape(-1).tolist()
                descriptors.append({**common, "controller": algorithm + "_calibrated",
                                    "source": "same_checkpoint_zero_observation_action_calibration",
                                    "raw_controller": algorithm, "zero_action_normalized": zero_action,
                                    "calibration_rule": "clip(predict(measured_observation)-predict(zero_observation), -1, 1)",
                                    "object": calibrated_controller})
        rows, records = _export_results(output_dir / axis, axis, scenes, descriptors)
        for row in rows:
            row["trace_file"] = f"{axis}/" + row["trace_file"]
        all_rows.extend(rows)
        controller_records[axis] = records
    write_csv(output_dir / "episode_metrics.csv", all_rows)
    write_csv(output_dir / "summary.csv", summarize(all_rows))
    pairwise, improvements = paired_tables(all_rows)
    write_csv(output_dir / "pairwise_results.csv", pairwise)
    write_csv(output_dir / "paired_improvements.csv", improvements)
    calibration = policy_calibration_pairs(all_rows)
    write_csv(output_dir / "policy_calibration_pairs.csv", calibration)
    manifest = {"evaluation_date": "2026-10-12", "split": "test", "samples_per_case": samples_per_case,
                "scene_counts": scene_counts, "episode_count": len(all_rows), "training_seeds": seeds,
                "controllers": controller_records, "pid_configs_sha256": sha256(pid_file),
                "paired_scene_comparison_count": len(pairwise),
                "policy_calibration_pair_count": len(calibration),
                "policy_calibration_validation_sha256": sha256(calibration_manifest_file),
                "selection_policy": "PID gains and learning checkpoints use validation only; test results cannot select controllers",
                "holding_policy": "20 seconds of deployment, then simultaneous error <= 1 degree and rate <= 0.5 degree/s throughout final 5 seconds",
                "seed_statistics": "Standard deviation across independent training seeds; pairwise CSV averages seeds within each held-out scene",
                "historical_policy": "Historical model deployment results are exported separately and excluded from fresh-training ranking",
                "policy_calibration": "All algorithms/seeds use the same predeclared zero-observation action subtraction, validated before held-out test; calibrated variants are separately labeled and never silently replace raw models",
                "policy_calibration_pairing": "Same axis, algorithm, checkpoint training seed, initial state, disturbance and noise realization; raw and calibrated traces saved for every pair"}
    manifest["evaluation_source_sha256"] = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifest["test_scenario_sha256"] = {axis: sha256(output_dir / axis / "test_scenarios.json") for axis in ("single", "three")}
    if include_historical:
        manifest["historical"] = evaluate_historical(output_dir / "historical", samples_per_case)
    write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest


def evaluate_historical(output_dir, samples_per_case=20):
    """Deploy the four archived policies on identical plants; do not retrain or rank budgets."""
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    from .archive_compat import load_archived_model

    torch.set_num_threads(1)
    package = Path(__file__).resolve().parents[1]
    archive = package / "01_遴选原始参考" / "05_中期前任务收尾新增材料" / "原样工程" / "大一立项AI在卫星姿态调整中的应用"
    specification = [("single", "PPO", archive / "单轴强化学习" / "ppo_satellite_final.zip"),
                     ("three", "PPO", archive / "三轴强化学习" / "ppo_satellite3d.zip"),
                     ("three", "SAC", archive / "三轴强化学习" / "sac_satellite3d.zip"),
                     ("three", "TD3", archive / "三轴强化学习" / "td3_satellite3d.zip")]
    missing = [str(path) for _, _, path in specification if not path.exists()]
    if missing:
        raise FileNotFoundError("Historical archive models missing: " + ", ".join(missing))
    classes, all_rows, metadata = {"PPO": PPO, "SAC": SAC, "TD3": TD3}, [], {}
    for axis in ("single", "three"):
        scenes = make_scenarios(axis, split="test", samples_per_case=samples_per_case)
        descriptors = []
        for candidate_axis, algorithm, path in specification:
            if candidate_axis != axis:
                continue
            model = load_archived_model(classes[algorithm], path)
            descriptors.append({"controller": f"historical_{algorithm}", "model_seed": "unrecorded",
                                "source": "archived_model_deployment", "model_path": str(path),
                                "model_sha256": sha256(path), "actual_training_steps": int(model.num_timesteps),
                                "original_action_is_physical_torque": True,
                                "object": ModelController(model, axis, physical_action=True)})
        rows, records = _export_results(Path(output_dir) / axis, axis, scenes, descriptors)
        for row in rows:
            row["trace_file"] = f"{axis}/" + row["trace_file"]
        all_rows.extend(rows)
        metadata[axis] = records
    write_csv(Path(output_dir) / "episode_metrics.csv", all_rows)
    write_csv(Path(output_dir) / "summary.csv", summarize(all_rows))
    manifest = {"episode_count": len(all_rows), "controllers": metadata,
                "archive_loading_adapter": "NumPy 2 metadata numpy._core maps to numpy.core; Box bounds/shape/dtype restored without unused space RNG; source ZIPs and policy weights remain unchanged",
                "purpose": "Historical deployment robustness, not an equal-training-budget algorithm comparison",
                "deployment_shift": "Single-axis historical torque limit 2 N m becomes 0.5 N m; three-axis limit remains 2 N m; all policies use new fixed-horizon integration and deployment observation timing"}
    write_json(Path(output_dir) / "evaluation_manifest.json", manifest)
    return manifest
