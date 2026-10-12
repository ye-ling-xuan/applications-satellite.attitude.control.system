"""Paired, deterministic validation and held-out evaluation for every controller.

Training/checkpoint selection and PID tuning only use the validation split.
Test scenarios are generated from a separate random seed and never select models.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .config import EnvConfig, PIDConfig
from .controllers import PIDController
from .env import SatelliteEnv
from .metrics import compute_metrics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_scenarios(split: str = "test", samples_per_case: int = 50,
                   base_config: EnvConfig | None = None) -> list[dict]:
    """16 cases; the same initial states/noise seeds are paired across controllers.

    Each factor is varied independently before two explicitly out-of-training-
    distribution combined cases. No large Cartesian product is constructed.
    """
    if split not in {"test", "validation"}:
        raise ValueError("split must be test or validation")
    if samples_per_case < 1:
        raise ValueError("samples_per_case must be positive")
    base = base_config or EnvConfig()
    # All evaluation configurations disable training randomization explicitly.
    base = replace(base, randomized=False, inertia=1.0, disturbance_bias=0.0,
                   disturbance_amplitude=0.0, noise_theta_deg=0.0,
                   noise_omega_deg=0.0, pulse_torque=0.0)
    cases: list[tuple[str, float, dict, bool]] = [("nominal", 0.0, {}, False)]
    cases += [("inertia", j, {"inertia": j}, j < 0.7 or j > 1.3)
              for j in (0.6, 0.8, 1.0, 1.2, 1.4)]
    cases += [("disturbance", d, {"disturbance_bias": d,
                                  "disturbance_amplitude": d}, d > 0.02)
              for d in (0.01, 0.03, 0.06)]
    cases += [("noise", n, {"noise_theta_deg": n, "noise_omega_deg": n}, n > 0.15)
              for n in (0.1, 0.3, 0.6)]
    cases += [("pulse", p, {"pulse_torque": p}, True) for p in (0.1, 0.25)]
    cases += [
        ("combined", 1.0, {"inertia": 0.6, "disturbance_bias": 0.03,
                           "disturbance_amplitude": 0.03, "noise_theta_deg": 0.3,
                           "noise_omega_deg": 0.3, "pulse_torque": 0.1}, True),
        ("combined", 2.0, {"inertia": 1.4, "disturbance_bias": -0.06,
                           "disturbance_amplitude": 0.06, "noise_theta_deg": 0.6,
                           "noise_omega_deg": 0.6, "pulse_torque": 0.25}, True),
    ]
    split_seed = 20261009 if split == "test" else 20260901
    rng = np.random.default_rng(split_seed)
    initial_theta = rng.uniform(-60.0, 60.0, samples_per_case)
    initial_omega = rng.uniform(-5.0, 5.0, samples_per_case)
    # Include both signs and nonzero rates even in small demonstration runs.
    if samples_per_case >= 2:
        initial_theta[:2] = (-45.0, 45.0)
        initial_omega[:2] = (-3.0, 3.0)
    records = []
    for case_idx, (family, level, changes, outside) in enumerate(cases):
        case_id = f"{family}_{level:g}"
        config = replace(base, **changes)
        for sample_idx in range(samples_per_case):
            records.append({
                "scenario_id": f"{split}_{case_idx:02d}_{sample_idx:03d}",
                "split": split, "case_id": case_id, "family": family,
                "level": level, "sample_id": sample_idx,
                "seed": split_seed + sample_idx * 101,
                "theta_deg": float(initial_theta[sample_idx]),
                "omega_deg_s": float(initial_omega[sample_idx]),
                "outside_training_range": outside, "env_config": asdict(config),
            })
    return records


def validation_scenarios(base_config: EnvConfig | None = None) -> list[dict]:
    """Ten fixed scenes used exclusively to tune PID/select PPO checkpoints."""
    scenes = make_scenarios("validation", 2, base_config)
    selected = [("nominal_0", 0), ("nominal_0", 1), ("inertia_0.8", 0),
                ("inertia_1.2", 1), ("disturbance_0.01", 0),
                ("disturbance_0.03", 1), ("noise_0.1", 0), ("noise_0.3", 1),
                ("pulse_0.1", 0), ("combined_1", 1)]
    records = [s for s in scenes if (s["case_id"], s["sample_id"]) in selected]
    for scene in records:
        # These fixed validation initial states are distinct from the test's
        # +/-45-degree demonstration pair and its independently drawn states.
        scene["theta_deg"] = -37.0 if scene["sample_id"] == 0 else 53.0
        scene["omega_deg_s"] = -2.5 if scene["sample_id"] == 0 else 4.5
    return records


def evaluate_batch(scenarios: list[dict], model: Any = None,
                   pid_config: PIDConfig | None = None) -> list[dict]:
    """One model prediction for all active scenes per step, preserving raw traces."""
    if not scenarios:
        return []
    if (model is None) == (pid_config is None):
        raise ValueError("provide exactly one of model or pid_config")
    envs, observations, traces, pids = [], [], [], []
    keys = ("t", "theta_deg", "omega_deg_s", "torque", "commanded_torque",
            "disturbance", "inertia", "failed")
    for scene in scenarios:
        config = EnvConfig(**scene["env_config"])
        env = SatelliteEnv(config)
        obs, info = env.reset(seed=scene["seed"], options={
            "theta_deg": scene["theta_deg"], "omega_deg_s": scene["omega_deg_s"],
            "inertia": config.inertia,
        })
        envs.append(env)
        observations.append(obs)
        initial = {"t": 0.0, "theta_deg": scene["theta_deg"],
                   "omega_deg_s": scene["omega_deg_s"], "torque": 0.0,
                   "commanded_torque": 0.0, "disturbance": 0.0,
                   "inertia": config.inertia, "failed": False}
        initial.update({k: info[k] for k in keys if k in info})
        traces.append({key: [initial[key]] for key in keys})
        if pid_config is not None:
            pid = PIDController(pid_config, max_torque=config.max_torque, dt=config.dt)
            pid.reset()
            pids.append(pid)
    active = list(range(len(envs)))
    while active:
        if model is not None:
            actions, _ = model.predict(np.asarray([observations[i] for i in active]),
                                       deterministic=True)
        else:
            actions = [pids[i].act(observations[i]) for i in active]
        next_active = []
        for batch_idx, env_idx in enumerate(active):
            obs, _, terminated, truncated, info = envs[env_idx].step(
                np.asarray(actions[batch_idx], dtype=np.float32).reshape(1))
            observations[env_idx] = obs
            for key in keys:
                traces[env_idx][key].append(info.get(key, False if key == "failed" else 0.0))
            if not (terminated or truncated):
                next_active.append(env_idx)
        active = next_active
    results = []
    for scene, env, raw in zip(scenarios, envs, traces):
        trace = {key: np.asarray(value) for key, value in raw.items()}
        trace["error_deg"] = trace["theta_deg"] - env.config.target_deg
        # Alias makes exported raw traces convenient for external plotting.
        trace["time_s"] = trace["t"]
        results.append({"scenario": scene, "trace": trace,
                        "metrics": compute_metrics(trace, env.config)})
        env.close()
    return results


def validation_score(results: list[dict]) -> tuple[float, dict]:
    """Success dominates; RMS error and control cost break ties."""
    metrics = [r["metrics"] for r in results]
    success = float(np.mean([m["success"] for m in metrics]))
    rms = float(np.mean([m["rms_error_deg"] for m in metrics]))
    cost = float(np.mean([m["control_cost"] for m in metrics]))
    score = 100.0 * success - rms - 0.2 * cost
    return score, {"success_rate": success, "rms_error_deg": rms,
                   "control_cost": cost, "score": score}


def tune_pid(output_dir: Path, base_config: EnvConfig | None = None,
             kp_values=(0.8, 1.5, 2.5, 4.0, 6.0, 10.0), ki_values=(0.0, 0.08, 0.2, 0.5),
             kd_values=(1.0, 2.0, 3.0, 4.0, 6.0)) -> PIDConfig:
    """Search 120 gain combinations on validation data only; export all trials."""
    output_dir = Path(output_dir)
    scenes = validation_scenarios(base_config)
    rows = []
    best_score = -np.inf
    best_config = None
    for kp, ki, kd in product(kp_values, ki_values, kd_values):
        config = PIDConfig(kp=float(kp), ki=float(ki), kd=float(kd), integral_limit=2.0)
        score, stats = validation_score(evaluate_batch(scenes, pid_config=config))
        rows.append({**asdict(config), **stats, "split": "validation"})
        if score > best_score:
            best_score, best_config = score, config
    if best_config is None:
        raise ValueError("PID search grid is empty")
    _write_csv(output_dir / "pid_search_validation.csv", rows)
    _write_json(output_dir / "pid_best.json", {
        "config": asdict(best_config), "validation_score": float(best_score),
        "selection_split": "validation", "scenario_count": len(scenes),
        "scenario_ids": [s["scenario_id"] for s in scenes],
    })
    _write_json(output_dir / "pid_validation_scenarios.json", scenes)
    return best_config


def summarize(rows: list[dict]) -> list[dict]:
    """Across-seed SD is separate from scene variability; failures stay failures."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["controller"], row["family"], row["case_id"],
                           row["level"]), []).append(row)
    summary = []
    for (controller, family, case_id, level), group in groups.items():
        seed_groups: dict[int, list[dict]] = {}
        for row in group:
            seed_groups.setdefault(int(row["model_seed"]), []).append(row)
        seed_success = [np.mean([r["success"] for r in g]) for g in seed_groups.values()]
        seed_rms = [np.mean([r["rms_error_deg"] for r in g]) for g in seed_groups.values()]
        settling = [r["settling_time_s"] for r in group if r["settling_time_s"] is not None]
        record = {
            "controller": controller, "family": family, "case_id": case_id,
            "level": level, "n_episodes": len(group), "n_model_seeds": len(seed_groups),
            "success_rate": float(np.mean([r["success"] for r in group])),
            "success_seed_std": float(np.std(seed_success, ddof=1)) if len(seed_success) > 1 else None,
            "rms_error_deg_mean": float(np.mean([r["rms_error_deg"] for r in group])),
            "rms_error_seed_std": float(np.std(seed_rms, ddof=1)) if len(seed_rms) > 1 else None,
            "steady_error_deg_mean": float(np.mean([r["steady_error_deg"] for r in group])),
            "control_cost_mean": float(np.mean([r["control_cost"] for r in group])),
            "settling_time_s_mean_successful": float(np.mean(settling)) if settling else None,
            "settled_episodes": len(settling),
            "failed_rate": float(np.mean([r["failed"] for r in group])),
            "outside_training_range": bool(group[0]["outside_training_range"]),
        }
        summary.append(record)
    return summary


def evaluate_models(output_dir: Path, models_dir: Path, samples_per_case: int = 50,
                    seeds=(11, 22, 33), base_config: EnvConfig | None = None,
                    pid_config: PIDConfig | None = None,
                    require_all_models: bool = True) -> dict:
    """Evaluate PID and all selected PPO checkpoints, never selecting on test."""
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique; repeated seeds duplicate episodes and overwrite traces")
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "raw_traces"
    trace_dir.mkdir(exist_ok=True)
    if pid_config is None:
        pid_path = models_dir / "pid_best.json"
        if pid_path.exists():
            pid_config = PIDConfig(**json.loads(pid_path.read_text(encoding="utf-8"))["config"])
        else:
            pid_config = tune_pid(models_dir, base_config)
    scenes = make_scenarios("test", samples_per_case, base_config)
    _write_json(output_dir / "test_scenarios.json", scenes)
    descriptors = [{"controller": "PID", "model_seed": 0, "path": None}]
    missing = []
    for mode in ("nominal", "randomized"):
        for seed in seeds:
            path = models_dir / mode / f"seed_{seed}" / "best_model.zip"
            if not path.exists():
                missing.append(str(path))
            else:
                descriptors.append({"controller": f"PPO_{mode}", "model_seed": seed,
                                    "path": str(path)})
    if require_all_models and missing:
        raise FileNotFoundError("Missing trained models: " + ", ".join(missing))
    rows = []
    for descriptor in descriptors:
        if descriptor["path"] is None:
            results = evaluate_batch(scenes, pid_config=pid_config)
        else:
            model = PPO.load(descriptor["path"], device="cpu")
            results = evaluate_batch(scenes, model=model)
        for result in results:
            scene = result["scenario"]
            trace_name = (f"{descriptor['controller']}_seed_{descriptor['model_seed']}_"
                          f"{scene['scenario_id']}.npz")
            relative = Path("raw_traces") / trace_name
            metadata = {**descriptor, "scenario": scene, "metrics": result["metrics"]}
            np.savez_compressed(output_dir / relative, **result["trace"],
                                metadata_json=json.dumps(metadata, ensure_ascii=False))
            rows.append({
                "scenario_id": scene["scenario_id"], "case_id": scene["case_id"],
                "family": scene["family"], "level": scene["level"],
                "sample_id": scene["sample_id"], "scenario_seed": scene["seed"],
                "controller": descriptor["controller"], "model_seed": descriptor["model_seed"],
                "theta_deg": scene["theta_deg"], "omega_deg_s": scene["omega_deg_s"],
                "outside_training_range": scene["outside_training_range"],
                **result["metrics"], "trace_file": relative.as_posix(),
            })
        print(f"Evaluated {descriptor['controller']} seed={descriptor['model_seed']}: "
              f"{len(results)} held-out scenes", flush=True)
    _write_csv(output_dir / "episode_metrics.csv", rows)
    summary = summarize(rows)
    _write_csv(output_dir / "summary.csv", summary)
    manifest = {
        "split": "test", "samples_per_case": samples_per_case,
        "scenario_count": len(scenes), "episode_count": len(rows),
        "controllers": descriptors, "missing_models": missing,
        "pid_config": asdict(pid_config), "env_base_config": asdict(base_config or EnvConfig()),
        "selection_policy": "PID gains and PPO checkpoints selected on fixed validation scenes only",
        "settling_policy": "Unsettled episodes have null settling time; never replaced by zero",
        "seed_std_policy": "Sample SD across independent training seeds; PID has no seed SD",
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest
