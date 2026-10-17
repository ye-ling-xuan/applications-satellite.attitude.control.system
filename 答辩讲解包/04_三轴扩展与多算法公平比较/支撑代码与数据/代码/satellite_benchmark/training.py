"""Reproducible PPO training with validation-only checkpoint selection."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import SatelliteEnv
from .evaluation import (_write_csv, _write_json, evaluate_batch,
                         validation_scenarios, validation_score)


def training_config(mode: str, base_config: EnvConfig | None = None) -> EnvConfig:
    base = base_config or EnvConfig()
    if mode == "nominal":
        return replace(base, randomized=False, inertia=1.0, disturbance_bias=0.0,
                       disturbance_amplitude=0.0, noise_theta_deg=0.0,
                       noise_omega_deg=0.0, pulse_torque=0.0)
    if mode == "randomized":
        return replace(base, randomized=True, inertia_range=(0.7, 1.3),
                       disturbance_range=(-0.02, 0.02), noise_range_deg=(0.0, 0.15),
                       disturbance_amplitude=0.02, pulse_torque=0.0)
    raise ValueError("mode must be nominal or randomized")


def train_model(output_dir: Path, mode: str, seed: int, total_timesteps: int = 200000,
                base_config: EnvConfig | None = None, n_envs: int = 8,
                validation_interval: int = 20000, preset: str = "full",
                learning_rate: float = 3e-4, resume: bool = False) -> dict:
    """Train one independent seed; caller may run modes/seeds in separate processes.

    Both modes use exactly the same PPO hyperparameters. `best_model.zip` is
    selected by ten fixed validation scenes; `final_model.zip` is always saved.
    `preset` is descriptive (quick/full); total_timesteps is the authoritative
    requested budget and is never silently reduced. With ``resume=True`` this
    budget is additional, not a new cumulative target. Resuming restores the
    policy/optimizer and step count, but not the environment RNG or an unfinished
    rollout. Each continuation gets its own monitor files and source manifest.
    """
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv

    if total_timesteps <= 0 or n_envs <= 0 or validation_interval <= 0:
        raise ValueError("timesteps, n_envs and validation_interval must be positive")
    torch.set_num_threads(1)
    set_random_seed(seed, using_cuda=False)
    config = training_config(mode, base_config)
    run_dir = Path(output_dir) / mode / f"seed_{seed}"
    scenes = validation_scenarios(base_config)
    hyperparameters = {
        "policy": "MlpPolicy", "learning_rate": learning_rate,
        "n_steps": max(64, 2048 // n_envs), "batch_size": 256, "n_epochs": 10,
        "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2,
        "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -0.7}, "device": "cpu",
        "seed": seed, "verbose": 0,
    }
    if hyperparameters["n_steps"] * n_envs < hyperparameters["batch_size"]:
        hyperparameters["batch_size"] = hyperparameters["n_steps"] * n_envs
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    manifest = {
        "mode": mode, "seed": seed, "requested_timesteps": total_timesteps,
        "n_envs": n_envs, "env_config": asdict(config),
        "hyperparameters": hyperparameters, "source_sha256": source_hashes,
        "selection_split": "validation", "test_data_used_in_training_or_selection": False,
    }
    final_path = run_dir / "final_model.zip"
    prior_rows = []
    best_score, best_timesteps = -np.inf, 0
    prior_metadata = None
    if resume:
        # Refuse an ambiguous restart before creating any files. In particular,
        # an interrupted run can continue from latest_model without a final zip.
        manifest_path = run_dir / "training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without the original manifest: {manifest_path}")
        initial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        canonical = lambda value: json.dumps(value, sort_keys=True)
        for field in ("mode", "seed", "n_envs", "env_config", "hyperparameters"):
            if canonical(initial_manifest.get(field)) != canonical(manifest[field]):
                raise ValueError(f"Cannot resume with changed {field}; use a new output directory")
        stored_scenes = run_dir / "validation_scenarios.json"
        if not stored_scenes.exists() or canonical(json.loads(stored_scenes.read_text(encoding="utf-8"))) != canonical(scenes):
            raise ValueError("Cannot resume with a different or missing validation split")
        candidates = []
        # The final checkpoint wins a step-count tie: its last PPO update has
        # finished, whereas a callback checkpoint can be from inside a rollout.
        for filename in ("latest_model.zip", "final_model.zip"):
            path = run_dir / filename
            if path.exists():
                checkpoint = PPO.load(path, device="cpu")
                candidates.append((int(checkpoint.num_timesteps), filename == "final_model.zip", path, checkpoint))
        if not candidates:
            raise FileNotFoundError(f"No final/latest checkpoint to resume in {run_dir}")
        _, _, resume_path, model = max(candidates, key=lambda item: item[:2])
        history_path = run_dir / "validation_history.csv"
        if history_path.exists():
            with history_path.open(encoding="utf-8-sig", newline="") as stream:
                prior_rows = list(csv.DictReader(stream))
        best_path = run_dir / "best_checkpoint.json"
        if not prior_rows or not best_path.exists() or not (run_dir / "best_model.zip").exists():
            raise ValueError("Cannot resume without validation history and the selected best checkpoint")
        best_record = json.loads(best_path.read_text(encoding="utf-8"))
        best_score = float(best_record["score"])
        best_timesteps = int(best_record["timesteps"])
        if max(int(row["timesteps"]) for row in prior_rows) > model.num_timesteps or best_timesteps > model.num_timesteps:
            raise ValueError("Validation records are ahead of the available checkpoint; restore a consistent run")
        metadata_path = run_dir / "training_metadata.json"
        if metadata_path.exists():
            prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Keep the initial manifest and all old monitors byte-for-byte. Separate
        # segments also retain monitor time origins and failed-session evidence.
        index = 1
        while True:
            session_dir = run_dir / f"resume_{index:04d}"
            try:
                session_dir.mkdir()
                break
            except FileExistsError:
                index += 1
        manifest.update({"resume_from": resume_path.name,
                         "resume_checkpoint_sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
                         "initial_timesteps": int(model.num_timesteps),
                         "initial_source_sha256": initial_manifest["source_sha256"]})
        _write_json(session_dir / "training_manifest.json", manifest)
        if prior_metadata is not None:
            _write_json(session_dir / "previous_training_metadata.json", prior_metadata)
    else:
        # Even a directory from an incomplete run must not be silently replaced.
        # mkdir is atomic, so two fresh workers cannot claim the same seed path.
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(f"Training directory already exists: {run_dir}; use resume or a new output directory") from exc
        initial_manifest = manifest
        session_dir = run_dir
        _write_json(run_dir / "training_manifest.json", manifest)
        _write_json(run_dir / "validation_scenarios.json", scenes)

    def factory(rank: int):
        def make():
            env = SatelliteEnv(config)
            env.reset(seed=seed + rank * 1009)
            return Monitor(env, filename=str(session_dir / f"monitor_{rank}.monitor.csv"))
        return make

    vec_env = DummyVecEnv([factory(i) for i in range(n_envs)])
    vec_env.seed(seed)
    if resume:
        model.set_env(vec_env)
    else:
        model = PPO(**hyperparameters, env=vec_env)

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = prior_rows.copy()
            self.best_score = best_score
            self.best_timesteps = best_timesteps
            self.next_eval = int(model.num_timesteps) + validation_interval

        def evaluate(self):
            score, stats = validation_score(evaluate_batch(scenes, model=self.model))
            row = {"timesteps": int(self.num_timesteps), **stats}
            self.rows.append(row)
            _write_csv(run_dir / "validation_history.csv", self.rows)
            self.model.save(run_dir / "latest_model.zip")
            if score > self.best_score:
                self.best_score = score
                self.best_timesteps = int(self.num_timesteps)
                self.model.save(run_dir / "best_model.zip")
                _write_json(run_dir / "best_checkpoint.json", {
                    **row, "selection_split": "validation",
                    "scenario_count": len(scenes),
                    "training_session": session_dir.relative_to(run_dir).as_posix(),
                    "source_sha256": source_hashes,
                })
            print(f"[{mode} seed={seed}] steps={self.num_timesteps} "
                  f"validation_success={stats['success_rate']:.2f} "
                  f"score={score:.3f}", flush=True)

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self.evaluate()
                self.next_eval = self.num_timesteps + validation_interval
            return True

        def _on_training_end(self):
            if not self.rows or int(self.rows[-1]["timesteps"]) != self.num_timesteps:
                self.evaluate()

    callback = ValidationCallback()
    start = time.perf_counter()
    initial_steps = int(model.num_timesteps)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback,
                    reset_num_timesteps=not resume, progress_bar=False)
        elapsed = time.perf_counter() - start
        model.save(final_path)
    except BaseException as exc:
        _write_json(session_dir / "session_status.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "initial_timesteps": initial_steps, "actual_timesteps": int(model.num_timesteps),
            "error_type": type(exc).__name__,
        })
        raise
    finally:
        vec_env.close()
    metadata = {
        "mode": mode, "seed": seed, "preset": preset,
        "requested_timesteps": total_timesteps, "initial_timesteps": initial_steps,
        "actual_timesteps": int(model.num_timesteps), "wall_time_s": elapsed,
        "n_envs": n_envs, "env_config": asdict(config),
        "hyperparameters": hyperparameters, "validation_interval": validation_interval,
        "selection_split": "validation", "validation_scene_count": len(scenes),
        "best_validation_score": float(callback.best_score),
        "best_checkpoint_timesteps": callback.best_timesteps,
        "selected_model": "best_model.zip", "final_model": "final_model.zip",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "gymnasium": gymnasium.__version__,
                     "stable_baselines3": stable_baselines3.__version__},
        "source_sha256": source_hashes,
        "initial_source_sha256": initial_manifest["source_sha256"],
        "resumed": resume, "training_session": session_dir.relative_to(run_dir).as_posix(),
        "resume_state_limit": "policy/optimizer/step count only; environment RNG and incomplete rollout are restarted",
        "test_data_used_in_training_or_selection": False,
    }
    if resume:
        _write_json(session_dir / "training_metadata.json", metadata)
    _write_json(run_dir / "training_metadata.json", metadata)
    _write_json(session_dir / "session_status.json", {"status": "completed",
                "initial_timesteps": initial_steps, "actual_timesteps": int(model.num_timesteps)})
    return metadata
