"""Fresh matched SAC training, common validation and auditable checkpoints.

Only validation_scenarios is imported; no held-out responses are used here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np

from .envs import InertiaSatelliteEnv, axis_name, mode_name


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_rank(stats, step):
    """Higher calibrated score, lower RMS/cost, then earlier checkpoint wins."""
    values = (float(stats["score"]), -float(stats["rms_error_deg"]),
              -float(stats["control_cost"]), -int(step))
    if not np.all(np.isfinite(values)):
        raise ValueError("checkpoint metrics must be finite")
    return values


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    inputs = [Path(__file__), Path(__file__).with_name("envs.py"),
              Path(__file__).with_name("vector_training.py"),
              Path(__file__).with_name("scenarios.py"), Path(__file__).with_name("protocol.py")]
    inputs.extend(root/"multi_algorithm_benchmark"/name for name in
                  ("envs.py", "vector_training.py", "controllers.py", "metrics.py", "improvements.py"))
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in inputs}


def train_model(output_dir, axis, mode, seed, steps=1000000, n_envs=8,
                validation_interval=100000, development=False, env_config=None):
    """Train one fresh SAC and return its complete metadata.

    output_dir is the *models root*, not a run path. Formal runs strictly enforce
    the predeclared budget. Existing run directories are never overwritten.
    A developer run requires development=True and is recorded as ineligible for
    final research. It can pass a short env_config horizon for interface tests.
    """
    axis, mode, seed = axis_name(axis), mode_name(mode), int(seed)
    if steps <= 0 or n_envs < 1 or validation_interval < 1:
        raise ValueError("training budget, n_envs and validation interval must be positive")
    if not development and (steps != 1000000 or n_envs != 8 or validation_interval != 100000
                            or seed not in (11, 22, 33) or env_config):
        raise ValueError("Formal runs require 1,000,000 steps, 8 envs, 100,000 validation interval, seed 11/22/33 and unchanged physics")
    run = Path(output_dir) / axis / mode / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"status": "initializing", "axis": axis, "mode": mode, "seed": seed,
              "requested_steps": int(steps), "actual_steps": 0, "development": bool(development),
              "eligible_for_final_research": not development}
    write_json(run/"session_status.json", status)
    env, model, callback = None, None, None
    try:
        import gymnasium
        import stable_baselines3
        import torch
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import DummyVecEnv
        from multi_algorithm_benchmark.controllers import ModelController
        from multi_algorithm_benchmark.improvements import CalibratedModelController
        from multi_algorithm_benchmark.metrics import evaluate_batch, validation_score
        from .scenarios import validation_scenarios
        from .vector_training import FastThreeVecEnv

        torch.set_num_threads(1)
        set_random_seed(seed, using_cuda=False)
        scenes = validation_scenarios(axis, nominal=False)
        if len(scenes) != 20:
            raise ValueError("SAC selection requires the common 20-scene validation set")
        write_json(run/"validation_scenarios.json", scenes)

        class RecordInitialStates(gymnasium.Wrapper):
            def __init__(self, wrapped, rank):
                super().__init__(wrapped)
                self.rank = rank
                self.stream = (run/f"training_episode_initial_states_{rank}.csv").open("x", newline="", encoding="utf-8-sig")
                self.writer = csv.DictWriter(self.stream, fieldnames=["rank", "episode_index", "stream_seed", "mode", "theta_deg", "omega_deg_s", "inertia_scale", "noise_seed"])
                self.writer.writeheader()

            def reset(self, **kwargs):
                observation, info = self.env.reset(**kwargs)
                record = info["episode_initial_state"]
                row = {key: record[key] for key in ("episode_index", "stream_seed", "mode", "theta_deg", "omega_deg_s", "inertia_scale", "noise_seed")}
                self.writer.writerow({"rank": self.rank, **row})
                self.stream.flush()
                return observation, info

            def close(self):
                if not self.stream.closed:
                    self.stream.close()
                return self.env.close()

        def factory(rank):
            def create():
                plant = InertiaSatelliteEnv(axis, mode, env_config)
                recorded = RecordInitialStates(plant, rank)
                return Monitor(recorded, filename=str(run/f"monitor_{rank}.monitor.csv"))
            return create

        if axis == "three":
            env = FastThreeVecEnv(n_envs, seed, run, mode, env_config)
        else:
            env = DummyVecEnv([factory(rank) for rank in range(n_envs)])
        env.seed(seed)
        common = dict(policy="MlpPolicy", env=env, learning_rate=3e-4, gamma=.995,
                      policy_kwargs={"net_arch": [64, 64]}, seed=seed, device="cpu", verbose=0)
        specific = dict(buffer_size=200000, learning_starts=20000, batch_size=256,
                        tau=.005, train_freq=(4, "step"), gradient_steps=2,
                        ent_coef="auto_0.01", target_entropy=-float(env.action_space.shape[0]))
        model = SAC(**common, **specific)
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "gymnasium": gymnasium.__version__,
                    "stable_baselines3": stable_baselines3.__version__}
        probe = InertiaSatelliteEnv(axis, mode, env_config)
        manifest = {**status, "algorithm": "SAC", "label": "SAC_N" if mode == "nominal" else "SAC_R",
                    "n_envs": n_envs, "validation_interval": validation_interval,
                    "validation_scene_count": len(scenes), "validation_scenarios_sha256": sha256(run/"validation_scenarios.json"),
                    "env_config": probe.config_dict(), "versions": versions,
                    "common_hyperparameters": {key: value for key, value in common.items() if key != "env"},
                    "algorithm_hyperparameters": specific, "source_sha256": source_hashes(),
                    "selection_split": "validation", "selection_policy": "calibrated",
                    "selection_score": "100*success_rate-rms_error_deg-0.2*control_cost",
                    "tie_break": ["higher_score", "lower_rms", "lower_control_cost", "earlier_checkpoint"],
                    "calibration": "clip(predict(obs)-predict(zeros),-1,1)",
                    "raw_ablation_uses_calibrated_selected_checkpoint": True,
                    "fresh_training": True, "test_used_for_selection": False,
                    "rank_seed_rule": "seed + rank; SeedSequence.spawn(initial,inertia,noise_seed)",
                    "initial_state_stream_independent_of_inertia": True,
                    "vector_engine": "V2 FastThreeVecEnv RK4 + pure inertia sampler" if axis == "three" else "DummyVecEnv over V2 batch exact integration"}
        probe.close()
        write_json(run/"training_manifest.json", manifest)

        class Validation(BaseCallback):
            def __init__(self):
                super().__init__()
                self.raw_rows, self.calibrated_rows = [], []
                self.best_rank = None
                self.best_steps = 0
                self.best_zero_action = None
                self.next_eval = validation_interval

            def evaluate(self):
                raw = ModelController(self.model, axis)
                calibrated = CalibratedModelController(self.model, axis)
                raw_score, raw_stats = validation_score(evaluate_batch(axis, scenes, raw, keep_traces=False))
                cal_score, cal_stats = validation_score(evaluate_batch(axis, scenes, calibrated, keep_traces=False))
                step = int(self.num_timesteps)
                zero_action = calibrated.zero_action.tolist()
                common_row = {"timesteps": step, "zero_action_normalized": json.dumps(zero_action), "validation_scene_count": 20}
                raw_row = {**common_row, **raw_stats}
                cal_row = {**common_row, **cal_stats}
                self.raw_rows.append(raw_row)
                self.calibrated_rows.append(cal_row)
                write_csv(run/"validation_history_raw.csv", self.raw_rows)
                write_csv(run/"validation_history_calibrated.csv", self.calibrated_rows)
                self.model.save(run/"latest_model.zip")
                rank = checkpoint_rank(cal_stats, step)
                if self.best_rank is None or rank > self.best_rank:
                    self.best_rank, self.best_steps, self.best_zero_action = rank, step, zero_action
                    self.model.save(run/"best_model.zip")
                    write_json(run/"best_checkpoint.json", {"timesteps": step, "selection_split": "validation", "selection_policy": "calibrated",
                        "rank": list(rank), "raw": raw_row, "calibrated": cal_row,
                        "zero_action_normalized": zero_action, "model_sha256": sha256(run/"best_model.zip")})
                write_json(run/"latest_checkpoint.json", {"timesteps": step, "raw": raw_row, "calibrated": cal_row,
                    "zero_action_normalized": zero_action, "model_sha256": sha256(run/"latest_model.zip")})
                write_json(run/"session_status.json", {**status, "status": "running", "actual_steps": step,
                    "last_validation_steps": step, "best_checkpoint_steps": self.best_steps,
                    "wall_time_s": time.perf_counter()-started})
                print(f"[{axis} {mode} SAC seed={seed}] steps={step} calibrated_success={cal_stats['success_rate']:.3f} score={cal_score:.3f} raw_score={raw_score:.3f}", flush=True)

            def _on_step(self):
                if self.num_timesteps >= self.next_eval:
                    self.evaluate()
                    self.next_eval += validation_interval
                return True

            def _on_training_end(self):
                if not self.calibrated_rows or self.calibrated_rows[-1]["timesteps"] != self.num_timesteps:
                    self.evaluate()

        callback = Validation()
        write_json(run/"session_status.json", {**status, "status": "running"})
        model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
        model.save(run/"final_model.zip")
        # Training-end validation already saved latest and its matching hash;
        # avoid rewriting a ZIP whose timestamps could invalidate that record.
        model.save_replay_buffer(run/"replay_buffer.pkl")
        artifacts = {name: {"sha256": sha256(run/name), "bytes": (run/name).stat().st_size,
                            "timesteps": callback.best_steps if name == "best_model.zip" else int(model.num_timesteps)}
                     for name in ("best_model.zip", "final_model.zip", "latest_model.zip", "replay_buffer.pkl")}
        source_after = source_hashes()
        metadata = {**manifest, "status": "complete", "actual_steps": int(model.num_timesteps),
                    "gradient_updates": int(model._n_updates), "best_checkpoint_steps": callback.best_steps,
                    "best_validation_score": float(callback.best_rank[0]),
                    "best_zero_action_normalized": callback.best_zero_action,
                    "validation_checkpoint_steps": [row["timesteps"] for row in callback.calibrated_rows],
                    "checkpoint_count": len(callback.calibrated_rows), "artifacts": artifacts,
                    "model_sha256": artifacts["best_model.zip"]["sha256"],
                    "source_sha256_after": source_after,
                    "runtime_source_unchanged": source_after == manifest["source_sha256"],
                    "wall_time_s": time.perf_counter()-started}
        if not development and model.num_timesteps != steps:
            raise RuntimeError("formal training interaction budget was not exact")
        if source_after != manifest["source_sha256"]:
            raise RuntimeError("runtime source files changed during training; formal evidence must be audited")
        write_json(run/"training_metadata.json", metadata)
        write_json(run/"session_status.json", {**status, "status": "complete", "actual_steps": int(model.num_timesteps),
            "best_checkpoint_steps": callback.best_steps, "model_sha256": metadata["model_sha256"],
            "checkpoint_count": metadata["checkpoint_count"], "wall_time_s": metadata["wall_time_s"]})
        return metadata
    except BaseException as exc:
        failure = {**status, "status": "incomplete", "actual_steps": int(model.num_timesteps) if model is not None else 0,
                   "error_type": type(exc).__name__, "error": str(exc), "wall_time_s": time.perf_counter()-started}
        if model is not None:
            try:
                model.save(run/"interrupted_model.zip")
                model.save_replay_buffer(run/"interrupted_replay_buffer.pkl")
                failure["interrupted_model_sha256"] = sha256(run/"interrupted_model.zip")
                failure["interrupted_replay_sha256"] = sha256(run/"interrupted_replay_buffer.pkl")
            except Exception as save_exc:
                failure["recovery_save_error"] = repr(save_exc)
        write_json(run/"session_status.json", failure)
        raise
    finally:
        if env is not None:
            env.close()
