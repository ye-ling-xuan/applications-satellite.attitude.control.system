"""Matched fresh training for PPO, SAC and TD3 with validation-only selection."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_model(output_dir, axis, algorithm, seed, steps=1000000, n_envs=8,
                validation_interval=100000):
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .envs import make_env
    from .controllers import ModelController
    from .metrics import evaluate_batch, validation_score
    from .scenarios import validation_scenarios

    if axis not in ("single", "three") or algorithm not in ("PPO", "SAC", "TD3"):
        raise ValueError("Unsupported axis or algorithm")
    if steps <= 0 or n_envs < 1 or validation_interval < 1:
        raise ValueError("Training budgets must be positive")
    torch.set_num_threads(1)
    set_random_seed(seed, using_cuda=False)
    run = Path(output_dir) / axis / algorithm / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=False)
    scenes = validation_scenarios(axis)
    write_json(run / "validation_scenarios.json", scenes)

    def factory(rank):
        def create():
            env = make_env(axis, randomized=True)
            env.reset(seed=seed + 1009 * rank)
            return Monitor(env, filename=str(run / f"monitor_{rank}.monitor.csv"))
        return create

    if axis == "three":
        from .vector_training import FastThreeVecEnv
        env = FastThreeVecEnv(n_envs=n_envs, seed=seed, monitor_dir=run, randomized=True)
    else:
        env = DummyVecEnv([factory(rank) for rank in range(n_envs)])
    env.seed(seed)
    common = dict(policy="MlpPolicy", env=env, learning_rate=3e-4, gamma=0.995,
                  policy_kwargs={"net_arch": [64, 64]}, seed=seed, device="cpu", verbose=0)
    if algorithm == "PPO":
        specific = dict(n_steps=256, batch_size=256, n_epochs=10, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.0)
        common["policy_kwargs"]["log_std_init"] = -0.7
        model = PPO(**common, **specific)
    else:
        specific = dict(buffer_size=200000, learning_starts=20000, batch_size=256,
                        tau=0.005, train_freq=(4, "step"), gradient_steps=2)
        if algorithm == "SAC":
            specific.update(ent_coef="auto_0.01", target_entropy=-float(env.action_space.shape[0]))
            model = SAC(**common, **specific)
        else:
            sigma = 0.08
            specific.update(policy_delay=2, target_policy_noise=0.1, target_noise_clip=0.2)
            model = TD3(**common, **specific,
                        action_noise=NormalActionNoise(np.zeros(env.action_space.shape[0]),
                                                       sigma * np.ones(env.action_space.shape[0])))
            specific["action_noise_normalized_sigma"] = sigma
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(__file__).parent.glob("*.py"))}
    probe = make_env(axis, randomized=True)
    env_config = probe.config_dict() if callable(probe.config_dict) else probe.config_dict
    probe.close()
    manifest = {"axis": axis, "algorithm": algorithm, "seed": seed,
                "requested_steps": steps, "n_envs": n_envs, "env_config": env_config,
                "common_hyperparameters": {k: v for k, v in common.items() if k != "env"},
                "algorithm_hyperparameters": specific, "validation_interval": validation_interval,
                "source_sha256": source, "selection_split": "validation",
                "fresh_training": True, "test_used_for_selection": False,
                "vector_engine": "FastThreeVecEnv batch RK4" if axis == "three" else "DummyVecEnv single-axis exact integration",
                "fairness": "Same axis physics, observations, action mapping, reward, randomized training distribution and interaction budget; algorithm-specific update rules are recorded, not identical."}
    write_json(run / "training_manifest.json", manifest)

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.best_score = -np.inf
            self.best_steps = 0
            self.next_eval = validation_interval

        def evaluate(self):
            controller = ModelController(self.model, axis)
            score, stats = validation_score(evaluate_batch(axis, scenes, controller))
            row = {"timesteps": int(self.num_timesteps), **stats}
            self.rows.append(row)
            write_csv(run / "validation_history.csv", self.rows)
            self.model.save(run / "latest_model.zip")
            if score > self.best_score:
                self.best_score, self.best_steps = score, int(self.num_timesteps)
                self.model.save(run / "best_model.zip")
                write_json(run / "best_checkpoint.json", {**row, "selection_split": "validation"})
            print(f"[{axis} {algorithm} seed={seed}] steps={self.num_timesteps} "
                  f"validation_success={stats['success_rate']:.3f} score={score:.3f}", flush=True)

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.evaluate()
                self.next_eval = self.num_timesteps + validation_interval
            return True

        def _on_training_end(self):
            if not self.rows or self.rows[-1]["timesteps"] != self.num_timesteps:
                self.evaluate()

    callback = Validation()
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
        model.save(run / "final_model.zip")
        if algorithm != "PPO":
            # Save replay state, making this completed off-policy run reviewable.
            model.save_replay_buffer(run / "replay_buffer.pkl")
        metadata = {**manifest, "actual_steps": int(model.num_timesteps),
                    "gradient_updates": int(model._n_updates),
                    "best_checkpoint_steps": callback.best_steps,
                    "best_validation_score": float(callback.best_score),
                    "wall_time_s": time.perf_counter() - started,
                    "model_sha256": hashlib.sha256((run / "best_model.zip").read_bytes()).hexdigest(),
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "torch": torch.__version__, "gymnasium": gymnasium.__version__,
                                 "stable_baselines3": stable_baselines3.__version__}}
        write_json(run / "training_metadata.json", metadata)
        return metadata
    except BaseException as exc:
        write_json(run / "session_status.json", {"status": "incomplete", "actual_steps": int(model.num_timesteps),
                                                 "error_type": type(exc).__name__})
        raise
    finally:
        env.close()
