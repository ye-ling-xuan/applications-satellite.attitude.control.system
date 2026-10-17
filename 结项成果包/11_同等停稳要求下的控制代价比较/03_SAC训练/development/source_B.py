"""Direct SAC with time observation and an identical 8 s settling requirement.

Training and checkpoint selection use development/validation scenes only.  This
module deliberately offers no test command.  Physics is copied from the prior
verified vector engine; reward and observations are specific to this study.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
from gymnasium import spaces
from multi_algorithm_benchmark.envs import euler_to_quaternion, normalize_quaternion
from multi_algorithm_benchmark.vector_training import FastThreeVecEnv
from .evaluation import evaluate, rank
from .protocol import STUDY, PACKAGE, DT, LIMIT, DURATION, scenarios, write_json, sha256


REWARD_CONFIGS = {
    "A": {"effort_weight": .5, "early_angle": .10, "ramp_angle": 6.,
          "hold_angle": 500., "hold_rate": 80., "early_rate": .02},
    "B": {"effort_weight": .3, "early_angle": .2, "ramp_angle": 3.,
          "hold_angle": 60., "hold_rate": 6., "early_rate": .1,
          "reward_multiplier": "inverse_dt"},
}


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def augment_observation(obs, elapsed):
    """Physical error in rad, rate scaled by 2, remaining time in [-1.5,1]."""
    obs = np.asarray(obs, np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    d = obs.shape[-1] // 2
    physical = obs.copy()
    physical[:, d:] *= 2.
    t = np.broadcast_to(np.asarray(elapsed), (len(obs),))
    return np.column_stack((physical, (8.-t)/8.)).astype(np.float32)


class SACController:
    """Time-dependent direct neural action, with explicit optional calibration."""
    def __init__(self, model, axis, calibrated=True):
        self.model, self.axis, self.calibrated = model, axis, bool(calibrated)

    def act(self, observation, time):
        augmented = augment_observation(observation, time)
        action = self.model.predict(augmented, deterministic=True)[0]
        if self.calibrated:
            zero = np.zeros_like(augmented)
            zero[:, -1] = augmented[:, -1]
            action = action - self.model.predict(zero, deterministic=True)[0]
        return np.clip(action, -1., 1.)

    def reset(self, n):
        return None

    @classmethod
    def load(cls, path, axis, calibrated=True):
        from stable_baselines3 import SAC
        return cls(SAC.load(path, device="cpu"), axis, calibrated)


FreshSACController = SACController


class DeadlineVecEnv(FastThreeVecEnv):
    """Same exact single-axis / RK4 three-axis physics, fresh reset streams."""
    def __init__(self, axis, reward_config="A", n_envs=8, seed=0, monitor_dir=None):
        self.axis = axis
        self.d = 1 if axis == "single" else 3
        self.reward_config = dict(REWARD_CONFIGS[reward_config]) if isinstance(reward_config, str) else dict(reward_config)
        super().__init__(n_envs, seed, monitor_dir, randomized=False,
                         config={"dt": DT[axis], "duration": DURATION, "max_torque": LIMIT[axis]})
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2*self.d+1,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(self.d,), dtype=np.float32)
        if self.d == 1:
            self.theta = np.zeros((n_envs, 1))
            for name in ("omega", "inertia", "torque", "disturbance"):
                setattr(self, name, np.zeros((n_envs, 1)))
        self._episode_cost = np.zeros(n_envs)
        self._episode_hold_ok = np.ones(n_envs, bool)
        self._episode_indices = np.zeros(n_envs, dtype=np.int64)
        self._initial_stream = self._initial_writer = None
        if monitor_dir is not None:
            self._initial_stream = (Path(monitor_dir)/"training_initial_states.csv").open("x", newline="", encoding="utf-8-sig")
            self._initial_writer = csv.DictWriter(self._initial_stream,
                fieldnames=["rank", "episode_index", "angle_deg", "omega_deg_s"])
            self._initial_writer.writeheader()

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self._seeds = [int(np.random.SeedSequence([int(seed), rank]).generate_state(1)[0])
                       for rank in range(self.num_envs)]
        return self._seeds

    @property
    def error_vector(self):
        if self.d == 1:
            return (self.theta+np.pi) % (2*np.pi)-np.pi
        return super().error_vector

    def _observation(self, error=None):
        error = self.error_vector if error is None else error
        return augment_observation(np.concatenate((error, self.omega), axis=1), self.time)

    def _reset_row(self, rank, seed=None, options=None):
        if seed is not None:
            self._rngs[rank] = np.random.default_rng(int(seed))
        rng = self._rngs[rank]
        angles = rng.uniform(-60., 60., self.d)
        rates = rng.uniform(-5., 5., self.d)
        opts = dict(options or {})
        if self.d == 1:
            angles = np.atleast_1d(opts.get("theta_deg", angles))
            rates = np.atleast_1d(opts.get("omega_deg_s", rates))
            self.theta[rank] = np.radians(angles)
            self.inertia[rank] = 1.
        else:
            angles = np.asarray(opts.get("euler_deg", angles))
            rates = np.asarray(opts.get("omega_deg_s", rates))
            self.q[rank] = (normalize_quaternion(opts["quaternion"]) if "quaternion" in opts
                            else euler_to_quaternion(np.radians(angles)))
            self.inertia[rank] = [1.2, 1., .8]
        self.omega[rank] = np.radians(rates)
        self.time[rank] = 0.
        self.step_count[rank] = 0
        self.failed[rank] = False
        self.torque[rank] = 0.
        self.disturbance[rank] = 0.
        self._episode_returns[rank] = 0.
        self._episode_lengths[rank] = 0
        self._episode_cost[rank] = 0.
        self._episode_hold_ok[rank] = True
        self.reset_infos[rank] = {"t": 0., "failed": False}
        self._scenes[rank] = {"angle_deg": angles.tolist(), "omega_deg_s": rates.tolist()}
        if self._initial_writer is not None:
            self._initial_writer.writerow({"rank": rank, "episode_index": int(self._episode_indices[rank]),
                "angle_deg": json.dumps(angles.tolist()), "omega_deg_s": json.dumps(rates.tolist())})
            self._initial_stream.flush()
        self._episode_indices[rank] += 1

    def step_async(self, actions):
        if self._closed or not self._ready or self._actions is not None:
            raise RuntimeError("reset or finish the pending step first")
        actions = np.asarray(actions, float)
        if actions.shape != (self.num_envs, self.d) or not np.isfinite(actions).all():
            raise ValueError("invalid action shape or nonfinite action")
        self._actions = actions.copy()

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("no pending action")
        actions, self._actions = self._actions, None
        self.torque = np.clip(actions, -1., 1.)*self.max_torque
        next_time = np.minimum(self.duration, (self.step_count+1)*self.dt)
        h = (next_time-self.time)[:, None]
        if self.d == 1:
            acceleration = self.torque/self.inertia
            self.theta += self.omega*h+.5*acceleration*h*h
            self.omega += acceleration*h
        else:
            q, w = self.q, self.omega
            a1, b1 = self._derivative(q, w, self.torque)
            a2, b2 = self._derivative(q+.5*h*a1, w+.5*h*b1, self.torque)
            a3, b3 = self._derivative(q+.5*h*a2, w+.5*h*b2, self.torque)
            a4, b4 = self._derivative(q+h*a3, w+h*b3, self.torque)
            self.q = normalize_quaternion(q+h*(a1+2*a2+2*a3+a4)/6)
            self.omega = w+h*(b1+2*b2+2*b3+b4)/6
        self.time, self.step_count = next_time, self.step_count+1
        error = self.error_vector
        e2, w2 = np.sum(error**2, axis=1), np.sum(self.omega**2, axis=1)
        edeg, wdeg = np.degrees(np.sqrt(e2)), np.degrees(np.sqrt(w2))
        terminated = wdeg > 300.
        if self.d == 1:
            terminated |= np.abs(np.degrees(self.theta[:, 0])) > 170.
        self.failed = terminated.copy()
        truncated = self.time >= self.duration-1e-10
        hold = self.time >= 8.-1e-10
        inside = (edeg <= 1.) & (wdeg <= .5)
        self._episode_hold_ok &= (~hold | inside) & ~terminated
        cfg = self.reward_config
        progress = np.minimum(self.time/8., 1.)
        angle_weight = np.where(hold, cfg["hold_angle"], cfg["early_angle"]+cfg["ramp_angle"]*progress**4)
        rate_weight = np.where(hold, cfg["hold_rate"], cfg["early_rate"])
        effort = np.sum(self.torque**2, axis=1)
        # Per-second integral; no reward for arriving before the required time.
        # Dense late penalties give the terminal constraint a learnable signal;
        # hard pass/fail remains the independent evaluator's job.
        rewards = -(cfg["effort_weight"]*effort+angle_weight*e2+rate_weight*w2)*self.dt
        if cfg.get("reward_multiplier") == "inverse_dt":
            rewards /= self.dt
        rewards -= terminated*1000.
        self._episode_returns += rewards
        self._episode_lengths += 1
        self._episode_cost += effort*self.dt
        dones = terminated | truncated
        observations = self._observation(error)
        infos = [{"t": float(self.time[i]), "error_deg": float(edeg[i]), "rate_deg_s": float(wdeg[i]),
                  "torque": self.torque[i].copy(), "failed": bool(terminated[i]),
                  "TimeLimit.truncated": bool(truncated[i] and not terminated[i])} for i in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            i = int(i)
            episode = {"r": float(self._episode_returns[i]), "l": int(self._episode_lengths[i]),
                       "t": round(time.time()-self._started, 6)}
            infos[i].update(episode=episode, terminal_observation=observations[i].copy(),
                            control_cost=float(self._episode_cost[i]), success_8s=bool(self._episode_hold_ok[i]))
            if self._monitor_writers:
                self._monitor_writers[i].writerow(episode)
                self._monitor_streams[i].flush()
            self._reset_row(i)
        if dones.any():
            observations[dones] = self._observation()[dones]
        return observations, rewards.astype(np.float32), dones, infos

    def close(self):
        if self._initial_stream is not None and not self._initial_stream.closed:
            self._initial_stream.close()
        super().close()


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("protocol.py"), Path(__file__).with_name("evaluation.py"),
             PACKAGE/"multi_algorithm_benchmark"/"envs.py", PACKAGE/"multi_algorithm_benchmark"/"vector_training.py"]
    return {str(p.relative_to(PACKAGE)).replace("\\", "/"): sha256(p) for p in paths}


def train(axis, seed, steps=1000000, config="A", stage="development", n_envs=8, validation_interval=100000):
    """Fresh training; run paths cannot be reused or overwritten."""
    import gymnasium
    import torch
    import stable_baselines3
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed

    if axis not in DT or config not in REWARD_CONFIGS or stage not in ("development", "formal", "smoke"):
        raise ValueError("invalid axis/config/stage")
    if stage == "formal" and (seed not in (41, 42, 43) or steps != 1000000 or n_envs != 8 or validation_interval != 100000):
        raise ValueError("formal runs require seeds41/42/43, 1m steps, 8 envs and 100k interval")
    run = STUDY/"03_SAC训练"/stage/axis/f"config_{config}"/f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    hashes = source_hashes()
    status = {"axis": axis, "seed": int(seed), "config": config, "stage": stage, "requested_steps": int(steps),
              "actual_steps": 0, "fresh_training": True, "eligible_for_formal_results": stage == "formal"}
    write_json(run/"session_status.json", {**status, "status": "initializing"})
    env, model = None, None
    try:
        torch.set_num_threads(1)
        set_random_seed(seed, using_cuda=False)
        env = DeadlineVecEnv(axis, config, n_envs, seed, run)
        env.seed(seed)
        scenes = scenarios(axis, "validation")
        write_json(run/"validation_scenarios.json", scenes)
        settings = dict(learning_rate=3e-4, gamma=.999 if axis == "single" else .9995,
                        policy_kwargs={"net_arch": [64, 64]}, buffer_size=200000,
                        learning_starts=10000, batch_size=256, tau=.005,
                        train_freq=(4, "step"), gradient_steps=2, ent_coef="auto_0.01",
                        target_entropy=-float(env.d), seed=seed, device="cpu", verbose=0)
        model = SAC("MlpPolicy", env, **settings)
        manifest = {**status, "algorithm": "SAC", "n_envs": n_envs, "validation_interval": validation_interval,
                    "observation": "physical error_rad, 2*omega_rad_s, (8-time)/8",
                    "duration_s": 20., "deadline_s": 8., "dt": env.dt, "max_torque": env.max_torque,
                    "reward": REWARD_CONFIGS[config], "reward_formula": "-dt*(effort_weight*sum(torque^2)+angle_weight(t)*sum(error_rad^2)+rate_weight(t)*sum(omega_rad_s^2)); divergence -1000",
                    "reward_multiplier": "inverse_dt for B; multiplies all integral terms by same positive constant within each task",
                    "angle_weight": "early_angle+ramp_angle*(t/8)^4 before8s; hold_angle at/after8s",
                    "rate_weight": "early_rate before8s; hold_rate at/after8s",
                    "validation_selection": "maximize success_8s count, minimize complete-horizon mean control_cost, then prefer earlier checkpoint; compare raw/calibrated deployment",
                    "calibration": "clip(pi(error,rate,t)-pi(0,0,t),-1,1)",
                    "formal_test_used_for_selection": False, "hyperparameters": settings,
                    "environment_stream_seed_rule": "uint32 SeedSequence([training_seed, vector_rank]).generate_state(1)[0]",
                    "environment_stream_seeds": env.seed(seed),
                    "source_sha256": hashes, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__}}
        write_json(run/"training_manifest.json", manifest)

        class Validation(BaseCallback):
            def __init__(self):
                super().__init__()
                self.history, self.episodes = [], []
                self.next_eval = validation_interval
                self.best_rank = None
                self.best = None

            def evaluate(self):
                step = int(self.num_timesteps)
                checkpoint = run/f"checkpoint_{step:07d}.zip"
                self.model.save(checkpoint)
                for calibrated in (False, True):
                    mode = "calibrated" if calibrated else "raw"
                    rows = evaluate(axis, scenes, SACController(self.model, axis, calibrated))["metrics"]
                    mean_cost = float(np.mean([r["control_cost"] for r in rows]))
                    summary = {"timesteps": step, "deployment": mode, "validation_count": len(rows),
                        "success_4s": sum(r["success_4s"] for r in rows), "success_6s": sum(r["success_6s"] for r in rows),
                        "success_8s": sum(r["success_8s"] for r in rows), "mean_control_cost": mean_cost,
                        "mean_rms_error_deg": float(np.mean([r["rms_error_deg"] for r in rows]))}
                    self.history.append(summary)
                    self.episodes.extend({"timesteps": step, "deployment": mode, "scenario_id": s["scenario_id"], **r} for s, r in zip(scenes, rows))
                    candidate_rank = (*rank(rows, 8.), -step, int(calibrated))
                    if self.best_rank is None or candidate_rank > self.best_rank:
                        self.best_rank = candidate_rank
                        self.model.save(run/"best_model.zip")
                        self.best = {**summary, "rank": list(candidate_rank), "checkpoint": checkpoint.name,
                                     "model_path": str((run/"best_model.zip").relative_to(STUDY)).replace("\\", "/"),
                                     "seed": seed, "axis": axis, "calibrated": calibrated,
                                     "model_sha256": sha256(run/"best_model.zip"), "selection_split": "validation"}
                        write_json(run/"best_checkpoint.json", self.best)
                    print(f"[{stage} {axis} config={config} seed={seed}] steps={step} {mode}:8s={summary['success_8s']}/{len(rows)} cost={mean_cost:.5f}", flush=True)
                write_csv(run/"validation_history.csv", self.history)
                write_csv(run/"validation_episode_metrics.csv", self.episodes)
                write_json(run/"session_status.json", {**status, "status": "running", "actual_steps": step,
                    "best": self.best, "wall_time_s": time.perf_counter()-started})

            def _on_step(self):
                if self.num_timesteps >= self.next_eval:
                    self.evaluate()
                    self.next_eval += validation_interval
                return True

            def _on_training_end(self):
                if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
                    self.evaluate()

        callback = Validation()
        model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
        model.save(run/"final_model.zip")
        after = source_hashes()
        metadata = {**manifest, "status": "complete", "actual_steps": int(model.num_timesteps),
                    "gradient_updates": int(model._n_updates), "best": callback.best,
                    "wall_time_s": time.perf_counter()-started, "source_sha256_after": after,
                    "runtime_source_unchanged": hashes == after,
                    "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in run.glob("*.zip")}}
        write_json(run/"training_metadata.json", metadata)
        if stage == "formal" and (hashes != after or model.num_timesteps != steps):
            raise RuntimeError("formal source hashes changed or interaction budget differs from frozen design")
        write_json(run/"session_status.json", {**status, "status": "complete", "actual_steps": int(model.num_timesteps),
                   "best": callback.best, "wall_time_s": metadata["wall_time_s"]})
        return metadata
    except BaseException as exc:
        write_json(run/"session_status.json", {**status, "status": "incomplete", "actual_steps": int(model.num_timesteps) if model is not None else 0,
                   "error": repr(exc), "wall_time_s": time.perf_counter()-started})
        if model is not None:
            model.save(run/"interrupted_model.zip")
        raise
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=tuple(DT), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--config", choices=tuple(REWARD_CONFIGS), default="A")
    parser.add_argument("--stage", choices=("development", "formal", "smoke"), default="development")
    parser.add_argument("--validation-interval", type=int, default=100000)
    args = parser.parse_args(argv)
    train(args.axis, args.seed, args.steps, args.config, args.stage, validation_interval=args.validation_interval)


if __name__ == "__main__":
    main()
