"""Train either direct PPO or a bounded PPO correction to a fixed PD controller."""
import argparse
import hashlib
from pathlib import Path
import torch
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from .env import AttitudeEnv
from .experiments import create_output, load_config, contract, provenance, save_json
from .tasks import Task
from .validation import ValidationCallback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pure", "residual"], default="pure")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument("--profile", choices=["v1", "precision_nominal", "precision_robust"], default="v1")
    parser.add_argument("--torque-penalty", type=float, default=0.5)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--validation-interval", type=int, default=20000)
    args = parser.parse_args(argv)
    if min(args.steps, args.n_envs, args.n_steps, args.batch_size, args.validation_interval) <= 0:
        parser.error("steps, env counts, batch size and validation interval must be positive")
    if args.n_steps * args.n_envs < args.batch_size or args.n_steps * args.n_envs % args.batch_size:
        parser.error("rollout size must be a multiple of batch size")
    if not 0 < args.gamma <= 1 or args.learning_rate <= 0 or args.ent_coef < 0:
        parser.error("invalid optimizer or discount settings")
    out = create_output(args.output or f"artifacts/{args.mode}_{args.profile}_seed{args.seed}")
    cfg = load_config(args.tuning)
    torch.set_num_threads(1)
    task = Task(args.profile, args.torque_penalty)
    probe = AttitudeEnv(cfg, args.mode, randomize=not args.nominal_only, task=task)
    check_env(probe, warn=True)
    probe.close()
    env = DummyVecEnv([lambda i=i: Monitor(
        AttitudeEnv(cfg, args.mode, randomize=not args.nominal_only, task=task),
        str(out / f"training_{i}")) for i in range(args.n_envs)])
    params = dict(learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
                  gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef)
    robust = not args.nominal_only and args.profile != "precision_nominal"
    angle_bound = min(60 if args.profile == "v1" else 40, cfg.failure_angle_deg)
    metadata = {"mode": args.mode, "seed": args.seed, "requested_timesteps": args.steps,
                "contract": contract(cfg), "hyperparameters": params,
                "task": task.to_dict(), "n_envs": args.n_envs,
                "training_domain": {"initial_error_deg": [-angle_bound, angle_bound],
                   "initial_rate_deg_s": [0.0, 0.0] if args.profile == "precision_nominal" else [-10, 10],
                   "inertia": [0.7 * cfg.inertia, 1.3 * cfg.inertia] if robust else [cfg.inertia, cfg.inertia],
                   "bias_torque_nm": [-0.03, 0.03] if robust else [0.0, 0.0]},
                "hidden_parameters": ["inertia", "bias_torque"] if robust else [],
                "note": "Hidden random parameters (if enabled) make the task partially observed; no online identification claim.",
                "selection_rule": "validation success, failure, tail error, cost (lexicographic)",
                "versions": {"torch": torch.__version__, "gymnasium": gymnasium.__version__,
                             "stable_baselines3": stable_baselines3.__version__},
                "provenance": provenance(), "status": "training"}
    save_json(out / "metadata.json", metadata)
    try:
        model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0, **params)
        callback = ValidationCallback(cfg, args.mode, task, out, args.validation_interval)
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(out / "final_model")
        metadata.update(status="complete", actual_timesteps=model.num_timesteps,
                        selected_checkpoint_timesteps=callback.best_step,
                        selected_validation_key=callback.best_key,
                        model_sha256=hashlib.sha256((out / "model.zip").read_bytes()).hexdigest(),
                        final_model_sha256=hashlib.sha256((out / "final_model.zip").read_bytes()).hexdigest())
        save_json(out / "metadata.json", metadata)
        print(f"Saved {args.mode} PPO: {out / 'model.zip'} ({model.num_timesteps} steps)")
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=str(exc))
        save_json(out / "metadata.json", metadata)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
