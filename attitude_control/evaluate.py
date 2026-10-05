"""Paired held-out evaluation of PD, PID, pure PPO, and residual PPO."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import Config
from .controllers import PID, action_command
from .experiments import (create_output, load_config, save_json, contract, provenance,
                          scenario_suite, scenario_dict, check_contract)
from .simulation import rollout
from .tasks import Task, PolicyAdapter


def load_policy(directory, config, mode):
    from stable_baselines3 import PPO
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    check_contract(metadata, config, mode)
    model_path = directory / "model.zip"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if metadata.get("status") != "complete" or digest != metadata.get("model_sha256"):
        raise ValueError("incomplete model or checkpoint hash mismatch")
    model = PPO.load(model_path, device="cpu")
    if model.observation_space.shape != (3,) or model.action_space.shape != (1,):
        raise ValueError("policy dimensions do not match the current environment")
    if not np.allclose(model.action_space.low, -1) or not np.allclose(model.action_space.high, 1):
        raise ValueError("policy action bounds do not match normalized torque mapping")
    task = Task.from_dict(metadata["task"]) if "task" in metadata else Task()
    return PolicyAdapter(model, task), {"directory": str(directory.resolve()), "sha256": digest,
                   "training_seed": metadata["seed"], "training_steps": metadata["actual_timesteps"],
                   "task": task.to_dict()}


def summarize(rows):
    summaries = []
    for controller in sorted({row["controller"] for row in rows}):
        selected = [row for row in rows if row["controller"] == controller]
        for group in ["all"] + sorted({row["group"] for row in selected}):
            subset = selected if group == "all" else [r for r in selected if r["group"] == group]
            settling = [r["settling_time_s"] for r in subset if r["success"]]
            summaries.append({"controller": controller, "group": group, "episodes": len(subset),
                "success_rate": float(np.mean([r["success"] for r in subset])),
                "failure_rate": float(np.mean([r["failed"] for r in subset])),
                "mean_settling_time_success_only_s": float(np.mean(settling)) if settling else None,
                **{f"mean_{k}": float(np.mean([r[k] for r in subset])) for k in [
                    "overshoot_deg", "tail_mean_abs_error_deg", "torque_effort_nm2_s", "cost_integral"]}})
    return summaries


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_example(traces, path, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for label, trace in traces.items():
        axes[0].plot(trace["time_s"], trace["angle_deg"], label=label)
        axes[1].plot(trace["time_s"], trace["rate_deg_s"], label=label)
        axes[2].step(trace["time_s"][:-1], trace["torque_nm"], where="post", label=label)
    axes[0].axhline(config.target_deg, color="gray", linestyle="--")
    axes[0].set_ylabel("Angle (deg)")
    axes[1].set_ylabel("Rate (deg/s)")
    axes[2].set_ylabel("Applied torque (N m)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("Held-out nominal maneuver: same state and actuator limits")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/evaluation_v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--pure", type=Path)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.tuning)
    pid_config = replace(config, ki=0.3)
    if args.tuning:
        pid_config = Config(**json.loads(args.tuning.read_text(encoding="utf-8"))["pid_config"])
    policies, models = {}, {}
    for mode, directory in [("pure", args.pure), ("residual", args.residual)]:
        if directory:
            policies[mode], models[mode] = load_policy(directory, config, mode)
    out = create_output(args.output)
    suite = list(scenario_suite(config, args.seed))
    rows, example_traces = [], {}
    for group, scenario in suite:
        methods = {
            "PD": PID(config.kp, config.kd, config.ki),
            "PID": PID(pid_config.kp, pid_config.kd, pid_config.ki),
            **{"PPO" if m == "pure" else "PD+PPO": p for m, p in policies.items()},
        }
        for name, controller in methods.items():
            if name in ("PD", "PID"):
                command = lambda ep, obs, c=controller: c.command(ep)
            else:
                mode = "pure" if name == "PPO" else "residual"
                command = lambda ep, obs, c=controller, m=mode: action_command(
                    c.predict(obs, deterministic=True)[0], ep, m)
            result, trace = rollout(config, scenario, command)
            rows.append({"controller": name, "group": group, "scenario": scenario.name, **result})
            np.savez_compressed(out / f"{scenario.name}_{name.replace('+', '_')}.npz", **trace)
            if scenario.name == "nominal_4":
                example_traces[name] = trace
    summaries = summarize(rows)
    write_csv(out / "episodes.csv", rows)
    write_csv(out / "summary.csv", summaries)
    save_json(out / "summary.json", summaries)
    save_json(out / "manifest.json", {"evaluation_seed": args.seed, "contract": contract(config),
               "pid_config": pid_config.to_dict(), "models": models,
               "scenarios": [{"group": group, **scenario_dict(s)} for group, s in suite],
               "provenance": provenance(),
               "note": "Single-seed model evaluation is a pilot, not statistical evidence of RL superiority."})
    if args.plot:
        plot_example(example_traces, out / "response.png", config)
    for row in summaries:
        if row["group"] == "all":
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
