"""Multi-seed PPO versus independently tuned PID/PD on a frozen paired suite."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .core import Scenario
from .controllers import PID, action_command
from .evaluate import load_policy, summarize, write_csv
from .experiments import (create_output, load_config, provenance, save_json, scenario_dict, contract)
from .simulation import rollout


def held_out_suite(config, seed, count):
    rng = np.random.default_rng(seed)
    groups = [
        ("in_distribution", 40, 0, 1.0, 0.0, 0.0, 0.0),
        ("initial_state_shift", 60, 10, 1.0, 0.0, 0.0, 0.0),
        ("low_inertia", 60, 10, 0.5, 0.0, 0.0, 0.0),
        ("high_inertia", 60, 10, 1.5, 0.0, 0.0, 0.0),
        ("bias_positive", 60, 10, 1.0, 0.08, 0.0, 0.0),
        ("bias_negative", 60, 10, 1.0, -0.08, 0.0, 0.0),
        ("pulse_noise", 60, 10, 1.0, 0.0, 0.3, 0.01),
    ]
    suite = []
    for group, bound, rate, inertia, bias, pulse, noise in groups:
        for i in range(count):
            # Balance positive/negative maneuvers even for small sample counts.
            angle = float(rng.uniform(5, bound)) * (-1 if i % 2 == 0 else 1)
            scenario = Scenario(name=f"{group}_{i}", angle_deg=config.target_deg + angle,
                rate_deg_s=float(rng.uniform(-rate, rate)), inertia=inertia * config.inertia,
                bias_torque=bias, pulse_torque=pulse, noise_std=noise,
                seed=int(rng.integers(0, 2**31)))
            suite.append((group, scenario))
    return suite


def across_seed_summary(summaries, seed_names):
    rows = []
    for group in sorted({r["group"] for r in summaries}):
        subset = [r for r in summaries if r["group"] == group and r["controller"] in seed_names]
        row = {"group": group, "training_seeds": len(subset)}
        for key in ["success_rate", "mean_tail_mean_abs_error_deg", "mean_torque_effort_nm2_s", "mean_cost_integral"]:
            values = [r[key] for r in subset]
            row[f"ppo_{key}_mean_across_seeds"] = float(np.mean(values))
            row[f"ppo_{key}_sample_sd_across_seeds"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        for controller in ["PD", "PID"]:
            base = next(r for r in summaries if r["controller"] == controller and r["group"] == group)
            for key in ["success_rate", "mean_tail_mean_abs_error_deg", "mean_torque_effort_nm2_s"]:
                row[f"{controller.lower()}_{key}"] = base[key]
        rows.append(row)
    return rows


def make_plots(traces, model_dirs, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for col, group in enumerate(["in_distribution", "bias_positive"]):
        for label, trace in traces[group].items():
            t = trace["time_s"]
            axes[0, col].plot(t, trace["angle_deg"], label=label)
            axes[1, col].plot(t, trace["rate_deg_s"], label=label)
            axes[2, col].step(t[:-1], trace["torque_nm"], where="post", label=label)
        axes[0, col].axhspan(config.target_deg - config.angle_tolerance_deg,
                            config.target_deg + config.angle_tolerance_deg, color="gray", alpha=0.12)
        axes[1, col].axhspan(-config.rate_tolerance_deg_s, config.rate_tolerance_deg_s, color="gray", alpha=0.12)
        axes[0, col].set_title(group.replace("_", " ").title() + ": predetermined case 4")
        axes[2, col].set_xlabel("Time (s)")
        for row, label in enumerate(["Angle (deg)", "Rate (deg/s)", "Torque (N m)"]):
            axes[row, col].set_ylabel(label)
            axes[row, col].grid(True)
        axes[0, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "responses.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, directory in model_dirs.items():
        with (directory / "validation.csv").open(encoding="utf-8") as handle:
            data = list(csv.DictReader(handle))
        steps = [int(row["timesteps"]) for row in data]
        axes[0].plot(steps, [float(r["success_rate"]) for r in data], marker="o", label=name)
        axes[1].plot(steps, [float(r["tail_mean_abs_error_deg"]) for r in data], marker="o", label=name)
    axes[0].set_ylabel("Validation success rate")
    axes[1].set_ylabel("Validation tail error (deg)")
    axes[1].set_xlabel("Environment samples")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--output", default="artifacts/benchmark_nominal_v1")
    parser.add_argument("--test-seed", type=int, default=9040)
    parser.add_argument("--cases-per-group", type=int, default=12)
    args = parser.parse_args(argv)
    if args.cases_per_group < 5:
        parser.error("at least 5 cases per group are required")
    torch.set_num_threads(1)
    config = load_config(args.tuning)
    pid_config = Config(**json.loads(args.tuning.read_text(encoding="utf-8"))["pid_config"])
    policies, manifests, directories = {}, {}, {}
    seen_seeds = set()
    for directory in args.models:
        policy, metadata = load_policy(directory, config, "pure")
        seed = metadata["training_seed"]
        if seed in seen_seeds:
            raise ValueError("each independent training seed may appear only once")
        seen_seeds.add(seed)
        name = f"PPO_seed{seed}"
        policies[name], manifests[name], directories[name] = policy, metadata, directory
    tasks = {json.dumps(p.task.to_dict(), sort_keys=True) for p in policies.values()}
    if len(tasks) != 1:
        raise ValueError("multi-seed aggregation requires the same training task")
    output = create_output(args.output)
    suite = held_out_suite(config, args.test_seed, args.cases_per_group)
    save_json(output / "manifest.json", {"test_seed": args.test_seed, "contract": contract(config),
        "pid_config": pid_config.to_dict(), "models": manifests,
        "scenarios": [{"group": group, **scenario_dict(s)} for group, s in suite],
        "provenance": provenance(), "status": "evaluating",
        "note": "Seed SD uses independent trained models, not repeated episodes as independent training runs. Small-seed exploratory evidence; no superiority claim."})
    rows, traces = [], {"in_distribution": {}, "bias_positive": {}}
    for index, (group, scenario) in enumerate(suite):
        methods = {"PD": PID(config.kp, config.kd),
                   "PID": PID(pid_config.kp, pid_config.kd, pid_config.ki), **policies}
        for name, controller in methods.items():
            if name in ("PD", "PID"):
                command = lambda ep, obs, c=controller: c.command(ep)
            else:
                command = lambda ep, obs, c=controller: action_command(c.predict(obs)[0], ep, "pure")
            result, trace = rollout(config, scenario, command)
            rows.append({"controller": name, "group": group, "scenario": scenario.name, **result})
            np.savez_compressed(output / f"{scenario.name}_{name}.npz", **trace)
            if scenario.name in ("in_distribution_4", "bias_positive_4"):
                traces[group][name] = trace
        if (index + 1) % args.cases_per_group == 0:
            print(f"Evaluated group {group}", flush=True)
    summaries = summarize(rows)
    seed_rows = across_seed_summary(summaries, policies)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "seed_summary.csv", seed_rows)
    save_json(output / "summary.json", summaries)
    save_json(output / "seed_summary.json", seed_rows)
    make_plots(traces, directories, output, config)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)
    for row in summaries:
        if row["group"] in ("all", "in_distribution"):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
