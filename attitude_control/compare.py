"""Check the single-factor training contract and compare paired PPO reward ablations."""
import argparse
import csv
import json
from pathlib import Path, PureWindowsPath
import numpy as np
from .experiments import create_output, save_json
from .tasks import Task
from .evaluate import write_csv


def check_paired_manifests(left, right):
    for key in ["test_seed", "contract", "pid_config", "scenarios"]:
        if left[key] != right[key]:
            raise ValueError(f"unpaired evaluation: {key} differs")
    if left.get("status") != "complete" or right.get("status") != "complete":
        raise ValueError("both evaluations must be complete")
    lseeds = {m["training_seed"] for m in left["models"].values()}
    rseeds = {m["training_seed"] for m in right["models"].values()}
    if lseeds != rseeds:
        raise ValueError("training seed sets differ")


def load_recorded_metadata(record, evaluation_directory=None):
    directory = Path(record["directory"])
    if not (directory / "metadata.json").is_file() and evaluation_directory is not None:
        # Archived manifests preserve the machine's original absolute paths.
        # After cloning/moving the project, models are siblings of evaluations.
        leaf = (PureWindowsPath(record["directory"]).name if "\\" in record["directory"]
                else directory.name)
        directory = Path(evaluation_directory).resolve().parent / leaf
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("model_sha256") != record["sha256"]:
        raise ValueError("recorded model metadata does not match evaluation checkpoint")
    return metadata


def verify_training_factor(left, right, left_directory=None, right_directory=None):
    checks = []
    for name in left["models"]:
        l = load_recorded_metadata(left["models"][name], left_directory)
        r = load_recorded_metadata(right["models"][name], right_directory)
        for key in ["mode", "seed", "contract", "hyperparameters", "n_envs", "requested_timesteps", "actual_timesteps",
                    "training_domain", "selection_rule"]:
            if l[key] != r[key]:
                raise ValueError(f"training factor mismatch: {name} {key}")
        lt, rt = Task.from_dict(l["task"]), Task.from_dict(r["task"])
        if lt.profile != rt.profile or lt.torque_penalty == rt.torque_penalty:
            raise ValueError("expected identical profiles with different torque penalties")
        checks.append({"seed": l["seed"], "left_torque_penalty": lt.torque_penalty,
                       "right_torque_penalty": rt.torque_penalty,
                       "left_selected_step": l["selected_checkpoint_timesteps"],
                       "right_selected_step": r["selected_checkpoint_timesteps"]})
    return checks


def compare_summaries(left, right):
    groups = sorted({r["group"] for r in left})
    rows = []
    for group in groups:
        ls = {r["controller"]: r for r in left if r["group"] == group}
        rs = {r["controller"]: r for r in right if r["group"] == group}
        names = sorted(name for name in ls if name.startswith("PPO_seed"))
        if set(ls) != set(rs):
            raise ValueError("summary controller sets differ")
        row = {"group": group, "training_seeds": len(names)}
        for metric in ["success_rate", "mean_tail_mean_abs_error_deg", "mean_torque_effort_nm2_s"]:
            lv = np.array([ls[n][metric] for n in names])
            rv = np.array([rs[n][metric] for n in names])
            row.update({f"left_{metric}_mean": float(lv.mean()),
                        f"left_{metric}_sample_sd": float(lv.std(ddof=1)) if len(lv) > 1 else None,
                        f"right_{metric}_mean": float(rv.mean()),
                        f"right_{metric}_sample_sd": float(rv.std(ddof=1)) if len(rv) > 1 else None,
                        f"paired_delta_{metric}_mean": float((rv - lv).mean())})
            for baseline in ["PD", "PID"]:
                if ls[baseline][metric] != rs[baseline][metric]:
                    raise ValueError("paired baseline metrics unexpectedly differ")
                row[f"{baseline.lower()}_{metric}"] = ls[baseline][metric]
        rows.append(row)
    return rows


def paired_settling(left_path, right_path):
    def load(path):
        with path.open(encoding="utf-8-sig") as handle:
            return {(r["controller"], r["scenario"]): r for r in csv.DictReader(handle)
                    if r["controller"].startswith("PPO_seed")}
    left, right = load(left_path), load(right_path)
    if set(left) != set(right):
        raise ValueError("episode pair sets differ")
    result = []
    for name in sorted({key[0] for key in left}):
        pairs = [(left[k], right[k]) for k in left if k[0] == name
                 and left[k]["success"] == "True" and right[k]["success"] == "True"]
        result.append({"controller": name, "jointly_successful_cases": len(pairs),
                       "right_minus_left_settling_s": float(np.mean([
                           float(r["settling_time_s"]) - float(l["settling_time_s"]) for l, r in pairs])) if pairs else None})
    return result


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r["group"] != "all"]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, key, label in zip(axes,
         ["success_rate", "mean_tail_mean_abs_error_deg", "mean_torque_effort_nm2_s"],
         ["Success rate", "Tail error (deg)", "Torque effort (N^2 m^2 s)"]):
        for offset, side, name in [(-0.3, "left", "PPO: torque penalty 0.5"),
                                   (-0.1, "right", "PPO: torque penalty 0.05")]:
            ax.bar(x + offset, [r[f"{side}_{key}_mean"] for r in selected], width=0.2,
                   yerr=[r[f"{side}_{key}_sample_sd"] or 0 for r in selected], capsize=2, label=name)
        for offset, baseline in [(0.1, "pd"), (0.3, "pid")]:
            ax.bar(x + offset, [r[f"{baseline}_{key}"] for r in selected], width=0.2, label=baseline.upper())
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Paired torque-penalty ablation; error bars: SD across 5 training seeds")
    axes[-1].set_xticks(x, [r["group"].replace("_", "\n") for r in selected])
    fig.tight_layout()
    fig.savefig(output / "ablation.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", default="artifacts/torque_ablation_v1")
    args = parser.parse_args(argv)
    left = json.loads((args.left / "manifest.json").read_text(encoding="utf-8"))
    right = json.loads((args.right / "manifest.json").read_text(encoding="utf-8"))
    check_paired_manifests(left, right)
    checks = verify_training_factor(left, right, args.left, args.right)
    rows = compare_summaries(json.loads((args.left / "summary.json").read_text(encoding="utf-8")),
                             json.loads((args.right / "summary.json").read_text(encoding="utf-8")))
    settling = paired_settling(args.left / "episodes.csv", args.right / "episodes.csv")
    output = create_output(args.output)
    write_csv(output / "comparison.csv", rows)
    save_json(output / "comparison.json", rows)
    save_json(output / "checks.json", {"factor_checks": checks, "paired_settling": settling,
        "left": str(args.left.resolve()), "right": str(args.right.resolve()),
        "note": "Paired exploratory simulation study; SD is across trained models. Settling deltas only use cases where both policies succeeded."})
    plot(rows, output)
    for row in rows:
        if row["group"] in ("all", "in_distribution"):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
