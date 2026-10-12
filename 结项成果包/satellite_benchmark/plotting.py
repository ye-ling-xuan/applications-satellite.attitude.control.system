"""Six titled figures generated directly from saved data (no invented results)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


LABELS = {"PID": "PID", "PPO_nominal": "普通 PPO", "PPO_randomized": "随机化 PPO"}
COLORS = {"PID": "#2675ad", "PPO_nominal": "#df8725", "PPO_randomized": "#23986e"}
LINESTYLES = {"PID": "-", "PPO_nominal": "--", "PPO_randomized": ":"}
MARKERS = {"PID": "o", "PPO_nominal": "s", "PPO_randomized": "^"}
FAMILIES = {
    "nominal": ("正常工况", "工况编号"),
    "inertia": ("惯量变化", "惯量 J / kg·m²"),
    "disturbance": ("持续扰动", "偏置及振幅 / N·m"),
    "noise": ("测量噪声", "角度/角速度噪声标准差 / °、°/s"),
    "pulse": ("中途脉冲扰动", "脉冲力矩 / N·m"),
    "combined": ("组合工况（训练范围外）", "组合工况编号"),
}


def _style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    selected = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                 "WenQuanYi Micro Hei") if n in names), "DejaVu Sans")
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False,
                         "figure.dpi": 120, "savefig.dpi": 180,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2,
                         "font.size": 10, "axes.titlesize": 12})


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value, default=np.nan):
    if value is None or value == "":
        return default
    return float(value)


def _success(row):
    return str(row["success"]).lower() in ("true", "1", "1.0")


def _save(fig, path: Path) -> str:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _annotate_missing(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_plots(results_dir: Path, models_dir: Path | None = None,
                   output_dir: Path | None = None) -> list[str]:
    results_dir = Path(results_dir)
    models_dir = Path(models_dir) if models_dir else None
    output_dir = Path(output_dir) if output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    rows = _read_csv(results_dir / "episode_metrics.csv")
    summaries = _read_csv(results_dir / "summary.csv")
    manifest_path = results_dir / "evaluation_manifest.json"
    config = json.loads(manifest_path.read_text(encoding="utf-8")).get("env_base_config", {}) if manifest_path.exists() else {}
    theta_tol = float(config.get("theta_tol_deg", 1.0))
    omega_tol = float(config.get("omega_tol_deg_s", 0.5))
    controllers = [c for c in LABELS if any(r["controller"] == c for r in rows)]
    paths = []

    # 1. Actual episode rewards and fixed-validation checkpoint performance.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("01 · PPO 训练与独立验证趋势")
    lines = 0
    if models_dir:
        for controller in controllers:
            if controller == "PID":
                continue
            mode = controller.removeprefix("PPO_")
            for seed_index, run_dir in enumerate(sorted((models_dir / mode).glob("seed_*"))):
                history = run_dir / "validation_history.csv"
                seed = run_dir.name.removeprefix("seed_")
                seed_style = ("-", "--", ":")[seed_index % 3]
                if history.exists():
                    records = _read_csv(history)
                    axes[1].plot([_number(r["timesteps"]) for r in records],
                                 [_number(r["success_rate"]) for r in records],
                                 color=COLORS[controller], alpha=0.8, linestyle=seed_style,
                                 label=f"{LABELS[controller]} · seed {seed}")
                    lines += 1
                for segment_index, segment in enumerate([run_dir, *sorted(run_dir.glob("resume_*"))]):
                    monitor = next((segment / name for name in ("monitor_0.monitor.csv", "monitor_0.csv.monitor.csv", "monitor_0.csv")
                                    if (segment / name).exists()), None)
                    if monitor is None:
                        continue
                    with monitor.open(encoding="utf-8") as stream:
                        content = [line for line in stream if not line.startswith("#")]
                    records = list(csv.DictReader(content))
                    if records:
                        rewards = np.asarray([_number(r["r"]) for r in records])
                        window = min(15, len(rewards))
                        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
                        n_envs = 8
                        meta_path = segment / "training_manifest.json"
                        offset = 0
                        if meta_path.exists():
                            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                            n_envs = metadata.get("n_envs", n_envs)
                            offset = metadata.get("initial_timesteps", 0)
                        steps = offset + np.cumsum([_number(r["l"]) for r in records]) * n_envs
                        axes[0].plot(steps[window - 1:], smooth, color=COLORS[controller],
                                     alpha=0.7, linestyle=seed_style,
                                     label=f"{LABELS[controller]} · seed {seed}" if segment_index == 0 else None)
    axes[0].set(title="实际 episode 奖励（环境 0，滑动平均）", ylabel="奖励",
                xlabel="近似总交互步数（环境 0 步数 × 环境数）")
    axes[1].set(title="固定 10 个验证场景的成功率（用于选模型）", ylabel="验证成功率",
                xlabel="总交互步数", ylim=(-0.05, 1.05))
    if lines:
        axes[1].legend(ncol=2, fontsize=8)
    else:
        _annotate_missing(axes[1], "没有训练验证日志；本图不代表训练已经完成")
    if not axes[0].lines:
        _annotate_missing(axes[0], "没有 Monitor 奖励日志")
    else:
        axes[0].legend(ncol=2, fontsize=8)
    paths.append(_save(fig, output_dir / "01_training_validation.png"))

    # 2. A fixed, paired example; smallest seed is disclosed, not chosen by test.
    example_id = next((r["scenario_id"] for r in rows
                       if r["family"] == "nominal" and _number(r["sample_id"]) == 1),
                      rows[0]["scenario_id"])
    example = next(r for r in rows if r["scenario_id"] == example_id)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    fig.suptitle(f"02 · 固定配对响应：{example_id}，初态 {float(example['theta_deg']):.1f}°\n"
                 "PPO 使用最小训练 seed；该场景不用于选模型")
    for controller in controllers:
        subset = [r for r in rows if r["controller"] == controller
                  and r["scenario_id"] == example_id]
        row = min(subset, key=lambda r: int(r["model_seed"]))
        with np.load(results_dir / row["trace_file"], allow_pickle=False) as trace:
            for ax, key in zip(axes, ("error_deg", "omega_deg_s", "torque")):
                ax.plot(trace["t"], trace[key], color=COLORS[controller], linestyle=LINESTYLES[controller],
                        label=f"{LABELS[controller]} (seed {row['model_seed']})")
    axes[0].axhspan(-theta_tol, theta_tol, color="grey", alpha=0.1, label=f"±{theta_tol:g}° 阈值")
    axes[1].axhspan(-omega_tol, omega_tol, color="grey", alpha=0.1)
    axes[0].set(ylabel="姿态误差 / °")
    axes[1].set(ylabel="角速度 / °/s")
    axes[2].set(ylabel="实际控制力矩 / N·m", xlabel="时间 / s")
    axes[0].legend(ncol=2, fontsize=8)
    paths.append(_save(fig, output_dir / "02_paired_response.png"))

    # 3. Six factor families; uncertainty is across training seeds.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("03 · 各测试工况成功率（误差线为独立训练 seed 的样本标准差）")
    for ax, (family, (title, xlabel)) in zip(axes.flat, FAMILIES.items()):
        for controller in controllers:
            subset = sorted([r for r in summaries if r["family"] == family
                             and r["controller"] == controller],
                            key=lambda r: float(r["level"]))
            if subset:
                ax.errorbar([float(r["level"]) for r in subset],
                            [float(r["success_rate"]) for r in subset],
                            yerr=[_number(r["success_seed_std"], 0.0) for r in subset],
                            marker=MARKERS[controller], markersize={"PID": 8, "PPO_nominal": 6, "PPO_randomized": 4}[controller],
                            markerfacecolor="white" if controller != "PPO_randomized" else COLORS[controller],
                            linestyle=LINESTYLES[controller], capsize=3, color=COLORS[controller],
                            label=LABELS[controller])
        ax.set(title=title, xlabel=xlabel, ylabel="成功率", ylim=(-0.08, 1.08))
    axes[0, 0].legend(fontsize=8)
    paths.append(_save(fig, output_dir / "03_success_robustness.png"))

    # 4. All cases and their macro average, so costs include unsuccessful runs.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("04 · 精度、成功率与控制代价（含全部成功和失败 episode）")
    for controller in controllers:
        subset = [r for r in summaries if r["controller"] == controller]
        cost = np.asarray([float(r["control_cost_mean"]) for r in subset])
        rms = np.asarray([float(r["rms_error_deg_mean"]) for r in subset])
        success = np.asarray([float(r["success_rate"]) for r in subset])
        axes[0].scatter(cost, rms, color=COLORS[controller], alpha=0.4, s=25)
        axes[1].scatter(cost, success, color=COLORS[controller], alpha=0.4, s=25)
        for ax, values in zip(axes, (rms, success)):
            ax.scatter([cost.mean()], [values.mean()], marker="*", s=220,
                       color=COLORS[controller], edgecolor="white", linewidth=0.5,
                       label=f"{LABELS[controller]} · 工况宏平均")
    axes[0].set(xlabel="力矩平方积分 / (N·m)²·s", ylabel="RMS 姿态误差 / °")
    axes[1].set(xlabel="力矩平方积分 / (N·m)²·s", ylabel="成功率", ylim=(-0.05, 1.05))
    axes[0].legend(fontsize=8)
    paths.append(_save(fig, output_dir / "04_performance_cost.png"))

    # 5. Deliberately disclose the worst failed case for each controller.
    fig, axes = plt.subplots(len(controllers), 1, figsize=(11, 3 * len(controllers)),
                             squeeze=False, constrained_layout=True)
    fig.suptitle("05 · 失败案例检查（各控制器失败中末段误差最大者）")
    failure_records = []
    for ax, controller in zip(axes[:, 0], controllers):
        failures = [r for r in rows if r["controller"] == controller and not _success(r)]
        if not failures:
            _annotate_missing(ax, f"{LABELS[controller]}：本次已测试场景中没有失败")
            continue
        row = max(failures, key=lambda r: _number(r["steady_error_deg"]))
        with np.load(results_dir / row["trace_file"], allow_pickle=False) as trace:
            ax.plot(trace["t"], trace["error_deg"], color=COLORS[controller])
        ax.axhspan(-theta_tol, theta_tol, color="grey", alpha=0.12)
        ax.set(title=f"{LABELS[controller]} · {row['case_id']} · seed {row['model_seed']} · "
                     f"{row['scenario_id']} · 末段误差 {float(row['steady_error_deg']):.2f}°",
               xlabel="时间 / s", ylabel="姿态误差 / °")
        failure_records.append(row)
    (output_dir / "failure_examples.json").write_text(
        json.dumps(failure_records, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(_save(fig, output_dir / "05_failure_cases.png"))

    # 6. Aggregate each independent model seed; do not conflate scene count and seeds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("06 · 独立训练 seed 的结果波动（相同测试场景）")
    for controller in controllers:
        subset = [r for r in rows if r["controller"] == controller]
        model_seeds = sorted({int(r["model_seed"]) for r in subset})
        success, rms = [], []
        for seed in model_seeds:
            group = [r for r in subset if int(r["model_seed"]) == seed]
            success.append(float(np.mean([_success(r) for r in group])))
            rms.append(float(np.mean([float(r["rms_error_deg"]) for r in group])))
        if controller == "PID":
            axes[0].axhline(success[0], color=COLORS[controller], linestyle="--", label="PID")
            axes[1].axhline(rms[0], color=COLORS[controller], linestyle="--", label="PID")
        else:
            for ax, values in zip(axes, (success, rms)):
                ax.plot(model_seeds, values, marker="o", color=COLORS[controller],
                        label=f"{LABELS[controller]} ({len(model_seeds)} seeds)")
    axes[0].set(xlabel="训练 seed", ylabel="全部工况成功率", ylim=(-0.05, 1.05))
    axes[1].set(xlabel="训练 seed", ylabel="全部工况 RMS 误差 / °")
    axes[0].legend(fontsize=8)
    paths.append(_save(fig, output_dir / "06_training_seed_variability.png"))
    (output_dir / "figure_manifest.json").write_text(json.dumps({
        "source_episode_metrics": str(results_dir / "episode_metrics.csv"),
        "source_summary": str(results_dir / "summary.csv"), "figures": paths,
        "representative_scenario": example_id,
        "representative_seed_policy": "smallest available training seed, fixed without testing selection",
        "failure_policy": "worst steady error among failed episodes, explicitly disclosed",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
