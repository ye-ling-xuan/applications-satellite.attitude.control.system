"""Publication-style static figures generated only from saved comparison data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["baseline_PID", "tuned_PID", "improved_PID", "PPO", "SAC", "TD3"]
LABELS = {"baseline_PID": "原始 PID", "tuned_PID": "整定 PID", "improved_PID": "改进 PID",
          "PPO": "PPO", "SAC": "SAC", "TD3": "TD3",
          "PPO_calibrated": "PPO（零动作校准）", "SAC_calibrated": "SAC（零动作校准）",
          "TD3_calibrated": "TD3（零动作校准）"}
COLORS = dict(zip(ORDER, ["#8b8b8b", "#a66b1d", "#067b64", "#3e69a8", "#8d50a0", "#d45a4d"]))
LINE_STYLES = {"baseline_PID": ":", "tuned_PID": "--", "improved_PID": "-"}
# Broad dotted/dashed strokes remain visible around the narrower solid stroke
# when the validation-selected PID variants have exactly coincident responses.
LINE_WIDTHS = {"baseline_PID": 2.8, "tuned_PID": 2.0, "improved_PID": 1.2}
AXES = {"single": "单轴", "three": "三轴"}


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    return rows


def setup():
    matplotlib.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                               "axes.unicode_minus": False, "font.size": 10,
                               "axes.spines.top": False, "axes.spines.right": False,
                               "savefig.dpi": 170})


def mean(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else np.nan


def save(fig, output_dir, name):
    path = Path(output_dir) / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def generate_plots(results_dir, output_dir=None):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup()
    rows = load_rows(results_dir / "episode_metrics.csv")
    paired = load_rows(results_dir / "paired_improvements.csv")
    cases = list(dict.fromkeys(row["case_id"] for row in rows))
    paths = []

    for axis in AXES:
        selected = [r for r in rows if r["axis"] == axis and r["case_id"] == "nominal" and r["sample_id"] == 0]
        fig, plots = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for controller in ORDER:
            traces = []
            for row in selected:
                if row["controller"] == controller:
                    with np.load(results_dir / row["trace_file"], allow_pickle=False) as trace:
                        traces.append({key: trace[key].copy() for key in ("time", "error_deg", "rate_deg_s", "torque")})
            if not traces:
                continue
            times = traces[0]["time"]
            label = LABELS[controller] + (f"（{len(traces)} 种子均值）" if len(traces) > 1 else "")
            error = np.mean([t["error_deg"] for t in traces], axis=0)
            rate = np.mean([t["rate_deg_s"] for t in traces], axis=0)
            torque = np.mean([np.linalg.norm(t["torque"], axis=1) for t in traces], axis=0)
            for plot, values in zip(plots, (error, rate, torque)):
                plot.plot(times, values, color=COLORS[controller], label=label,
                          linestyle=LINE_STYLES.get(controller, "-"),
                          linewidth=LINE_WIDTHS.get(controller, 1.4))
        for plot, threshold, ylabel in zip(plots, (1., .5, None), ("指向误差 (°)", "角速度范数 (°/s)", "控制力矩范数 (N·m)")):
            plot.set_ylabel(ylabel)
            plot.grid(alpha=.22)
            plot.axvspan(15, 20, color="#067b64", alpha=.07)
            if threshold is not None:
                plot.axhline(threshold, color="#333333", linestyle="--", linewidth=.9)
                plot.set_yscale("symlog", linthresh=1.)
                plot.set_ylim(bottom=0.)
        plots[0].legend(ncol=3, fontsize=8, loc="upper right")
        plots[-1].set_xlabel("部署时间 (s)")
        plots[-1].set_xlim(0, 20)
        fig.suptitle(f"{AXES[axis]}同条件代表轨迹：正常工况、第 1 个初始姿态\n阴影为末 5 秒保持验收窗口；前两图用对称对数刻度", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        paths.append(save(fig, output_dir, f"01_response_{axis}.png"))

    fig, plots = plt.subplots(2, 1, figsize=(13, 9))
    for axis, plot in zip(AXES, plots):
        matrix = np.zeros((len(ORDER), len(cases)))
        for i, controller in enumerate(ORDER):
            for j, case in enumerate(cases):
                group = [r for r in rows if r["axis"] == axis and r["controller"] == controller and r["case_id"] == case]
                matrix[i, j] = 100 * mean(group, "success")
        plot.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for i in range(len(ORDER)):
            for j in range(len(cases)):
                plot.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center", fontsize=9,
                          color="white" if matrix[i,j] >= 65 else "#172a3a")
        plot.set_xticks(range(len(cases)), cases, rotation=20, ha="right", fontsize=8)
        plot.set_yticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
        plot.set_title(f"{AXES[axis]}：完整 20 秒 + 末 5 秒误差/角速度同时达标")
    fig.suptitle("同条件成功率：强化学习保留全部 3 个训练种子", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    paths.append(save(fig, output_dir, "02_success_by_condition.png"))

    fig, plots = plt.subplots(1, 2, figsize=(12, 5))
    for axis, plot in zip(AXES, plots):
        for controller in ORDER:
            group = [r for r in rows if r["axis"] == axis and r["controller"] == controller]
            x, y = mean(group, "control_cost"), mean(group, "tail_mean_error_deg")
            label = f"{LABELS[controller]}（成功 {100*mean(group,'success'):.1f}%）"
            if controller == "tuned_PID":
                # A ring stays visible around the coincident improved-PID point.
                plot.scatter(x, y, s=145, facecolors="none", edgecolors=COLORS[controller],
                             linewidths=2., label=label)
            else:
                plot.scatter(x, y, s=65 if controller == "improved_PID" else 85,
                             color=COLORS[controller], label=label)
        plot.set_title(AXES[axis] + "：9 工况等权汇总")
        plot.set_xlabel("控制代价 ∫‖τ‖²dt (N²·m²·s)")
        plot.set_ylabel("末 5 秒平均误差 (°)")
        plot.set_yscale("symlog", linthresh=.1)
        plot.grid(alpha=.22)
        plot.margins(.25)
        plot.legend(fontsize=8, loc="best")
    fig.suptitle("精度与控制代价：所有失败回合纳入均值，控制代价不是实际能耗", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    paths.append(save(fig, output_dir, "03_accuracy_control_cost.png"))

    fig, plots = plt.subplots(2, 2, figsize=(14, 10))
    for i, axis in enumerate(AXES):
        for j, field in enumerate(("tail_mean_error_deg", "success_rate")):
            plot = plots[i, j]
            positions = np.arange(len(cases))
            for offset, baseline, color in ((-.19, "baseline_PID", "#a66b1d"), (.19, "tuned_PID", "#067b64")):
                values = []
                for case in cases:
                    group = [r for r in paired if r["axis"] == axis and r["baseline"] == baseline and r["case_id"] == case]
                    value = mean(group, field + "_change")
                    values.append(value * 100 if field == "success_rate" else value)
                plot.bar(positions + offset, values, width=.36, color=color, label="改进 PID − " + LABELS[baseline])
            plot.axhline(0, color="#333333", linewidth=.8)
            plot.set_xticks(positions, cases, rotation=25, ha="right", fontsize=8)
            plot.set_title(AXES[axis] + ("：末段误差变化（负值更好）" if field != "success_rate" else "：成功率变化（正值更好）"))
            plot.set_ylabel("平均误差差值 (°)" if field != "success_rate" else "成功率差值（百分点）")
            plot.legend(fontsize=8)
            plot.grid(axis="y", alpha=.22)
    fig.suptitle("改进必须在相同初态、相同外扰和相同噪声下比较；改善与退化均显示", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    paths.append(save(fig, output_dir, "04_paired_pid_improvement.png"))

    fig, plots = plt.subplots(2, 2, figsize=(11, 9))
    for i, axis in enumerate(AXES):
        for j, field in enumerate(("success", "tail_mean_error_deg")):
            plot = plots[i, j]
            success_bounds = []
            for x, controller in enumerate(("PPO", "SAC", "TD3")):
                group = [r for r in rows if r["axis"] == axis and r["controller"] == controller]
                seeds = sorted(set(r["model_seed"] for r in group))
                values = [mean([r for r in group if r["model_seed"] == seed], field) for seed in seeds]
                values = np.asarray(values) * (100 if field == "success" else 1)
                deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
                success_bounds.extend([*values, float(np.mean(values))-deviation, float(np.mean(values))+deviation])
                plot.bar(x, float(np.mean(values)), color=COLORS[controller], alpha=.28, width=.55)
                plot.errorbar(x, float(np.mean(values)), yerr=deviation,
                              color=COLORS[controller], capsize=6)
                for seed, value, offset in zip(seeds, values, np.linspace(-.15, .15, len(seeds))):
                    plot.scatter(x + offset, value, color=COLORS[controller], edgecolor="white", s=50)
                    plot.annotate(str(int(seed)), (x + offset, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            plot.set_xticks(range(3), ["PPO", "SAC", "TD3"])
            plot.set_title(AXES[axis] + ("：成功率" if field == "success" else "：末 5 秒误差"))
            plot.set_ylabel("成功率 (%)" if field == "success" else "末段平均误差 (°)")
            if field == "success":
                plot.set_ylim(min([0., *success_bounds])-4, max([100., *success_bounds])+8)
            plot.grid(axis="y", alpha=.22)
    fig.suptitle("随机种子波动：点为独立训练种子，柱与误差棒为均值 ± 样本标准差\nSD 是种子波动而非置信区间；均值减 SD 越界不代表物理负误差", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    paths.append(save(fig, output_dir, "05_learning_seed_variation.png"))

    historical_path = results_dir / "historical" / "episode_metrics.csv"
    if historical_path.exists():
        historical = load_rows(historical_path)
        fig, plots = plt.subplots(2, 2, figsize=(11, 8))
        for i, axis in enumerate(AXES):
            controllers = list(dict.fromkeys(r["controller"] for r in historical if r["axis"] == axis))
            for j, field in enumerate(("success", "tail_mean_error_deg")):
                plot = plots[i, j]
                values = [mean([r for r in historical if r["axis"] == axis and r["controller"] == c], field) for c in controllers]
                if field == "success":
                    values = np.asarray(values) * 100
                plot.bar(range(len(controllers)), values, color=[COLORS[c.replace("historical_", "")] for c in controllers])
                plot.set_xticks(range(len(controllers)), [c.replace("historical_", "") for c in controllers])
                plot.set_title(AXES[axis] + "历史策略迁移部署")
                plot.set_ylabel("成功率 (%)" if field == "success" else "末 5 秒平均误差 (°)")
                if field == "success":
                    plot.set_ylim(0, 110)
                plot.grid(axis="y", alpha=.22)
        fig.suptitle("历史模型迁移复测：执行器、时步与积分器按统一部署环境；独立于新模型排名", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        paths.append(save(fig, output_dir, "06_historical_deployment.png"))
    if any(str(row["controller"]).endswith("_calibrated") for row in rows):
        fig, plots = plt.subplots(2, 2, figsize=(12, 9))
        for i, axis in enumerate(AXES):
            for j, field in enumerate(("success", "tail_mean_error_deg")):
                plot = plots[i, j]
                success_bounds = []
                for offset, suffix, color, label in ((-.18, "", "#78848e", "原策略"),
                                                     (.18, "_calibrated", "#067b64", "零动作校准")):
                    means, deviations = [], []
                    for algorithm in ("PPO", "SAC", "TD3"):
                        group = [r for r in rows if r["axis"] == axis and r["controller"] == algorithm + suffix]
                        seeds = sorted(set(r["model_seed"] for r in group))
                        values = np.asarray([mean([r for r in group if r["model_seed"] == seed], field) for seed in seeds])
                        values *= 100 if field == "success" else 1
                        means.append(float(np.mean(values)))
                        deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.)
                    success_bounds.extend([v-d for v,d in zip(means,deviations)])
                    success_bounds.extend([v+d for v,d in zip(means,deviations)])
                    plot.bar(np.arange(3) + offset, means, yerr=deviations, width=.34,
                             color=color, label=label, capsize=4)
                plot.set_xticks(range(3), ["PPO", "SAC", "TD3"])
                plot.set_title(AXES[axis] + ("：成功率" if field == "success" else "：末 5 秒误差"))
                plot.set_ylabel("成功率 (%)" if field == "success" else "末段平均误差 (°)")
                if field == "success":
                    plot.set_ylim(min([0., *success_bounds])-4, max([100., *success_bounds])+8)
                else:
                    plot.set_yscale("symlog", linthresh=.1)
                plot.legend(fontsize=9)
                plot.grid(axis="y", alpha=.22)
        fig.suptitle("RL 零动作校准：同一个 checkpoint 的校准前后配对；均值 ± 3 种子 SD\n校准策略独立报告；SD 不是置信区间，越界不代表物理负误差", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .92))
        paths.append(save(fig, output_dir, "07_policy_calibration.png"))
    (output_dir / "figure_manifest.json").write_text(json.dumps({"figures": [Path(p).name for p in paths]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
