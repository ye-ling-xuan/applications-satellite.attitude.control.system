"""Freeze controllers, run one held-out evaluation, and explain every result."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time
import numpy as np

from .protocol import STUDY, PACKAGE, DEADLINES, SEEDS, scenarios, sha256, write_json
from .evaluation import evaluate, metric_key


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def freeze():
    path = STUDY/"01_实验方案"/"部署冻结.json"
    if path.exists():
        result = read(path)
        check_frozen(result)
        return result
    if (STUDY/"04_独立测试"/"episode_metrics.csv").exists():
        raise RuntimeError("cannot construct a pre-test freeze after test outputs exist")
    pid_path = STUDY/"02_PID整定"/"selected_pid.json"
    legacy_path = STUDY/"03_SAC训练"/"旧模型验证与动作缩放"/"selected_legacy.json"
    formal_design = STUDY/"03_SAC训练"/"正式训练配置冻结.json"
    if not formal_design.is_file():
        raise RuntimeError("formal training configuration must be frozen first")
    formal_settings = read(formal_design)
    for relative, expected in formal_settings["source_sha256"].items():
        if sha256(PACKAGE/relative) != expected:
            raise RuntimeError("formal training source changed: "+relative)
    pid, legacy = read(pid_path), read(legacy_path)
    inputs = [pid_path, legacy_path, formal_design, STUDY/"01_实验方案"/"任务定义.json"]
    controls = []
    for axis in ("single", "three"):
        for deadline in DEADLINES:
            value = pid[axis]["deadlines"][str(int(deadline))]
            controls.append({"id": f"{axis}_pid_{int(deadline)}s", "axis": axis, "kind": "pid", "group": "pid",
                             "seed": -1, "parameters": value["parameters"], "optimized_deadlines": [deadline],
                             "validation": value["validation"]})
            margin = pid[axis].get("auxiliary_validation_time_margin_0_25s", {}).get(str(int(deadline)))
            if margin:
                controls.append({"id": f"{axis}_pid_margin_{int(deadline)}s", "axis": axis, "kind": "pid", "group": "pid_margin",
                                 "seed": -1, "parameters": margin["parameters"], "optimized_deadlines": [deadline],
                                 "validation": margin["validation"], "validation_time_margin_s": .25})
    for entry in legacy["models"]:
        axis, seed, scale = entry["axis"], entry["seed"], entry["action_scale"]
        key = f"{axis}_legacy_{seed}_scale_{str(scale).replace('.', 'p')}"
        prior = next((c for c in controls if c["id"] == key), None)
        if sha256(PACKAGE/entry["model_path"]) != entry["model_sha256"]:
            raise RuntimeError("legacy model changed after validation")
        if prior:
            if prior["model_path"] != entry["model_path"] or prior["model_sha256"] != entry["model_sha256"]:
                raise RuntimeError("inconsistent legacy model deduplication")
            prior["optimized_deadlines"].append(entry["deadline_s"])
            continue
        controls.append({"id": key, "axis": axis, "kind": "legacy", "group": "legacy_sac", "seed": seed,
                         "model_path": entry["model_path"], "model_sha256": entry["model_sha256"],
                         "calibrated": True, "scale": scale, "optimized_deadlines": [entry["deadline_s"]],
                         "fresh_training": False})
    metadata_files = sorted((STUDY/"03_SAC训练"/"formal").rglob("training_metadata.json"))
    if len(metadata_files) != 6:
        raise RuntimeError(f"exactly six formal training records required, got {len(metadata_files)}")
    for axis in ("single", "three"):
        matching = [(p, read(p)) for p in metadata_files if read(p)["axis"] == axis]
        if sorted(m["seed"] for _, m in matching) != list(SEEDS):
            raise RuntimeError("all three formal seeds are required per axis")
        if len({m["config"] for _, m in matching}) != 1:
            raise RuntimeError("cannot choose a reward configuration using test responses")
        for metadata_path, metadata in matching:
            if metadata["status"] != "complete" or metadata["actual_steps"] != 1000000 or not metadata["runtime_source_unchanged"]:
                raise RuntimeError("formal training did not complete unchanged exact budget")
            if metadata["config"] != formal_settings["config"]:
                raise RuntimeError("formal reward configuration differs from its freeze")
            for relative, expected in metadata["source_sha256"].items():
                if sha256(PACKAGE/relative) != expected:
                    raise RuntimeError("training metadata source differs from current input: "+relative)
            if metadata["source_sha256"] != metadata["source_sha256_after"]:
                raise RuntimeError("formal runtime inputs were not identical")
            best = metadata["best"]
            model_path = STUDY/best["model_path"]
            if sha256(model_path) != best["model_sha256"]:
                raise RuntimeError("selected neural model changed")
            seed = metadata["seed"]
            for calibrated, group, suffix in ((best["calibrated"], "fresh_sac", "selected"),
                                               (not best["calibrated"], "fresh_sac_ablation", "ablation")):
                controls.append({"id": f"{axis}_fresh_{seed}_{suffix}", "axis": axis, "kind": "fresh", "group": group,
                                 "seed": seed, "model_path": str(model_path.relative_to(PACKAGE)).replace("\\", "/"),
                                 "model_sha256": best["model_sha256"], "calibrated": calibrated,
                                 "optimized_deadlines": [8.], "fresh_training": True,
                                 "checkpoint_steps": best["timesteps"], "reward_config": metadata["config"],
                                 "validation": best, "ablation_same_model": group.endswith("ablation")})
            inputs.append(metadata_path)
    for axis in ("single", "three"):
        inputs.append(STUDY/"01_实验方案"/f"test_{axis}.json")
    inputs.extend((STUDY/"代码").rglob("*.py"))
    inputs.extend((STUDY/"03_SAC训练").glob("*.py"))
    inputs.extend((STUDY/"03_SAC训练").glob("*恢复*.json"))
    inputs.extend(PACKAGE/"multi_algorithm_benchmark"/f for f in ("envs.py", "vector_training.py"))
    inputs.extend(PACKAGE/c["model_path"] for c in controls if c["kind"] != "pid")
    frozen = {str(p.relative_to(PACKAGE)).replace("\\", "/"): sha256(p) for p in sorted(set(inputs))}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "test_controller_responses_read": False,
              "controllers": controls, "frozen_sha256": frozen,
              "formal_training_runs": 6, "formal_training_interactions": 6000000,
              "test_scenes_per_axis": 256, "all_seeds_and_deadlines_retained": True,
              "selection_of_PID_and_legacy_separately_for_each_deadline": True,
              "fresh_SAC_optimized_for_8s_only": True,
              "primary_PID": "minimum validation cost among maximal validation-pass candidates",
              "additional_PID": "chosen from evaluated candidates; validation settling <= deadline-0.25s; not retuned after test",
              "observation_difference": "fresh SAC also reads elapsed time; PID remains fixed-gain feedback; this is not an equal-architecture comparison"}
    write_json(path, result)
    print(f"部署已冻结：{len(controls)}个控制实例，正式测试尚未运行。", flush=True)
    return result


def check_frozen(record):
    for relative, expected in record["frozen_sha256"].items():
        if sha256(PACKAGE/relative) != expected:
            raise RuntimeError("frozen input changed: "+relative)


def load_controller(entry):
    if entry["kind"] == "pid":
        from .baselines import PIDController
        return PIDController(entry["axis"], entry["parameters"])
    from stable_baselines3 import SAC
    model = SAC.load(PACKAGE/entry["model_path"], device="cpu")
    if entry["kind"] == "legacy":
        from .legacy import LegacySAC
        return LegacySAC(model, entry["axis"], entry["scale"])
    from .training import FreshSACController
    return FreshSACController(model, entry["axis"], entry["calibrated"])


def test():
    import torch
    torch.set_num_threads(1)
    record = read(STUDY/"01_实验方案"/"部署冻结.json")
    check_frozen(record)
    directory = STUDY/"04_独立测试"
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"trajectories").mkdir()
    all_rows, artifacts = [], {}
    start = time.perf_counter()
    for entry in record["controllers"]:
        axis, name = entry["axis"], entry["id"]
        scenes = scenarios(axis, "test")
        if scenes != read(STUDY/"01_实验方案"/f"test_{axis}.json"):
            raise RuntimeError("held-out split changed")
        result = evaluate(axis, scenes, load_controller(entry), keep_traces=True)
        path = directory/"trajectories"/f"{name}.npz"
        np.savez_compressed(path, **result["traces"])
        artifacts[str(path.relative_to(directory)).replace("\\", "/")] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        for scene, row in zip(scenes, result["metrics"]):
            all_rows.append({"controller_id": name, "axis": axis, "group": entry["group"], "seed": entry["seed"],
                             "scenario_id": scene["scenario_id"], "sample_id": scene["sample_id"], **row})
        passes = {int(d): sum(row[metric_key(d)] for row in result["metrics"]) for d in DEADLINES}
        print(f"[独立测试 {name}] {passes} / {len(scenes)}, cost={np.mean([r['control_cost'] for r in result['metrics']]):.6f}", flush=True)
    csv_write(directory/"episode_metrics.csv", all_rows)
    check_frozen(record)
    write_json(directory/"test_run.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "controllers": len(record["controllers"]), "episodes": len(all_rows), "scene_count_per_axis": 256,
               "unique_initial_states_total": 512, "wall_time_s": time.perf_counter()-start,
               "deployment_freeze_sha256": sha256(STUDY/"01_实验方案"/"部署冻结.json"),
               "episode_metrics_sha256": sha256(directory/"episode_metrics.csv"), "artifacts": artifacts,
               "post_test_training_or_selection": False})
    return all_rows


def parse_rows():
    with (STUDY/"04_独立测试"/"episode_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    boolean = ["failed", "complete", *(metric_key(d) for d in DEADLINES)]
    integers = ("sample_id", "seed")
    text = ("controller_id", "axis", "group", "scenario_id")
    for row in rows:
        for k in row:
            if k in boolean:
                row[k] = row[k] == "True"
            elif k in integers:
                row[k] = int(row[k])
            elif k not in text:
                row[k] = float(row[k]) if row[k] else None
    return rows


def label(entry):
    names = {"pid": "按代价整定PID", "pid_margin": "预留0.25秒裕量PID",
             "legacy_sac": "旧SAC＋动作缩放", "fresh_sac": "新训练SAC",
             "fresh_sac_ablation": "新SAC同模型部署消融"}
    value = names[entry["group"]]
    if entry["kind"] != "pid":
        value += f"（种子{entry['seed']}）"
    return value


def summarize(record, rows):
    expected_ids = {c["id"] for c in record["controllers"]}
    if len(expected_ids) != len(record["controllers"]):
        raise RuntimeError("duplicate controller identifiers in freeze")
    if {r["controller_id"] for r in rows} != expected_ids:
        raise RuntimeError("CSV controller membership differs from freeze")
    by_id = {c["id"]: sorted([r for r in rows if r["controller_id"] == c["id"]], key=lambda r: r["sample_id"]) for c in record["controllers"]}
    summaries = []
    for c in record["controllers"]:
        values = by_id[c["id"]]
        expected = [(s["sample_id"], s["scenario_id"]) for s in scenarios(c["axis"], "test")]
        identities = [(r["sample_id"], r["scenario_id"]) for r in values]
        if identities != expected or len(set(identities)) != len(expected):
            raise RuntimeError("CSV initial states missing, duplicated or mismatched: "+c["id"])
        if any(r["axis"] != c["axis"] or r["group"] != c["group"] or r["seed"] != c["seed"] for r in values):
            raise RuntimeError("CSV controller metadata mismatch: "+c["id"])
        if any(not np.isfinite(r["control_cost"]) or r["control_cost"] < 0. for r in values):
            raise RuntimeError("invalid control cost")
        for deadline in DEADLINES:
            settle = [r["settling_time_s"] for r in values if r[metric_key(deadline)]]
            summaries.append({"axis": c["axis"], "controller_id": c["id"], "label": label(c), "group": c["group"],
                              "seed": c["seed"], "deadline_s": deadline, "test_count": len(values),
                              "passes": sum(r[metric_key(deadline)] for r in values),
                              "success_rate": float(np.mean([r[metric_key(deadline)] for r in values])),
                              "mean_control_cost_all_episodes": float(np.mean([r["control_cost"] for r in values])),
                              "mean_settling_time_success_only_s": float(np.mean(settle)) if settle else None,
                              "mean_rms_error_all_episodes_deg": float(np.mean([r["rms_error_deg"] for r in values])),
                              "physical_failures": sum(r["failed"] for r in values),
                              "optimized_for_this_deadline": deadline in c["optimized_deadlines"]})
    return by_id, summaries


def comparisons(record, by_id):
    rng = np.random.default_rng(2026101791)
    out = []
    for axis in ("single", "three"):
        for deadline in DEADLINES:
            for baseline_group in ("pid", "pid_margin"):
                baseline = next((c for c in record["controllers"] if c["axis"] == axis and c["group"] == baseline_group and deadline in c["optimized_deadlines"]), None)
                if baseline is None:
                    continue
                pid_rows = by_id[baseline["id"]]
                pid_cost = np.array([r["control_cost"] for r in pid_rows])
                pid_pass = sum(r[metric_key(deadline)] for r in pid_rows)
                for group in ("legacy_sac", "fresh_sac"):
                    models = [c for c in record["controllers"] if c["axis"] == axis and c["group"] == group and
                              (group == "fresh_sac" or deadline in c["optimized_deadlines"])]
                    models.sort(key=lambda c: c["seed"])
                    if len(models) != 3:
                        raise RuntimeError("must report three models for every comparison")
                    pid_keys = [(r["sample_id"], r["scenario_id"]) for r in pid_rows]
                    if any([(r["sample_id"], r["scenario_id"]) for r in by_id[c["id"]]] != pid_keys for c in models):
                        raise RuntimeError("paired comparison initial-state identities differ")
                    costs = np.array([[r["control_cost"] for r in by_id[c["id"]]] for c in models])
                    passes = [sum(r[metric_key(deadline)] for r in by_id[c["id"]]) for c in models]
                    differences = costs-pid_cost[None, :]
                    mean_differences = differences.mean(axis=1)
                    # Resample initial states, keeping all three fixed models together.
                    paired_difference = differences.mean(axis=0)
                    draws = rng.integers(0, len(pid_cost), (5000, len(pid_cost)))
                    ci = np.quantile(paired_difference[draws].mean(axis=1), [.025, .975])
                    gate = pid_pass == len(pid_cost) and all(p == len(pid_cost) for p in passes)
                    direction = bool(np.all(mean_differences < 0.))
                    advantage = bool(gate and direction and ci[1] < 0.)
                    out.append({"axis": axis, "deadline_s": deadline, "baseline_group": baseline_group,
                                "baseline_id": baseline["id"], "method": group,
                                "model_ids": [c["id"] for c in models], "seeds": [c["seed"] for c in models],
                                "n_initial_states": len(pid_cost), "PID_passes": pid_pass,
                                "SAC_passes_each_seed": passes, "PID_mean_cost": float(pid_cost.mean()),
                                "SAC_mean_cost_each_seed": costs.mean(axis=1).tolist(),
                                "SAC_mean_cost": float(costs.mean()),
                                "relative_cost_change_pct": float(100*(costs.mean()/pid_cost.mean()-1)),
                                "mean_paired_cost_difference_each_seed": mean_differences.tolist(),
                                "paired_cost_difference_95pct_bootstrap_CI": ci.tolist(),
                                "bootstrap_unit": "initial state; 3 fixed model outcomes retained together",
                                "all_controllers_passed_all_scenes": bool(gate),
                                "all_seed_cost_differences_negative": direction,
                                "supports_lower_effort_under_identical_requirements": advantage,
                                "primary_deadline": deadline == 8.,
                                "CI_scope": "fixed models and sampled initial-state distribution; not all training randomness"})
    return out


def make_figures(record, by_id, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [name for name in ("Microsoft YaHei", "SimHei", "Arial Unicode MS") if name in fonts] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    directory = STUDY/"05_图表与分析"
    directory.mkdir(parents=True, exist_ok=True)
    selected = [s for s in summaries if s["deadline_s"] == 8. and s["group"] in ("pid", "pid_margin", "fresh_sac", "legacy_sac") and
                8. in next(c for c in record["controllers"] if c["id"] == s["controller_id"])["optimized_deadlines"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, axis in enumerate(("single", "three")):
        subset = [s for s in selected if s["axis"] == axis]
        x = np.arange(len(subset))
        labels = [s["label"].replace("（种子", "\n种子").replace("）", "") for s in subset]
        color = ["#3c78a8" if s["group"].startswith("pid") else "#e18b37" if s["group"] == "legacy_sac" else "#408f68" for s in subset]
        axes[0, col].bar(x, [100*s["success_rate"] for s in subset], color=color)
        axes[0, col].set_ylim(0, 110); axes[0, col].set_ylabel("8秒期限通过率 / %")
        axes[0, col].set_title(("单轴" if axis == "single" else "三轴")+"：同一验收要求")
        axes[1, col].bar(x, [s["mean_control_cost_all_episodes"] for s in subset], color=color)
        axes[1, col].set_ylabel("平均力矩平方积分 / (N·m)²·s")
        for a in axes[:, col]:
            a.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            a.grid(axis="y", alpha=.2); a.set_axisbelow(True)
        axes[1, col].set_title("全部回合均保留；低代价须结合通过率判断")
    fig.tight_layout(); fig.savefig(directory/"01_八秒期限通过率与控制代价.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, axis in enumerate(("single", "three")):
        c_pid = next(c for c in record["controllers"] if c["axis"] == axis and c["group"] == "pid" and 8. in c["optimized_deadlines"])
        c_old = next(c for c in record["controllers"] if c["axis"] == axis and c["group"] == "legacy_sac" and c["seed"] == 11 and 8. in c["optimized_deadlines"])
        c_new = next(c for c in record["controllers"] if c["axis"] == axis and c["group"] == "fresh_sac" and c["seed"] == 41)
        for c in (c_pid, c_old, c_new):
            with np.load(STUDY/"04_独立测试"/"trajectories"/f"{c['id']}.npz") as z:
                axes[0, col].plot(z["time"], z["error_deg"][:, 0], label=label(c))
                axes[1, col].plot(z["time"], z["rate_deg_s"][:, 0], label=label(c))
        for row, tolerance, text in ((0, 1., "姿态误差 / °"), (1, .5, "角速度范数 / (°/s)")):
            axes[row, col].axhline(tolerance, color="gray", ls=":", lw=1)
            axes[row, col].axvline(8., color="black", ls="--", lw=1)
            axes[row, col].set_ylabel(text); axes[row, col].set_xlabel("时间 / s")
            axes[row, col].grid(alpha=.2); axes[row, col].legend(fontsize=8)
        axes[0, col].set_title(("单轴" if axis == "single" else "三轴")+"：预定测试初态0，非挑选最佳案例")
    fig.tight_layout(); fig.savefig(directory/"02_固定初态的完整响应.png", dpi=170); plt.close(fig)


def report():
    record = read(STUDY/"01_实验方案"/"部署冻结.json")
    check_frozen(record)
    rows = parse_rows()
    by_id, summaries = summarize(record, rows)
    compare = comparisons(record, by_id)
    output = STUDY/"05_图表与分析"
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output/"全部控制器与期限汇总.csv", summaries)
    write_json(output/"配对比较与结论.json", compare)
    attempts = []
    for manifest_path in sorted((STUDY/"03_SAC训练").rglob("training_manifest.json")):
        status_path = manifest_path.with_name("session_status.json")
        if status_path.is_file():
            status = read(status_path)
            attempts.append({"path": str(manifest_path.parent.relative_to(STUDY)).replace("\\", "/"),
                             "axis": status["axis"], "seed": status["seed"], "stage": status["stage"],
                             "status": status["status"], "actual_steps": status["actual_steps"]})
    write_json(output/"全部训练尝试统计.json", attempts)
    incomplete = [a for a in attempts if a["status"] != "complete"]
    incomplete_steps = sum(a["actual_steps"] for a in incomplete)
    make_figures(record, by_id, summaries)
    lines = ["# 同等停稳要求下的控制代价比较：研究结果", "",
             "本轮把问题改成：两种方法都必须按时停稳，然后比较完整20秒的力矩平方积分。它不是电能测量。原惯量研究的结论保持原有适用条件。", "",
             "## 先看8秒期限的独立测试", "",
             "每个控制器使用同一轴的256个全新初态。从第8秒到第20秒，每个采样点都要满足角误差≤1°、角速度≤0.5°/s。失败回合全部保留。", "",
             "| 任务 | 控制器 | 通过 / 256 | 平均力矩平方积分 | 成功回合平均停稳时间 / s | 全程RMS / ° |",
             "|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        c = next(c for c in record["controllers"] if c["id"] == s["controller_id"])
        if s["deadline_s"] != 8. or 8. not in c["optimized_deadlines"] or c["group"] == "fresh_sac_ablation":
            continue
        settle = s["mean_settling_time_success_only_s"]
        lines.append(f"| {'单轴' if s['axis']=='single' else '三轴'} | {s['label']} | {s['passes']} | {s['mean_control_cost_all_episodes']:.6f} | {settle:.2f} | {s['mean_rms_error_all_episodes_deg']:.2f} |" if settle is not None else
                     f"| {'单轴' if s['axis']=='single' else '三轴'} | {s['label']} | {s['passes']} | {s['mean_control_cost_all_episodes']:.6f} | 无通过回合 | {s['mean_rms_error_all_episodes_deg']:.2f} |")
    primary = [c for c in compare if c["deadline_s"] == 8. and c["baseline_group"] == "pid"]
    wins = [c for c in primary if c["supports_lower_effort_under_identical_requirements"]]
    lines += ["", "## 是否得到了强化学习更优的结论", ""]
    if wins:
        lines.append("观察到以下有限条件下的控制代价优势。它不表示强化学习全面优于PID，也不表示同样的实际停稳时间、全过程误差或真实用电量。")
    else:
        lines.append("**本轮8秒主要任务没有获得满足预定判据的强化学习控制代价优势。** 不能把更低的某个代价数值单独当成优越结论；双方可靠停稳、三个模型方向一致也是预先规定的条件。")
    for value in primary:
        task = "单轴" if value["axis"] == "single" else "三轴"
        method = "旧SAC加验证选取的动作缩放" if value["method"] == "legacy_sac" else "新训练SAC"
        passes = "、".join(map(str, value["SAC_passes_each_seed"]))
        delta = value["relative_cost_change_pct"]
        phrase = f"平均代价相对PID{'降低' if delta<0 else '增加'}{abs(delta):.1f}%"
        reason = "达到本轮有限条件下的优势判据" if value["supports_lower_effort_under_identical_requirements"] else "未达到本轮优势判据"
        lines.append(f"- {task}，{method}：PID通过{value['PID_passes']}/256，三个SAC模型分别通过{passes}/256；{phrase}；{reason}。")
    lines += ["", "## 为什么这次比较与旧结果不同", "",
              "旧PID选择偏重快速响应。本轮将低增益和逐轴增益纳入搜索，允许控制器利用规定的时间完成动作，再按代价选参。因此，旧研究中SAC代价较低，不能直接推出本轮也会更低。", "",
              "同一验证集多次细化的参数可能贴近截止边界，在新初态上出现失败。本轮另在测试前保留预留0.25秒验证时间裕量的PID作辅助对照；该对照只从已有候选中选择，没有看测试再调参。主选择和辅助选择均完整报告。", "",
              "新SAC在误差、角速度之外增加了时钟观测，并使用针对8秒任务的时变奖励；PID仍为固定增益结构。它们满足相同物理条件和验收要求，不是相同结构或完全相同输入处理的比较。若未来要判断学习是否必要，还应与轨迹规划、时变反馈或模型预测控制等比较。", "",
              "训练奖励仍是包含误差和力矩项的软惩罚，并不是已经求解了带硬约束的最小力矩平方积分问题。开发只尝试两种奖励配置，正式采用固定预算。本轮未取得优势时，结论只限于这些具体训练与部署方案，不能推断强化学习在此类任务中永远不可能更好。", "",
              "## 研究实际做了多少", "",
              "新增正式训练为单轴、三轴各三个种子，共6份从零训练SAC、600万交互步；开发A/B各200k步、每轴两份，共4份开发记录和80万开发步，不混计为正式训练。已有6份SAC只调整部署动作缩放，不算新增训练。检查点、raw/calibrated部署实例均不算额外训练。", "",
              f"此外保留{len(incomplete)}次未完成训练尝试，实际交互{incomplete_steps:,}步，不计作完成模型；全部训练尝试合计实际交互{sum(a['actual_steps'] for a in attempts):,}步。若因文件暂时占用而原种子重跑，恢复记录与未完成目录保留，不按结果好坏筛除。", "",
              f"正式独立测试包含{len(record['controllers'])}个控制实例、{len(rows)}个完整回合，只有512个独立初态；多个控制器复用初态，不能把回合数当成独立初态数。", "",
              "PID的开发广搜、验证筛选和局部细化都保留日志，后续重复评估含同一候选，不当作独立参数数量或正式测试回合。搜索到的是所考察候选中的结果，不是PID全局最优证明。", "",
              "## 完整结果与统计限制", "",
              "4、6、8秒全部结果见[汇总CSV](全部控制器与期限汇总.csv)，逐条件配对结果见[比较JSON](配对比较与结论.json)。PID与旧SAC部署系数分别针对每一期限选择；新训练SAC只针对8秒优化，4/6秒结果是附加观察。", "",
              "主判定要求指定PID和三个SAC模型均在测试中全部通过，三个模型代价差方向一致，且按初态配对自举的95%区间支持降低。区间保留同一初态的三个模型结果，只描述固定模型对初态抽样的不确定性，不充分覆盖训练随机性；没有对多期限作普遍显著性结论。", "",
              "停稳时间只对相应期限通过的回合取均值，失败较多时不能拿这个均值直接说明更快。控制代价和RMS包含全部完整回合；如有物理失败，必须结合失败记录解释。", "",
              "## 图怎么看", "",
              "先看通过率，再看代价。未达到持续停稳要求的控制器，即使代价小，也不能按本轮规则认定更好。代表曲线固定使用初态编号0、旧种子11、新种子41，不是从结果中挑最好的一条。", "",
              "![8秒期限](01_八秒期限通过率与控制代价.png)", "",
              "![固定初态](02_固定初态的完整响应.png)", "",
              "## 复核入口", "",
              "[独立核验报告](../06_核验/独立核验报告.json)须单独运行核验后生成；生成本文不等于核验通过。冻结记录约束模型、参数、代码和测试初态；最终测试后不续训、不改奖励、不重新选参。"]
    markdown = "\n".join(lines)+"\n"
    (output/"研究结果说明.md").write_text(markdown, encoding="utf-8")
    import markdown as md
    rendered = md.markdown(markdown, extensions=["tables", "fenced_code"])
    webpage = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同等停稳要求下的控制代价比较</title><style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.85;color:#243247}h1,h2{line-height:1.45}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #c9d3df;padding:7px;text-align:left}th{background:#edf3f8}img{max-width:100%;height:auto}a{color:#196398}strong{color:#15263c}</style><body>'+rendered+'</body></html>'
    (output/"结果阅读本.html").write_text(webpage, encoding="utf-8")
    print(str(output/"研究结果说明.md"), flush=True)
    return compare


def main(command):
    if command == "freeze":
        freeze()
    elif command == "test":
        test()
    elif command == "report":
        report()
    elif command == "verify":
        from .verification import verify
        result = verify()
        print(json.dumps({"status": result["status"], "counts": result.get("counts"),
                          "issue_count": result["issue_count"]}, ensure_ascii=False), flush=True)
        if result["status"] != "passed":
            raise RuntimeError("独立核验未通过，请查看06_核验/独立核验报告.json")
