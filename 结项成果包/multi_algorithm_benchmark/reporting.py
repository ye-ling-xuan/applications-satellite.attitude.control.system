"""Build the integrated explanation from actual models and saved test evidence."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np

from .evaluation import write_json
from .plotting import AXES, LABELS, ORDER, load_rows, mean


def number(value, digits=3):
    return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def relative(path, parent):
    return Path(os.path.relpath(path, parent)).as_posix()


def aggregate(rows):
    summaries = []
    for axis in AXES:
        controllers = [c for c in ORDER if any(r["axis"] == axis and r["controller"] == c for r in rows)]
        controllers += [c for c in dict.fromkeys(r["controller"] for r in rows if r["axis"] == axis) if c not in controllers]
        for controller in controllers:
            group = [r for r in rows if r["axis"] == axis and r["controller"] == controller]
            seeds = set(r["model_seed"] for r in group)
            seed_success = [mean([r for r in group if r["model_seed"] == s], "success") for s in seeds]
            settled = [r["settling_time_s"] for r in group if r["settling_time_s"] is not None]
            summaries.append({"axis": axis, "controller": controller, "n_episodes": len(group),
                              "n_training_seeds": len(seeds), "success_rate": mean(group, "success"),
                              "success_seed_std": float(np.std(seed_success, ddof=1)) if len(seeds) > 1 else None,
                              "rms_error_deg": mean(group, "rms_error_deg"),
                              "tail_mean_error_deg": mean(group, "tail_mean_error_deg"),
                              "control_cost": mean(group, "control_cost"),
                              "settling_time_s_successful": float(np.mean(settled)) if settled else None,
                              "failed_rate": mean(group, "failed")})
    return summaries


def markdown_table(summaries, axis):
    lines = ["| 控制器 | 回合数 | 成功率 | 种子间成功率 SD | 全程 RMS (°) | 末 5 秒误差 (°) | 控制代价 | 成功回合稳定时间 (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["axis"] != axis:
            continue
        sd = number(100*row["success_seed_std"], 1) + " pp" if row["success_seed_std"] is not None else "—"
        label = LABELS.get(row["controller"], row["controller"])
        lines.append(f"| {label} | {row['n_episodes']} | {100*row['success_rate']:.1f}% | {sd} | {number(row['rms_error_deg'])} | {number(row['tail_mean_error_deg'])} | {number(row['control_cost'])} | {number(row['settling_time_s_successful'])} |")
    return "\n".join(lines)


def build_report(results_dir, models_dir, output_dir=None):
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    output_dir = Path(output_dir) if output_dir else results_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(results_dir / "episode_metrics.csv")
    paired = load_rows(results_dir / "paired_improvements.csv")
    manifest = json.loads((results_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    raw_rows = [row for row in rows if not str(row["controller"]).endswith("_calibrated")]
    calibrated_rows = [row for row in rows if str(row["controller"]).endswith("_calibrated")]
    summaries = aggregate(raw_rows)
    calibrated_summaries = aggregate(calibrated_rows)
    calibration_file = results_dir / "policy_calibration_pairs.csv"
    calibration_pairs = load_rows(calibration_file) if calibration_file.exists() else []
    calibration_summaries = []
    for axis in AXES:
        for algorithm in ("PPO", "SAC", "TD3"):
            group = [r for r in calibration_pairs if r["axis"] == axis and r["algorithm"] == algorithm]
            if not group:
                continue
            original, calibrated = mean(group, "tail_mean_error_deg_raw"), mean(group, "tail_mean_error_deg_calibrated")
            calibration_summaries.append({"axis": axis, "algorithm": algorithm, "paired_episode_count": len(group),
                                          "success_rate_raw": mean(group, "success_raw"),
                                          "success_rate_calibrated": mean(group, "success_calibrated"),
                                          "success_change_pp": 100*mean(group, "success_change"),
                                          "success_recovered_count": int(sum(r["success_recovered"] for r in group)),
                                          "success_lost_count": int(sum(r["success_lost"] for r in group)),
                                          "tail_mean_error_raw_deg": original,
                                          "tail_mean_error_calibrated_deg": calibrated,
                                          "tail_error_reduction_percent": 100*(original-calibrated)/original if original > 1e-12 else None,
                                          "rms_change_deg": mean(group, "rms_error_deg_change"),
                                          "control_cost_change": mean(group, "control_cost_change")})
    training = []
    for path in sorted(models_dir.glob("*/*/seed_*/training_metadata.json")):
        training.append(json.loads(path.read_text(encoding="utf-8")))
    pids = json.loads((models_dir / "pid_configs.json").read_text(encoding="utf-8"))
    history_file = results_dir / "historical" / "episode_metrics.csv"
    history_rows = load_rows(history_file) if history_file.exists() else []
    history_summary = aggregate(history_rows) if history_rows else []
    improvements = []
    for axis in AXES:
        for baseline in ("baseline_PID", "tuned_PID"):
            group = [r for r in paired if r["axis"] == axis and r["baseline"] == baseline]
            if not group:
                continue
            original = mean(group, "tail_mean_error_deg_baseline")
            improved = mean(group, "tail_mean_error_deg_improved")
            improvements.append({"axis": axis, "baseline": baseline,
                                 "paired_scene_count": len(group),
                                 "success_change_pp": 100*mean(group, "success_rate_change"),
                                 "tail_mean_error_original_deg": original,
                                 "tail_mean_error_improved_deg": improved,
                                 "tail_error_reduction_percent": 100*(original-improved)/original if original > 1e-12 else None,
                                 "rms_change_deg": mean(group, "rms_error_deg_change"),
                                 "control_cost_change": mean(group, "control_cost_change")})
    write_json(output_dir / "result_summary.json", {"fresh_results": summaries, "calibrated_results": calibrated_summaries,
                                                     "pid_paired_improvements": improvements,
                                                     "policy_calibration_improvements": calibration_summaries,
                                                     "historical_deployment": history_summary,
                                                     "fresh_episode_count": len(rows), "raw_episode_count": len(raw_rows),
                                                     "calibrated_episode_count": len(calibrated_rows),
                                                     "policy_calibration_pair_count": len(calibration_pairs),
                                                     "historical_episode_count": len(history_rows)})
    res = relative(results_dir, output_dir)
    mods = relative(models_dir, output_dir)
    module = relative(Path(__file__).parent, output_dir)
    lines = ["# 单轴与三轴 PID、PPO、SAC、TD3：新增材料融合与实际结果", "",
             "本报告由保存的模型元数据、正式测试 CSV 和配对轨迹自动生成。原项目文件及新增来源目录均保留；旧结项实验继续作为前一版基线。新增工程的 60 份有效材料原样归档，4 份历史模型单独做迁移部署复测。以下算法比较来自统一环境中的新训练与正式测试，不沿用历史图表的优势结论。", "",
             f"本次保存 **{len(training)} 份新训练记录**，正式评估 **{len(rows)} 回合**（原控制器 {len(raw_rows)}、同模型部署校准 {len(calibrated_rows)}），历史模型迁移复测 **{len(history_rows)} 回合**。每轴 {manifest['samples_per_case']} 组初态 × 9 工况形成 {manifest['scene_counts']['single']} 条环境配置，初态在工况间复用；这并非 {manifest['scene_counts']['single']} 个独立初态。相同配置的控制器共享初态、扰动函数和传感器噪声实现。", "",
             "## 01 融合了哪些工作", "",
             "新增材料带来了三轴 PPO/SAC/TD3 的环境、训练脚本、模型和横向对比经验。保留四元数、欧拉耦合动力学、共享仿真器及算法说明；把一次入阈早停改成完整部署测试，增加角速度与持续保持判据，补齐单轴 SAC/TD3，并把 PID 参数整定和测量滤波的作用分别展示。", "",
             "历史脚本的纠正时间可以只由最后一个阈值内采样点决定，历史稳态误差窗口也随控制器不同而变化。新报告统一统计最后 5 秒，并保留失败回合。这样可以验证控制器保持姿态的能力，也能判断已有成果在新部署条件下是否仍成立。", "",
             "## 02 数学模型与共同条件", "",
             "| 条件 | 单轴 | 三轴 |", "|---|---|---|",
             "| 惯量 | J = 1 kg·m² | diag(1.2, 1.0, 0.8) kg·m² |",
             "| 动力学 | θ̈ = (τ+d)/J | ω̇ = I⁻¹[τ+d−ω×(Iω)] |",
             "| 姿态运动学 | 常加速度精确位置更新 | q̇ = 0.5 q⊗[0,ω]，RK4 与四元数归一化 |",
             "| 时间步、总时长 | 0.05 s、20 s | 0.02 s、20 s |",
             "| 执行器 | ±0.5 N·m | 每轴 ±2 N·m |",
             "| 控制器可见观测 | 带噪声的角误差、角速度（2 维） | 带噪声的最短本体系旋转误差、角速度（6 维） |",
             "| 动作 | 归一化连续动作映射为实际力矩 | 3 维归一化连续动作映射为实际力矩 |",
             "| 验收 | 20 s 无发散，末 5 s 每个样本同时误差 ≤1°、角速度 ≤0.5°/s | 同左，以向量范数判定 |", "",
             "单轴和三轴分别比较各算法；二者执行器与状态维度不同，跨轴数的控制代价不直接解释为算法优劣。控制代价为 ∫‖τ‖²dt，单位 N²·m²·s；它是控制负担代理指标，不是实际电能。三轴误差处理 q 与 −q 同姿态及 180° 最短路径边界，角速度属于本体坐标系。", "",
             "共同训练奖励 r = −(6‖e‖² + 0.6‖ω‖² + 0.03‖τ‖²)，e、ω 使用弧度制；发散首次发生时额外惩罚 1000。训练域随机化包括惯量比例 0.8–1.2、偏置干扰 −0.02–0.02 N·m、正弦幅值 0.02 N·m、传感器噪声标准差 0–0.15°。三轴按各测量分量加噪声；测试中的更强噪声与组合条件用于压力评估。", "",
             "## 03 各算法的含义与改进点", "",
             "- 原始 PID：单轴沿用上一版的 PD(4,0,3)，三轴沿用新增材料的有限差分 PID(3,0.5,1)。两者是可追溯的原始基线。",
             "- 整定 PID：只在独立验证场景选择比例、积分、阻尼增益，导数项使用测得角速度，增加饱和条件下的积分保护。",
             "- 改进 PID：在相同可见观测上增加角速度低通滤波，并用验证集选择增益与滤波参数。若最优 alpha=1，说明该轴验证条件下未选择滤波，不能声称滤波本身带来提升。",
             "- PPO：通过新采样轨迹进行策略更新，裁剪策略变化以降低更新波动；直接输出连续控制动作。",
             "- SAC：复用经验回放，学习随机策略和价值函数，熵项维持探索；推理时使用确定性动作。",
             "- TD3：复用经验回放，采用双价值网络和延迟策略更新，显式动作噪声用于训练探索，推理时关闭探索。", "",
             "整定/改进 PID 的力矩形式为 τ* = Kp·e_c + Ki·z − Kd·ω_f，然后经过执行器饱和；e_c 是目标修正方向，单轴取负角误差、三轴使用本体系目标修正旋转向量。ω_f[k] = ω_f[k−1] + alpha·(ω_measured[k]−ω_f[k−1])，alpha=1 等于不滤波。积分候选值限幅，若增量进一步推高饱和则拒绝该积分更新。原三轴 PID 则保留误差有限差分导数，以展示方法改动的影响。", "",
             "三种 RL 在同一轴下使用相同动力学、观测、力矩映射、奖励、域随机化范围、训练交互预算与训练种子。算法的梯度更新次数和计算开销不同，不能把相同交互步数称作相同计算量。模型与 PID 均用固定验证场景选择，正式测试不用于选模型或改参数。", "",
             "### 实际选定的 PID 参数", "",
             "| 轴数 | 类型 | Kp | Ki | Kd | 滤波 alpha |", "|---|---|---:|---:|---:|---:|" ]
    for axis in AXES:
        for kind, label in (("legacy", "原始"), ("tuned", "整定"), ("improved", "改进")):
            g = pids[axis][kind]
            g = g.get("gains", g)
            lines.append(f"| {AXES[axis]} | {label} PID | {number(g['kp'])} | {number(g['ki'])} | {number(g['kd'])} | {number(g.get('alpha',1))} |")
    lines += ["", "### 实际训练记录", "",
              "| 轴数 | 算法 | 种子 | 请求步数 | 实际交互步数 | SB3 内部更新计数 | 入选 checkpoint 步数 |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in training:
        lines.append(f"| {AXES[m['axis']]} | {m['algorithm']} | {m['seed']} | {m['requested_steps']} | {m['actual_steps']} | {m['gradient_updates']} | {m['best_checkpoint_steps']} |")
    lines += ["", "PPO 按完整 rollout 结束，因此实际步数可能比请求步数略大；详细超参数、验证历史、模型 SHA256 和源码 SHA256 保存在训练目录。", "",
              "SB3 内部更新计数读取各算法的 `_n_updates`：PPO 按训练 epoch 计数，每个 epoch 内还包含多个 minibatch 更新；SAC/TD3 按梯度迭代计数，TD3 的策略更新另外延迟。因此此列口径随算法不同，不能横向比较为相同定义的优化器更新次数，也不代表相同计算量。", "",
              "## 04 同条件正式结果", "",
              "测试工况预先确定为正常、低/高惯量、两档正弦+偏置干扰、两档传感器噪声、脉冲和组合压力。每轴 9 工况等样本量汇总；三个训练种子全部保留，SD 为独立训练种子间样本标准差，PID 无训练种子 SD。稳定时间仅对成功回合求均值，必须结合成功率阅读。", "",
              "### 单轴", "", markdown_table(summaries, "single"), "", "### 三轴", "", markdown_table(summaries, "three"), "",
              "若发生物理发散，冻结发散状态并填充到 20 s，随后力矩置零；此填充不表示真实系统继续运行。未达持续保持标准但未发散的回合仍完整仿真 20 s，两者均计为不成功。RMS 包含发散后的填充状态，应先看成功率并核对发散标志；发散回合因有效时长较短产生的低控制代价不视为优点。轨迹另存发散时间和有效运行时长。", "",
              "## 05 最终结果究竟改善了多少", "",
              "以下使用同一个测试场景进行逐项配对，分别检验整套改进相对原始基线的作用、额外滤波方案相对整定 PID 的作用。正的误差下降百分比表示改善，负值表示退化；成功率变化单位为百分点。", "",
              "| 轴数 | 比较基线 | 配对场景数 | 成功率变化 | 末段误差：基线→改进 (°) | 末段误差下降 | RMS 变化 (°) | 控制代价变化 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in improvements:
        lines.append(f"| {AXES[row['axis']]} | {LABELS[row['baseline']]} | {row['paired_scene_count']} | {row['success_change_pp']:+.1f} pp | {number(row['tail_mean_error_original_deg'])} → {number(row['tail_mean_error_improved_deg'])} | {number(row['tail_error_reduction_percent'],1)}% | {row['rms_change_deg']:+.3f} | {row['control_cost_change']:+.3f} |")
    lines += ["", "这个表只描述本次预定测试集上的实测结果。若某轴相对整定 PID 没有继续提升，也按实际值报告。算法排名取决于工况、指标和训练种子，不能由某张代表曲线推断所有条件都更优。", ""]
    for item in improvements:
        if (item["axis"], item["baseline"]) not in (("single", "baseline_PID"), ("three", "tuned_PID")):
            continue
        lines += [f"{AXES[item['axis']]}改进 PID 相对{LABELS[item['baseline']]}：成功率变化 {item['success_change_pp']:+.1f} 个百分点，末段误差由 {item['tail_mean_error_original_deg']:.3f}° 变为 {item['tail_mean_error_improved_deg']:.3f}°，全程 RMS 变化 {item['rms_change_deg']:+.3f}°，控制代价变化 {item['control_cost_change']:+.3f}。各指标存在取舍，不能统称全部性能改善。", ""]
    lines += [
              "## 06 RL 部署校准：同一模型能否进一步改善", "",
              "神经策略在目标静止观测 `[0,…,0]` 上可能仍输出非零动作，使名义目标状态不再是控制器的零动作平衡点。校准统一采用 a_cal(o) = clip(a_raw(o) − a_raw(0), −1, 1)：只减去该模型的零观测动作，不重新训练、不读取真实状态、不改变执行器边界。偏置向量及原 checkpoint 哈希写入每个控制器描述与 NPZ 元数据。", "",
              "该规则针对名义零误差、零角速度的固定点，不估计或补偿未知外扰；原策略在扰动下需要的偏置也可能被减掉，因此可以发生退化。先在固定验证配置上检查全部 18 个模型的原策略与校准策略（共 36 个版本），再把所有版本共同纳入正式测试，未按正式测试结果选取改进版本。原6类控制器的算法表保持原口径，下表独立解释部署后处理的作用。", "",
              "配对键严格包含轴数、算法、同一个训练种子和同一环境配置。各回合保存校准前后两条轨迹；没有把不同种子互配，也没有用跨种子的平均轨迹替代本表的原始配对。", "",
              "| 轴数 | 算法 | 配对回合数 | 成功率：原→校准 | 变化 | 末段误差：原→校准 (°) | 误差下降 | 挽回/失去成功回合 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in calibration_summaries:
        lines.append(f"| {AXES[row['axis']]} | {row['algorithm']} | {row['paired_episode_count']} | {100*row['success_rate_raw']:.1f}% → {100*row['success_rate_calibrated']:.1f}% | {row['success_change_pp']:+.1f} pp | {number(row['tail_mean_error_raw_deg'])} → {number(row['tail_mean_error_calibrated_deg'])} | {number(row['tail_error_reduction_percent'],1)}% | {row['success_recovered_count']} / {row['success_lost_count']} |")
    calibration_reference = [row for row in summaries if row["controller"] == "improved_PID"]
    lines += ["", "### 校准策略的完整指标与 PID 参考", "",
              "下表使用同一组预定工况，独立展示校准策略；改进 PID 为参考。PID 每配置一次，RL 每配置保留三个训练种子。成功回合稳定时间的均值取自各策略自己的成功子集，成功率变化时不能将该均值差直接解释为同一批回合的速度变化。", "",
              "#### 单轴", "", markdown_table(calibration_reference + calibrated_summaries, "single"), "",
              "#### 三轴", "", markdown_table(calibration_reference + calibrated_summaries, "three"), ""]
    for axis in AXES:
        candidates = [row for row in calibrated_summaries if row["axis"] == axis]
        reference = next((row for row in calibration_reference if row["axis"] == axis), None)
        if candidates and reference:
            best = max(candidates, key=lambda row: row["success_rate"])
            relation = "仍低于" if best["success_rate"] < reference["success_rate"] else "高于" if best["success_rate"] > reference["success_rate"] else "等于"
            lines += [f"这批 9 工况等权测试中，{AXES[axis]}校准 RL 的最高总体成功率为 {100*best['success_rate']:.1f}%（{LABELS[best['controller']]}），{relation}改进 PID 的 {100*reference['success_rate']:.1f}%；较小末段误差也不能替代持续保持成功率。", ""]
    worsened_rms = [row for row in calibration_summaries if row["rms_change_deg"] > 0]
    if worsened_rms:
        details = "；".join(f"{AXES[row['axis']]} {row['algorithm']} 的全程 RMS 增加 {row['rms_change_deg']:.3f}°" for row in worsened_rms)
        lines += [f"校准的取舍也保留：{details}。应结合上表中的成功率、末段误差、RMS、控制代价与成功回合稳定时间共同阅读。", ""]
    lines += ["校准变化是原模型加部署后处理的效果，不能把其结果冒充原始 PPO/SAC/TD3 的训练性能。零动作补偿也不保证对所有工况有益，成功率、误差和控制代价变化均保留原始配对证据。", "",
              "## 07 历史模型迁移部署复测", "",
              "原档案 4 份模型保持不变，只加载推理。它们输出的是物理力矩，新统一环境将动作换算后经过相同执行器饱和。单轴执行器从原历史 ±2 N·m 改为当前 ±0.5 N·m；时步、积分器和持续保持逻辑也按新部署环境统一。因此下表是迁移部署复测，不能解释为原实验的逐数值复现，也不与新等预算模型混合排名。", "",
              "### 历史单轴", "", markdown_table(history_summary, "single") if history_summary else "未执行历史部署评估。", "",
              "### 历史三轴", "", markdown_table(history_summary, "three") if history_summary else "未执行历史部署评估。", "",
              "## 08 可交付成果与证据位置", "",
              f"- [正式逐回合结果]({res}/episode_metrics.csv)、[按工况统计]({res}/summary.csv)，含成功与失败。",
              f"- [全控制器配对结果]({res}/pairwise_results.csv)、[PID 改进配对结果]({res}/paired_improvements.csv)，学习策略先按相同场景跨种子取均值。",
              f"- [同种子 RL 校准前后配对]({res}/policy_calibration_pairs.csv)：逐模型、逐配置的一一配对，保留各指标变化及两条轨迹路径。",
              f"- [校准验证记录]({mods}/policy_calibration_validation.csv)、[校准规则与验证清单]({mods}/policy_calibration_manifest.json)：正式测试前保留全部模型的原/校准对照。",
              f"- `{res}/single/raw_traces` 与 `{res}/three/raw_traces` 保存完整 NPZ：误差、角速度、实际控制力矩、扰动、失败标志及元数据。",
              f"- `{mods}` 保存 {len(training)} 组正式模型目录、验证日志、训练元数据、验证 PID 搜索与选定参数。",
              f"- `{res}/historical` 独立保存 4 个历史策略的迁移部署结果和轨迹。",
              f"- [离线结果页面](结果总览.html)、`{res}/figures` 展示代表轨迹、成功率、精度/控制代价、配对改进和种子波动。", "",
              "## 09 仍需要团队完成的结项工作", "",
              "本包完成可复现实验与代码证据，团队仍需把数学模型和算法选择用自己的语言解释、整理正式综述/学校模板、制作 PPT 并演练答辩。没有开展硬件实验、飞行验证或实际能源测量；三轴仍是理想刚体与简化扰动，不包含反作用轮动量饱和、通信延迟和轨道任务。本次 3 种训练种子、有限场景样本不能代表所有卫星任务，也不能保证学校优秀等级。", "",
              "训练数据只包含预定域随机化，超出该范围的压力结果需如实报告。若以后修改奖励或控制器，需另建版本并重新划分正式测试；不要使用这批测试数据反复调参后继续称其为独立测试。", ""]
    (output_dir / "新融合分析.md").write_text("\n".join(lines), encoding="utf-8")

    readme = f"""# 单轴与三轴多算法对比：新增成果入口

这里把新增的 PID/PPO/SAC/TD3 工作融入统一的结项实验。原文件与存档模型保留，新训练模型、正式测试和历史迁移复测分开保存。

先看 [新融合分析](新融合分析.md) 和 [离线结果总览](结果总览.html)。本次正式测试 {len(rows)} 回合；历史策略迁移复测 {len(history_rows)} 回合。实际改进、失败与种子波动均由保存的数据生成。

| 工作 | 目录/文件 | 用途 |
|---|---|---|
| 01 数学模型与共同条件 | `{module}/envs.py` | 单轴刚体、三轴四元数与耦合动力学 |
| 02 PID 基线、整定、改进 | `{module}/controllers.py`、`{mods}/pid_configs.json` | 验证集寻优、饱和积分保护与测量滤波 |
| 03 PPO/SAC/TD3 新训练 | `{mods}/single`、`{mods}/three` | 同条件、同交互预算、3 种子 |
| 04 正式对比 | `{res}/episode_metrics.csv`、`{res}/summary.csv` | 全回合和按工况统计 |
| 05 配对改进 | `{res}/paired_improvements.csv`、`{res}/pairwise_results.csv` | 相同初态、外扰和噪声下的效果变化 |
| 06 全部原始轨迹 | `{res}/single/raw_traces`、`{res}/three/raw_traces` | 独立复算指标和绘图 |
| 07 历史模型迁移部署 | `{res}/historical` | 单独评估，不混入新模型排名 |
| 08 图表与报告 | `{res}/figures`、`新融合分析.md`、`结果总览.html` | 答辩和结项素材 |
| 09 RL 部署校准 | `{module}/improvements.py`、`{res}/policy_calibration_pairs.csv` | 同模型、同种子、同配置比较零动作补偿前后 |

在结项成果包根目录运行：

```powershell
python run_multi.py check
python run_multi.py evaluate --samples 20
python run_multi.py plot
python run_multi.py report
```

正式模型已经保存，可直接复测。重新训练时请指定新的输出目录，防止把已发布实验覆盖：

```powershell
python run_multi.py tune-pid --output "07_单轴三轴多算法对比/另一次训练"
python run_multi.py train --steps 1000000 --jobs 3 --output "07_单轴三轴多算法对比/另一次训练"
python run_multi.py calibrate --models "07_单轴三轴多算法对比/另一次训练"
python run_multi.py evaluate --models "07_单轴三轴多算法对比/另一次训练" --output "07_单轴三轴多算法对比/另一次结果"
python run_multi.py plot --results "07_单轴三轴多算法对比/另一次结果"
python run_multi.py report --models "07_单轴三轴多算法对比/另一次训练" --results "07_单轴三轴多算法对比/另一次结果" --output "07_单轴三轴多算法对比/另一次结果/报告"
```

固定验收：完整部署 20 秒，末 5 秒每个样本误差 ≤1° 且角速度 ≤0.5°/s，无发散。此指标比一次入阈即提前终止更严格，历史结果与本次数字不能直接横向拼接。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    figures = sorted((results_dir / "figures").glob("*.png"))
    cards = "".join(f'<figure><img src="{html.escape(relative(path,output_dir),quote=True)}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem)}</figcaption></figure>' for path in figures)
    payload = json.dumps(load_rows(results_dir / "summary.csv"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    improvement_items = "".join(f"<li>{AXES[r['axis']]}，改进 PID 相对{LABELS[r['baseline']]}：成功率 {r['success_change_pp']:+.1f} 个百分点，末段误差 {r['tail_mean_error_original_deg']:.3f}° → {r['tail_mean_error_improved_deg']:.3f}°。</li>" for r in improvements)
    calibration_items = "".join(f"<li>{AXES[r['axis']]} {r['algorithm']}：成功率 {100*r['success_rate_raw']:.1f}% → {100*r['success_rate_calibrated']:.1f}%，末段误差 {r['tail_mean_error_raw_deg']:.3f}° → {r['tail_mean_error_calibrated_deg']:.3f}°；挽回 {r['success_recovered_count']} 回合，失去 {r['success_lost_count']} 回合。</li>" for r in calibration_summaries)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单轴与三轴多算法对比</title><style>
body{font-family:'Microsoft YaHei',system-ui,sans-serif;background:#f2f5f8;color:#172a3a;margin:0}main{max-width:1250px;margin:auto;padding:28px}h1{font-size:28px}p,li{line-height:1.8}.box,figure{background:white;border:1px solid #dbe3e9;border-radius:10px;padding:20px;margin:18px 0}label{margin-right:20px}select{padding:8px;border:1px solid #bac8d3;border-radius:6px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dbe3e9;text-align:right}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}th{background:#edf3f8}.scroll{overflow:auto}img{display:block;width:100%;height:auto}figcaption{color:#516675;font-size:13px}a{color:#246097}.note{color:#52687c}</style><main>
<h1>单轴与三轴：PID / PPO / SAC / TD3</h1><p>正式实验 EPISODES 回合；历史迁移部署 HISTORY 回合。相同轴数下使用共同动力学、初态、力矩限幅、扰动和传感器噪声；新学习模型全部保留 3 个训练种子。</p><p><a href="新融合分析.md">完整分析与边界</a> · <a href="README.md">运行与目录说明</a> · <a href="RAW">逐回合原始数据</a></p>
<div class="box"><strong>实际配对改进</strong><ul>IMPROVEMENTS</ul><p class="note">改善与退化均由实测数据计算；PID 整定与 checkpoint 选择只使用验证集。</p></div>
<div class="box"><strong>RL 零动作校准：同 checkpoint、同种子逐回合配对</strong><ul>CALIBRATION</ul><p class="note">动作 = clip(原动作 − 零观测动作, −1, 1)。原模型保留；校准是部署后处理，可能改善也可能退化，独立于原算法排名。</p></div>
<div class="box"><label>轴数 <select id="axis"><option value="single">单轴</option><option value="three">三轴</option></select></label><label>工况 <select id="case"></select></label><label>版本 <select id="version"><option value="raw">原6类控制器</option><option value="calibrated">校准后3种RL</option></select></label><p class="note">成功要求完整 20 秒无发散，末 5 秒误差 ≤1°且角速度 ≤0.5°/s。稳定时间仅对成功回合求均值；末段误差包含失败回合。控制代价不是实际能源。</p><div class="scroll"><table><thead><tr><th>控制器</th><th>工况</th><th>回合数</th><th>成功率</th><th>RMS (°)</th><th>末段误差 (°)</th><th>控制代价</th><th>成功稳定时间 (s)</th></tr></thead><tbody id="rows"></tbody></table></div></div>
FIGURES<p class="note">历史图只展示迁移部署能力，不与同预算新模型混合排名。本实验为仿真刚体，未开展硬件或飞行验证。</p></main><script>
const data=PAYLOAD;const labels={baseline_PID:'原始 PID',tuned_PID:'整定 PID',improved_PID:'改进 PID',PPO:'PPO',SAC:'SAC',TD3:'TD3',PPO_calibrated:'PPO（校准）',SAC_calibrated:'SAC（校准）',TD3_calibrated:'TD3（校准）'};const axis=document.getElementById('axis'),cases=document.getElementById('case'),version=document.getElementById('version'),body=document.getElementById('rows');const f=(v,n=3)=>v===null?'—':Number(v).toFixed(n);
function fillCases(){cases.innerHTML='';[...new Set(data.filter(r=>r.axis===axis.value).map(r=>r.case_id))].forEach(c=>{let o=document.createElement('option');o.value=c;o.textContent=c;cases.appendChild(o)});draw()}
function draw(){body.innerHTML='';data.filter(r=>r.axis===axis.value&&r.case_id===cases.value&&(r.controller.endsWith('_calibrated')===(version.value==='calibrated'))).forEach(r=>{let tr=document.createElement('tr');[labels[r.controller]||r.controller,r.case_id,r.n_episodes,f(r.success_rate*100,1)+'%',f(r.rms_error_deg_mean),f(r.tail_mean_error_deg_mean),f(r.control_cost_mean),f(r.settling_time_s_mean_successful)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});body.appendChild(tr)})}axis.addEventListener('change',fillCases);cases.addEventListener('change',draw);version.addEventListener('change',draw);fillCases();</script></html>"""
    replacements = {"EPISODES": str(len(rows)), "HISTORY": str(len(history_rows)),
                    "RAW": html.escape(f"{res}/episode_metrics.csv", quote=True),
                    "IMPROVEMENTS": improvement_items, "CALIBRATION": calibration_items,
                    "FIGURES": cards, "PAYLOAD": payload}
    for key, value in replacements.items():
        page = page.replace(key, value)
    (output_dir / "结果总览.html").write_text(page, encoding="utf-8")
    print(f"Generated actual comparison report and offline overview: {output_dir}", flush=True)
    return {"output_dir": str(output_dir), "episode_count": len(rows), "training_run_count": len(training),
            "historical_episode_count": len(history_rows), "pid_improvements": improvements,
            "policy_calibration_improvements": calibration_summaries}
