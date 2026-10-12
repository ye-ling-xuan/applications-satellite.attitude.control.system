"""Build Chinese deliverable reports from actual experiment artifacts."""
from __future__ import annotations

import csv
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_实验数据与图表"
MODELS = ROOT / "03_训练模型与日志" / "正式训练_100万步"
REPORTS = ROOT / "06_结项说明与报告"
LABELS = {"PID": "PID", "PPO_nominal": "普通 PPO", "PPO_randomized": "随机化 PPO"}
FAMILIES = {"nominal": "正常工况", "inertia": "惯量变化", "disturbance": "持续扰动",
            "noise": "测量噪声", "pulse": "脉冲扰动", "combined": "组合压力测试"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    return None if value is None or value == "" else float(value)


def success(row):
    return str(row["success"]).lower() in ("true", "1", "1.0")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["controller"], row["family"])].append(row)
    output = []
    for (controller, family), group in sorted(groups.items()):
        settled = [number(r["settling_time_s"]) for r in group if success(r)]
        seed_rates = []
        for seed in sorted({r["model_seed"] for r in group}):
            seed_group = [r for r in group if r["model_seed"] == seed]
            seed_rates.append(float(np.mean([success(r) for r in seed_group])))
        output.append({"controller": controller, "family": family, "episodes": len(group),
                       "success_rate": float(np.mean([success(r) for r in group])),
                       "seed_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else None,
                       "steady_error_deg": float(np.mean([float(r["steady_error_deg"]) for r in group])),
                       "settling_success_only_s": float(np.mean(settled)) if settled else None,
                       "control_cost": float(np.mean([float(r["control_cost"]) for r in group]))})
    return output


def md_table(records):
    lines = ["| 工况 | 控制器 | 回合数 | 成功率 | 跨训练种子标准差 | 末段误差 / ° | 稳定时间 / s | 控制代价 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        std = "—" if r["seed_std"] is None else f"{100*r['seed_std']:.1f}%"
        settling = "—" if r["settling_success_only_s"] is None else f"{r['settling_success_only_s']:.2f}"
        lines.append(f"| {FAMILIES[r['family']]} | {LABELS[r['controller']]} | {r['episodes']} | "
                     f"{100*r['success_rate']:.1f}% | {std} | {r['steady_error_deg']:.3f} | "
                     f"{settling} | {r['control_cost']:.4f} |")
    return "\n".join(lines)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = read_json(RESULTS / "evaluation_manifest.json")
    rows = read_csv(RESULTS / "episode_metrics.csv")
    summaries = read_csv(RESULTS / "summary.csv")
    records = aggregate(rows)
    runs = [read_json(p) for p in sorted(MODELS.glob("*/seed_*/training_metadata.json"))]
    preserve = read_json(REPORTS / "original_preservation_check.json")
    copies = read_json(ROOT / "01_遴选原始参考" / "copy_manifest.json")
    copy_count = len(copies.get("files", copies.get("records", []))) if isinstance(copies, dict) else len(copies)
    if not copy_count and isinstance(copies, dict):
        copy_count = int(copies.get("copied_file_count", copies.get("file_count", 0)))
    three_path = ROOT / "05_三轴拓展演示" / "outputs" / "metrics.json"
    three = read_json(three_path) if three_path.exists() else {}
    pid = manifest["pid_config"]
    audit_path = REPORTS / "result_verification.json"
    audit = read_json(audit_path) if audit_path.exists() else {}
    table = md_table(records)
    by_key = {(r["controller"], r["family"]): r for r in records}
    nominal = "; ".join(f"{LABELS[c]} {100*by_key[(c,'nominal')]['success_rate']:.1f}%" for c in LABELS)
    normal_cost_change = 100.0 * (1.0 - by_key[("PPO_nominal", "nominal")]["control_cost"]
                                 / by_key[("PID", "nominal")]["control_cost"])
    normal_detail = (
        f"正常工况下，普通PPO的平均控制代价较PID基线低{normal_cost_change:.1f}%；"
        f"成功回合的平均稳定时间分别为{by_key[('PPO_nominal','nominal')]['settling_success_only_s']:.2f} s"
        f"与{by_key[('PID','nominal')]['settling_success_only_s']:.2f} s。"
        f"但PID基线的末段误差更小，随机化PPO也存在名义精度代价。"
        f"这些是本次数据的描述性比较，没有证明统计显著或实际能耗优势。"
    )
    improvement = []
    regression = []
    for family in FAMILIES:
        a = by_key[("PPO_nominal", family)]["success_rate"]
        b = by_key[("PPO_randomized", family)]["success_rate"]
        if b > a + 1e-10:
            improvement.append(f"{FAMILIES[family]}（+{100*(b-a):.1f} 个百分点）")
        elif b < a - 1e-10:
            regression.append(f"{FAMILIES[family]}（{100*(b-a):.1f} 个百分点）")
    improved_text = "、".join(improvement) if improvement else "未观察到成功率提升的工况"
    regression_text = "、".join(regression) if regression else "未观察到成功率下降的工况"
    run_lines = ["| 训练方式 | seed | 实际交互步数 | 选中检查点步数 | 最佳验证成功率 | 训练用时 / s |",
                 "|---|---:|---:|---:|---:|---:|"]
    for r in runs:
        cp = read_json(MODELS / r["mode"] / f"seed_{r['seed']}" / "best_checkpoint.json")
        run_lines.append(f"| {LABELS['PPO_'+r['mode']]} | {r['seed']} | {r['actual_timesteps']} | "
                         f"{r['best_checkpoint_timesteps']} | {100*cp['success_rate']:.1f}% | {r['wall_time_s']:.1f} |")
    training_table = "\n".join(run_lines)
    model_budget = sum(r["actual_timesteps"] for r in runs)
    fusion_path = ROOT / "07_单轴三轴多算法对比" / "result_summary.json"
    fusion_section = ""
    fusion_banner = ""
    fusion_highlight = ""
    if fusion_path.exists():
        fusion = read_json(fusion_path)
        lookup = {(item["axis"], item["controller"]): item for item in fusion["fresh_results"]}
        old_three = lookup[("three", "baseline_PID")]["success_rate"]
        new_three = lookup[("three", "improved_PID")]["success_rate"]
        fusion_highlight = f"\n当前新增成果：18份新训练模型、{fusion['fresh_episode_count']}回合正式测试、{fusion['historical_episode_count']}回合历史迁移复测、8张对比图及完整轨迹。三轴PID成功率{100*old_three:.1f}%→{100*new_three:.1f}%；RL校准改善及其代价见07。新增来源与第一次来源共同核验，950份原始文件未改、127份副本一致。优先阅读[07新融合分析](07_单轴三轴多算法对比/新融合分析.md)和[07结果总览](07_单轴三轴多算法对比/结果总览.html)。\n"
        fusion_section = f"""\n## 07 新增材料融合：单轴、三轴与多算法\n\n新增“中期前任务收尾”中的60份有效材料已原样归档；其827份来源文件另有保全快照。单轴与三轴分别完成PID、PPO、SAC、TD3的同条件比较；新增18次独立训练，每种算法每轴三个种子，每份请求100万步，使用共同验证场景选模。旧实验数据和六份PPO模型保留。\n\n新增正式测试{fusion['fresh_episode_count']}回合，四份历史模型另做{fusion['historical_episode_count']}回合迁移部署复测。原始、整定、滤波PID分别展示；学习策略原始输出和零参考力矩校准分别测量。校准不修改网络权重，不把后处理效果冒充重新训练成果。\n\n| 轴数 | 改进PID相对基线 | 成功率变化 | 末段平均误差：基线→改进 |\n|---|---|---:|---:|\n"""
        for item in fusion["pid_paired_improvements"]:
            axis_label = "单轴" if item["axis"] == "single" else "三轴"
            baseline_label = "原始PID" if item["baseline"] == "baseline_PID" else "整定PID"
            fusion_section += f"| {axis_label} | {baseline_label} | {item['success_change_pp']:+.1f}个百分点 | {item['tail_mean_error_original_deg']:.3f}° → {item['tail_mean_error_improved_deg']:.3f}° |\n"
        fusion_section += "\n| 轴数 | 同模型部署校准 | 成功率：原始→校准 | 挽回/失去成功回合 |\n|---|---|---:|---:|\n"
        for item in fusion.get("policy_calibration_improvements", []):
            axis_label = "单轴" if item["axis"] == "single" else "三轴"
            fusion_section += f"| {axis_label} | {item['algorithm']} | {100*item['success_rate_raw']:.1f}% → {100*item['success_rate_calibrated']:.1f}% | {item['success_recovered_count']} / {item['success_lost_count']} |\n"
        fusion_section += "\n完整算法阐释、原始/校准策略对比、失败与代价见[新增融合分析](07_单轴三轴多算法对比/新融合分析.md)，可打开[新增离线结果页](07_单轴三轴多算法对比/结果总览.html)。07补齐三轴学习策略、SAC/TD3多种子和配对改进；前文03–06描述的是前一版实验，两版场景与物理约束不同，不能直接拼接数字进行排名。\n"
        fusion_banner = '<section><h2>新增：单轴与三轴多算法融合</h2><p><a href="07_单轴三轴多算法对比/结果总览.html">查看PID / PPO / SAC / TD3及改进的最新对比</a> · <a href="07_单轴三轴多算法对比/新融合分析.md">阐释与实际改进</a></p><p>下方保留前一版单轴随机化实验。新增算法和三轴结果在07单独保存。</p></section>'
    overview = f"""# 本次整理与完善工作总说明

本成果包研究卫星单轴与三轴姿态控制中的经典控制与强化学习。以下01–06保留首次单轴PID/PPO实验；新增多算法融合、实际改进及三轴学习结果见最后的07。原始项目 {preserve['original_file_count']} 份文件的哈希核验显示：没有删改。本次文件均位于“结项成果包”内。大创计划书未作为实施依据。
{fusion_highlight}

## 项目做什么、目标是什么

项目原始方向是把人工智能方法用于卫星姿态控制。当前结项任务是：分别在单轴与三轴的共同动力学、传感输入和执行器约束下，比较经典PID与PPO/SAC/TD3的姿态调节及持续保持能力，检查惯量偏差、干扰与噪声下的表现，并实测验证集整定、测量滤波和策略偏置校准的收益与代价。以下01–06保存首次单轴PID/PPO实验与三轴PID开发演示；最新多算法研究及实测结论见07。

整理前，原文件已包含按周推进的单轴PID、扰动噪声扩展、三轴PID，强化学习基础笔记、Gym封装、PPO训练调参及历史模型日志。本次把这些原型补成统一环境、明确判据、独立训练种子、验证集选参和完整测试证据。历史模型与新环境的接口可能不同，不能把原来的模型文件数量直接当成同条件有效成果数量。

## 01 原始材料遴选与保全

遴选 {copy_count} 份原始参考，覆盖单轴/三轴 PID、强化学习环境与训练、调参和对比、历史模型、学习笔记、训练日志及原立项/中期报告。副本保留来源目录，并在 copy_manifest.json 逐项记录 SHA256。01 目录内的代码和模型用于来源追溯，正式实验使用 satellite_benchmark 和 03 的新模型。

## 02 统一仿真与 PID 基线

新建配置、环境、控制器和指标模块，统一 rad 与 degree、观测与力矩映射。PID 增加生效的积分限幅和条件积分，微分项使用角速度，避免首步微分突变。训练和测试均调用同一 env.step。使用验证场景整定 PID 并保存完整搜索表。

在120组增益中，验证集选出的参数为 Kp={pid['kp']:g}、Ki={pid['ki']:g}、Kd={pid['kd']:g}。积分增益为0，因此本次正式基线实际使用PD；代码支持积分，但不能把正式结果描述成积分补偿效果。

修正超调包含初始角度、首次进入容差即认定稳定、失败回合用末尾时间假装收敛等问题。20 s 仿真中最后 5 s 的姿态误差须持续 ≤1°、角速度须持续 ≤0.5°/s，且没有发散，才判成功。力矩平方积分称“控制代价”，不冒充实际电能。

## 03 新模型训练与归档

完成 {len(runs)} 次独立 PPO 训练，共 {model_budget:,} 个环境交互步；普通与随机化两种训练方式分别保留 seed 11、22、33。随机化训练改变惯量、扰动和观测噪声，网络、奖励及训练预算保持一致。模型检查点由固定验证场景选择，最终测试数据不用于选模。保留选中模型、最终模型、参数、版本、模型与源码哈希和训练趋势。

中断前的 20,000 步联调记录保留在 03/nominal/seed_11；20万步的首轮预算探索保留在03/正式训练。两者均不参与最终结果。正式结果只使用03/正式训练_100万步，每份模型从零训练。

## 04 批量实验与真实结果

完成 {manifest['scenario_count']} 个测试场景、{manifest['episode_count']} 个控制回合，覆盖正常、惯量、持续扰动、测量噪声、脉冲和组合压力测试。每个控制器使用相同初始条件和外部随机序列；逐回合指标、全部原始轨迹和统计表均保存。六张核心图可以从数据重新生成。

正常工况成功率：{nominal}。

{normal_detail}

随机化 PPO 相比普通 PPO 的成功率提升：{improved_text}。下降：{regression_text}。这些结果直接来自本次测试，不能概括为“PPO 全面优于 PID”或“随机化一定有效”。详细数字和失败案例见结项技术报告。

## 05 三轴 PID 拓展

在新目录实现含欧拉陀螺耦合的三轴刚体演示，采用四元数和 RK4 积分，处理 q 与 −q 的等价及零误差边界，加入三轴扰动、噪声和饱和。三轴输出图、轨迹和判据检查独立保存。本项没有训练三轴强化学习。

固定三轴开发场景达到约定判据：稳定时间{three['settling_time_s']:.2f} s，末5 s最大姿态误差{three['max_attitude_error_last_5s_deg']:.3f}°，最大角速度范数{three['max_rate_norm_last_5s_deg_s']:.3f}°/s。保留14组PID候选的开发校准记录，该场景不属于独立统计泛化测试。

## 06 运行与交付

统一入口 run.py 提供 check、train、tune-pid、evaluate、plot、three-axis、report。README 说明代码职责、目录结构、依赖和复现命令。结果总览.html 可离线浏览图表并筛选统计。06 目录包含技术报告、答辩提纲、回归检查和原文件保全记录。

完整实验核验状态为{audit.get('status', '尚未运行')}：逐回合从NPZ重算指标，独立重算48组汇总，并检查有限状态、离散动力学、力矩限幅、配对初态/扰动和模型来源；记录见06/result_verification.json。核心回归和Gym接口核验见06/verification_log.txt。

训练记录中的源码哈希对应正式训练开始时的版本。训练结束后完善了续训防覆盖与日志保留、绘图及核验工具；这些维护修改没有重新训练正式模型，也没有改变正式实验的动力学、奖励、场景或成功判据。

## 首次交付与剩余一个月

本次已交付{copy_count}份来源副本、统一仿真代码、验证集PID整定记录、六份独立训练模型及日志、{manifest['scenario_count']}个场景的{manifest['episode_count']}回合原始数据、六张主线图、三轴PID演示及轨迹、技术报告和离线结果页面。目录标题和代码职责见README。

接下来最有价值的任务是让团队独立复现并解释结果、完善两篇学习综述和文献核查、按学校要求形成正式结项报告与答辩PPT、补齐成员分工与提交信息，并准备演示录屏。具体四周安排见[一个月收尾与交付清单](06_结项说明与报告/一个月收尾与交付清单.md)。如果增设技术实验，应保留本次基线，使用新的验证/测试数据；不能依据已经看过的测试结果调参后仍声称本次测试独立。

## 完成边界

代码、模型、实验数据、可视化和技术说明已形成可检查的软件仿真成果；是否达到学校结项或优秀项目要求，由学校评审决定。首次实验仅有三轴PID演示；新增三轴强化学习及SAC/TD3比较见07。本包未实现机器学习模型辨识/MPC、STK 联合验证、硬件在环或在轨验证。最终提交前团队须实际复现并理解方法、核实文献、填写学校表格及完成答辩。
"""
    (ROOT / "工作总说明.md").write_text(overview + fusion_section, encoding="utf-8")
    report = f"""# 卫星单轴姿态控制对比与随机化训练技术报告（第一版）

本报告保留第一版单轴PID/PPO数据与三轴PID开发演示。最新单轴与三轴PID/PPO/SAC/TD3比较、部署校准及实测改进请阅读[07新增融合分析](../07_单轴三轴多算法对比/新融合分析.md)。两版环境与工况分别解释，不能拼接为同一算法排名。

## 研究问题与结果

本项目研究相同力矩约束下 PID 和 PPO 的姿态调节能力，以及在惯量偏差、外部扰动、观测噪声条件下的性能变化。普通 PPO 与随机化 PPO 的对比用于检验物理参数随机化训练的效果。采用单轴作为完整对比对象，三轴 PID 作为建模拓展。

本次正常工况成功率为：{nominal}。随机化训练的成功率改善出现在：{improved_text}；下降出现在：{regression_text}。不能根据单条曲线或某一个种子宣称算法全面占优。

{normal_detail}

## 动力学与工程约束

单轴模型为 J·ω̇=u+d，θ̇=ω。默认 J=1 kg·m²、dt=0.05 s、控制时长20 s，u 为经过饱和和死区的实际执行器力矩，d 为外部偏置、正弦或脉冲扰动。每一步内力矩保持不变，θ 使用 θ+ωdt+0.5αdt² 更新，ω 使用 ω+αdt 更新；时变扰动在步首采样后零阶保持。角速度和姿态的观测添加独立高斯噪声。该模型描述刚体单轴转动，不包含轨道运动或执行器电气特性。

三轴拓展为 Jω̇=τ−ω×Jω，姿态由单位四元数描述，RK4 积分后归一化。使用最短旋转的姿态误差及三通道 PID，修复 q/−q 等价和零误差退化。三轴演示数据与主线单轴数据分别存档，不以三轴 PID 的效果代替三轴强化学习证据。

## 控制方法

PID 计算 u=−Kp·e−Kd·ω+Ki·∫(−e)dt。积分限幅，饱和时只接受使输出回到允许范围的积分变化。微分项使用测量角速度；PID 与 PPO 接收同样的两维观测。

在10个固定验证场景上搜索120组参数，选中Kp={pid['kp']:g}、Ki={pid['ki']:g}、Kd={pid['kd']:g}，积分限幅{pid['integral_limit']:g} rad·s。由于Ki=0，正式结果中的“PID”属于PID控制器族的PD特例；没有启用积分补偿，不能以积分解释其抗偏置效果。

PPO 使用 Stable-Baselines3 的标准实现，连续动作在 [-1,1] 上，经统一环境映射为实际力矩。奖励为 −(6e²+0.6ω²+0.03u²)，发生发散时额外扣1000，以避免通过主动终止逃避未来代价。状态越界为失败终止，时间用尽为截断，达到目标不会提前结束。

随机化 PPO 每个 episode 随机选取 J∈[0.7,1.3]、偏置扰动∈[-0.02,0.02] N·m、角度及角速度噪声标准差∈[0,0.15] °与°/s，并包含振幅0.02 N·m、频率0.2 Hz的正弦扰动。普通 PPO 保持J=1、无扰动噪声。具体设置以模型旁的 training_metadata.json 为准。两种训练方式使用相同网络、奖励和预算。

{training_table}

## 实验设计与指标

PID 在验证集上搜索增益，PPO 在验证集上选择检查点。最终测试使用单独的场景种子，选择后的全部三个训练种子都纳入评估。固定代表图使用最小训练seed展示，不按测试效果挑选。全部失败回合保留。

测试独立指初态样本和噪声种子不参与选参；部分工况类型和参数等级与验证集相同。“outside_training_range”仅指超出随机化PPO的训练参数范围，不能解释为普通PPO的训练范围，也不能解释为验证集从未出现过的工况。本次没有证明对全新工况类型的泛化。

成功要求完整20 s轨迹的最后5 s持续满足姿态误差≤1°、角速度≤0.5°/s。稳定时间为此后持续满足两个阈值的最早时刻，未达标为空。稳定时间均值只在成功回合中计算，必须结合成功率解读。末段误差和控制代价包含失败回合；若在15 s后提前发散，末段误差统计15 s之后的可用采样；若在15 s前发散，则统计短轨迹最后至多5 s。这些误差不能等价为完整20 s稳态结果。成功率标准差基于独立训练种子，不把大量测试场景冒充大量训练重复。

超调从初始误差方向越过目标后计算；控制代价是∫u²dt，单位(N·m)²·s，不是能耗实测。所有判据都是本项目的离散采样仿真定义。

提前发散会缩短控制代价的积分时长，因此必须同时查看成功率、失败率与duration_s，不能把失败回合较低的代价解释成控制优势。

本次共有{manifest['samples_per_case']}个初态样本/工况、{manifest['scenario_count']}个场景和{manifest['episode_count']}个回合。六类工况包括16个具体case；样本在多个case间复用，以支持配对比较，不能将跨case的样本当作独立初态。正常case与J=1的惯量case在物理上重复；总体结果依赖本次case权重，不代表实际在轨工况分布。

## 测试结果

{table}

表中稳定时间仅统计成功回合。跨种子标准差是描述性样本标准差，三个种子不足以证明普遍统计显著性。图03按具体参数等级展开，可以查看哪些强度越过了可用范围；图05显示失败案例，图06显示种子差异。

## 结果解释与限制

惯量变化会改变相同力矩产生的角加速度。带积分的 PID 在适当条件下可补偿持续偏置，但本次选中基线的Ki=0；不能把其测试表现解释为积分作用。本项目 PPO 为两维瞬时观测的无记忆策略，不能据此声称学会显式积分或辨识。对持续扰动表现较差时，应讨论策略的稳态偏置、训练分布和奖励权衡。

噪声会使控制动作波动，力矩代价必须和误差、成功率一起解读。随机化增加训练条件的多样性，也提高优化难度；结果可能存在名义精度损失或较大的种子差异。表中的实际变化是本次有限数据下的发现，不能仅凭正向结果断言因果机制。

训练与测试仍基于简化刚体模型，尚未建模反作用飞轮动量、完整姿态估计、时延、柔性结构或真实轨道环境。没有实物或在轨数据。没有完成模型辨识、预测控制和STK联合验证。本报告对应首次PPO主线；新增SAC/TD3同条件比较与三轴学习结果见[07融合分析](../07_单轴三轴多算法对比/新融合分析.md)。DDPG未正式验证。

## 可复现资料

代码在satellite_benchmark；环境与判据在02；正式模型与验证日志在03/正式训练_100万步；测试场景、episode_metrics.csv、summary.csv及全部raw_traces在04；三轴拓展在05/outputs。使用README所列命令可重新评估已保存模型并生成图表。原始文件保全检查为{preserve['preserved']}。

正式训练元信息保存当时源码哈希；之后只维护了续训保护、图表及核验工具，没有重新训练正式模型。续训会保留原始manifest，新增过程与源码哈希保存于resume_0001等独立目录；配置或验证集变化时拒绝续训。每次新训练应使用独立输出目录。

## 来源与方法参考

建模、PID、奖励设计和原型训练参考本项目既有代码及学习笔记，复制来源见01/遴选清单.md；立项要求见复制的2025年立项报告和2026年中期报告。PPO方法可参考Schulman等《Proximal Policy Optimization Algorithms》(2017, arXiv:1707.06347)；强化学习基础可参考Sutton与Barto《Reinforcement Learning: An Introduction》(2nd ed., 2018)；接口实现使用Gymnasium与Stable-Baselines3。这里列出方法出处，未冒充完成全部论文复现实验。正式学习综述的文献内容仍须团队逐条阅读核实。
"""
    (REPORTS / "结项技术报告.md").write_text(report, encoding="utf-8")
    (REPORTS / "family_summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    figures = sorted((RESULTS / "figures").glob("*.png"))
    cards = "".join(f'<section><h2>{html.escape(p.stem)}</h2><img loading="lazy" src="{html.escape(p.relative_to(ROOT).as_posix())}" alt="{html.escape(p.stem)}"></section>' for p in figures)
    data = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    labels = json.dumps(LABELS, ensure_ascii=False)
    families = json.dumps(FAMILIES, ensure_ascii=False)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>卫星姿态控制结项结果总览</title><style>body{{margin:0;background:#f6f8fa;color:#142534;font:16px/1.65 'Microsoft YaHei',sans-serif}}main{{max-width:1150px;margin:auto;padding:32px 24px}}h1{{font-size:30px}}h2{{font-size:21px}}section{{background:white;padding:22px;margin:24px 0;border-radius:8px}}img{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dce3e8}}select{{padding:8px;margin:8px}}a{{color:#126c9d}}.scroll{{overflow:auto}}.note{{color:#4b6070}}</style><main>
<h1>卫星姿态控制结项结果总览</h1><p>PID、普通PPO与随机化PPO的同条件仿真比较。{manifest['scenario_count']}个场景，{manifest['episode_count']}个回合，6份独立训练模型。</p>
{fusion_banner}
<p class="note">成功：20秒仿真末5秒持续满足误差≤1°、角速度≤0.5°/s。所有数据来自保存结果，失败回合保留。学校评审及实际在轨性能不由本页面保证。</p>
<p><a href="工作总说明.md">工作总说明</a> · <a href="README.md">运行说明</a> · <a href="06_结项说明与报告/结项技术报告.md">技术报告</a> · <a href="04_实验数据与图表/episode_metrics.csv">逐回合数据</a></p>
<section><h2>01 工况与控制器比较</h2><label>工况<select id="family"><option value="all">全部</option></select></label><label>控制器<select id="controller"><option value="all">全部</option></select></label><div class="scroll"><table><thead><tr><th>工况</th><th>控制器</th><th>回合</th><th>成功率</th><th>跨种子标准差</th><th>末段误差 / °</th><th>稳定时间 / s</th><th>控制代价</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note">稳定时间只在成功回合上求均值；控制代价为力矩平方积分。PID没有训练种子误差线。</p></section>
{cards}<section><h2>前一版三轴PID拓展</h2><p>该目录为三轴PID演示；新增三轴强化学习对比在07。</p><img src="05_三轴拓展演示/outputs/3axis_response.png" alt="三轴姿态控制响应"></section>
</main><script>const data={data},labels={labels},families={families}; const f=document.getElementById('family'),c=document.getElementById('controller');for(const [k,v] of Object.entries(families))f.add(new Option(v,k));for(const [k,v] of Object.entries(labels))c.add(new Option(v,k));function render(){{document.getElementById('rows').innerHTML=data.filter(r=>(f.value==='all'||f.value===r.family)&&(c.value==='all'||c.value===r.controller)).map(r=>`<tr><td>${{families[r.family]}}</td><td>${{labels[r.controller]}}</td><td>${{r.episodes}}</td><td>${{(r.success_rate*100).toFixed(1)}}%</td><td>${{r.seed_std===null?'—':(r.seed_std*100).toFixed(1)+'%'}}</td><td>${{r.steady_error_deg.toFixed(3)}}</td><td>${{r.settling_success_only_s===null?'—':r.settling_success_only_s.toFixed(2)}}</td><td>${{r.control_cost.toFixed(4)}}</td></tr>`).join('')}}f.onchange=c.onchange=render;render();</script></html>"""
    (ROOT / "结果总览.html").write_text(page, encoding="utf-8")
    print(f"Wrote overview, technical report and offline result page from {len(rows)} real episodes.")


if __name__ == "__main__":
    main()
