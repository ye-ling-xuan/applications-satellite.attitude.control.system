"""从已冻结的正式结果重绘结题报告前期结果图；不训练、不改写原始资料。"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "结项成果包").is_dir())
OUT = ROOT / "结题报告配图/05_完整图表版"
RIGID = ROOT / "结项成果包/07_单轴三轴多算法对比"
INERTIA = ROOT / "结项成果包/08_惯量域随机化与泛化边界研究"
ACT = ROOT / "结项成果包/12_执行器滞后与纯延迟对比研究"
RIGID_CSV = RIGID / "results/episode_metrics.csv"
RIGID_SUM = RIGID / "result_summary.json"
INERTIA_CSV = INERTIA / "03_测试数据与配对结果/episode_metrics.csv"
INERTIA_SUM = INERTIA / "05_研究分析与核验/核心结论.json"
ACT_CSV = ACT / "04_独立测试/episode_metrics.csv"
ACT_SUM = ACT / "06_结果与说明/分组汇总.csv"
SOURCES = [
    RIGID_CSV, RIGID_SUM, RIGID / "实验约定.md",
    INERTIA_CSV, INERTIA_SUM, INERTIA / "01_实验协议与配置/protocol.json",
    ACT_CSV, ACT_SUM, ACT / "01_实验方案/执行配置.json",
    ACT / "06_结果与说明/研究结果说明.md",
]
COLORS = ["#506a89", "#248c89", "#e5aa42", "#9c76b7"]


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def success(row):
    return row["success"].lower() == "true"


def summarize(rows, cost="control_cost"):
    assert rows
    passes = sum(success(r) for r in rows)
    return {
        "count": len(rows), "passes": passes,
        "success_rate": passes / len(rows),
        "mean_cost": float(np.mean([float(r[cost]) for r in rows])),
        "mean_rms_error_deg": float(np.mean([float(r["rms_error_deg"]) for r in rows])),
        "mean_settling_time_success_only_s": float(np.mean([
            float(r["settling_time_s"]) for r in rows if success(r)
        ])) if passes else None,
    }


def style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 12.5,
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def clean(ax, ylabel="成功率 / %", ymax=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce2e8", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def label_bars(ax, bars, labels, dy=1.5, fs=11):
    for b, label in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                label, ha="center", va="bottom", fontsize=fs)


def export(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=240)
    plt.close(fig)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    style()
    rigid_rows = load_csv(RIGID_CSV)
    inertia_rows = load_csv(INERTIA_CSV)
    actuator_rows = load_csv(ACT_CSV)
    rigid_ref = json.loads(RIGID_SUM.read_text(encoding="utf-8"))
    inertia_ref = json.loads(INERTIA_SUM.read_text(encoding="utf-8"))
    act_ref = load_csv(ACT_SUM)
    numeric = {"figure06": [], "figure07_inertia": [], "figure07_actuator": [], "comparisons": []}

    rigid = {}
    for axis in ("single", "three"):
        for controller in ("baseline_PID", "tuned_PID", "improved_PID", "PPO", "SAC", "TD3",
                           "PPO_calibrated", "SAC_calibrated", "TD3_calibrated"):
            selected = [r for r in rigid_rows if r["axis"] == axis and r["controller"] == controller]
            d = summarize(selected)
            ref = next(r for r in rigid_ref["fresh_results"] + rigid_ref["calibrated_results"]
                       if r["axis"] == axis and r["controller"] == controller)
            assert d["count"] == ref["n_episodes"]
            assert abs(d["success_rate"] - ref["success_rate"]) < 1e-12
            assert abs(d["mean_cost"] - ref["control_cost"]) < 1e-12
            seed_rates = {str(seed): summarize([r for r in selected if r["model_seed"] == str(seed)])["success_rate"]
                          for seed in (11, 22, 33)} if "PID" not in controller else {}
            d.update(axis=axis, controller=controller, per_seed_success_rates=seed_rates)
            numeric["figure06"].append(d)
            rigid[axis, controller] = d

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.3))
    fig.suptitle("刚体姿态控制：先检查PID整定，再比较同模型部署修正", fontsize=20, fontweight="bold", y=.975)
    fig.subplots_adjust(left=.07, right=.975, bottom=.115, top=.88, wspace=.23, hspace=.46)
    for j, (axis, title) in enumerate((("single", "单轴"), ("three", "三轴耦合"))):
        ax = axes[0, j]
        ds = [rigid[axis, c] for c in ("baseline_PID", "tuned_PID", "improved_PID")]
        bars = ax.bar(np.arange(3), [d["success_rate"] * 100 for d in ds], width=.57,
                      color=COLORS[:3], zorder=3)
        label_bars(ax, bars, [f'{d["success_rate"]*100:.1f}%\n{d["passes"]}/{d["count"]}' for d in ds], fs=11)
        ax.set_xticks(np.arange(3), ["原始PID", "验证集整定PID", "扩展滤波候选PID"])
        ax.set_title(f"（{chr(65+j)}）{title}：PID整定对照", pad=14)
        clean(ax, ymax=105)
        if axis == "single":
            ax.text(.5, -.20, "单轴入选滤波系数α=1，实际未启用滤波。", transform=ax.transAxes,
                    ha="center", fontsize=10.5, color="#4b5a66")
        else:
            ax.text(.5, -.20, "滤波候选较整定PID多通过3/180场景。", transform=ax.transAxes,
                    ha="center", fontsize=10.5, color="#4b5a66")
        ax = axes[1, j]
        xs = np.arange(3)
        width = .30
        for shift, suffix, color, label in ((-.17, "", "#9aa8ba", "原始部署"),
                                            (.17, "_calibrated", "#248c89", "零观测动作偏置修正")):
            ds = [rigid[axis, c + suffix] for c in ("PPO", "SAC", "TD3")]
            bars = ax.bar(xs + shift, [d["success_rate"] * 100 for d in ds], width=width,
                          color=color, zorder=3, label=label)
            label_bars(ax, bars, [f'{d["success_rate"]*100:.1f}%' for d in ds], dy=2, fs=11)
            for x, d in zip(xs + shift, ds):
                vals = list(d["per_seed_success_rates"].values())
                ax.scatter(x + np.linspace(-.065, .065, 3), np.array(vals)*100, marker="o", s=22,
                           color="#263b4e", edgecolor="white", linewidth=.5, zorder=5)
        best = rigid[axis, "improved_PID"]["success_rate"] * 100
        ax.axhline(best, color="#c58920", linewidth=1.2, linestyle="--", label=f"滤波候选PID {best:.1f}%")
        ax.set_xticks(xs, ["PPO", "SAC", "TD3"])
        ax.set_title(f"（{chr(67+j)}）{title}：同一网络权重，修正前后", pad=14)
        clean(ax, ymax=113)
        ax.legend(loc="upper left", ncol=1, fontsize=9.5, frameon=False)
    fig.text(.07, .055, "共同协议：每回合20秒，末5秒持续满足角误差≤1°、角速度≤0.5°/s。PID各180场景；RL每算法3种子×180场景。", fontsize=11.5)
    fig.text(.07, .025, "柱为合并成功率，深色点为各训练种子的成功率；场景跨种子复用。0%表示未通过持续停稳验收，不等同于物理发散。", fontsize=11.5, color="#435564")
    export(fig, "图06_刚体多算法与部署修正")

    inertia = {}
    for axis in ("single", "three"):
        for controller in ("PID_N", "SAC_N", "SAC_R"):
            selected = [r for r in inertia_rows if r["axis"] == axis and r["controller"] == controller
                        and r["experiment"] == "scalar" and r["domain"] == "out_of_domain"]
            d = summarize(selected)
            ref = next(r for r in inertia_ref["aggregate"] if r["axis"] == axis and
                       r["controller"] == controller and r["region"] == "out_of_domain")
            assert d["count"] == ref["n_episodes"]
            assert abs(d["success_rate"] - ref["success_rate"]) < 1e-12
            assert abs(d["mean_cost"] - ref["control_cost"]) < 1e-12
            d.update(axis=axis, controller=controller)
            d["per_seed_success_rates"] = {str(seed): summarize([r for r in selected if r["model_seed"] == str(seed)])["success_rate"]
                                             for seed in (11,22,33)} if controller != "PID_N" else {}
            numeric["figure07_inertia"].append(d)
            inertia[axis, controller] = d

    actuator = {}
    for kind in ("lag", "delay"):
        for controller in (f"{kind}_pid",) + tuple(f"{kind}_residual_formal_seed{s}" for s in (101,202,303)):
            selected = [r for r in actuator_rows if r["kind"] == kind and r["controller_id"] == controller
                        and r["region"] == "in_domain"]
            d = summarize(selected, cost="actual_cost")
            ref = next(r for r in act_ref if r["controller_id"] == controller and r["region"] == "in_domain")
            assert d["count"] == int(ref["count"]) == 1536
            assert d["passes"] == int(ref["passes"])
            assert abs(d["mean_cost"] - float(ref["mean_actual_cost"])) < 1e-12
            d.update(kind=kind, controller=controller)
            numeric["figure07_actuator"].append(d)
            actuator[kind, controller] = d

    fig, axes = plt.subplots(2, 2, figsize=(16.2, 11.0))
    fig.suptitle("惯量与执行器探索：本次配置下未获得稳定的学习优势", fontsize=20, fontweight="bold", y=.978)
    fig.subplots_adjust(left=.07, right=.975, bottom=.125, top=.88, wspace=.25, hspace=.44)
    ax = axes[0,0]
    xs = np.arange(2)
    for j,(controller,label,color) in enumerate((("PID_N","标称整定PID",COLORS[0]),
                                              ("SAC_N","标称训练SAC（校准）",COLORS[1]),
                                              ("SAC_R","随机化训练SAC（校准）",COLORS[2]))):
        ds=[inertia[a,controller] for a in ("single","three")]
        pos=xs+(j-1)*.25
        bars=ax.bar(pos,[d["success_rate"]*100 for d in ds], width=.23, color=color,label=label,zorder=3)
        label_bars(ax,bars,[f'{d["success_rate"]*100:.1f}%' for d in ds],dy=2,fs=10.5)
        for x,d in zip(pos,ds):
            vals=list(d["per_seed_success_rates"].values())
            if vals:ax.scatter(x+np.linspace(-.05,.05,3),np.array(vals)*100,s=24,c="#23394b",edgecolor="white",linewidth=.5,zorder=5)
    ax.set_xticks(xs,["单轴","三轴耦合"])
    clean(ax,ymax=116)
    ax.set_title("（A）惯量保留域外：持续停稳成功率",pad=14)
    ax.legend(frameon=False,fontsize=9.5,loc="lower left")
    ax.text(.5,-.21,"域外ρ=0.5、0.6、1.4、1.5；PID各120场景，SAC各3×120回合。",transform=ax.transAxes,ha="center",fontsize=10.2,color="#435564")
    ax=axes[0,1]
    ratio_specs=(("mean_cost","力矩平方积分",COLORS[1]),("mean_settling_time_success_only_s","停稳时间",COLORS[2]),("mean_rms_error_deg","全程RMS误差",COLORS[3]))
    for j,(key,label,color) in enumerate(ratio_specs):
        vals=[100*inertia[a,"SAC_N"][key]/inertia[a,"PID_N"][key] for a in ("single","three")]
        for axis,value in zip(("single","three"),vals):numeric["comparisons"].append({"axis":axis,"metric":key,"nominal_SAC_over_PID_percent":value})
        bars=ax.bar(xs+(j-1)*.25,vals,width=.23,color=color,label=label,zorder=3)
        label_bars(ax,bars,[f"{v:.1f}%" for v in vals],dy=3.0,fs=10.5)
    ax.axhline(100,ls="--",lw=1.2,color="#536575")
    ax.set_xticks(xs,["单轴","三轴耦合"])
    clean(ax,"相对PID的指标 / %（PID=100%）",ymax=235)
    ax.set_title("（B）标称SAC：代价下降伴随响应取舍",pad=14)
    ax.legend(frameon=False,fontsize=10,loc="upper left")
    ax.text(.5,-.21,"仅比较两者均全部成功的标称SAC；三项指标均为越低越好。",transform=ax.transAxes,ha="center",fontsize=10.2,color="#435564")

    labels=["整定PID","PID＋SAC\n种子101","PID＋SAC\n种子202","PID＋SAC\n种子303"]
    width=.18
    for metric,ax,title,ylabel in (("success_rate",axes[1,0],"（C）原12号执行器实验：域内成功率","成功率 / %"),
                                 ("mean_cost",axes[1,1],"（D）同批完整时域实际力矩代价","平均力矩平方积分 / (N·m)²·s")):
        for j,label in enumerate(labels):
            cs=[f"{kind}_pid" if j==0 else f"{kind}_residual_formal_seed{(101,202,303)[j-1]}" for kind in ("lag","delay")]
            ds=[actuator[kind,c] for kind,c in zip(("lag","delay"),cs)]
            vals=[d[metric]*(100 if metric=="success_rate" else 1) for d in ds]
            bars=ax.bar(xs+(j-1.5)*.21,vals,width=width,color=COLORS[j],label=label.replace("\n"," "),zorder=3)
            label_bars(ax,bars,[(f"{v:.1f}%" if metric=="success_rate" else f"{v:.4f}") for v in vals],
                       dy=(2 if metric=="success_rate" else .005),fs=9.5)
        ax.set_xticks(xs,["一阶滞后","纯延迟"])
        clean(ax,ylabel,ymax=126 if metric=="success_rate" else .35)
        ax.set_title(title,pad=14)
        ax.legend(frameon=False,fontsize=9.2,ncol=2,loc="upper left")
    fig.text(.07,.064,"惯量：20秒、末5秒验收；随机化训练ρ∈[0.7,1.3]，域外测试冻结策略。深色点为三个训练种子，不是额外独立初态。",fontsize=11)
    fig.text(.07,.038,"执行器：仅原12号正式版本，参数0.05～0.30秒，8～22秒持续验收；各控制器1536回合=256初态×6参数。",fontsize=11)
    fig.text(.07,.012,"控制代价不等于电能；不同研究协议的代价不横向排名。15号修正版为单列开发诊断，未与原12号正式数值合并。",fontsize=11,color="#435564")
    export(fig,"图07_惯量与执行器探索结果")

    (HERE/"图表数值.json").write_text(json.dumps(numeric,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    provenance={"purpose":"正式记录重绘，不训练、不调参、不修改原件", "date":"2026-10-19",
                "assertions":"各图数值由逐回合CSV重算，成功数、回合数和均值与原正式摘要逐项核对，容差1e-12。",
                "sources":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,
                            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES],
                "outputs":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,
                            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in sorted(OUT.glob("图0[67]_*.?*"))]}
    (HERE/"来源与核对记录.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("完成图06与图07（PNG/SVG），逐回合重算与正式摘要全部一致。")


if __name__=="__main__":
    main()
