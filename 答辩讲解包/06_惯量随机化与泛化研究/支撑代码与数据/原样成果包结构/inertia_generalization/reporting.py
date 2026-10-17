"""Five predeclared scientific figures and an offline V3 evidence report."""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_algorithm_benchmark.evaluation import write_json
from .protocol import STUDY_ROOT,RESULTS_DIR,MODELS_DIR,ANALYSIS_DIR,TEST_RHOS,sha256
from .analysis import MAIN_CONTROLLERS

FIG_DIR=STUDY_ROOT/"04_图表与结果页面"
LABELS={"PID_N":"标称整定PID","PID_R":"鲁棒整定PID","SAC_N":"标称训练SAC","SAC_R":"惯量随机化SAC",
        "SAC_N_raw":"标称SAC原始动作","SAC_R_raw":"随机化SAC原始动作"}
AXES={"single":"单轴","three":"三轴"}
COLORS={"PID_N":"#526579","PID_R":"#111827","SAC_N":"#d37524","SAC_R":"#007e87","SAC_N_raw":"#d37524","SAC_R_raw":"#007e87"}
FILES=("01_成功率与惯量倍率","02_经验可用范围","03_精度速度与控制代价","04_非等比例惯量迁移","05_校准消融与固定代表轨迹")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    bools={"success","failed","complete","success_recovered","success_lost","passed_90_percent","nominal_passed",
           "reaches_test_lower_limit","reaches_test_upper_limit"}
    strings={"axis","controller","model_seed","region","experiment","domain","case_id","family","scenario_id","trace_file","definition","comparison","controller_a","controller_b","trace_file_a","trace_file_b"}
    for r in rows:
        for k,v in r.items():
            if v=="":r[k]=None
            elif k in bools:r[k]=v.lower()=="true"
            elif k not in strings:
                try:r[k]=float(v)
                except ValueError:pass
    return rows


def style():
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,
                         "font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.2,"savefig.facecolor":"white"})


def save(fig,name):
    fig.savefig(FIG_DIR/(name+".png"),dpi=160,bbox_inches="tight")
    fig.savefig(FIG_DIR/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def stats_by_rho(rows,axis,controller,field):
    means,sd=[],[]
    for rho in TEST_RHOS:
        group=[r for r in rows if r["axis"]==axis and r["controller"]==controller and r["experiment"]=="scalar" and r["rho"]==rho]
        values=[]
        for seed in sorted({r["model_seed"] for r in group}):
            sample=[r[field] for r in group if r["model_seed"]==seed and r[field] is not None]
            if sample:values.append(np.mean(sample))
        means.append(np.mean(values) if values else np.nan)
        sd.append(np.std(values,ddof=1) if len(values)>1 else 0.)
    return np.array(means),np.array(sd)


def figures(rows,points,boundaries):
    style();FIG_DIR.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for ax,axis in zip(axs,AXES):
        ax.axvspan(.7,1.3,color="#dcecef",alpha=.5,label="随机化训练域")
        for c in MAIN_CONTROLLERS:
            y,s=stats_by_rho(rows,axis,c,"success")
            ax.plot(TEST_RHOS,100*y,"o-",label=LABELS[c],color=COLORS[c],lw=2,ms=4)
            if c.startswith("SAC"):
                ax.fill_between(TEST_RHOS,100*np.clip(y-s,0,1),100*np.clip(y+s,0,1),color=COLORS[c],alpha=.13)
                for seed in ("11","22","33"):
                    values=[next(p["success_rate"] for p in points if p["axis"]==axis and p["controller"]==c and p["model_seed"]==seed and p["rho"]==rho) for rho in TEST_RHOS]
                    ax.plot(TEST_RHOS,100*np.array(values),color=COLORS[c],lw=.6,alpha=.35)
        ax.axhline(90,color="#6b7280",ls="--",lw=1);ax.set(xlabel="惯量倍率 ρ",title=AXES[axis],ylim=(-3,103),xticks=TEST_RHOS)
    axs[0].set_ylabel("持续保持成功率 / %")
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=5,bbox_to_anchor=(.5,-.02))
    fig.suptitle("末5秒持续保持：均值、三个种子与样本SD（阴影截至0–100%）")
    fig.tight_layout(rect=(0,.08,1,.96));save(fig,FILES[0])

    fig,axs=plt.subplots(1,2,figsize=(13,6.8),sharex=True)
    for ax,axis in zip(axs,AXES):
        entries=[r for r in boundaries if r["axis"]==axis and r["controller"] in MAIN_CONTROLLERS and
                 (r["model_seed"]=="common" or r["controller"].startswith("SAC"))]
        for y,r in enumerate(entries):
            c=r["controller"];common=r["model_seed"]=="common"
            if r["nominal_passed"]:
                ax.plot([r["lower_rho"],r["upper_rho"]],[y,y],"o-",color=COLORS[c],lw=5 if common else 2,ms=5)
            else:ax.text(.51,y,"标称未通过90%阈值",color=COLORS[c],va="center",fontsize=9)
        ax.set(yticks=range(len(entries)),yticklabels=[LABELS[r["controller"]]+(" 共同" if r["model_seed"]=="common" else " s"+r["model_seed"]) for r in entries],
               xlabel="包含标称点的连续已测网格范围",title=AXES[axis],xlim=(.48,1.52),xticks=TEST_RHOS)
        ax.axvline(1,color="gray",ls=":",lw=1);ax.invert_yaxis()
    fig.suptitle("经验可用范围：每种子至少27/30成功，共同范围要求三个种子均通过")
    fig.tight_layout();save(fig,FILES[1])

    fig,axs=plt.subplots(2,3,figsize=(15,8.2))
    for i,axis in enumerate(AXES):
        for j,(field,title) in enumerate((("rms_error_deg","全程RMS误差 / °"),("settling_time_s","成功回合稳定时间 / s"),("control_cost","控制代价 ∫‖τ‖²dt / (N·m)²·s"))):
            ax=axs[i,j];ax.axvspan(.7,1.3,color="#dcecef",alpha=.5)
            for c in MAIN_CONTROLLERS:
                y,s=stats_by_rho(rows,axis,c,field);ax.plot(TEST_RHOS,y,"o-",color=COLORS[c],label=LABELS[c],ms=3)
                if c.startswith("SAC"):ax.fill_between(TEST_RHOS,np.maximum(y-s,0),y+s,color=COLORS[c],alpha=.12)
            ax.set(title=AXES[axis]+" "+title,xlabel="惯量倍率 ρ",xticks=TEST_RHOS)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="lower center",ncol=4)
    fig.suptitle("精度与代价保留全部回合；稳定时间仅统计成功回合")
    fig.tight_layout(rect=(0,.05,1,.96));save(fig,FILES[2])

    anis=[r for r in rows if r["experiment"]=="anisotropic"]
    cases=list(dict.fromkeys(r["case_id"] for r in anis));fig,axs=plt.subplots(1,2,figsize=(13.5,4.8))
    for ax,field,title in zip(axs,("success","control_cost"),("成功率 / %","控制代价 / (N·m)²·s")):
        for k,c in enumerate(MAIN_CONTROLLERS):
            y=[];sd=[]
            for case in cases:
                sample=[r for r in anis if r["controller"]==c and r["case_id"]==case]
                vals=[np.mean([r[field] for r in sample if r["model_seed"]==seed]) for seed in sorted({r["model_seed"] for r in sample})]
                y.append(np.mean(vals)*(100 if field=="success" else 1));sd.append((np.std(vals,ddof=1) if len(vals)>1 else 0)*(100 if field=="success" else 1))
            errors=np.array([np.minimum(sd,y),np.minimum(sd,100-np.array(y))]) if field=="success" else sd
            ax.bar(np.arange(4)+(k-1.5)*.19,y,.19,label=LABELS[c],color=COLORS[c],yerr=errors,capsize=2)
        ax.set(xticks=range(4),xticklabels=["(0.6,1,1)","(1.4,1,1)","(1,0.6,1.2)","(1,1.4,0.8)"],xlabel="三轴惯量乘数",title=title)
        if field=="success":ax.set_ylim(0,105)
    fig.legend(*axs[0].get_legend_handles_labels(),loc="lower center",ncol=4)
    fig.suptitle("非等比例迁移：相同30初态，不计入标量边界；成功率SD线截至0–100%",fontsize=13)
    fig.tight_layout(rect=(0,.08,1,.96));save(fig,FILES[3])

    fig,axs=plt.subplots(2,3,figsize=(15,8.5))
    for i,axis in enumerate(AXES):
        ax=axs[i,0];groups=["SAC_N_raw","SAC_N","SAC_R_raw","SAC_R"]
        values=[];sds=[]
        for c in groups:
            sample=[r for r in rows if r["axis"]==axis and r["controller"]==c and r["domain"]=="out_of_domain"]
            vals=[np.mean([r["success"] for r in sample if r["model_seed"]==seed]) for seed in ("11","22","33")]
            values.append(100*np.mean(vals));sds.append(100*np.std(vals,ddof=1))
        errors=np.array([np.minimum(sds,values),np.minimum(sds,100-np.array(values))])
        ax.bar(range(4),values,color=[COLORS[c] for c in groups],alpha=.8,yerr=errors,capsize=3)
        ax.set(xticks=range(4),xticklabels=["N原始","N校准","R原始","R校准"],ylim=(0,105),title=AXES[axis]+" 域外校准消融",ylabel="成功率 / %")
        for j,rho in enumerate((.6,1.4),1):
            ax=axs[i,j]
            for c in MAIN_CONTROLLERS:
                row=next(r for r in rows if r["axis"]==axis and r["controller"]==c and r["experiment"]=="scalar" and r["rho"]==rho and r["sample_id"]==0 and r["model_seed"]==("0" if c.startswith("PID") else "11"))
                with np.load(RESULTS_DIR/row["trace_file"],allow_pickle=False) as trace:
                    ax.plot(trace["time"],np.maximum(trace["error_deg"],1e-7),color=COLORS[c],label=LABELS[c],lw=1.6)
            ax.axhline(1,color="gray",ls="--",lw=1);ax.axvspan(15,20,color="#dcecef",alpha=.4)
            ax.set(yscale="log",xlabel="时间 / s",ylabel="姿态误差 / °",title=f"{AXES[axis]} ρ={rho}, 初态0, seed11")
    fig.legend(*axs[0,1].get_legend_handles_labels(),loc="lower center",ncol=4)
    fig.suptitle("同检查点校准消融（SD线截至0–100%）；代表初态、种子与倍率在测试前约定",fontsize=13)
    fig.tight_layout(rect=(0,.05,1,.96));save(fig,FILES[4])


def percentage(x):return f"{100*x:.1f}%"


def interval_text(r):
    if not r["nominal_passed"]:return "无包含标称点的可用区间"
    suffix="（覆盖至已测范围）" if r["reaches_test_lower_limit"] and r["reaches_test_upper_limit"] else ""
    return f"{r['lower_rho']:.1f}–{r['upper_rho']:.1f}，{int(r['grid_points'])}个网格点"+suffix


def build_report():
    rows=read_csv(RESULTS_DIR/"episode_metrics.csv")
    points=read_csv(RESULTS_DIR/"boundary_points.csv");boundaries=read_csv(RESULTS_DIR/"empirical_boundaries.csv")
    agg=json.loads((RESULTS_DIR/"aggregate_results.json").read_text(encoding="utf-8"))
    manifest=json.loads((RESULTS_DIR/"evaluation_manifest.json").read_text(encoding="utf-8"))
    by={(r["axis"],r["controller"],r["region"]):r for r in agg}
    common={(r["axis"],r["controller"]):r for r in boundaries if r["model_seed"]=="common"}
    figures(rows,points,boundaries)
    lines=["# 惯量域随机化与泛化边界研究分析","",
           "本轮是结项前最后一轮技术研究。三轴为主，单轴用于机制对照；采用从零训练的SAC，比较标称训练与仅惯量随机化，并与同预算充分整定的两类PID对照。下列结论来自预先冻结的测试，保留全部训练种子、原始动作消融和失败回合。","",
           "## 实测核心结论",""]
    findings=[]
    for axis in AXES:
        n=by[axis,"SAC_N","out_of_domain"];r=by[axis,"SAC_R","out_of_domain"]
        delta=100*(r["success_rate"]-n["success_rate"])
        bn,br=common[axis,"SAC_N"],common[axis,"SAC_R"]
        sentence=f"{AXES[axis]}域外成功率由{percentage(n['success_rate'])}变为{percentage(r['success_rate'])}，随机化相对标称训练变化{delta:+.1f}个百分点。三个种子共同可用网格由{int(bn['grid_points'])}个变为{int(br['grid_points'])}个。"
        findings.append({"axis":axis,"out_of_domain_success_change_pp":delta,"common_grid_change":int(br["grid_points"]-bn["grid_points"]),"text":sentence})
        lines.extend([sentence,""])
    lines.extend(["域外仅指ρ=0.5、0.6、1.4、1.5；训练域端点0.7/1.3未算作域外。这里的变化是本次三个训练种子、每轴30组复用初态下的描述性结果，未作统计显著性或真实卫星可用性保证。","",
        "本次未观察到仅惯量随机化扩大免重整定范围。两轴充分整定PID和标称训练SAC均覆盖至已测0.5–1.5倍网格，因而只获得该有限范围内可用的证据，没有找到它们的真实失效边界。三轴随机化SAC的共同范围缩至0.5–1.1，说明平均表现不能掩盖较弱种子。固定PID在惯量变化后并未崩溃，不能用预设叙事替代实测。","",
        "## 统一方法与冻结条件","",
        "单轴J₀=1 kg·m²，dt=0.05 s，力矩±0.5 N·m；三轴I₀=diag(1.2,1.0,0.8) kg·m²，dt=0.02 s，每轴力矩±2 N·m。两组SAC保持相同64×64网络、奖励、优化参数、初态和每份100万步预算，只改变每回合惯量为ρ=1或ρ∼U[0.7,1.3]。关闭外扰、脉冲与噪声，回合内惯量恒定。","",
        "每轴共同20场景域内验证集每10万步选模，以校准部署分数100×成功率−RMS−0.2×控制代价排名，平分按RMS、代价和较早检查点。两种训练模式均用共同域内验证，标称组虽不在随机惯量下训练，也利用域内验证选模。原始动作消融使用同一检查点。","",
        "固定部署校准为clip(π(o)−π(0),−1,1)。这是输出偏置后处理，不是新训练算法。权重、PID参数和偏置跨惯量冻结，模型不读真实惯量，也不含记忆或在线辨识，因此结论属于固定反馈策略的泛化和参数鲁棒性。","",
        "成功要求完整20 s的末5 s每个采样点同时满足真实姿态误差≤1°、角速度≤0.5°/s，三轴采用测地角和角速度范数。经验可用区间采用每点≥27/30成功、包含ρ=1的连续已测网格；SAC共同区间要求三个种子各自通过，未作网格间插值。","",
        "## 核心结果与两端差异","",
        "| 轴 | 控制器 | 标称成功率 | 域外成功率 | 域外末段均误差 / ° | 域外控制代价 | 共同经验区间 |",
        "|---|---|---:|---:|---:|---:|---|"])
    for axis in AXES:
        for c in MAIN_CONTROLLERS:
            a=by[axis,c,"nominal"];b=by[axis,c,"out_of_domain"]
            lines.append(f"| {AXES[axis]} | {LABELS[c]} | {percentage(a['success_rate'])} | {percentage(b['success_rate'])} | {b['tail_mean_error_deg']:.4f} | {b['control_cost']:.4f} | {interval_text(common[axis,c])} |")
    lines.extend(["","| 轴 | 控制器 | 低惯量端0.5/0.6成功率 | 高惯量端1.4/1.5成功率 | 最差种子域外成功率 | 域外种子SD |","|---|---|---:|---:|---:|---:|"])
    for axis in AXES:
        for c in MAIN_CONTROLLERS:
            vals=[np.mean([r["success"] for r in rows if r["axis"]==axis and r["controller"]==c and r["rho"] in levels]) for levels in ((.5,.6),(1.4,1.5))]
            a=by[axis,c,"out_of_domain"]
            lines.append(f"| {AXES[axis]} | {LABELS[c]} | {percentage(vals[0])} | {percentage(vals[1])} | {percentage(a['worst_seed_success_rate'])} | {percentage(a['success_seed_std']) if a['success_seed_std'] is not None else '不适用'} |")
    lines.extend(["","SD为三个训练种子成功率的样本标准差，不是置信区间。PID是单个固定控制器，未报告虚构的训练种子波动。","",
        "aggregate_results.json中的nominal单列ρ=1，in_domain标签对应域内其余六个网格点；二者合并才是完整域内汇总。下表及核心结论的域外统计始终只含四个严格保留点。","",
        "## 控制代价收益与保持失败",""])
    for axis in AXES:
        pid=by[axis,"PID_N","out_of_domain"];sac=by[axis,"SAC_N","out_of_domain"]
        reduction=100*(1-sac["control_cost"]/pid["control_cost"])
        lines.extend([f"{AXES[axis]}域外测试中，校准后的标称训练SAC与PID-N均为100%成功；SAC平均控制代价相对PID-N降低{reduction:.1f}%，成功回合平均稳定时间由{pid['settling_time_s_successful']:.2f} s变为{sac['settling_time_s_successful']:.2f} s，全程RMS由{pid['rms_error_deg']:.2f}°变为{sac['rms_error_deg']:.2f}°。较低力矩平方积分伴随较慢响应及较大全程误差，是条件性取舍，不等于省电实测或全指标改善。",""])
    lines.extend(["以下对主控制器的全部标量测试失败按末段阈值直接分类：角度超限、角速度超限、两者超限或物理发散。它们属于验收失败，不都表示动力学发散。","",
        "| 轴 | 控制器 | 种子 | 角度超限 | 角速度超限 | 两者超限 | 物理发散 |","|---|---|---:|---:|---:|---:|---:|"])
    from collections import defaultdict
    counts=defaultdict(lambda:[0,0,0,0])
    for row in rows:
        if row["controller"] not in MAIN_CONTROLLERS or row["experiment"]!="scalar" or row["success"]:continue
        angle=row["tail_max_error_deg"]>1;rate=row["tail_max_rate_deg_s"]>.5
        reason=3 if row["failed"] else 2 if angle and rate else 0 if angle else 1
        counts[row["axis"],row["controller"],row["model_seed"]][reason]+=1
    for (axis,c,seed),values in counts.items():lines.append(f"| {AXES[axis]} | {LABELS[c]} | {seed} | "+" | ".join(str(v) for v in values)+" |")
    lines.extend(["","三轴随机化组失败集中在种子33的角速度保持超限；该种子域内验证最高成功率为85%，低于其他种子的100%，但仍按预定规则保留。现有证据支持训练波动与高惯量端保持不足的描述，未通过额外因果实验确定具体优化机制。固定代表曲线使用预先约定的种子11，不能代表种子33或替代全体失败统计。","",
        "## 充分整定PID与训练交付","",
        "两类PID每轴各144个候选，合计576配置、11,520个域内验证回合。采用角速度阻尼、条件抗积分饱和和角速度滤波结构；标称/鲁棒仅整定惯量分布不同。Ki=0时是PID族的PD特例，alpha=1时不平滑观测。","",
        "| 轴 | 整定模式 | Kp | Ki | Kd | alpha |","|---|---|---:|---:|---:|---:|"])
    pids=json.loads((MODELS_DIR/"pid_configs.json").read_text(encoding="utf-8"))
    for axis in AXES:
        for mode in ("nominal","randomized"):
            g=pids[axis][mode];g=g.get("gains",g)
            lines.append(f"| {AXES[axis]} | {mode} | {g['kp']} | {g['ki']} | {g['kd']} | {g['alpha']} |")
    lines.extend(["","| 轴 | SAC训练模式 | 种子 | 实际交互步数 | 选中步数 | 校准验证成功率 |","|---|---|---:|---:|---:|---:|"])
    for axis in AXES:
        for mode in ("nominal","randomized"):
            for seed in (11,22,33):
                folder=MODELS_DIR/axis/mode/f"seed_{seed}"
                meta=json.loads((folder/"training_metadata.json").read_text(encoding="utf-8"));best=json.loads((folder/"best_checkpoint.json").read_text(encoding="utf-8"))
                lines.append(f"| {AXES[axis]} | {mode} | {seed} | {meta['actual_steps']:,} | {meta['best_checkpoint_steps']:,} | {percentage(best['calibrated']['success_rate'])} |")
    lines.extend(["","## 校准消融与非等比例迁移","",
        "| 轴 | 训练模式 | 原始域外成功率 | 校准域外成功率 |","|---|---|---:|---:|"])
    for axis in AXES:
        for c in ("SAC_N","SAC_R"):
            lines.append(f"| {AXES[axis]} | {LABELS[c]} | {percentage(by[axis,c+'_raw','out_of_domain']['success_rate'])} | {percentage(by[axis,c,'out_of_domain']['success_rate'])} |")
    lines.extend(["","校准可改善目标点的输出偏置，也可能损失部分回合或改变饱和区动作。全部配对损益在calibration_pairs.csv中，不能只报告恢复的成功回合。原始动作并非按其自身分数选出的最佳检查点，因此仅用于该部署规则的同检查点消融。","",
        "三轴四种非等比例乘数为(0.6,1,1)、(1.4,1,1)、(1,0.6,1.2)、(1,1.4,0.8)，对应主惯量满足正值与三角不等式。它们改变惯量比例和陀螺耦合，只是预先约定的迁移压力测试，不纳入标量泛化区间。","",
        "| 三轴控制器 | 非等比例成功率 | 最差种子成功率 | 全程RMS / ° | 控制代价 |","|---|---:|---:|---:|---:|"])
    for c in MAIN_CONTROLLERS:
        a=by["three",c,"anisotropic"]
        lines.append(f"| {LABELS[c]} | {percentage(a['success_rate'])} | {percentage(a['worst_seed_success_rate'])} | {a['rms_error_deg']:.3f} | {a['control_cost']:.4f} |")
    lines.extend(["","## 五张核心图",""])
    for f in FILES:lines.extend([f"### {f[3:]}","",f"![{f[3:]}](../04_图表与结果页面/{f}.png)",""])
    lines.extend(["## 证据清单与解释限制","",
        f"正式测试含{manifest['scene_counts']['single']}单轴场景与{manifest['scene_counts']['three']}三轴场景，每场景14个控制实例，总计{manifest['episode_count']:,}完整回合。正式模型12份、总交互12,000,000步；24个原始/校准动作实例不等于24份模型。配对表分别为{manifest['training_mode_pair_count']:,}组训练分布比较、{manifest['calibration_pair_count']:,}组校准比较和{manifest['pid_pair_count']:,}组PID比较。","",
        "[逐回合数据](../03_测试数据与配对结果/episode_metrics.csv)、[边界点](../03_测试数据与配对结果/boundary_points.csv)、[全部区间](../03_测试数据与配对结果/empirical_boundaries.csv)、[部署冻结](../03_测试数据与配对结果/deployment_freeze.json)、[实验清单](../03_测试数据与配对结果/evaluation_manifest.json)与全部NPZ互相对应。独立核验的实际状态请查看同目录核验报告，本文生成不代表核验自动通过。","",
        "控制代价为力矩平方积分，不是实测能耗；失败回合保留在误差和代价统计中，须结合失败率与有效时长。稳定时间仅对成功回合统计，成功样本变化会影响均值。跨倍率复用初态支持配对，不增加独立初态数量。","",
        "惯量变化在每个回合内恒定，不模拟燃料流出、帆板展开动力学或任意完整惯量张量。仿真未包含姿态估计、飞轮动量、柔性结构及实物验证。研究允许域随机化无收益或强PID更好，不以最终测试续训、改奖励或追踪更极端的边界。","",
        "本轮完成后冻结技术范围。后续只开展总结、两篇学习综述、结项报告、答辩和真实贡献核对；在线自适应、辨识、预测控制、STK与硬件留作未完成承诺或未来研究，据实解释。",""])
    ANALYSIS_DIR.mkdir(parents=True,exist_ok=True)
    report=ANALYSIS_DIR/"研究分析.md";report.write_text("\n".join(lines),encoding="utf-8")
    write_json(ANALYSIS_DIR/"核心结论.json",{"findings":findings,"aggregate":agg,"common_boundaries":[r for r in boundaries if r["model_seed"]=="common"],
        "representative_traces":{"sample_id":0,"seed":11,"rhos":[.6,1.4]},"report_sha256":sha256(report)})
    import markdown
    # Report links are relative to 05. Preserve that base when rendering in 04.
    body=markdown.markdown(report.read_text(encoding="utf-8"),extensions=["tables","fenced_code"])
    body=body.replace('href="../03_','href="../03_')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>惯量泛化研究结果</title><style>body{max-width:1120px;margin:40px auto;padding:0 24px;color:#172533;font:17px/1.8 "Microsoft YaHei",sans-serif}h1,h2{line-height:1.4}h2{margin-top:42px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}th,td{padding:9px;border:1px solid #ccd5dd}th{background:#eef4f7}a{color:#006d78}nav{margin:20px 0}@media print{body{margin:0;font-size:11pt}h2{break-after:avoid}img{max-height:165mm;object-fit:contain}}</style><nav><a href="../../README.md">成果包入口</a>　<a href="../05_研究分析与核验/研究分析.md">完整分析源文</a></nav>'+body+'</html>'
    (FIG_DIR/"结果总览.html").write_text(page,encoding="utf-8")
    return {"figures":len(FILES),"episodes":len(rows),"report":str(report)}


if __name__=="__main__":print(json.dumps(build_report(),ensure_ascii=False))
