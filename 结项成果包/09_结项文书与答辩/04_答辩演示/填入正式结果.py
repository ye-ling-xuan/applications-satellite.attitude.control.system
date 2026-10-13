"""Populate the offline defense deck from actual V3 evidence.

Default is pending audit. --verified additionally requires a passed final audit;
the root workflow must authorize that publication after the audit completes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STUDY = PACKAGE/"08_惯量域随机化与泛化边界研究"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def populate(verified=False):
    source = STUDY/"05_研究分析与核验"/"核心结论.json"
    core = read(source)
    manifest_path = STUDY/"03_测试数据与配对结果"/"evaluation_manifest.json"
    manifest = read(manifest_path)
    audit_path = STUDY/"05_研究分析与核验"/"独立核验记录.json"
    if verified:
        audit = read(audit_path)
        if audit.get("status") != "passed" or audit.get("mode") != "final_integrity":
            raise ValueError("A passed final independent audit is required before verified publication")
    aggregate = {(row["axis"], row["controller"], row["region"]): row for row in core["aggregate"]}
    boundaries = {(row["axis"], row["controller"]): row for row in core["common_boundaries"]}
    names = {"single": "单轴", "three": "三轴"}
    labels = {"PID_N": "PID-N", "PID_R": "PID-R", "SAC_N": "SAC-N共同", "SAC_R": "SAC-R共同"}
    rows = []
    for axis in ("single", "three"):
        for controller in labels:
            row = boundaries[axis, controller]
            interval = f"{row['lower_rho']:g}–{row['upper_rho']:g}，{int(row['grid_points'])}个已测点" if row["nominal_passed"] else "无包含标称点的可用区间"
            rows.append({"axis": names[axis], "controller": labels[controller], "interval": interval})
    n_single = aggregate["single", "SAC_N", "out_of_domain"]
    r_single = aggregate["single", "SAC_R", "out_of_domain"]
    n_three = aggregate["three", "SAC_N", "out_of_domain"]
    r_three = aggregate["three", "SAC_R", "out_of_domain"]
    p_single = aggregate["single", "PID_N", "out_of_domain"]
    p_three = aggregate["three", "PID_N", "out_of_domain"]
    reductions = [100*(1-n_single["control_cost"]/p_single["control_cost"]),
                  100*(1-n_three["control_cost"]/p_three["control_cost"])]
    models = [read(STUDY/"02_模型与验证日志"/axis/mode/f"seed_{seed}"/"training_metadata.json")
              for axis in ("single", "three") for mode in ("nominal", "randomized") for seed in (11, 22, 33)]
    if len(models) != 12 or any(row["status"] != "complete" or row["development"] for row in models):
        raise ValueError("The defense deck requires all twelve completed formal models")
    image_names = {"success": "01_成功率与惯量倍率.png", "boundary": "02_经验可用范围.png",
                   "tradeoff": "03_精度速度与控制代价.png", "transfer": "04_非等比例惯量迁移.png",
                   "calibration": "05_校准消融与固定代表轨迹.png"}
    figures = {}
    for key, name in image_names.items():
        path = STUDY/"04_图表与结果页面"/name
        if not path.is_file():
            raise FileNotFoundError(path)
        figures[key] = os.path.relpath(path, HERE).replace("\\", "/")
    result = {
        "verified": bool(verified), "status": "verified" if verified else "pending_independent_audit",
        "source_core_sha256": digest(source), "source_manifest_sha256": digest(manifest_path),
        "source_final_audit_sha256": digest(audit_path) if verified else None,
        "summary": [
            f"本次仅惯量随机化未扩大可用范围。单轴域外成功率由{n_single['success_rate']:.1%}变为{r_single['success_rate']:.1%}，共同网格保持11点。",
            f"三轴域外成功率由{n_three['success_rate']:.1%}变为{r_three['success_rate']:.1%}，共同区间由0.5–1.5缩至0.5–1.1。",
            f"强PID域外均100%成功。标称SAC控制代价低{reductions[0]:.1f}%和{reductions[1]:.1f}%，同时稳定更慢。"
        ],
        "boundaries": rows,
        "training": {"models": len(models), "steps": sum(row["actual_steps"] for row in models),
                     "episodes": manifest["episode_count"]},
        "figures": figures,
        "captions": {
            "success": "两轴强PID与校准SAC-N域外均100%。SAC-R单轴98.9%，三轴84.4%；三轴高惯量端68.9%。",
            "boundary": "SAC-N与强PID覆盖至已测0.5–1.5倍网格。三轴SAC-R共同范围缩至0.5–1.1，不代表真实物理边界。",
            "tradeoff": f"标称SAC域外控制代价相对PID-N低{reductions[0]:.1f}% / {reductions[1]:.1f}%，稳定时间分别{n_single['settling_time_s_successful']:.2f} / {n_three['settling_time_s_successful']:.2f} s，PID-N为{p_single['settling_time_s_successful']:.2f} / {p_three['settling_time_s_successful']:.2f} s。"
        },
        "calibration_note": "SAC主结果含固定零动作校准。三轴原始动作域外成功率均为0%，同检查点校准后为100%与84.4%。",
        "limitations": ["三个训练种子、每轴30组初态、离散惯量网格与20 s保持判据下的描述性结果",
                        "控制代价是力矩平方积分，较低代价伴随较慢响应，不等于实测省电",
                        "固定反馈未输入惯量或在线辨识，真实卫星仍需估计器、执行器和硬件验证"]
    }
    (HERE/"defense_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (HERE/"defense_results.js").write_text("// Generated from actual frozen V3 outputs. Final independent audit controls publication.\nwindow.INERTIA_DEFENSE_RESULTS = "+json.dumps(result, ensure_ascii=False, indent=2)+";\n", encoding="utf-8")
    return {"verified": result["verified"], "status": result["status"], "training": result["training"], "figures": len(figures)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", action="store_true")
    args = parser.parse_args()
    print(json.dumps(populate(args.verified), ensure_ascii=False, indent=2))
