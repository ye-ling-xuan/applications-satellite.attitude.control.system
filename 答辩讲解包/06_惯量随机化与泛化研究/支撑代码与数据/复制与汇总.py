"""Curate unchanged source copies and derive a compact table from saved CSV only."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from statistics import mean

SUPPORT = Path(__file__).resolve().parent
STEP = SUPPORT.parent
PROJECT = SUPPORT.parents[2]
SOURCE = PROJECT / "结项成果包"
STUDY_NAME = "08_惯量域随机化与泛化边界研究"
STUDY = SOURCE / STUDY_NAME
TARGET = SUPPORT / "原样成果包结构"
records = []

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = digest(source)
    assert source_hash == digest(target), str(target)
    records.append({"source": source.relative_to(PROJECT).as_posix(),
                    "copy": target.relative_to(STEP).as_posix(), "sha256": source_hash})

def same_layout(source):
    copy(source, TARGET / source.relative_to(SOURCE))

def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    for package in ("inertia_generalization", "multi_algorithm_benchmark"):
        for source in sorted((SOURCE / package).glob("*.py")):
            same_layout(source)
    for name in ("__init__.py", "config.py", "env.py", "controllers.py", "metrics.py"):
        same_layout(SOURCE / "satellite_benchmark" / name)
    for name in ("run_inertia.py", "requirements.txt", "requirements-tested.txt"):
        same_layout(SOURCE / name)
    same_layout(SOURCE / "tools/verify_inertia_results.py")
    for folder in ("01_实验协议与配置", "03_测试数据与配对结果", "04_图表与结果页面", "05_研究分析与核验"):
        for source in sorted((STUDY / folder).iterdir()):
            if source.is_file() and source.name != "研究执行与冻结.md":
                same_layout(source)
    for axis in ("single", "three"):
        same_layout(STUDY / f"03_测试数据与配对结果/{axis}/controllers.json")
    models = STUDY / "02_模型与验证日志"
    for source in sorted(models.iterdir()):
        if source.is_file():
            same_layout(source)
    keep = {"best_model.zip", "best_checkpoint.json", "session_status.json", "training_episode_initial_states.csv",
            "training_manifest.json", "training_metadata.json", "validation_history_calibrated.csv",
            "validation_history_raw.csv", "validation_scenarios.json"}
    for source in sorted(models.rglob("*")):
        if source.is_file() and source.parent != models and source.name in keep:
            same_layout(source)
    figures = ["01_成功率与惯量倍率.png", "02_经验可用范围.png", "03_精度速度与控制代价.png", "05_校准消融与固定代表轨迹.png"]
    for name in figures:
        copy(STUDY / "04_图表与结果页面" / name, STEP / name)

    with (STUDY / "03_测试数据与配对结果/episode_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r["axis"] == "three" and r["experiment"] == "scalar"
                and float(r["rho"]) == 1.4 and r["sample_id"] == "0"
                and ((r["controller"] in ("PID_N", "PID_R") and r["model_seed"] == "0")
                     or (r["controller"] in ("SAC_N", "SAC_R") and r["model_seed"] == "11"))]
    failing = sorted([r for r in rows if r["axis"] == "three" and r["experiment"] == "scalar"
                      and r["controller"] == "SAC_R" and r["model_seed"] == "33"
                      and r["success"].lower() == "false" and float(r["tail_max_rate_deg_s"]) > 0.5],
                     key=lambda r: r["scenario_id"])[0]
    samples = []
    for row in selected + [failing]:
        source = STUDY / "03_测试数据与配对结果" / row["trace_file"]
        same_layout(source)
        samples.append({"role": "posthoc_failure_example" if row is failing else "predeclared_representative",
                        "rule": "first failing scenario_id for calibrated three SAC_R seed33" if row is failing else "three, rho1.4, sample0, SACseed11/PIDseed0",
                        "metrics": row, "copy": (TARGET / source.relative_to(SOURCE)).relative_to(SUPPORT).as_posix()})
    json_write(SUPPORT / "轨迹样例索引.json", {"examples_are_not_population_statistics": True, "samples": samples})

    table = []
    for axis in ("single", "three"):
        for controller in ("PID_N", "PID_R", "SAC_N", "SAC_R"):
            group = [r for r in rows if r["axis"] == axis and r["controller"] == controller
                     and r["experiment"] == "scalar" and r["domain"] == "out_of_domain"]
            good = [r for r in group if r["success"].lower() == "true"]
            table.append({"axis": axis, "controller": controller, "region": "strict_scalar_out_of_domain_rho_0.5_0.6_1.4_1.5",
                          "successful_episodes": len(good), "episodes": len(group), "success_rate": len(good)/len(group),
                          "settling_time_successful_only_s": mean(float(r["settling_time_s"]) for r in good),
                          "rms_all_episodes_deg": mean(float(r["rms_error_deg"]) for r in group),
                          "control_cost_all_episodes": mean(float(r["control_cost"]) for r in group)})
    with (STEP / "结果速查表.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    json_write(SUPPORT / "来源清单.json", {"purpose": "readable defense curation; no new experiment", "sources": records,
               "derived": [{"copy": "结果速查表.csv", "source": f"结项成果包/{STUDY_NAME}/03_测试数据与配对结果/episode_metrics.csv",
                            "rule": "strict scalar out-of-domain only; successful settling, all-episode RMS/cost; SAC pools seeds 11/22/33 over reused initial states"}],
               "omissions": ["full raw NPZ population", "replay buffers", "final/latest/intermediate models"],
               "all_12_best_models_and_metadata_included": True})
    print(json.dumps({"copied_files": len(records), "copied_bytes": sum((STEP / r["copy"]).stat().st_size for r in records),
                      "models": len(list(TARGET.glob(f"{STUDY_NAME}/02_模型与验证日志/**/best_model.zip"))),
                      "trace_samples": len(samples), "metric_rows": len(rows)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
