"""Copy existing V2 evidence into steps 04/05. Never train or edit originals."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

PROJECT = Path(__file__).resolve().parents[3]
PACKAGE = PROJECT / "结项成果包"
V2 = PACKAGE / "07_单轴三轴多算法对比"
DEST = PROJECT / "答辩讲解包"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(step_name, specifications):
    step = DEST / step_name
    entries = []
    for source, relative in specifications:
        target = step / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        entries.append({"source": source.relative_to(PROJECT).as_posix(),
                        "copy": relative, "sha256": sha(target)})
    (step / "支撑代码与数据" / "来源清单.json").write_text(json.dumps(
        {"purpose": "原样遴选副本；非新实验，原始CSV中的路径保留历史记录含义",
         "sources": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return step


def specs04():
    specs = [(V2 / name, "支撑代码与数据/原始报告与核验/" + name)
             for name in ("新融合分析.md", "实验约定.md", "核验记录.json", "result_summary.json")]
    specs += [(V2 / "results" / name, "支撑代码与数据/结果数据/" + name)
              for name in ("episode_metrics.csv", "summary.csv", "evaluation_manifest.json")]
    specs += [(V2 / "results/figures" / name, name)
              for name in ("01_response_three.png", "02_success_by_condition.png", "05_learning_seed_variation.png")]
    for module in ("multi_algorithm_benchmark", "satellite_benchmark"):
        specs += [(p, "支撑代码与数据/代码/" + module + "/" + p.name)
                  for p in sorted((PACKAGE / module).glob("*.py"))]
    specs += [(PACKAGE / name, "支撑代码与数据/代码/" + name)
              for name in ("run_multi.py", "requirements.txt", "requirements-tested.txt")]
    for axis in ("single", "three"):
        for algorithm in ("PPO", "SAC", "TD3"):
            for seed in (11, 22, 33):
                source = V2 / "models" / axis / algorithm / f"seed_{seed}"
                relative = f"支撑代码与数据/模型索引/训练记录/{axis}/{algorithm}/seed_{seed}/"
                for name in ("training_metadata.json", "training_manifest.json", "best_checkpoint.json",
                             "validation_history.csv", "validation_scenarios.json"):
                    specs.append((source / name, relative + name))
                if seed == 11 and (axis == "three" or algorithm == "PPO"):
                    specs.append((source / "best_model.zip", relative + "best_model.zip"))
    for name in ("baseline_PID_seed_0_test_three_nominal_000.npz",
                 "improved_PID_seed_0_test_three_nominal_000.npz",
                 "PPO_seed_11_test_three_nominal_000.npz",
                 "SAC_seed_11_test_three_nominal_000.npz"):
        specs.append((V2 / "results/three/raw_traces" / name, "支撑代码与数据/代表轨迹/" + name))
    return specs


def specs05():
    specs = [(V2 / "results" / name, "支撑代码与数据/结果数据/" + name)
             for name in ("episode_metrics.csv", "paired_improvements.csv", "policy_calibration_pairs.csv")]
    specs += [(V2 / "models" / name, "支撑代码与数据/参数与验证/" + name)
              for name in ("pid_configs.json", "pid_search_single.csv", "pid_search_three.csv",
                           "policy_calibration_manifest.json", "policy_calibration_validation.csv")]
    specs += [(V2 / "results/figures" / name, name)
              for name in ("04_paired_pid_improvement.png", "07_policy_calibration.png")]
    specs += [(PACKAGE / "multi_algorithm_benchmark" / name, "支撑代码与数据/关键代码/" + name)
              for name in ("controllers.py", "improvements.py", "metrics.py")]
    for name in ("SAC_seed_11_test_three_nominal_000.npz",
                 "SAC_calibrated_seed_11_test_three_nominal_000.npz",
                 "tuned_PID_seed_0_test_three_combined_000.npz",
                 "improved_PID_seed_0_test_three_combined_000.npz"):
        specs.append((V2 / "results/three/raw_traces" / name, "支撑代码与数据/代表轨迹/" + name))
    return specs


def write_index(step):
    runs = []
    for meta_path in sorted((V2 / "models").glob("*/*/seed_*/training_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        directory = meta_path.parent
        local = step / "支撑代码与数据/模型索引/训练记录" / directory.relative_to(V2 / "models")
        runs.append({"axis": meta["axis"], "algorithm": meta["algorithm"], "seed": meta["seed"],
                     "requested_steps": meta["requested_steps"], "actual_steps": meta["actual_steps"],
                     "best_checkpoint_steps": meta["best_checkpoint_steps"],
                     "best_model_source": (directory / "best_model.zip").relative_to(PROJECT).as_posix(),
                     "best_model_sha256": sha(directory / "best_model.zip"),
                     "final_model_source": (directory / "final_model.zip").relative_to(PROJECT).as_posix(),
                     "final_model_sha256": sha(directory / "final_model.zip"),
                     "best_model_copied": (local / "best_model.zip").is_file(),
                     "local_records": local.relative_to(step).as_posix()})
    assert len(runs) == 18
    (step / "支撑代码与数据/模型索引/18份模型索引.json").write_text(json.dumps(
        {"model_count": len(runs), "copied_best_models": 4,
         "selection": "四份seed_11模型仅演示存储与推理结构，不是按正式测试挑选优秀种子；统计仍保留全部18份模型",
         "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    step04 = prepare("04_三轴扩展与多算法公平比较", specs04())
    step05 = prepare("05_控制方法改进与效果分析", specs05())
    write_index(step04)
    for step in (step04, step05):
        record = json.loads((step / "支撑代码与数据/来源清单.json").read_text(encoding="utf-8"))
        print(f"{step.name}: copied and SHA256-checked {len(record['sources'])} files")


if __name__ == "__main__":
    main()
