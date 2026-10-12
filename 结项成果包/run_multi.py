"""Version 2: matched single/three-axis PID/PPO/SAC/TD3 comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "07_单轴三轴多算法对比"
MODELS = PACKAGE / "models"
RESULTS = PACKAGE / "results"


def train_job(arguments):
    from multi_algorithm_benchmark.training import train_model
    return train_model(**arguments)


def main():
    parser = argparse.ArgumentParser(description="单轴与三轴：统一PID/PPO/SAC/TD3对比")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    train = sub.add_parser("train")
    train.add_argument("--axes", nargs="+", choices=("single", "three"), default=["single", "three"])
    train.add_argument("--algorithms", nargs="+", choices=("PPO", "SAC", "TD3"), default=["PPO", "SAC", "TD3"])
    train.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    train.add_argument("--steps", type=int, default=1000000)
    train.add_argument("--validation-interval", type=int, default=100000)
    train.add_argument("--jobs", type=int, default=3)
    train.add_argument("--output", type=Path, default=MODELS)
    tune = sub.add_parser("tune-pid")
    tune.add_argument("--output", type=Path, default=MODELS)
    calibrate = sub.add_parser("calibrate", help="在验证集复核零参考动作偏置校准；原始/校准策略全部保留")
    calibrate.add_argument("--models", type=Path, default=MODELS)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--models", type=Path, default=MODELS)
    evaluate.add_argument("--output", type=Path, default=RESULTS)
    evaluate.add_argument("--samples", type=int, default=20)
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    history = sub.add_parser("historical")
    history.add_argument("--output", type=Path, default=PACKAGE / "历史模型部署复测")
    plot = sub.add_parser("plot")
    plot.add_argument("--results", type=Path, default=RESULTS)
    report = sub.add_parser("report")
    report.add_argument("--results", type=Path, default=RESULTS)
    report.add_argument("--models", type=Path, default=MODELS)
    report.add_argument("--output", type=Path, help="报告另存目录；默认结果目录的父目录")
    args = parser.parse_args()
    if args.command == "check":
        from stable_baselines3.common.env_checker import check_env
        from multi_algorithm_benchmark.envs import make_env
        for axis in ("single", "three"):
            check_env(make_env(axis), warn=True)
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT,
                                capture_output=True, text=True)
        text = "Both multi-algorithm Gym interfaces: passed\n" + result.stdout + result.stderr
        print(text)
        PACKAGE.mkdir(parents=True, exist_ok=True)
        (PACKAGE / "verification_log.txt").write_text(text, encoding="utf-8")
        if result.returncode:
            raise SystemExit(result.returncode)
    elif args.command == "train":
        for name in ("axes", "algorithms", "seeds"):
            values = getattr(args, name)
            if len(set(values)) != len(values):
                parser.error(f"--{name} cannot contain duplicates")
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        jobs = [{"output_dir": args.output.resolve(), "axis": axis, "algorithm": algorithm,
                 "seed": seed, "steps": args.steps, "validation_interval": args.validation_interval}
                for axis in args.axes for algorithm in args.algorithms for seed in args.seeds]
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            for result in pool.map(train_job, jobs):
                print(json.dumps({k: result[k] for k in ("axis", "algorithm", "seed", "actual_steps", "wall_time_s")}), flush=True)
    elif args.command == "tune-pid":
        from multi_algorithm_benchmark.controllers import tune_pid
        for axis in ("single", "three"):
            print(tune_pid(axis, args.output.resolve()))
    elif args.command == "calibrate":
        from multi_algorithm_benchmark.improvements import validate_calibration
        validate_calibration(args.models.resolve())
    elif args.command == "evaluate":
        from multi_algorithm_benchmark.evaluation import evaluate_models
        print(evaluate_models(args.models.resolve(), args.output.resolve(), args.samples, args.seeds))
    elif args.command == "historical":
        from multi_algorithm_benchmark.evaluation import evaluate_historical
        print(evaluate_historical(args.output.resolve()))
    elif args.command == "plot":
        from multi_algorithm_benchmark.plotting import generate_plots
        print(generate_plots(args.results.resolve()))
    elif args.command == "report":
        from multi_algorithm_benchmark.reporting import build_report
        build_report(args.results.resolve(), args.models.resolve(), args.output.resolve() if args.output else None)


if __name__ == "__main__":
    main()
