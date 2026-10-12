"""One entry point for checks, training, experiments and reports."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "03_训练模型与日志" / "正式训练_100万步"
RESULTS = ROOT / "04_实验数据与图表"


def train_job(arguments):
    from satellite_benchmark.training import train_model
    return train_model(**arguments)


def check() -> None:
    from stable_baselines3.common.env_checker import check_env
    from satellite_benchmark.env import SatelliteEnv
    check_env(SatelliteEnv(), warn=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                            cwd=ROOT, capture_output=True, text=True)
    text = "Stable-Baselines3 check_env: passed\n" + result.stdout + result.stderr
    print(text)
    (ROOT / "06_结项说明与报告" / "verification_log.txt").write_text(text, encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)
    subprocess.run([sys.executable, str(ROOT / "tools" / "verify_originals.py"), "check"],
                   cwd=ROOT, check=True)
    if (RESULTS / "evaluation_manifest.json").exists():
        from tools.verify_results import verify_results
        report = verify_results()
        print(json.dumps({"formal_result_verification": report["status"]}, ensure_ascii=False))
        if report["status"] != "passed":
            raise SystemExit("正式结果核验未通过，详见06_结项说明与报告/result_verification.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="卫星姿态控制结项成果包：统一运行入口")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="环境接口、关键回归与原文件保全核验")
    train = sub.add_parser("train", help="训练普通/随机化 PPO；新实验请指定独立输出目录")
    train.add_argument("--mode", choices=["nominal", "randomized", "both"], default="both")
    train.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    train.add_argument("--steps", type=int, default=1000000)
    train.add_argument("--jobs", type=int, default=1)
    train.add_argument("--output", type=Path, default=MODELS)
    train.add_argument("--validation-interval", type=int, default=50000)
    train.add_argument("--resume", action="store_true")
    tune = sub.add_parser("tune-pid", help="仅使用验证集搜索 PID 参数")
    tune.add_argument("--output", type=Path, default=MODELS)
    evaluate = sub.add_parser("evaluate", help="独立测试集上评估所有控制器并保存原始轨迹")
    evaluate.add_argument("--models", type=Path, default=MODELS)
    evaluate.add_argument("--output", type=Path, default=RESULTS)
    evaluate.add_argument("--samples", type=int, default=50)
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    evaluate.add_argument("--allow-missing", action="store_true", help="仅用于联调；正式实验要求六个模型齐全")
    plot = sub.add_parser("plot", help="从保存数据重新生成结果图")
    plot.add_argument("--results", type=Path, default=RESULTS)
    plot.add_argument("--models", type=Path, default=MODELS)
    sub.add_parser("three-axis", help="运行独立三轴 PID 拓展演示")
    sub.add_parser("report", help="从实际数据生成结项技术报告与离线结果页面")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "train":
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds 不能重复，否则并行作业会写入相同模型目录")
        modes = ["nominal", "randomized"] if args.mode == "both" else [args.mode]
        jobs = [{"output_dir": args.output.resolve(), "mode": mode, "seed": seed,
                 "total_timesteps": args.steps, "validation_interval": args.validation_interval,
                 "preset": "full", "resume": args.resume}
                for mode in modes for seed in args.seeds]
        if args.jobs < 1:
            parser.error("--jobs 必须大于零")
        if args.jobs == 1:
            for job in jobs:
                print(json.dumps(train_job(job), ensure_ascii=False), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
                for result in pool.map(train_job, jobs):
                    print(json.dumps(result, ensure_ascii=False), flush=True)
    elif args.command == "tune-pid":
        from satellite_benchmark.evaluation import tune_pid
        print(tune_pid(args.output.resolve()))
    elif args.command == "evaluate":
        from satellite_benchmark.evaluation import evaluate_models
        print(json.dumps(evaluate_models(args.output.resolve(), args.models.resolve(),
                        samples_per_case=args.samples, seeds=tuple(args.seeds),
                        require_all_models=not args.allow_missing), ensure_ascii=False))
    elif args.command == "plot":
        from satellite_benchmark.plotting import generate_plots
        print(json.dumps(generate_plots(args.results.resolve(), args.models.resolve()), ensure_ascii=False))
    elif args.command == "three-axis":
        subprocess.run([sys.executable, str(ROOT / "05_三轴拓展演示" / "run_three_axis.py")],
                       cwd=ROOT, check=True)
    elif args.command == "report":
        from tools.build_report import main as build_report
        build_report()


if __name__ == "__main__":
    main()
