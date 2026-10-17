"""Independent V3 CLI. Existing V1/V2 code and outputs are preserved."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path


def _train_one(kwargs):
    from inertia_generalization.training import train_model
    result = train_model(**kwargs)
    return {key: result[key] for key in ("axis", "mode", "seed", "actual_steps", "checkpoint_count", "best_checkpoint_steps", "model_sha256")}


def main(argv=None):
    from inertia_generalization.protocol import CONFIG_DIR, MODELS_DIR, STUDY_ROOT, RESULTS_DIR
    parser = argparse.ArgumentParser(description="V3 pure-inertia research; no implicit held-out test")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze protocol and predeclared scenario inputs")
    freeze.add_argument("--output", type=Path, default=CONFIG_DIR)
    pid = commands.add_parser("tune-pid", help="equal-budget validation-only PID-N/PID-R search")
    pid.add_argument("--axes", nargs="+", choices=("single", "three"), default=["single", "three"])
    pid.add_argument("--output", type=Path, default=MODELS_DIR)
    evaluate = commands.add_parser("evaluate", help="one frozen held-out test, requiring passed independent pre-test audit")
    evaluate.add_argument("--models", type=Path, default=MODELS_DIR)
    evaluate.add_argument("--output", type=Path, default=RESULTS_DIR)
    commands.add_parser("report", help="generate predeclared figures and offline report from completed tests")
    train = commands.add_parser("train", help="fresh SAC-N/SAC-R training; existing runs are never overwritten")
    train.add_argument("--axes", nargs="+", choices=("single", "three"), default=None)
    train.add_argument("--modes", nargs="+", choices=("nominal", "randomized"), default=None)
    train.add_argument("--seeds", nargs="+", type=int, default=None)
    train.add_argument("--axis", choices=("single", "three"))
    train.add_argument("--mode", choices=("nominal", "randomized"))
    train.add_argument("--seed", type=int)
    train.add_argument("--jobs", type=int, default=1)
    train.add_argument("--steps", type=int, default=1000000)
    train.add_argument("--n-envs", type=int, default=8)
    train.add_argument("--validation-interval", type=int, default=100000)
    train.add_argument("--output", type=Path)
    train.add_argument("--development", action="store_true", help="explicitly ineligible developer output in a separate directory")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        from inertia_generalization.protocol import freeze_protocol
        print(json.dumps(freeze_protocol(args.output), ensure_ascii=False, indent=2))
        return
    if args.command == "tune-pid":
        from inertia_generalization.pid import tune_pid
        for axis in args.axes:
            print(json.dumps(tune_pid(axis, args.output), ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        from inertia_generalization.evaluation import evaluate_models
        print(json.dumps(evaluate_models(args.models,args.output),ensure_ascii=False,indent=2))
        return
    if args.command == "report":
        from inertia_generalization.reporting import build_report
        print(json.dumps(build_report(),ensure_ascii=False,indent=2))
        return
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if (args.axis and args.axes) or (args.mode and args.modes) or (args.seed is not None and args.seeds):
        parser.error("use either singular or plural axis/mode/seed options")
    axes = args.axes or ([args.axis] if args.axis else ["single", "three"])
    modes = args.modes or ([args.mode] if args.mode else ["nominal", "randomized"])
    seeds = args.seeds or ([args.seed] if args.seed is not None else [11, 22, 33])
    if args.development:
        output = args.output or STUDY_ROOT/"06_开发联调"/"models"
        if MODELS_DIR.resolve() == output.resolve() or MODELS_DIR.resolve() in output.resolve().parents:
            parser.error("developer runs cannot be stored in the formal models directory")
    else:
        output = args.output or MODELS_DIR
    tasks = [dict(output_dir=str(output), axis=axis, mode=mode, seed=seed, steps=args.steps,
                  n_envs=args.n_envs, validation_interval=args.validation_interval, development=args.development)
             for axis in dict.fromkeys(axes) for mode in dict.fromkeys(modes) for seed in dict.fromkeys(seeds)]
    results = []
    if args.jobs == 1:
        for task in tasks:
            result = _train_one(task)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_train_one, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    from inertia_generalization.training import write_json
    write_json(output/"training_batch_summary.json", {"development": args.development, "runs": results})


if __name__ == "__main__":
    main()
