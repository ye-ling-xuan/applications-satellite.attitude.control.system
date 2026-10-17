import argparse
from .protocol import STUDY, freeze_design


def main():
    parser = argparse.ArgumentParser(description="同等停稳要求下的控制代价比较；原成果不改动")
    parser.add_argument("command", choices=("design", "legacy", "pid", "train", "freeze", "test", "report", "verify"))
    args, remaining = parser.parse_known_args()
    freeze_design()
    if args.command in ("legacy", "pid", "train") and (STUDY/"01_实验方案"/"部署冻结.json").exists():
        raise RuntimeError("本轮已冻结，禁止覆盖模型或重新选参；新实验请使用独立目录。")
    if args.command == "design":
        print("任务定义和独立数据划分已保存。")
    elif args.command == "legacy":
        from .legacy import tune_legacy
        tune_legacy()
    elif args.command == "pid":
        from .baselines import main as run
        run(remaining)
    elif args.command == "train":
        from .training import main as run
        run(remaining)
    else:
        from .delivery import main as run
        run(args.command)
