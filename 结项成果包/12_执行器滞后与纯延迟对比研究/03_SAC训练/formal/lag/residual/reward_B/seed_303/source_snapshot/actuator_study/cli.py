"""Research entry points, with separate selection and held-out response stages."""
import argparse
from .protocol import design


def main():
    parser = argparse.ArgumentParser(description='一阶滞后/纯延迟：单轴PID与SAC比较')
    parser.add_argument('command', choices=('design', 'pid', 'train', 'development', 'formal', 'freeze', 'test', 'verify', 'report'))
    parser.add_argument('--kind', choices=('lag', 'delay'))
    args, remaining = parser.parse_known_args()
    design()
    if args.command == 'design':
        print('已保存执行配置和独立初态。')
    elif args.command == 'pid':
        from .delivery import selection_open
        selection_open()
        from .baselines import tune
        for kind in (args.kind,) if args.kind else ('lag', 'delay'):
            tune(kind)
    else:
        from .delivery import dispatch
        dispatch(args.command, args.kind, remaining)
