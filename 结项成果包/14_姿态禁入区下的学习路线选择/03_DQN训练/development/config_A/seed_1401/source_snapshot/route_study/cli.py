"""Explicit stage entry points; implementation never implies training started."""
import argparse
import json
from . import protocol as p


def development():
    from .learning import train_run, decide
    for config_name, seed in p.DEVELOPMENT_SEEDS.items():
        value = train_run('development', config_name, seed)
        stats = value['best']['validation']
        if stats['collisions'] == 0 and stats['success_rate'] >= .95:
            break
    return decide()


def formal():
    from .learning import train_run
    decision = p.read_json(p.STUDY/'03_DQN训练'/'development_decision.json')
    if not decision['passed_gate']:
        return dict(status='not_started', reason='development reliability gate not passed')
    return [train_run('formal', decision['config_name'], seed) for seed in p.FORMAL_SEEDS]


def main(argv=None):
    parser = argparse.ArgumentParser(description='单轴禁入弧：真实单步DQN路线选择与同一固定PID')
    parser.add_argument('command', choices=('design', 'baseline', 'prepare', 'development', 'formal', 'freeze', 'test', 'verify', 'report', 'all'))
    args = parser.parse_args(argv)
    if args.command == 'design':
        from .learning import predeclare
        value = dict(protocol=p.design(), learning=predeclare())
    elif args.command == 'baseline':
        from .baselines import tune
        value = tune()
    elif args.command == 'prepare':
        from .learning import prepare_database
        value = prepare_database()
    elif args.command == 'development':
        value = development()
    elif args.command == 'formal':
        value = formal()
    elif args.command in ('freeze', 'test', 'report'):
        from . import delivery
        value = getattr(delivery, 'test_responses' if args.command == 'test' else args.command)()
    elif args.command == 'verify':
        from .verify import verify
        value = verify()
    else:
        # Every run/selection directory is exclusive; unfinished attempts are
        # retained, never silently resumed or extended beyond their budget.
        p.design()
        from .learning import predeclare, prepare_database
        from .baselines import tune
        from .delivery import freeze, test_responses, report
        from .verify import verify
        predeclare()
        if not (p.STUDY/'02_传统对照'/'selected_pid.json').exists():
            tune()
        prepare_database()
        development()
        formal()
        freeze()
        test_responses()
        verify()
        value = report()
    print(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value, flush=True)


if __name__ == '__main__':
    main()
