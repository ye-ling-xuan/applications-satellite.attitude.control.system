"""Run declared continuous-control experiments; never retune after freezing."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import sys
from datetime import datetime, timezone
import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY/'代码'))
from continuous_study import protocol as p

FREEZE = STUDY/'04_独立测试'/'freeze.json'


def metadata_path(stage, kind, algorithm, reward, seed, mode='residual'):
    return STUDY/'03_训练记录'/stage/kind/mode/algorithm/f'reward_{reward}'/f'seed_{seed}'/'training_metadata.json'


def selection_open():
    if FREEZE.exists():
        raise RuntimeError('controllers are frozen; tuning and training are closed for this study')


def completed(path):
    if not path.exists():
        return False
    run = p.read_json(path)
    if run['status'] != 'complete':
        raise RuntimeError('existing incomplete run retained: '+str(path))
    if run.get('source_sha256') and run['source_sha256'] != p.source_hashes():
        raise RuntimeError('training production source changed: '+str(path))
    return True


def base(kind):
    return p.read_json(STUDY/'02_传统控制'/f'selected_{kind}.json')['pid']


def train(args):
    from continuous_study.learning import train_run, predeclare
    selection_open()
    p.design()
    predeclare()
    stage, kind, algorithm, reward = args.stage, args.kind, args.algorithm, args.reward
    mode = 'direct' if stage == 'sanity' else 'residual'
    if stage == 'sanity':
        kind, reward = 'ideal', 'A'
        seed, steps = p.SANITY_SEEDS[algorithm], p.SANITY_STEPS
        parameters = None
    elif stage == 'development':
        seed, steps = p.DEVELOPMENT_SEEDS[reward], p.DEVELOPMENT_STEPS
        parameters = base(kind)['parameters']
    else:
        decision = p.read_json(STUDY/'03_训练记录'/'decisions'/f'{kind}.json')
        if not decision['formal_eligible']:
            raise RuntimeError('development did not produce an eligible learned candidate')
        algorithm, reward = decision['selected']['algorithm'], decision['selected']['reward_name']
        seed, steps = args.seed, p.FORMAL_STEPS
        parameters = base(kind)['parameters']
    path = metadata_path(stage, kind, algorithm, reward, seed, mode)
    if completed(path):
        print('completed run retained: '+str(path.relative_to(STUDY)), flush=True)
        return
    train_run(kind=kind, algorithm=algorithm, seed=seed, steps=steps, stage=stage,
              base_parameters=parameters, validation_scenes=p.scenes(kind, 'validation'),
              mode=mode, reward_name=reward)


def decide(kind):
    from continuous_study.evaluation import ranking
    selection_open()
    pid = base(kind)['validation']
    runs = []
    for reward in ('A', 'B'):
        for algorithm in p.ALGORITHMS:
            path = metadata_path('development', kind, algorithm, reward, p.DEVELOPMENT_SEEDS[reward])
            if path.exists():
                if not completed(path):
                    raise RuntimeError('incomplete development')
                run = p.read_json(path)
                runs.append(dict(metadata_path=path.relative_to(STUDY).as_posix(), algorithm=algorithm,
                                 reward_name=reward, seed=run['seed'], best=run['best'],
                                 base_parameters=run.get('base_parameters', base(kind)['parameters'])))
        if reward == 'A' and len(runs) < 3:
            raise RuntimeError('all three A development runs are required')
    a = [r for r in runs if r['reward_name'] == 'A']
    a_advantage = any(r['best']['steps'] > 0 and r['best']['validation']['physical_failures'] == 0
                      and r['best']['validation']['success_rate'] >= .95
                      and r['best']['validation']['mean_actual_cost'] < pid['mean_actual_cost'] for r in a)
    require_b = not a_advantage
    if require_b and len(runs) < 6:
        result = dict(kind=kind, reward_B_required=True, complete=False,
                      formal_eligible=False, selected=None, runs=runs,
                      test_responses_used=False)
    else:
        eligible = [r for r in runs if r['best']['steps'] > 0 and
                    r['best']['validation']['success_rate'] >= .95 and
                    r['best']['validation']['physical_failures'] == 0]
        chosen = max(eligible, key=lambda r: (*ranking(r['best']['validation']),
                     -r['best']['steps'], -p.ALGORITHMS.index(r['algorithm']))) if eligible else None
        result = dict(kind=kind, reward_B_required=require_b, complete=True,
                      formal_eligible=chosen is not None, selected=chosen,
                      runs=runs, test_responses_used=False)
    p.write_json(STUDY/'03_训练记录'/'decisions'/f'{kind}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, ensure_ascii=False), flush=True)
    return result


def make_controller(entry):
    from continuous_study.baselines import PIDController, ModelCompensatedController
    if entry['family'] == 'pid':
        return PIDController(entry['parameters'])
    if entry['family'] == 'model_compensated':
        return ModelCompensatedController(entry['kind'], entry['parameters'])
    from continuous_study.learning import load_controller
    return load_controller(STUDY/entry['model_path'], entry['algorithm'], entry['family'],
                           entry['base_parameters'])


def model_entry(run, entry_id, stage):
    best = run['best']
    return dict(id=entry_id, kind=run['kind'], family=run.get('mode', 'residual'),
                algorithm=run['algorithm'], stage=stage, seed=run['seed'],
                reward_name=run['reward_name'], selected_steps=best['steps'],
                model_path=best['model_path'], model_sha256=best['model_sha256'],
                base_parameters=run['base_parameters'])


def check_hashes(artifacts):
    for relative, expected in artifacts.items():
        if p.sha256(STUDY/relative) != expected:
            raise RuntimeError('artifact changed: '+relative)


def freeze():
    if FREEZE.exists():
        result = p.read_json(FREEZE)
        check_hashes(result['artifacts'])
        return result
    entries = []
    for kind in p.KINDS:
        baseline = p.read_json(STUDY/'02_传统控制'/f'selected_{kind}.json')
        for family in ('pid', 'model_compensated'):
            entries.append(dict(id=f'{kind}_{family}', kind=kind, family=family,
                                stage='baseline', parameters=baseline[family]['parameters']))
        decision = p.read_json(STUDY/'03_训练记录'/'decisions'/f'{kind}.json')
        if not decision['complete']:
            raise RuntimeError('development decisions incomplete')
        if decision['formal_eligible']:
            chosen = decision['selected']
            for seed in p.FORMAL_SEEDS:
                path = metadata_path('formal', kind, chosen['algorithm'], chosen['reward_name'], seed)
                if not completed(path):
                    raise RuntimeError('formal model missing')
                run = p.read_json(path)
                entries.append(model_entry(run, f'{kind}_{run["algorithm"]}_formal_seed{seed}', 'formal'))
        # Report all algorithms' selected development candidates, including failures.
        from continuous_study.evaluation import ranking
        for algorithm in p.ALGORITHMS:
            candidates = [r for r in decision['runs'] if r['algorithm'] == algorithm]
            selected = max(candidates, key=lambda r: (*ranking(r['best']['validation']), -r['best']['steps']))
            run = p.read_json(STUDY/selected['metadata_path'])
            entries.append(model_entry(run, f'{kind}_{algorithm}_development_seed{run["seed"]}', 'development_diagnostic'))
    artifacts = p.source_hashes()
    for directory in ('01_实验方案', '02_传统控制', '03_训练记录'):
        for path in (STUDY/directory).rglob('*.json'):
            if 'source_snapshot' not in path.parts:
                artifacts[path.relative_to(STUDY).as_posix()] = p.sha256(path)
    for entry in entries:
        if 'model_path' in entry:
            artifacts[entry['model_path']] = p.sha256(STUDY/entry['model_path'])
    artifacts['运行研究.py'] = p.sha256(Path(__file__))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), controllers=entries,
                  artifacts=artifacts, independent_initial_states=p.COUNTS['test'],
                  test_responses_used_for_selection=False, diagnostic_candidates_predeclared=True)
    p.write_json(FREEZE, result)
    return result


def test():
    from continuous_study.evaluation import evaluate, summary, write_csv
    frozen = p.read_json(FREEZE)
    check_hashes(frozen['artifacts'])
    target = FREEZE.with_name('test_manifest.json')
    if target.exists():
        manifest = p.read_json(target)
        check_hashes(manifest['artifacts'])
        return manifest
    entries, artifacts, all_rows = [], {}, []
    freeze_sha = p.sha256(FREEZE)
    for entry in frozen['controllers']:
        controller = make_controller(entry)
        for region in ('in_domain', 'ideal', 'out_of_domain'):
            scenarios = p.scenes(entry['kind'], 'test', region)
            for parameter in sorted({r['parameter_s'] for r in scenarios}):
                selected = [r for r in scenarios if r['parameter_s'] == parameter]
                directory = FREEZE.parent/'trajectories'/entry['id']/region
                directory.mkdir(parents=True, exist_ok=True)
                trace_path = directory/f'parameter_{parameter:.2f}.npz'
                data_path = trace_path.with_suffix('.json')
                if trace_path.exists() and data_path.exists():
                    saved = p.read_json(data_path)
                    if saved['freeze_sha256'] != freeze_sha or saved['scenes'] != selected:
                        raise RuntimeError('existing shard belongs to other inputs')
                    rows = saved['rows']
                else:
                    result = evaluate(entry['kind'], selected, controller, keep_traces=True)
                    np.savez_compressed(trace_path, **result['traces'])
                    rows = [dict(r, controller_id=entry['id'], family=entry['family'],
                                 algorithm=entry.get('algorithm'), stage=entry['stage'], seed=entry.get('seed'))
                            for r in result['rows']]
                    p.write_json(data_path, dict(rows=rows, scenes=selected, freeze_sha256=freeze_sha))
                all_rows.extend(rows)
                for path in (trace_path, data_path):
                    artifacts[path.relative_to(STUDY).as_posix()] = p.sha256(path)
                entries.append(dict(controller_id=entry['id'], kind=entry['kind'], region=region,
                                    parameter_s=parameter, trace_path=trace_path.relative_to(STUDY).as_posix(),
                                    data_path=data_path.relative_to(STUDY).as_posix()))
                print(json.dumps(dict(event='independent test', controller=entry['id'],
                                      region=region, parameter=parameter, **summary(rows)), ensure_ascii=False), flush=True)
    output = FREEZE.parent/'episode_metrics.csv'
    write_csv(output, all_rows)
    artifacts[output.relative_to(STUDY).as_posix()] = p.sha256(output)
    manifest = dict(entries=entries, artifacts=artifacts, freeze_sha256=freeze_sha,
                    complete=True, episode_records=len(all_rows), independent_initial_states=p.COUNTS['test'])
    p.write_json(target, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('design', 'train', 'decide', 'freeze', 'test', 'verify', 'report'))
    parser.add_argument('--kind', choices=p.KINDS, default='lag')
    parser.add_argument('--algorithm', choices=p.ALGORITHMS, default='SAC')
    parser.add_argument('--reward', choices=('A', 'B'), default='A')
    parser.add_argument('--stage', choices=('sanity', 'development', 'formal'), default='development')
    parser.add_argument('--seed', type=int, choices=p.FORMAL_SEEDS, default=p.FORMAL_SEEDS[0])
    args = parser.parse_args()
    if args.command == 'design':
        p.design()
    elif args.command == 'train':
        train(args)
    elif args.command == 'decide':
        decide(args.kind)
    elif args.command == 'freeze':
        freeze()
    elif args.command == 'test':
        test()
    elif args.command == 'verify':
        from continuous_study.verification import verify_all
        result = verify_all()
        print(json.dumps({k: v for k, v in result.items() if k != 'shards'}, ensure_ascii=False))
        if not result['passed']:
            raise SystemExit(1)
    else:
        from 汇总研究结果 import report
        report()


if __name__ == '__main__':
    main()
