"""Run selection first, freeze deployment, then collect untouched test responses."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
from . import protocol as p

TEST = p.STUDY/'04_独立测试'
FREEZE = TEST/'freeze.json'


def selection_open():
    if FREEZE.exists() or (TEST/'test_manifest.json').exists():
        raise RuntimeError('deployment already frozen; selection/training cannot continue in this study')


def predeclare_delivery():
    selection_open()
    declaration = dict(
        failed_development_test='selected best development candidate is tested diagnostically; no formal three-seed claim',
        diagnostic_selection='same predeclared validation ranking; never choose from test responses',
        regions=['in_domain', 'ideal', 'out_of_domain'],
        shard_unit='one controller, kind, region and actuator parameter; all 256 common initial states',
        bootstrap_seed=202610170901, bootstrap_repetitions=20000,
        bootstrap_unit='initial state, retaining all six actuator parameters',
        primary_family_size=12, confidence_for_primary_intervals=1-.05/12,
        acceptance='all in-domain episodes pass in both methods and all three formal seeds; all corrected difference intervals below zero',
        emphasized_reduction=.05, test_initials_independent=256,
        all_searches_and_failed_runs_retained=True,
        online_learning_in_tests=False,
    )
    path = p.STUDY/'01_实验方案'/'执行补充声明.json'
    if path.exists() and p.read_json(path) != declaration:
        raise RuntimeError('delivery/statistical declaration changed')
    if not path.exists():
        p.write_json(path, declaration)
    return declaration


def _metadata(kind, method, reward, seed, stage):
    return p.STUDY/'03_SAC训练'/stage/kind/method/f'reward_{reward}'/f'seed_{seed}'/'training_metadata.json'


def _completed(path):
    if not path.exists():
        return False
    data = p.read_json(path)
    if data['status'] != 'complete':
        raise RuntimeError(f'recorded incomplete attempt requires an explicit new documented run: {path}')
    if p.sha256(p.STUDY/data['best']['model_path']) != data['best']['model_sha256']:
        raise RuntimeError('saved model checksum differs from training record')
    if data['source_sha256_after'] != p.source_hashes():
        raise RuntimeError('source differs from completed run; do not silently reuse mixed implementations')
    return True


def development(kinds):
    from .learning import predeclare, train_run, decide_development
    selection_open()
    predeclare_delivery()
    predeclare()
    for kind in kinds:
        for method in p.METHODS:
            for reward, seed in p.DEVELOPMENT_SEEDS.items():
                path = _metadata(kind, method, reward, seed, 'development')
                if not _completed(path):
                    train_run(kind, method, reward, seed, p.DEVELOPMENT_STEPS, 'development')
            print(json.dumps(decide_development(kind, method), ensure_ascii=False), flush=True)


def formal(kinds):
    from .learning import train_run
    selection_open()
    for kind in kinds:
        for method in p.METHODS:
            decision = p.read_json(p.STUDY/'03_SAC训练'/'development_decisions'/f'{kind}_{method}.json')
            if not decision['passed_gate']:
                print(f'formal skipped: {kind}/{method}: development gate not passed', flush=True)
                continue
            reward = decision['reward_name']
            for seed in p.FORMAL_SEEDS:
                path = _metadata(kind, method, reward, seed, 'formal')
                if not _completed(path):
                    train_run(kind, method, reward, seed, p.FORMAL_STEPS, 'formal')


def controller_entries():
    result = []
    for kind in p.KINDS:
        baseline = p.read_json(p.STUDY/'02_传统控制'/f'selected_{kind}.json')
        for family in ('pid', 'model_compensated'):
            result.append(dict(id=f'{kind}_{family}', kind=kind, family=family,
                               stage='baseline', parameters=baseline[family]['parameters']))
        for method in p.METHODS:
            decision = p.read_json(p.STUDY/'03_SAC训练'/'development_decisions'/f'{kind}_{method}.json')
            if decision['passed_gate']:
                paths = [_metadata(kind, method, decision['reward_name'], seed, 'formal') for seed in p.FORMAL_SEEDS]
            else:
                # Predeclared diagnostic testing preserves visibility of unsuccessful methods.
                best = decision['best']
                if best is None:
                    continue
                paths = [next(p.STUDY/a['metadata_path'] for a in decision['attempts']
                              if a['reward_name'] == decision['reward_name'])]
            for path in paths:
                if not _completed(path):
                    raise RuntimeError('required training run is missing: '+str(path))
                run = p.read_json(path)
                stage = 'formal' if run['stage'] == 'formal' else 'development_diagnostic'
                result.append(dict(id=f'{kind}_{method}_{stage}_seed{run["seed"]}', kind=kind,
                                   family=method, stage=stage, seed=run['seed'], reward_name=run['reward_name'],
                                   model_path=run['best']['model_path'], model_sha256=run['best']['model_sha256'],
                                   metadata_path=path.relative_to(p.STUDY).as_posix(),
                                   base_parameters=run['base_parameters'], selected_steps=run['best']['steps']))
    return result


def check_artifacts(artifacts):
    for relative, expected in artifacts.items():
        path = p.STUDY/relative
        if not path.is_file() or p.sha256(path) != expected:
            raise RuntimeError('frozen artifact mismatch: '+relative)


def freeze():
    if FREEZE.exists():
        result = p.read_json(FREEZE)
        check_artifacts(result['artifacts'])
        return result
    selection_open()
    entries = controller_entries()
    artifacts = dict(p.source_hashes())
    for directory in ('01_实验方案', '02_传统控制'):
        for path in (p.STUDY/directory).rglob('*'):
            if path.is_file():
                artifacts[path.relative_to(p.STUDY).as_posix()] = p.sha256(path)
    for path in (p.STUDY/'03_SAC训练').rglob('*.json'):
        if 'source_snapshot' not in path.parts:
            artifacts[path.relative_to(p.STUDY).as_posix()] = p.sha256(path)
    for entry in entries:
        if 'model_path' in entry:
            artifacts[entry['model_path']] = p.sha256(p.STUDY/entry['model_path'])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), controllers=entries,
                  artifacts=artifacts, test_responses_used_for_selection=False,
                  independent_initial_states=256, code_source='代码',
                  diagnostic_candidates_predeclared=True)
    p.write_json(FREEZE, result)
    return result


def make_controller(entry):
    from .baselines import PIDController, ModelCompensatedController
    if entry['family'] == 'pid':
        return PIDController(entry['parameters'])
    if entry['family'] == 'model_compensated':
        return ModelCompensatedController(entry['kind'], entry['parameters'])
    from stable_baselines3 import SAC
    import torch
    from .learning import SACController
    torch.set_num_threads(1)
    return SACController(SAC.load(p.STUDY/entry['model_path'], device='cpu'),
                         entry['family'], entry['base_parameters'])


def test():
    from .evaluation import evaluate, write_csv, summary
    manifest_path = TEST/'test_manifest.json'
    frozen = p.read_json(FREEZE)
    check_artifacts(frozen['artifacts'])
    if manifest_path.exists():
        manifest = p.read_json(manifest_path)
        check_artifacts(manifest['artifacts'])
        if manifest['freeze_sha256'] != p.sha256(FREEZE):
            raise RuntimeError('test manifest belongs to a different freeze')
        return manifest
    entries, artifacts, all_rows = [], {}, []
    for entry in frozen['controllers']:
        controller = make_controller(entry)
        for region in ('in_domain', 'ideal', 'out_of_domain'):
            scenarios = p.scenes(entry['kind'], 'test', region)
            for parameter in sorted({s['parameter_s'] for s in scenarios}):
                subset = [s for s in scenarios if s['parameter_s'] == parameter]
                directory = TEST/'trajectories'/entry['id']/region
                directory.mkdir(parents=True, exist_ok=True)
                trace_path = directory/f'parameter_{parameter:.2f}.npz'
                data_path = trace_path.with_suffix('.json')
                if trace_path.exists() and data_path.exists():
                    # Deterministic interrupted collection may resume only the exact frozen controller.
                    saved = p.read_json(data_path)
                    if saved.get('freeze_sha256') != p.sha256(FREEZE) or saved['scenes'] != subset:
                        raise RuntimeError('existing test shard differs from frozen inputs')
                    rows = saved['rows']
                else:
                    response = evaluate(entry['kind'], subset, controller, keep_traces=True)
                    np.savez_compressed(trace_path, **response['traces'])
                    rows = [dict(row, controller_id=entry['id'], family=entry['family'],
                                 stage=entry['stage'], seed=entry.get('seed')) for row in response['rows']]
                    p.write_json(data_path, dict(scenes=subset, rows=rows, freeze_sha256=p.sha256(FREEZE)))
                all_rows.extend(rows)
                for path in (trace_path, data_path):
                    artifacts[path.relative_to(p.STUDY).as_posix()] = p.sha256(path)
                entries.append(dict(controller_id=entry['id'], kind=entry['kind'], region=region,
                                    parameter_s=parameter, trace_path=trace_path.relative_to(p.STUDY).as_posix(),
                                    data_path=data_path.relative_to(p.STUDY).as_posix()))
                print(json.dumps(dict(event='held-out test', controller=entry['id'], region=region,
                                      parameter=parameter, **summary(rows)), ensure_ascii=False), flush=True)
    csv_path = TEST/'episode_metrics.csv'
    write_csv(csv_path, all_rows)
    artifacts[csv_path.relative_to(p.STUDY).as_posix()] = p.sha256(csv_path)
    manifest = dict(entries=entries, artifacts=artifacts, freeze_sha256=p.sha256(FREEZE), complete=True,
                    completed_utc=datetime.now(timezone.utc).isoformat(), episode_records=len(all_rows),
                    independent_initial_states=256)
    p.write_json(manifest_path, manifest)
    return manifest


def dispatch(command, kind, remaining):
    kinds = (kind,) if kind else p.KINDS
    if remaining:
        raise ValueError('unknown arguments: '+str(remaining))
    if command in ('train', 'development'):
        development(kinds)
    elif command == 'formal':
        formal(kinds)
    elif command == 'freeze':
        freeze()
    elif command == 'test':
        test()
    elif command == 'verify':
        from .verification import verify_all
        print(json.dumps(verify_all(), ensure_ascii=False), flush=True)
    elif command == 'report':
        from .analysis import report
        report()
    else:
        raise ValueError(command)
