"""Immutable selection, held-out full traces, and candid report generation."""
from __future__ import annotations
import csv
import html
import numpy as np
from . import protocol as p
from .baselines import selected, write_csv
from .physics import simulate, analytic_actions, features, summary


def freeze():
    path = p.STUDY/'04_独立测试'/'freeze.json'
    if path.exists():
        raise FileExistsError('held-out freeze already exists')
    decision = p.read_json(p.STUDY/'03_DQN训练'/'development_decision.json')
    models = []
    if decision['passed_gate']:
        from .learning import run_path
        for seed in p.FORMAL_SEEDS:
            source = run_path('formal', decision['config_name'], seed)/'training_metadata.json'
            metadata = p.read_json(source)
            if metadata['status'] != 'complete' or metadata['actual_decisions'] != p.FORMAL_DECISIONS:
                raise RuntimeError('all three predeclared formal seeds must complete before test freeze')
            if metadata['source_sha256'] != p.source_hashes() or metadata['source_sha256_after'] != p.source_hashes():
                raise RuntimeError('formal training source differs from held-out source')
            if p.sha256(p.STUDY/metadata['best']['model_path']) != metadata['best']['model_sha256']:
                raise RuntimeError('selected model differs from its training checkpoint hash')
            models.append(dict(name=f'DQN_seed_{seed}', seed=seed, **metadata['best']))
    baseline_path = p.STUDY/'02_传统对照'/'selected_pid.json'
    fixed = [baseline_path, p.STUDY/'01_方案与场景'/'执行配置.json',
             p.STUDY/'01_方案与场景'/'DQN预声明.json',
             p.STUDY/'03_DQN训练'/'physical_environment_database'/'database.npz']
    fixed += [p.STUDY/'01_方案与场景'/f'{split}.json' for split in ('development', 'validation', 'test', 'detour', 'edge')]
    fixed += [p.STUDY/model['model_path'] for model in models]
    # Bind the learned archive to every retained attempt, optimizer/experience
    # log and physical-cache provenance, rather than re-hashing weights alone.
    fixed += [item for item in (p.STUDY/'03_DQN训练').rglob('*') if item.is_file() and item.suffix in ('.json', '.jsonl', '.csv', '.zip', '.npz')]
    if any(item.stat().st_size >= 100*1024**2 for item in fixed):
        raise RuntimeError('split artifacts must each remain below 100 MiB')
    value = dict(source_sha256=p.source_hashes(), artifact_sha256={item.relative_to(p.STUDY).as_posix(): p.sha256(item) for item in sorted(set(fixed))},
                 parameters=selected()['parameters'], models=models, development_passed=decision['passed_gate'],
                 no_test_responses_used_for_selection=True,
                 interpretation='learned high-level route choice with fixed PID; no low-level learned feedback claim')
    p.write_json(path, value)
    return value


def check_freeze():
    frozen = p.read_json(p.STUDY/'04_独立测试'/'freeze.json')
    if frozen['source_sha256'] != p.source_hashes():
        raise RuntimeError('source changed after held-out freeze')
    for relative, digest in frozen['artifact_sha256'].items():
        if p.sha256(p.STUDY/relative) != digest:
            raise RuntimeError('frozen artifact changed: '+relative)
    return frozen


def test_responses():
    frozen = check_freeze()
    directory = p.STUDY/'04_独立测试'/'responses'
    directory.mkdir(parents=True, exist_ok=False)
    methods = [('direct_PID', None), ('analytic_route_PID', None)]+[(item['name'], item) for item in frozen['models']]
    results = []
    for name, model_info in methods:
        model = None
        if model_info is not None:
            from stable_baselines3 import DQN
            model = DQN.load(p.STUDY/model_info['model_path'], device='cpu')
        for split in ('test', 'detour', 'edge'):
            scene_list = p.scenes(split)
            population = np.array([[row['theta_rad'], row['target_rad']] for row in scene_list])
            actions = (np.full(len(scene_list), -1) if name == 'direct_PID'
                       else analytic_actions(population) if model is None
                       else model.predict(features(population), deterministic=True)[0])
            output = simulate(scene_list, actions, frozen['parameters'], keep_traces=True)
            target = directory/name/split
            target.mkdir(parents=True)
            write_csv(target/'episodes.csv', output['rows'])
            np.savez_compressed(target/'traces.npz', **output['traces'])
            results.append(dict(method=name, region=split, **summary(output['rows'])))
    write_csv(directory/'summary.csv', results)
    p.write_json(directory/'response_manifest.json', dict(
        freeze_sha256=p.sha256(p.STUDY/'04_独立测试'/'freeze.json'),
        files={path.relative_to(p.STUDY).as_posix(): p.sha256(path) for path in sorted(directory.rglob('*')) if path.is_file()},
        total_scenario_responses=sum(row['count'] for row in results)))
    return results


def _csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def report():
    frozen = check_freeze()
    verify_path = p.STUDY/'05_独立核验'/'verification.json'
    verification = p.read_json(verify_path)
    if not verification['passed']:
        raise RuntimeError('independent verification must pass before reporting conclusions')
    directory = p.STUDY/'04_独立测试'/'responses'
    rows = _csv(directory/'summary.csv')
    lines = ['# 姿态禁入区下的学习路线选择', '',
             '本实验每回合仅由DQN选择一次顺／逆时针，之后由同一固定PID完成反馈控制。它是单步上下文决策，不是连续深度控制。', '',
             '禁入区固定为[160°,200°]；起点／目标在距禁区边界至少15°的允许域均匀抽样，起终点相隔至少15°，初始角速度为0。J=1，力矩限幅2 Nm，采样0.05秒，12秒完成，验收延续至22秒。', '',
             '解析安全路线是公开的完整传统规划基准；直接目标PID是省略规划模块的对照。学习相对直接PID的收益只能归因于路线选择，不能写成学习反馈胜过PID。', '',
             f"低层参数：`{frozen['parameters']}`。所有方法完全共享。PID搜索包括750组候选，40组进入独立验证。", '',
             '| 方法 | 集合 | 场景数 | 完成 | 碰撞 | 完成率 | 安全率 | 全场景平均力矩平方积分 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['region']} | {row['count']} | {row['successes']} | {row['collisions']} | {float(row['success_rate']):.2%} | {float(row['safety_rate']):.2%} | {float(row['mean_torque_cost']):.6f} |")
    lines += ['', '自然测试集512个初态为主结果。绕行压力集128个初态满足展开角差>180°；边缘压力集128个初态要求至少一个端点距采样边界不超过5°。压力集分别报告，不用于代替自然分布。', '',
              '碰撞轨迹在首次触碰所在的完整采样间隔后冻结；后续段标记为失败填充，代价不再累积。失败的较低代价不解释为节能收益。禁入弧穿越按每个恒力矩区间内部的二次轨迹检查，包含内部转向极值。', '']
    if frozen['models']:
        # Paired nonparametric bootstrap keeps each independent test pair intact.
        direct = _csv(directory/'direct_PID'/'test'/'episodes.csv')
        baseline_success = np.array([row['success'] == 'True' for row in direct], float)
        rng = np.random.default_rng(2026101701499)
        resamples = rng.integers(0, len(direct), size=(20000, len(direct)))
        intervals = []
        lines += ['三个正式种子相对直接PID的自然测试集完成率配对差：', '', '| 种子 | 配对差 | Bonferroni校正98.333%区间 |', '|---|---:|---|']
        for model in frozen['models']:
            learned = _csv(directory/model['name']/'test'/'episodes.csv')
            difference = np.array([row['success'] == 'True' for row in learned], float)-baseline_success
            interval = np.quantile(difference[resamples].mean(axis=1), [.05/6, 1-.05/6])
            intervals.append(dict(seed=model['seed'], difference=float(difference.mean()), lower=float(interval[0]), upper=float(interval[1])))
            lines.append(f"| {model['seed']} | {difference.mean():.2%} | [{interval[0]:.2%}, {interval[1]:.2%}] |")
        supported = all(row['lower'] > 0. for row in intervals)
        lines += ['', '三个种子的校正区间均高于0，支持本自然测试分布下相对直接目标PID的任务完成率提高。' if supported else '预定统计判据未支持三个种子均相对直接目标PID提高任务完成率；保留全部结果。']
        p.write_json(p.STUDY/'06_报告'/'paired_intervals.json', dict(intervals=intervals, bootstrap_samples=20000, family_alpha=.05, comparisons=3, supported=supported))
    else:
        lines += ['开发未达到预先规定的可靠性门槛，因此没有启动正式三种子比较；这不能推断DQN路线选择本身不可行。']
    database = p.read_json(p.STUDY/'03_DQN训练'/'physical_environment_database'/'database_metadata.json')
    lines += ['', f"训练环境缓存由{database['unique_independent_training_states']}个独立训练初态、两个方向的{database['simulated_trajectories']}条真实物理闭环轨迹构成；实际传播{database['physically_active_hold_intervals']}个恒力矩区间。开发／验证／测试随机流独立。", '',
              '缓存两动作的结果不作为监督标签。DQN仅接收自身epsilon-greedy所选动作的奖励，优化器使用真实DQN时序差分损失（所有决策终止，gamma=0）。每次路线决策对应整条轨迹；重复缓存重放次数、唯一训练状态及状态动作对数逐次训练记录。', '',
              '结论范围限于固定单弧、静态目标、准确测量和理想力矩执行器。固定单禁区存在解析解，学习方法没有已知的规划最优性优势。不外推至多禁区、移动障碍、三轴姿态或在轨安全。', '',
              f"独立核验通过：{verification['passed']}；源码、冻结模型与轨迹校验和均保留。"]
    text = '\n'.join(lines)+'\n'
    output = p.STUDY/'06_报告'
    output.mkdir(exist_ok=True)
    (output/'研究报告.md').write_text(text, encoding='utf-8')
    (output/'研究报告.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>学习路线选择研究</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui;background:#f8fafc;color:#16243b}pre{white-space:pre-wrap;background:white;padding:32px;border-radius:12px}</style><body><pre>'+html.escape(text)+'</pre></body></html>', encoding='utf-8')
    return text
