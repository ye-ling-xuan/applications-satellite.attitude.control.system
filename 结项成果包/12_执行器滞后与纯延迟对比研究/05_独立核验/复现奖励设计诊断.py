"""Constructed reward counterexample; no training or assessment responses read.

Run with D:/Anaconda2024.06-1/python.exe. This standalone diagnostic imports no
production plant, controller, reward function or training runtime. For two
constant commands it uses the continuous closed-form lag response, independently
reimplements the declared reward, and checks source hashes against the immutable
source_snapshot captured when the current study's training began.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal_assignment(source, name, keyword=None):
    def literal(node):
        if isinstance(node, ast.Dict):
            return {literal(key): literal(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
            return {item.arg: literal(item.value) for item in node.keywords}
        return ast.literal_eval(node)

    tree = ast.parse(Path(source).read_text(encoding='utf-8'))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                     for target in statement.targets):
            value = statement.value
            if keyword is not None:
                if not isinstance(value, ast.Call):
                    raise ValueError('expected a literal dict(...) declaration')
                value = next(item.value for item in value.keywords if item.arg == keyword)
            return literal(value)
    raise ValueError('missing literal declaration: '+name)


def independent_reward(theta, omega, end_time, actual_cost, new_failure,
                       weights, dt, deadline, limit):
    if new_failure:
        return float(np.float32(-1000.))
    angle, rate = abs(theta), abs(omega)
    angle_ratio, rate_ratio = angle/math.radians(1.), rate/math.radians(.5)
    outside = angle_ratio > 1. or rate_ratio > 1.
    violation = (float(outside)+min(max(angle_ratio-1., 0.)**2, 25.)
                 +min(max(rate_ratio-1., 0.)**2, 25.))
    cost = (weights['error_weight']*min((angle/math.radians(5.))**2, 100.)
            +weights['rate_weight']*min((rate/math.radians(5.))**2, 100.)
            +weights['torque_weight']*actual_cost/(dt*limit**2)
            +(end_time >= deadline-1e-10)*weights['violation_weight']*violation)
    return float(np.float32(-dt*cost))


def source_seal():
    # Read only immutable source snapshots, not episode logs, validation or test
    # results. The snapshot was made before the first optimizer update.
    snapshots = sorted((STUDY/'03_SAC训练'/'development').glob('*/*/reward_*/seed_*/source_snapshot'))
    if not snapshots:
        raise RuntimeError('a training-time immutable source snapshot is required')
    snapshot = snapshots[0]
    expected = {'代码/'+path.relative_to(snapshot).as_posix(): digest(path)
                for path in sorted(snapshot.rglob('*.py')) if '__pycache__' not in path.parts}
    current = {path.relative_to(STUDY).as_posix(): digest(path)
               for path in sorted((STUDY/'代码').rglob('*.py')) if '__pycache__' not in path.parts}
    if expected != current:
        raise RuntimeError('current source differs from the immutable training-time snapshot')
    return snapshot.relative_to(STUDY).as_posix(), current


def main():
    snapshot, sources = source_seal()
    protocol = STUDY/'代码'/'actuator_study'/'protocol.py'
    learning = protocol.with_name('learning.py')
    constants = {name: literal_assignment(protocol, name) for name in
                 ('DT', 'DURATION', 'DEADLINE', 'LIMIT', 'INERTIA', 'FAIL_ANGLE', 'FAIL_RATE')}
    weights = literal_assignment(learning, 'REWARDS')
    gamma = literal_assignment(learning, 'SAC_SETTINGS', 'gamma')
    dt, duration, deadline, limit, inertia = (constants[name] for name in
                                             ('DT', 'DURATION', 'DEADLINE', 'LIMIT', 'INERTIA'))
    initial_angle, time_constant = math.radians(60.), .05
    records, artifacts = [], {}
    for name, command in [('保持60度_零命令', 0.), ('向外饱和_提前失效', limit)]:
        times, angles, rates = [0.], [initial_angle], [0.]
        commands, actual_costs, tau_starts, tau_ends = [], [], [], []
        rewards = {reward: [] for reward in weights}

        def cost_primitive(t):
            one = -math.expm1(-t/time_constant)
            two = -math.expm1(-2.*t/time_constant)
            return command**2*(t-2.*time_constant*one+.5*time_constant*two)

        failed = False
        for k in range(round(duration/dt)):
            start, end = k*dt, (k+1)*dt
            one = -math.expm1(-end/time_constant)
            # Independent continuous constant-command solution from t=0:
            # tau=u(1-exp(-t/T)), omega=u/J*(t-T(1-exp(-t/T))).
            omega = command/inertia*(end-time_constant*one)
            theta = initial_angle+command/inertia*(.5*end**2-time_constant*end+time_constant**2*one)
            actual_cost = max(0., cost_primitive(end)-cost_primitive(start))
            failed = abs(theta) > math.radians(constants['FAIL_ANGLE']) or abs(omega) > math.radians(constants['FAIL_RATE'])
            commands.append(command)
            actual_costs.append(actual_cost)
            tau_starts.append(command*(-math.expm1(-start/time_constant)))
            tau_ends.append(command*one)
            times.append(end)
            angles.append(theta)
            rates.append(omega)
            for reward, parameters in weights.items():
                rewards[reward].append(independent_reward(theta, omega, end, actual_cost, failed,
                                                           parameters, dt, deadline, limit))
            if failed:
                break
        trace_name = name+'.npz'
        trace_path = OUTPUT/trace_name
        np.savez_compressed(trace_path, time_s=np.array(times), theta_rad=np.array(angles),
                            omega_rad_s=np.array(rates), command_Nm=np.array(commands),
                            tau_start_Nm=np.array(tau_starts), tau_end_Nm=np.array(tau_ends),
                            actual_cost_interval=np.array(actual_costs),
                            **{'reward_'+reward: np.array(values, dtype=np.float32)
                               for reward, values in rewards.items()})
        artifacts[trace_name] = digest(trace_path)
        records.append(dict(policy=name, constant_command_Nm=command, steps=len(commands),
                            failed=bool(failed), terminal_time_s=times[-1],
                            final_angle_deg=math.degrees(angles[-1]),
                            final_rate_deg_s=math.degrees(rates[-1]),
                            actual_torque_square_integral=sum(actual_costs), trace_file=trace_name,
                            returns={reward: dict(discounted_return=sum(gamma**k*value for k, value in enumerate(values)),
                                                  undiscounted_return=sum(values))
                                     for reward, values in rewards.items()}))
    bounds = {}
    for reward, parameters in weights.items():
        state_bound = 100.*parameters['error_weight']+100.*parameters['rate_weight']+parameters['torque_weight']
        before = dt*state_bound
        after = dt*(state_bound+51.*parameters['violation_weight'])
        worst = sum(gamma**k*(after if (k+1)*dt >= deadline-1e-10 else before)
                    for k in range(round(duration/dt)))
        bounds[reward] = dict(max_ordinary_step_penalty_before_deadline=before,
                              max_ordinary_step_penalty_after_deadline=after,
                              discounted_full_horizon_ordinary_penalty_upper_bound=worst)
    declaration = STUDY/'03_SAC训练'/'predeclared_learning.json'
    value = dict(diagnostic='constructed reward escape counterexample; not a controller evaluation',
                 created_utc=datetime.now(timezone.utc).isoformat(), training_performed=False,
                 heldout_responses_used=False, source_modified=False,
                 initial_state_constructed=True, initial_angle_deg=60., initial_rate_deg_s=0.,
                 kind='lag', parameter_s=time_constant, gamma=gamma, constants=constants,
                 reward_weights=weights, failure_reward=-1000., reward_dtype='float32',
                 source_snapshot=snapshot, source_sha256=sources,
                 reward_source_sha256=digest(learning),
                 learning_declaration_sha256=digest(declaration),
                 independent_script_sha256=digest(__file__), trajectories=records,
                 ordinary_penalty_bounds=bounds, artifact_sha256=artifacts,
                 limitations=['The example proves a reward preference exists, not that a trained policy intentionally exploits it.',
                              'Neither command is an optimized successful controller; successful control can achieve a better return.',
                              'The example does not prove SAC infeasible or generally inferior to PID.',
                              'All planned configurations, failed attempts and original results remain unchanged.',
                              'Any changed reward, initialization, exploration or absorbing failure state requires a separate declared study.'])
    (OUTPUT/'奖励设计诊断.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# 奖励设计诊断：提前失效的构造反例', '',
             '本诊断没有运行训练，没有读取独立测试集响应，也没有修改代码目录或原计划结果。初态 +60°、角速度 0、滞后时间常数 T=0.05 s 是专门构造的反例。两条固定命令轨迹用独立的连续解析解计算；奖励按照训练前声明独立重写，并保留原 float32 舍入。', '',
             f'γ={gamma}；采样 {dt} s；最长 {duration} s；限幅 {limit} Nm；单轴惯量 {inertia} kg·m²；首次采样角度超过170°或角速度超过300°/s时终止，失败奖励替代该步普通奖励为−1000。', '',
             '| 奖励 | 固定命令 | 步数 | 终止时间/s | 物理失败 | 折扣回报 | 未折扣回报 |',
             '|---|---:|---:|---:|---|---:|---:|']
    for reward in weights:
        for record in records:
            outcome = record['returns'][reward]
            lines.append(f"| {reward} | {record['constant_command_Nm']:+.1f} Nm | {record['steps']} | {record['terminal_time_s']:.2f} | {record['failed']} | {outcome['discounted_return']:.6f} | {outcome['undiscounted_return']:.6f} |")
    lines += ['', 'A、B两种奖励中，向外饱和并提前失效的回报都高于保持在60°不动。这表明−1000的有限终止惩罚可能低于继续承担状态与超时惩罚的代价，因此奖励存在逃避后续惩罚的激励。成功控制仍可取得更好回报；本反例比较的两个固定命令均不是优化后的成功控制器。', '',
              '这不是失败策略已主动利用该机制的因果证明。开发阶段未达到门槛，应表述为“在本配置、预算与开发规则下未成功”，并将该奖励激励列为可能机制。不能据此证明SAC不可行，也不能作为强化学习普遍不如PID的证据。', '',
              '普通奖励的状态平方截断上限和超限项上限给出以下保守界；这些是奖励函数的范围界，不保证某条物理轨迹能同时达到全部上限：', '',
              '| 奖励 | 截止前单步最大惩罚 | 截止后单步最大惩罚 | 22秒普通惩罚折扣和上界 |',
              '|---|---:|---:|---:|']
    for reward, bound in bounds.items():
        lines.append(f"| {reward} | {bound['max_ordinary_step_penalty_before_deadline']:.6f} | {bound['max_ordinary_step_penalty_after_deadline']:.6f} | {bound['discounted_full_horizon_ordinary_penalty_upper_bound']:.6f} |")
    lines += ['', '原实验结果、模型、失败尝试和预定门槛全部保留。本诊断不追溯更换奖励或重新挑选结果。若后续研究采用零残差初始化、更低探索、吸收失败状态或尾部惩罚，必须另立配置、记录预算与数据划分，并重新独立测试。', '',
              '复现：', '', '```powershell',
              "& 'D:/Anaconda2024.06-1/python.exe' './复现奖励设计诊断.py'", '```', '',
              '在本目录运行上述脚本。脚本会检查代码目录全部Python文件与训练启动时的不可变source_snapshot相同；不会导入生产物理、生产奖励、控制器或训练框架。JSON记录完整源码SHA256、奖励源码SHA256、训练前声明SHA256、脚本及两份轨迹校验和。', '',
              f'训练启动时源码快照：`{snapshot}`', '',
              f'奖励源码 SHA256：`{digest(learning)}`', '',
              f'独立脚本 SHA256：`{digest(__file__)}`', '']
    (OUTPUT/'奖励设计诊断.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(trajectories=records, ordinary_penalty_bounds=bounds,
                          source_matches_training_snapshot=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
