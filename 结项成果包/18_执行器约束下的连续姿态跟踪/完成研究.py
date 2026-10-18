"""Finalize the declared extra development without altering frozen sources."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

# NumPy's installed BLAS runtime initializes lazily.  Initialize it before
# Torch to avoid the demonstrated Windows duplicate-OpenMP initialization.
_runtime_warmup=float(np.trace(np.eye(100)@np.eye(100)))
import scipy.linalg
import scipy.integrate

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'代码'))
import study as b
import revision as r

_original_factory=b.controller_from_entry
_mpc_instances=[]


def factory(entry):
    if entry['family']=='MPC_SLSQP':
        controller=r.ExactMPC();_mpc_instances.append(controller);return controller
    if entry.get('revision'):
        from stable_baselines3 import PPO
        if b.sha256(ROOT/entry['model_path'])!=entry['model_sha256']:raise RuntimeError('model changed')
        return r.Controller(PPO.load(ROOT/entry['model_path'],device='cpu'),entry['parameters'])
    return _original_factory(entry)


def freeze():
    path=ROOT/'04_独立测试'/'freeze.json'
    if path.exists():raise RuntimeError('freeze already exists')
    decision=b.read_json(ROOT/'03_追加训练'/'decision.json')
    entries=[]
    for family in ('PID','PID_feedforward_preview'):
        metadata=ROOT/'02_传统控制'/f'{family}.json';data=b.read_json(metadata)
        if data['source_sha256']!=b.source_hash():raise RuntimeError('traditional production source differs')
        entries.append(dict(name=family,family=family,parameters=data['parameters'],metadata_path=b.relative(metadata),metadata_sha256=b.sha256(metadata)))
    entries.append(dict(name='MPC_SLSQP',family='MPC_SLSQP'))
    for algorithm in b.ALGORITHMS:
        metadata=ROOT/'03_训练记录'/'development'/algorithm/f'seed_{b.DEVELOPMENT_SEED}'/'training_metadata.json';data=b.read_json(metadata)
        if data['status']!='complete' or data['source_sha256']!=b.source_hash():raise RuntimeError('initial development incomplete')
        best=data['best']
        entries.append(dict(name=f'{algorithm}_initial_development_only',family='learned',algorithm=algorithm,seed=b.DEVELOPMENT_SEED,
                            model_path=best['model_path'],model_sha256=best['model_sha256'],selected_steps=best['steps'],parameters=data['base_parameters'],diagnostic=True))
    runs=[('formal',seed) for seed in b.FORMAL_SEEDS] if decision['formal_eligible'] else [('development',r.SEED)]
    for stage,seed in runs:
        metadata=ROOT/'03_追加训练'/stage/'PPO'/f'seed_{seed}'/'training_metadata.json';data=b.read_json(metadata)
        if data['status']!='complete' or data['actual_steps']!=r.STEPS or data['source_sha256']!=r.source_hash():raise RuntimeError('revision training incomplete')
        best=data['best']
        entries.append(dict(name=f'PPO_symmetry_seed_{seed}',family='learned',algorithm='PPO',seed=seed,revision=True,
                            diagnostic=stage!='formal',model_path=best['model_path'],model_sha256=best['model_sha256'],selected_steps=best['steps'],
                            parameters=data['base_parameters'],metadata_path=b.relative(metadata),metadata_sha256=b.sha256(metadata)))
    frozen=dict(controllers=entries,source_sha256=b.source_hash(),revision_sha256=r.source_hash(),
                declaration_sha256=b.sha256(ROOT/'01_实验方案'/'预声明.json'),revision_declaration_sha256=b.sha256(ROOT/'01_实验方案'/'开发修正追加声明.json'),
                test_scenarios_sha256=b.sha256(ROOT/'01_实验方案'/'test.json'),formal_eligible=decision['formal_eligible'],test_responses_used_for_selection=False)
    b.write_json(path,frozen)


def check():
    frozen=b.read_json(ROOT/'04_独立测试'/'freeze.json')
    if frozen['source_sha256']!=b.source_hash() or frozen['revision_sha256']!=r.source_hash():raise RuntimeError('frozen source changed')
    if frozen['revision_declaration_sha256']!=b.sha256(ROOT/'01_实验方案'/'开发修正追加声明.json'):raise RuntimeError('revision declaration changed')


def test():
    check();b.controller_from_entry=factory;b.test()
    if _mpc_instances:
        controller=_mpc_instances[-1]
        b.write_json(ROOT/'04_独立测试'/'MPC求解诊断.json',dict(
            method='SLSQP same convex QP',max_constraint_residual=controller.max_solver_residual,
            optimizer_unsuccessful_calls=controller.optimization_failures,max_iterations_observed=controller.maximum_iterations,
            total_calls=128*b.N_STEPS,physical_command_constraints_always_enforced=True))


def verify():
    check();b.controller_from_entry=factory
    result=b.verify()
    result.update(revision_source_verified=True,base_source_verified=True,
                  mpc_numerics=b.read_json(ROOT/'04_独立测试'/'MPC求解诊断.json'))
    b.write_json(ROOT/'05_独立核验'/'verification.json',result)
    return result


def report():
    check()
    audit=b.read_json(ROOT/'05_独立核验'/'verification.json')
    if not audit['passed']:raise RuntimeError('audit failed')
    frozen=b.read_json(ROOT/'04_独立测试'/'freeze.json')
    responses={entry['name']:b.read_json(ROOT/'04_独立测试'/entry['name']/'results.json') for entry in frozen['controllers']}
    formal=[entry for entry in frozen['controllers'] if entry.get('revision') and not entry.get('diagnostic')]
    intervals={entry['name']:{name:b.paired(responses[name]['rows'],responses[entry['name']]['rows'])
               for name in ('PID','PID_feedforward_preview','MPC_SLSQP')} for entry in formal}
    advantage=bool(len(formal)==3 and all(entry['selected_steps']>0 and intervals[entry['name']]['PID']['difference_interval'][0]>0
                   and responses[entry['name']]['summary']['physical_failures']<=responses['PID']['summary']['physical_failures'] for entry in formal))
    output=ROOT/'06_研究报告';output.mkdir(parents=True,exist_ok=True)
    b.write_json(output/'paired_intervals.json',dict(advantage_vs_PID=advantage,comparisons=intervals,
                 primary_confidence=1.-.05/3,secondary_intervals='exploratory; same per-reference correction, no global superiority claim'))
    lines=['# 执行器约束下的连续姿态跟踪结果','',
           '本研究在单轴连续扫描中比较相同幅值和变化速率受限执行器下的控制方法，所有方法可使用相同的测量和未来3秒已知参考指令。AI输出连续力矩补偿，基本反馈由已经加入模型前馈与预览的同一PID承担。','',
           '| 方法 | 平均角度RMS误差（°） | 平均力矩平方积分 | 物理失败 |','|---|---:|---:|---:|']
    for name,data in responses.items():
        stats=data['summary'];lines.append(f'| {name} | {stats["mean_rms_error_deg"]:.5f} | {stats["mean_actual_cost"]:.6f} | {stats["physical_failures"]}/{stats["count"]} |')
    lines+=['','原TD3/PPO初轮开发的最优模型均为0步，表中标注initial_development_only，它们只是前馈PID回退，不属于学习成功。追加PPO使用物理反号对称性、较小补偿和低学习率后，三个正式种子均选择了大于0步的检查点。','',
            '| 正式PPO种子 | 相对同一前馈PID的RMS降低 | 差值98.333%区间（°） |','|---|---:|---|']
    for entry in formal:
        item=intervals[entry['name']]['PID_feedforward_preview'];ci=item['difference_interval']
        lines.append(f'| {entry["seed"]} | {100*item["relative_reduction"]:.3f}% | [{ci[0]:.6f}, {ci[1]:.6f}] |')
    lines+=['','比较普通PID只能表明整个组合方法的效果。相对同一前馈PID的差异，才反映学习模块自身增加了什么。平均RMS下降不等于所有轨迹都变好，应同时查看区间、均方综合目标、峰值误差及力矩代价。','',
            'MPC是固定3秒预测时域和固定权重的凸二次规划SLSQP实现。虽核验求解残差，但未全面优化预测时域、终端代价和权重，不能由该实现的结果推断AI优于充分优化MPC。原ADMM结果仅作求解诊断，未充当正式优势依据。','',
            f'全量核验：{audit["episodes"]}回合、{audit["physical_intervals"]}个区间的物理、指令和指标回放通过；独立ODE积分代表区间核验通过。力矩平方积分不是实测电能。']
    (output/'研究报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(advantage_vs_PID=advantage,summary={name:data['summary'] for name,data in responses.items()},audit=audit)
    diagnostic=ROOT/'06_研究报告'/'开发过程说明.md'
    diagnostic.write_text(
        '# 开发过程与适用范围\n\n'
        '原TD3、PPO各131072步开发的最优综合验证目标均回退到0步，保留为开发诊断。'
        '随后在测试前追加低学习率、小补偿、物理反号对称性的PPO配置，原始数据和代码未修改。'
        '追加声明、验证曲线和所有模型均保留。若某正式种子的最优检查点仍为0步，它不能计作学习改进。\n\n'
        'AI基底是已经加入模型前馈和参考预览的PID；关闭学习补偿可严格恢复这一基底。'
        '因此只有相对PID加前馈也改善时，才能说学习增加了这类基底尚未实现的作用。'
        '仅优于普通PID时，实际价值应限定，不能把模型前馈本身作为AI贡献。\n\n'
        '原ADMM的验证残差较大，保留为数值求解诊断。正式传统约束对照使用同一凸二次规划的SLSQP解，'
        '求解残差、未成功调用次数和物理约束核验见独立测试/MPC求解诊断.json。'
        '即使AI优于普通PID，也不能由此推断其优于充分求解的模型预测控制。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','test','verify','report','finish']);args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='test':test()
    elif args.command=='verify':print(json.dumps(verify(),ensure_ascii=False),flush=True)
    elif args.command=='report':report()
    else:freeze();test();verify();report()


if __name__=='__main__':main()
