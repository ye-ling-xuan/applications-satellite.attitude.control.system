"""Report every retained training attempt, including the zero-action fallback."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'代码'))
from continuous_study import protocol as p


def generate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    paths = sorted((ROOT/'03_训练记录').rglob('training_metadata.json'))
    runs = [p.read_json(path) for path in paths if 'source_snapshot' not in path.parts]
    output = ROOT/'06_结果与说明'
    output.mkdir(exist_ok=True)
    lines = ['# 三种算法究竟训练到了什么程度', '',
             '“训练程序运行完了”和“控制任务已经学会了”是两件事。应看独立验证表现、检查点和不同种子，不能只报交互步数。', '',
             '|阶段|执行器|算法|奖励|种子|实际步数|选中步数|最佳通过数|最后通过数|',
             '|---|---|---|---|---|---|---|---|---|']
    for run in runs:
        best, last = run.get('best'), run.get('checkpoints', [])
        if best is None or not last:
            continue
        b, end = best['validation'], last[-1]['validation']
        lines.append('|'+ '|'.join(map(str, [run['stage'], run['kind'], run['algorithm'], run['reward_name'],
                     run['seed'], run['actual_steps'], best['steps'],
                     f'{b["passes"]}/{b["count"]}', f'{end["passes"]}/{end["count"]}']))+'|')
    zero = [r for r in runs if r.get('best') and r['mode'] == 'residual' and r['best']['steps'] == 0]
    sanity = [r for r in runs if r['stage'] == 'sanity' and r['status'] == 'complete']
    lines += ['', f'共{len(zero)}次补偿训练选中了第0步。第0步时网络补偿为零，行为就是原PID；它说明保留原控制器比已保存的学习检查点更合适，不能写成强化学习已经达到PID性能。', '']
    if sanity and all(r['best']['validation']['passes'] == 0 for r in sanity):
        lines += ['三个算法在本轮理想单轴检查中都未达到严格的持续停稳要求。这说明本轮观测、奖励、超参数和训练预算的组合仍不足，不能将阴性结果归结为这些算法理论上不适合卫星控制。', '']
    lines += ['SAC、TD3和PPO的输出均为连续补偿力矩。这里没有把顺时针/逆时针等简单方向选择包装成学习控制。', '',
              '全部失败尝试保留。新的工程场景另立假设、数据划分和验收，不能反复更改本轮测试条件直到出现预期结论。', '']
    panels = [('ideal', 'A', '理想单轴检查'), ('lag', 'A', '滞后，奖励A'),
              ('delay', 'A', '延迟，奖励A'), ('lag', 'B', '滞后，奖励B'), ('delay', 'B', '延迟，奖励B')]
    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 3.1*len(panels)), constrained_layout=True)
    for axis, (kind, reward, title) in zip(axes, panels):
        selected = [r for r in runs if r['kind'] == kind and r['reward_name'] == reward and r['stage'] != 'formal']
        for run in selected:
            points = run['checkpoints']
            steps = [c['steps'] for c in points]
            axis[0].plot(steps, [c['validation']['success_rate'] for c in points], marker='o', label=run['algorithm'])
            axis[1].plot(steps, [c['validation']['mean_actual_cost'] for c in points], marker='o', label=run['algorithm'])
        axis[0].set(title=title+'：验证通过率', xlabel='实际交互步数', ylim=(-.03, 1.03))
        axis[1].set(title=title+'：实际力矩平方积分', xlabel='实际交互步数', ylabel='(N·m)²·s')
        for ax in axis:
            ax.grid(alpha=.2)
            if selected:
                ax.legend()
    image = output/'全部开发训练曲线.png'
    fig.savefig(image, dpi=130)
    plt.close(fig)
    lines += ['![全部训练检查点](全部开发训练曲线.png)', '',
              '右图必须结合左图阅读：未完成任务时，代价低不能当作收益。', '']
    (output/'训练效果与局限.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(runs=len(runs), zero_step_residual_fallbacks=len(zero), sanity_runs=len(sanity))


if __name__ == '__main__':
    print(json.dumps(generate(), ensure_ascii=False))
