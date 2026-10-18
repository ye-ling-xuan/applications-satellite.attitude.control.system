"""Summarize frozen continuous-control tests without selecting new models."""
from pathlib import Path
import html
import json
import numpy as np
import sys

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY/'代码'))
from continuous_study import protocol as p
from continuous_study.evaluation import summary, write_csv


def interval(matrix, seed):
    matrix = np.asarray(matrix, float)
    independent = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(20000)
    for start in range(0, 20000, 500):
        index = rng.integers(0, len(independent), (500, len(independent)))
        draws[start:start+500] = independent[index].mean(axis=1)
    tail = .05/12/2
    return dict(mean=float(matrix.mean()), low=float(np.quantile(draws, tail)),
                high=float(np.quantile(draws, 1-tail)), confidence=1-.05/12,
                independent_initial_states=matrix.shape[0], parameters_per_initial=matrix.shape[1],
                bootstrap_samples=20000)


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str, row))+' |' for row in rows]


def label(entry):
    if entry['family'] == 'pid':
        return '重新整定PID'
    if entry['family'] == 'model_compensated':
        return '传统模型补偿'
    value = 'PID加'+entry['algorithm']+'补偿，种子'+str(entry['seed'])
    if entry['stage'] == 'development_diagnostic':
        value += '（开发模型，非三种子结论）'
    if entry.get('selected_steps') == 0:
        value += '（初始零补偿，未获得更好学习检查点）'
    return value


def report():
    audit = p.read_json(STUDY/'05_独立核验'/'核验结果.json')
    if audit.get('passed') is not True:
        raise RuntimeError('full independent audit must pass first')
    frozen = p.read_json(STUDY/'04_独立测试'/'freeze.json')
    manifest = p.read_json(STUDY/'04_独立测试'/'test_manifest.json')
    rows = []
    for shard in manifest['entries']:
        rows.extend(p.read_json(STUDY/shard['data_path'])['rows'])
    grouped = []
    for entry in frozen['controllers']:
        for region in ('in_domain', 'ideal', 'out_of_domain'):
            selected = [r for r in rows if r['controller_id'] == entry['id'] and r['region'] == region]
            grouped.append(dict(controller_id=entry['id'], kind=entry['kind'], family=entry['family'],
                                algorithm=entry.get('algorithm'), stage=entry['stage'], seed=entry.get('seed'),
                                region=region, selected_steps=entry.get('selected_steps'), **summary(selected)))
    comparisons = []
    for kind in p.KINDS:
        baseline = {(r['sample_id'], r['parameter_s']): r for r in rows
                    if r['controller_id'] == f'{kind}_pid' and r['region'] == 'in_domain'}
        formal = [e for e in frozen['controllers'] if e['kind'] == kind and e['stage'] == 'formal']
        each = []
        both_reliable = len(formal) == 3 and all(r['success'] for r in baseline.values())
        for entry in formal:
            learned = {(r['sample_id'], r['parameter_s']): r for r in rows
                       if r['controller_id'] == entry['id'] and r['region'] == 'in_domain'}
            assert learned.keys() == baseline.keys()
            safe = all(r['success'] for r in learned.values())
            both_reliable = both_reliable and safe
            cost_matrix = [[learned[(i, v)]['actual_cost']-baseline[(i, v)]['actual_cost']
                            for v in p.IN_DOMAIN] for i in range(p.COUNTS['test'])]
            success_matrix = [[float(learned[(i, v)]['success'])-float(baseline[(i, v)]['success'])
                               for v in p.IN_DOMAIN] for i in range(p.COUNTS['test'])]
            cost_ci = interval(cost_matrix, 151000+entry['seed']+1000*p.KINDS.index(kind))
            success_ci = interval(success_matrix, 152000+entry['seed']+1000*p.KINDS.index(kind))
            reduction = 1-np.mean([r['actual_cost'] for r in learned.values()])/np.mean([r['actual_cost'] for r in baseline.values()])
            each.append(dict(seed=entry['seed'], selected_steps=entry['selected_steps'],
                             all_learned_pass=bool(safe), cost_difference=cost_ci,
                             success_difference=success_ci, mean_cost_reduction=float(reduction),
                             physical_failures=sum(r['failed'] for r in learned.values())))
        cost_supported = bool(both_reliable and all(e['cost_difference']['high'] < 0 for e in each))
        reliable_improvement = bool(len(each) == 3 and all(e['success_difference']['low'] > 0
                                  and e['physical_failures'] == 0 for e in each))
        mean_reduction = float(np.mean([e['mean_cost_reduction'] for e in each])) if each else None
        comparisons.append(dict(kind=kind, algorithm=formal[0]['algorithm'] if formal else None,
                                formal_models=len(formal), all_in_domain_pass=bool(both_reliable),
                                cost_advantage_supported=cost_supported,
                                emphasized_cost_advantage=bool(cost_supported and mean_reduction >= .05),
                                success_advantage_supported=reliable_improvement,
                                mean_cost_reduction=mean_reduction, formal_seeds=each))
    output = STUDY/'06_结果与说明'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'分组汇总.csv', grouped)
    p.write_json(output/'主要比较与统计区间.json', comparisons)
    runs = [p.read_json(path) for path in (STUDY/'03_训练记录').rglob('training_metadata.json')]
    completed = [r for r in runs if r['status'] == 'complete']
    lines = ['# 修正训练后的连续控制多算法比较', '',
             '本研究分别比较一阶滞后和纯延迟下的连续力矩控制。SAC、TD3、PPO使用相同观测、相同连续补偿幅度、相同训练交互预算；不把相同步数称为相同算力。开发选择和正式三种子验证分别报告。', '',
             '主要验收：从8秒起直到22秒的全部采样点，角度误差≤1°、角速度≤0.5°/秒。实际力矩平方积分单位为(N·m)²·s；失败情况下的较低代价不作为收益。', '',
             f'实际完成{len(completed)}次训练，累计{sum(r["actual_steps"] for r in runs):,}交互步；独立测试{len(rows):,}回合，{p.COUNTS["test"]}个独立初态跨参数和控制器复用。', '',
             '## 修正了什么', '',
             '失败后保持固定440步计时并持续惩罚，不允许提前结束少受罚；学习补偿从确定性零动作开始，初始行为与PID相同；奖励更重视控制代价与截止后约束；历史窗口覆盖0.6秒。初始化、奖励及检查点选择在训练前声明。实际选中步数为0时，只能解释为回退原PID，不能称为学习成功。', '',
             '## 理想单轴训练检查', '']
    sanity = [r for r in completed if r['stage'] == 'sanity']
    lines += table(['算法', '实际步数', '最佳验证通过数', '选中步数'], [
        [r['algorithm'], r['actual_steps'], f'{r["best"]["validation"]["passes"]}/{r["best"]["validation"]["count"]}',
         r['best']['steps']] for r in sanity])
    lines += ['', '这部分仅检查直接策略能否学会简单稳定控制，不用于证明相对PID的正式优势。', '', '## 开发与正式选择', '']
    development_rows = []
    for kind in p.KINDS:
        decision = p.read_json(STUDY/'03_训练记录'/'decisions'/f'{kind}.json')
        for run in decision['runs']:
            stats = run['best']['validation']
            development_rows.append([kind, run['algorithm'], run['reward_name'],
                                     f'{stats["passes"]}/{stats["count"]}', f'{stats["mean_actual_cost"]:.6f}',
                                     run['best']['steps']])
    lines += table(['执行器', '算法', '奖励', '开发验证通过数', '平均实际代价', '选中步数'], development_rows)
    for kind in p.KINDS:
        lines += ['', '## '+{'lag': '一阶滞后', 'delay': '纯延迟'}[kind], '']
        cells = []
        for entry in frozen['controllers']:
            if entry['kind'] != kind:
                continue
            for g in [x for x in grouped if x['controller_id'] == entry['id']]:
                cells.append([label(entry), {'in_domain': '范围内', 'ideal': '理想', 'out_of_domain': '范围外'}[g['region']],
                              f'{g["passes"]}/{g["count"]}', f'{g["mean_actual_cost"]:.6f}',
                              f'{g["mean_rms_error_deg"]:.3f}', g['physical_failures']])
        lines += table(['方法', '测试范围', '完成数', '平均实际代价', '全程RMS误差/°', '物理失败'], cells)
        comp = next(c for c in comparisons if c['kind'] == kind)
        if comp['emphasized_cost_advantage']:
            statement = f'三个正式种子全部完成范围内测试，校正区间均支持相对PID的代价降低，平均降低{comp["mean_cost_reduction"]:.1%}，达到预定5%强调门槛。传统模型补偿结果仍须同时查看，不据此声称优于所有传统控制。'
        elif comp['cost_advantage_supported']:
            statement = '校正区间支持相对PID的代价降低，但未达到预定5%强调门槛。'
        elif comp['success_advantage_supported']:
            statement = '三个正式种子的配对成功率区间均支持改善；代价和其他传统对照仍完整报告。'
        elif not comp['formal_models']:
            statement = '开发没有产生符合门槛的已学习检查点，未进入正式三种子训练。不能把该结果写成算法能力上限。'
        else:
            statement = '没有满足本轮预定的可靠性与控制代价优势判据，不能宣称强化学习更优。'
        lines += ['', '**'+statement+'**', '']
    lines += ['## 核验与结论范围', '',
              '每个响应分片均独立重算动力学、连续实际力矩积分、历史特征、确定性网络动作、抗饱和PID状态及全部验收指标；检查源码、参数、模型、场景及响应哈希。主比较按初态聚类，12个区间作Bonferroni校正。', '',
              '本研究仍是单轴实验。三轴需要针对完整耦合模型另行联合训练，不以逐轴迁移代替。若本轮没有优势，按计划进入三个有工程机制的后续场景，保留本轮阴性结果。', '',
              '复核：`python -X utf8 -B 运行研究.py verify`；重建报告：`python -X utf8 -B 运行研究.py report`。', '']
    _figures(output, frozen['controllers'], rows)
    lines += ['![独立测试逐参数成功率](逐参数成功率.png)', '',
              '![独立测试逐参数实际代价](逐参数实际代价.png)', '',
              '失败方法的实际代价仅供诊断，不与成功方法解释为同等任务下的节省。', '']
    text = '\n'.join(lines)
    (output/'研究结果说明.md').write_text(text, encoding='utf-8')
    sys.path.insert(0, str(STUDY.parent/'新增研究汇总'))
    from 生成研究总览 import render
    (output/'结果总览.html').write_text(render(lines), encoding='utf-8')
    result = dict(completed_runs=len(completed), actual_training_steps=sum(r['actual_steps'] for r in runs),
                  formal_runs=sum(r['stage'] == 'formal' for r in completed),
                  test_records=len(rows), independent_initial_states=p.COUNTS['test'], comparisons=comparisons,
                  independent_verification_passed=True,
                  practical_advantage=any(c['emphasized_cost_advantage'] or c['success_advantage_supported'] for c in comparisons))
    p.write_json(output/'结果摘要.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _figures(output, entries, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    for metric, filename, y_label in [('success', '逐参数成功率.png', '持续停稳通过率'),
                                     ('actual_cost', '逐参数实际代价.png', '平均实际力矩平方积分 / (N·m)²·s')]:
        fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
        for ax, kind in zip(axes, p.KINDS):
            formal = any(e['stage'] == 'formal' and e['kind'] == kind for e in entries)
            for entry in entries:
                if entry['kind'] != kind or (formal and entry['stage'] == 'development_diagnostic'):
                    continue
                selected = [r for r in rows if r['controller_id'] == entry['id']]
                params = sorted({r['parameter_s'] for r in selected})
                values = [np.mean([r[metric] for r in selected if r['parameter_s'] == v]) for v in params]
                ax.plot(params, values, marker='o', label=label(entry))
            ax.axvline(.30, color='gray', linestyle='--', linewidth=1)
            ax.set(title={'lag': '一阶滞后', 'delay': '纯延迟'}[kind], xlabel='时间常数或延迟 / s', ylabel=y_label)
            if metric == 'success':
                ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.savefig(output/filename, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    report()
