"""Paired, initial-state clustered analysis; unsuccessful control is not a saving."""
from __future__ import annotations
import html
import json
import numpy as np
from . import protocol as p
from .evaluation import summary, write_csv

LABEL = {'pid': '重新整定PID', 'model_compensated': '传统模型补偿',
         'direct': '直接SAC', 'residual': 'PID＋SAC补偿'}
KIND = {'lag': '一阶滞后', 'delay': '纯延迟'}
REGION = {'in_domain': '训练范围内', 'ideal': '理想执行器', 'out_of_domain': '训练范围外'}


def paired_interval(differences, seed, repetitions=20000, confidence=1-.05/12):
    values = np.asarray(differences, float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError('expected initial-state x parameter matrix')
    # Collapse parameters WITHIN each independent initial state before resampling.
    means = values.mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        count = min(500, repetitions-start)
        indices = rng.integers(0, len(means), size=(count, len(means)))
        samples[start:start+count] = means[indices].mean(axis=1)
    tail = (1-confidence)/2
    return dict(mean_difference=float(values.mean()), confidence=confidence,
                low=float(np.quantile(samples, tail)), high=float(np.quantile(samples, 1-tail)),
                independent_initial_states=len(means), parameters_per_initial=values.shape[1],
                repetitions=repetitions)


def _table(headers, rows):
    return ['| '+' | '.join(map(str, headers))+' |', '| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str, row))+' |' for row in rows]


def load_results():
    frozen = p.read_json(p.STUDY/'04_独立测试'/'freeze.json')
    manifest = p.read_json(p.STUDY/'04_独立测试'/'test_manifest.json')
    rows = []
    for shard in manifest['entries']:
        rows.extend(p.read_json(p.STUDY/shard['data_path'])['rows'])
    return frozen, manifest, rows


def compare(frozen, rows):
    comparisons = []
    for kind in p.KINDS:
        baseline_rows = [r for r in rows if r['controller_id'] == f'{kind}_pid' and r['region'] == 'in_domain']
        baseline = {(r['sample_id'], r['parameter_s']): r for r in baseline_rows}
        for method in p.METHODS:
            members = [c for c in frozen['controllers'] if c['kind'] == kind and c['family'] == method]
            formal = [c for c in members if c['stage'] == 'formal']
            intervals = []
            reliable = len(formal) == 3 and all(r['success'] for r in baseline_rows)
            costs = []
            for member in formal:
                selected = [r for r in rows if r['controller_id'] == member['id'] and r['region'] == 'in_domain']
                lookup = {(r['sample_id'], r['parameter_s']): r for r in selected}
                if lookup.keys() != baseline.keys():
                    raise ValueError('unmatched paired test scenarios')
                reliable = reliable and all(r['success'] for r in selected)
                matrix = [[lookup[(i, q)]['actual_cost']-baseline[(i, q)]['actual_cost']
                           for q in p.IN_DOMAIN] for i in range(256)]
                interval = paired_interval(matrix, 202610170901+member['seed']+1000*p.KINDS.index(kind)+10000*p.METHODS.index(method))
                intervals.append(dict(seed=member['seed'], **interval))
                costs.append(np.mean([r['actual_cost'] for r in selected]))
            reduction = 1-float(np.mean(costs))/np.mean([r['actual_cost'] for r in baseline_rows]) if costs else None
            supported = bool(reliable and all(c['high'] < 0 for c in intervals))
            comparisons.append(dict(kind=kind, method=method, formal_models=len(formal),
                                    all_in_domain_pass=bool(reliable), corrected_intervals=intervals,
                                    mean_cost_reduction_fraction=reduction, cost_advantage_supported=supported,
                                    emphasized_improvement=bool(supported and reduction >= .05),
                                    diagnostic_only=not formal))
    return comparisons


def report():
    output = p.STUDY/'06_结果与说明'
    output.mkdir(parents=True, exist_ok=True)
    verification = p.read_json(p.STUDY/'05_独立核验'/'核验结果.json')
    if not (verification.get('passed') or verification.get('all_passed')):
        raise RuntimeError('independent verification must pass before final report')
    frozen, manifest, rows = load_results()
    entries = frozen['controllers']
    grouped = []
    for entry in entries:
        for region in REGION:
            selected = [r for r in rows if r['controller_id'] == entry['id'] and r['region'] == region]
            grouped.append(dict(controller_id=entry['id'], kind=entry['kind'], family=entry['family'],
                                stage=entry['stage'], seed=entry.get('seed'), region=region, **summary(selected)))
    write_csv(output/'分组汇总.csv', grouped)
    comparisons = compare(frozen, rows)
    p.write_json(output/'主要比较与统计区间.json', comparisons)
    development = [p.read_json(p.STUDY/'03_SAC训练'/'development_decisions'/f'{k}_{m}.json')
                   for k in p.KINDS for m in p.METHODS]
    runs = [p.read_json(path) for path in (p.STUDY/'03_SAC训练').rglob('training_metadata.json')]
    complete = [r for r in runs if r['status'] == 'complete']
    steps = sum(r['actual_steps'] for r in runs)
    lines = ['# 执行器一阶滞后与纯延迟：研究结果', '',
             '本研究分别改变一阶滞后时间常数和纯延迟时间，在固定惯量、无外部扰动和准确测量下，比较固定PID、传统模型补偿、直接SAC和PID加SAC。真实执行器参数不向控制器公开。', '',
             f'已完成训练{len(complete)}份，其中正式训练{sum(r["stage"] == "formal" for r in complete)}份；累计实际交互步{steps:,}。独立测试产生{len(rows):,}条回合记录，使用256个独立初态；同一初态在不同参数和控制器中复用，不是额外独立样本。', '',
             '任务：8秒起直到22秒的每个采样点，角度误差不超过1°、角速度不超过0.5°/s。代价为0至22秒实际力矩平方积分，单位为(N·m)²·s，不是电能。失败控制器的低代价不能视为节省。', '',
             '## 开发门槛与实际执行', '']
    lines += _table(['场景', '方法', '选定奖励', '验证通过', '是否进入正式训练'], [
        [KIND[d['kind']], LABEL[d['method']], d['reward_name'],
         f'{d["validation"]["passes"]}/{d["validation"]["count"]}' if d['validation'] else '无完整候选',
         '是' if d['passed_gate'] else '否，保留开发结果'] for d in development])
    lines += ['', '开发未通过的组合只对选定候选进行诊断测试；它们不构成正式三个种子的算法结论。开发门槛是95%验证通过且无物理失败。', '',
              '## 固定控制器的独立测试', '']
    for kind in p.KINDS:
        lines += [f'### {KIND[kind]}', '']
        data = []
        for g in grouped:
            if g['kind'] != kind:
                continue
            suffix = f'种子{g["seed"]}' if g['seed'] is not None else ''
            if g['stage'] == 'development_diagnostic':
                suffix += '（开发诊断）'
            data.append([LABEL[g['family']]+suffix, REGION[g['region']],
                         f'{g["passes"]}/{g["count"]}（{g["success_rate"]:.1%}）',
                         f'{g["mean_actual_cost"]:.6f}', f'{g["mean_rms_error_deg"]:.3f}', g['physical_failures']])
        lines += _table(['方法', '测试范围', '通过率', '平均实际力矩代价', '全程RMS误差/°', '物理失败'], data)+['']
    lines += ['## 能支持什么结论', '']
    for c in comparisons:
        name = KIND[c['kind']]+'：'+LABEL[c['method']]
        if c['diagnostic_only']:
            conclusion = '开发门槛未通过，没有进行正式三种子训练；诊断结果不能证明该方法获得可靠优势，也不能推断该算法在其他预算下不可能成功。'
        elif c['emphasized_improvement']:
            conclusion = f'全部范围内测试成功，三个种子的校正区间均支持实际力矩代价降低，平均降低{c["mean_cost_reduction_fraction"]:.1%}。结论仅针对本任务与已测范围；响应和误差仍须同时查看。'
        elif c['cost_advantage_supported']:
            conclusion = f'校正区间支持代价降低，但平均幅度{c["mean_cost_reduction_fraction"]:.1%}未达到预定5%强调门槛。'
        else:
            conclusion = '未满足预定的同等可靠性与代价优势判据。'
        lines += [f'- **{name}：**{conclusion}']
    lines += ['', '主要比较共四组、每组三个训练种子，使用12个Bonferroni校正区间（单区间约99.583%）。以初态为单位自举，保留全部参数结果。训练范围外和传统补偿结果均完整报告；不据局部好结果改换主要结论。', '',
              '## 核验与复现', '',
              '独立核验逐区间重算动力学、实际力矩积分、控制动作、成功判据与回合指标，并检查冻结文件哈希及场景覆盖。详细记录见 ../05_独立核验/核验结果.json。', '',
              '从研究目录运行：`python -X utf8 -B 运行研究.py verify`；报告重建：`python -X utf8 -B 运行研究.py report`。测试冻结后，训练与参数选择入口拒绝继续修改本轮研究。', '',
              '现阶段为单轴研究。单轴结果不能直接写成完成三轴耦合比较；三轴需另有完整模型、适配、测试和核验记录。', '']
    _figures(output, entries, rows)
    lines += ['## 图表', '', '![逐参数成功率](逐参数成功率.png)', '', '![逐参数控制代价](逐参数控制代价.png)', '',
              '图中的失败方法仍显示实际产生的代价，仅供诊断；不与成功方法解释为同等任务下的节省。', '']
    markdown = '\n'.join(lines)
    (output/'研究结果说明.md').write_text(markdown, encoding='utf-8')
    # Minimal semantic rendering, without a dependency on a Markdown package.
    rendered, in_table = [], False
    for line in lines:
        if line.startswith('|'):
            if set(line.replace('|', '').replace('-', '').replace(' ', '')) == set():
                continue
            if not in_table:
                rendered.append('<table>'); in_table = True
            rendered.append('<tr>'+''.join('<td>'+html.escape(cell.strip())+'</td>' for cell in line.strip('|').split('|'))+'</tr>')
            continue
        if in_table:
            rendered.append('</table>'); in_table = False
        if line.startswith('#'):
            level = len(line)-len(line.lstrip('#'))
            rendered.append(f'<h{level}>'+html.escape(line.lstrip('# '))+f'</h{level}>')
        elif line.startswith('!['):
            path = line.split('](', 1)[1].rstrip(')')
            rendered.append(f'<img src="{html.escape(path)}" alt="对比图">')
        elif line:
            rendered.append('<p>'+html.escape(line).replace('**', '')+'</p>')
    if in_table:
        rendered.append('</table>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>执行器研究结果</title><style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#17212d}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #d2dbe3;padding:8px}tr:first-child{background:#eaf0f6;font-weight:bold}img{max-width:100%;height:auto}h1,h2,h3{line-height:1.4}</style><body>'+''.join(rendered)+'</body></html>'
    (output/'结果总览.html').write_text(document, encoding='utf-8')
    p.write_json(output/'结果摘要.json', dict(completed_runs=len(complete), formal_runs=sum(r['stage'] == 'formal' for r in complete),
                                            actual_training_steps=steps, test_records=len(rows), comparisons=comparisons,
                                            independent_initial_states=256, independent_verification_passed=True))
    print(markdown[:500], flush=True)
    return comparisons


def _figures(output, entries, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    for metric, filename, ylabel in [('success', '逐参数成功率.png', '通过率'),
                                     ('actual_cost', '逐参数控制代价.png', '平均实际力矩平方积分 / (N·m)²·s')]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, kind in zip(axes, p.KINDS):
            for family in LABEL:
                selected = [r for r in rows if r['kind'] == kind and r['family'] == family]
                if not selected:
                    continue
                parameters = sorted({r['parameter_s'] for r in selected})
                values = [np.mean([r[metric] for r in selected if r['parameter_s'] == q]) for q in parameters]
                diagnostic = any(r['stage'] == 'development_diagnostic' for r in selected)
                ax.plot(parameters, values, marker='o', label=LABEL[family]+('（开发诊断）' if diagnostic else ''))
            ax.axvline(.30, color='gray', linestyle='--', linewidth=1)
            ax.set(title=KIND[kind], xlabel='时间常数T / s' if kind == 'lag' else '纯延迟d / s', ylabel=ylabel)
            if metric == 'success':
                ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.25); ax.legend(fontsize=9)
        fig.savefig(output/filename, dpi=160)
        plt.close(fig)
