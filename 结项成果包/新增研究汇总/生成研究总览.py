"""只读汇总执行器滞后、纯延迟与三轴迁移的核验后结果，不训练、不重新选择控制器。"""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S12 = ROOT / '12_执行器滞后与纯延迟对比研究'
S13 = ROOT / '13_执行器误差下的三轴耦合迁移验证'
SOURCES = []


def read_json(path):
    SOURCES.append(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    SOURCES.append(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def pct(value):
    return f'{float(value):.2%}'


def method12(row):
    label = {'pid': '调参后的PID', 'model_compensated': '传统模型补偿',
             'direct': '直接SAC', 'residual': 'PID加SAC补偿'}[row['family']]
    if row['seed']:
        label += '（种子' + row['seed'] + '）'
    if row['stage'] == 'development_diagnostic':
        label += '，开发未达标（未进行正式三种子训练）'
    return label


def inline(line):
    line = html.escape(line)
    line = re.sub(r'`([^`]+)`', r'<code>\1</code>', line)
    line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', line)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', line)


def render(lines):
    output, in_table = [], False
    for line in lines:
        if line.startswith('|'):
            if set(line.replace('|', '').replace('-', '').replace(' ', '')) == set():
                continue
            if not in_table:
                output.append('<div class="table-scroll"><table>')
                in_table = True
            output.append('<tr>' + ''.join('<td>' + inline(cell.strip()) + '</td>'
                                          for cell in line.strip('|').split('|')) + '</tr>')
            continue
        if in_table:
            output.append('</table></div>')
            in_table = False
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            output.append(f'<h{level}>' + inline(line.lstrip('# ')) + f'</h{level}>')
        elif line.startswith('!['):
            alt, path = line[2:].split('](', 1)
            output.append(f'<figure><img src="{html.escape(path[:-1])}" alt="{html.escape(alt)}"></figure>')
        elif line:
            output.append('<p>' + inline(line) + '</p>')
    if in_table:
        output.append('</table></div>')
    style = '''body{font:17px/1.85 "Microsoft YaHei",sans-serif;color:#172b3a;background:#f4f7fa;margin:0}
    main{max-width:1080px;margin:32px auto;background:white;padding:36px 44px;border-radius:12px}
    h1{font-size:29px;line-height:1.45}h2{font-size:23px;margin-top:36px;color:#15556b}
    h3{font-size:19px}a{color:#0b6781}strong{color:#14475c}code{background:#edf2f5;padding:2px 4px}
    .table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}
    td{border:1px solid #d8e2e8;padding:8px 10px}tr:first-child{background:#e9f1f5;font-weight:bold}
    tr:nth-child(2n+3){background:#f8fafc}img{width:100%;height:auto}figure{margin:24px 0}
    @media(max-width:700px){main{margin:0;padding:20px}body{font-size:16px}}
    @media print{body{background:white}main{margin:0;padding:0}a{color:inherit}}
    '''
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>后续研究结果说明</title><style>' + style + '</style><main>' + ''.join(output) + '</main></html>'


def historical_main():
    audit12 = read_json(S12 / '05_独立核验/核验结果.json')
    audit13 = read_json(S13 / '05_独立核验/all_saved_verification.json')
    summary_audit13 = read_json(S13 / '05_独立核验/summary_semantic_verification.json')
    assert audit12.get('passed') is True
    assert audit13['source_and_input_hashes_unchanged'] is True
    assert summary_audit13['passed'] is True
    assert audit13['expected_output_groups'] == audit13['verified_output_groups'] > 0
    assert all(item['passed'] for item in audit13['all_saved_trajectories_metric_cost_limit_checks'])
    a12 = read_json(S12 / '06_结果与说明/结果摘要.json')
    rows12 = read_csv(S12 / '06_结果与说明/分组汇总.csv')
    rows13 = read_csv(S13 / '04_独立测试/all_summaries.csv')
    registry13 = read_json(S13 / '04_独立测试/freeze.json')
    specs13 = {row['id']: row for row in registry13['controllers']}
    lines = ['# 后续研究结果说明：执行器滞后、纯延迟与三轴迁移', '',
        '**本轮没有获得SAC在相同可靠性要求下优于PID的控制代价证据。奖励设计和训练还存在不足，当前任务也未完成多种强化学习算法比较，因此不能写成“强化学习普遍不如PID”。**', '',
        '## 一、实际研究了什么', '',
        '第一步，分别给单轴执行器加入一阶滞后和纯延迟，比较调参后的PID、传统模型补偿、直接SAC、PID加SAC补偿。先用开发集和验证集选参数、奖励与检查点，再冻结控制器进行独立测试。', '',
        '第二步，把已训练的单轴控制器逐轴用于带完整刚体耦合的三轴模型，并加入重新整定的三轴PID作比较。这一步是迁移验证，没有另训三轴联合SAC。', '',
        '## 二、一阶滞后和纯延迟的结果', '',
        '一阶滞后：指令改变后，实际力矩逐渐靠近指令。纯延迟：指令先等待一段时间，再原样执行。每回合的时间常数或延迟在训练范围0.05～0.30秒内抽取，回合内固定；测试时控制器不知道真实值。', '',
        '验收要求是从第8秒起直到第22秒，每个采样点都满足角度误差不超过1°、角速度不超过0.5°/秒。所有控制器使用同一批初态、同一力矩限制和同一运行时长。', '',
        '直接SAC在开发阶段没有达到95%的验证通过门槛，只保留开发候选的独立测试结果，未进行正式三种子训练。PID加SAC补偿达到开发门槛，随后完成三个全新随机种子的正式训练；所有种子均保留，不根据测试成绩删选。', '']
    for kind, title in [('lag', '一阶滞后'), ('delay', '纯延迟')]:
        lines += [f'### {title}：训练范围内的独立测试', '']
        selected = [row for row in rows12 if row['kind'] == kind and row['region'] == 'in_domain']
        lines += table(['方法', '完成数', '完成率', '平均实际力矩平方积分', '全程RMS误差/°'], [
            [method12(row), f'{row["passes"]}/{row["count"]}', pct(row['success_rate']),
             f'{float(row["mean_actual_cost"]):.6f}', f'{float(row["mean_rms_error_deg"]):.3f}'] for row in selected]) + ['']
    lines += ['本轮补偿模型即使在某些场景响应更快，也要同时查看力矩代价和持续停稳，不能只挑较好的一项写“全面更优”。', '',
        '这里的代价是0～22秒实际力矩平方积分，单位为(N·m)²·s。它衡量施加力矩的强度与持续时间，不是实测电能。任务失败时较小的代价不能解释为节能。', '',
        '独立诊断还发现奖励设计不足：某些提前失效的固定动作，其回报高于长时间不完成任务。这表明奖励可能给出不恰当的激励。诊断只是构造反例，没有证明某个模型确实主动利用了它；但在排查这类问题前，不能把训练失败简单归因于SAC算法。原模型和原结果保留，未在测试后更换奖励。', '',
        '[完整单轴结果、训练范围外结果与图表](../12_执行器滞后与纯延迟对比研究/06_结果与说明/研究结果说明.md)；[奖励设计诊断](../12_执行器滞后与纯延迟对比研究/05_独立核验/奖励设计诊断.md)。', '',
        '## 三、三轴完成了什么', '',
        '三轴模型保留欧拉刚体动力学中的陀螺耦合项，采用四元数表示姿态，惯量为[1.2, 1.0, 0.8] kg·m²。不能把它看成三份完全独立的单轴仿真。', '',
        '下表是训练参数范围内的独立迁移结果。每个方法使用128个新初态与6个执行器参数，共768回合；不同参数下复用同一初态。', '']
    data13 = []
    order13 = {'retuned': 0, 'pid_transfer': 1, 'compensated_transfer': 2, 'direct': 3, 'residual': 4}
    for row in sorted((r for r in rows13 if r['region'] == 'in_domain'),
                      key=lambda r: (r['kind'] != 'lag', order13[r['method']], r['controller'])):
        spec = specs13[row['controller']]
        label = {'retuned': '重新整定三轴PID', 'pid_transfer': '单轴PID逐轴迁移',
                 'compensated_transfer': '单轴传统补偿逐轴迁移',
                 'direct': '单轴直接SAC逐轴迁移', 'residual': '单轴PID加SAC逐轴迁移'}[row['method']]
        if spec.get('seed') is not None:
            label += f'（种子{spec["seed"]}）'
        elif 'seed' in row['controller']:
            label += '（种子' + re.search(r'seed(\d+)', row['controller']).group(1) + '）'
        if row['diagnostic_only'] == 'True':
            label += '，开发未达标（未进行正式三种子训练）'
        data13.append([{'lag': '一阶滞后', 'delay': '纯延迟'}[row['kind']], label,
                       f'{row["passes"]}/{row["count"]}', f'{float(row["mean_actual_cost"]):.6f}'])
    lines += table(['执行器', '方法', '完成数', '平均实际力矩平方积分'], data13) + ['',
        '三轴迁移没有表现出跨种子的稳定优势。一阶滞后中，补偿种子202完成768/768，重新整定三轴PID为766/768，但该学习模型的平均代价约为PID的4.45倍，另外两个学习种子的通过率也不一致。纯延迟中，重新整定PID完成768/768，三个学习补偿模型分别只完成119、3、0回合。不能只挑种子202宣称强化学习更好。', '',
        '迁移条件同时改变了惯量、多轴姿态几何与耦合，并使用整体姿态角和角速度向量范数验收。通过率下降不能单独归因于“耦合”；也不能据此推断三轴联合训练的效果。', '',
        '[三轴迁移研究报告及全部测试范围](../13_执行器误差下的三轴耦合迁移验证/研究报告.md)。', '',
        '## 四、为什么选SAC、是否训练充分、随机种子是什么', '',
        '**选SAC的依据是任务适用性和已有实现，不是本任务中的优胜证据。** 力矩是连续变量，SAC支持连续动作，也能重复使用既有交互数据训练；项目已有SAC实现经验。这些只能支持“可以把它作为候选”。本轮没有在同一执行器任务、数据划分和整定预算下比较SAC、TD3、PPO，不能宣称SAC最好。其他算法可以使用。', '',
        'TD3同样适合连续动作，也能重复使用交互数据；PPO也支持连续动作，但训练方式和数据使用方式不同。执行器状态部分不可观测，除了现有的历史输入，也可以研究带循环网络的策略；循环网络是否有用仍须独立验证，不能预先保证它更好。', '',
        '**当前证据不足以声称“SAC已经训练得很好”。** 完成规定训练步数只是执行完成；代码和轨迹核验通过表示实现与记录一致，不等于控制效果优秀。直接SAC开发验证未达标，纯延迟补偿的正式模型还存在明显种子差异，奖励函数也发现不合理激励。后续若继续，应先纠正训练目标与验收要求的不一致、检查奖励和训练曲线，再用新的验证及独立测试比较算法。当前阴性结果只适用于这套配置和预算。', '',
        '**随机种子是控制伪随机数序列的整数。** 它会影响网络初始参数、探索动作、训练场景抽样和经验回放抽样。同一个方法用不同种子从头训练，可能得到不同模型，因此需要重复运行检查结果是否稳定；它不是模型等级或第几代算法。', '',
        '本轮正式训练使用随机种子101、202、303，开发训练使用9101、9102。“共14次训练”指8次开发加6次正式训练；三轴的“14个控制器条目”也是数量，均不表示随机种子14。两类执行器分别训练三个正式模型，不能把它误读为只训练过一个模型。', '',
        '## 五、现有结论与实际交付', '',
        '目前可以准确表述为：在本次执行器误差模型、训练配置和有限预算下，SAC补偿没有获得预定的同等可靠性与控制代价优势；单轴模型向三轴迁移的表现也不稳定。当前结果还受奖励设计和训练质量限制，不能用于给强化学习算法整体排名。', '',
        f'单轴执行器研究共完成{a12["completed_runs"]}次训练，其中正式训练{a12["formal_runs"]}次，实际交互{a12["actual_training_steps"]:,}步；独立测试{a12["test_records"]:,}回合，使用256个独立初态并跨参数、控制器复用。', '',
        f'三轴迁移研究有{len(registry13["controllers"])}个控制器条目、{len(rows13)}组结果，共{sum(int(row["count"]) for row in rows13):,}回合，使用128个独立初态；没有新增三轴学习训练。两项研究共{a12["test_records"] + sum(int(row["count"]) for row in rows13):,}回合独立测试记录。', '',
        '单轴逐段独立核验全部响应；三轴对全部轨迹重算指标、代价与限制，独立动力学求解仅覆盖代表工况。方案、参数搜索记录、学习模型、训练元数据、冻结清单、原始轨迹与核验记录均保留。', '',
        '汇总重建：在本目录运行 `python -X utf8 -B 生成研究总览.py`。脚本只读取12号和13号研究的结果，不训练、不改变控制器。', '']
    markdown = '\n'.join(lines)
    (HERE / '12与13阶段说明.md').write_text(markdown, encoding='utf-8')
    (HERE / '12与13阶段说明.html').write_text(render(lines), encoding='utf-8')
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'sources_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(SOURCES))},
        'single_axis': a12,
        'three_axis': {'controller_count': len(registry13['controllers']), 'output_groups': len(rows13),
                       'test_records': sum(int(row['count']) for row in rows13), 'joint_training': False},
        'scope': 'actuator lag and delay; SAC training limitations; coupled three-axis transfer only',
    }
    (HERE / '12与13阶段数据来源.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    historical_main()
    from 柔性研究总结 import generate_current
    generate_current(render)


if __name__ == '__main__':
    main()
