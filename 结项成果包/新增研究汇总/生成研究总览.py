"""只读汇总12、13、14号研究的核验后结果，不训练、不重新选择控制器。"""
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
S14 = ROOT / '14_姿态禁入区下的学习路线选择'
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
        label += '，仅开发诊断'
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


def main():
    audit12 = read_json(S12 / '05_独立核验/核验结果.json')
    audit13 = read_json(S13 / '05_独立核验/all_saved_verification.json')
    summary_audit13 = read_json(S13 / '05_独立核验/summary_semantic_verification.json')
    audit14 = read_json(S14 / '05_独立核验/verification.json')
    assert audit12.get('passed') is True
    assert audit13['source_and_input_hashes_unchanged'] is True
    assert summary_audit13['passed'] is True
    assert audit13['expected_output_groups'] == audit13['verified_output_groups']
    assert all(item['passed'] for item in audit13['all_saved_trajectories_metric_cost_limit_checks'])
    assert audit14['passed'] is True
    a12 = read_json(S12 / '06_结果与说明/结果摘要.json')
    rows12 = read_csv(S12 / '06_结果与说明/分组汇总.csv')
    rows13 = read_csv(S13 / '04_独立测试/all_summaries.csv')
    registry13 = read_json(S13 / '04_独立测试/freeze.json')
    specs13 = {row['id']: row for row in registry13['controllers']}
    rows14 = read_csv(S14 / '04_独立测试/responses/summary.csv')
    intervals14 = read_json(S14 / '06_报告/paired_intervals.json')
    resource14 = read_json(S14 / '06_报告/资源与可复现性.json')
    names14 = {'direct_PID': '直接目标PID', 'analytic_route_PID': '解析路线规划加PID',
               **{f'DQN_seed_{seed}': f'DQN路线选择加PID（种子{seed}）' for seed in (101, 202, 303)}}
    natural = [row for row in rows14 if row['region'] == 'test']
    direct = next(row for row in natural if row['method'] == 'direct_PID')
    learned = [row for row in natural if row['method'].startswith('DQN_')]
    lo, hi = min(float(row['success_rate']) for row in learned), max(float(row['success_rate']) for row in learned)
    lines = ['# 后续研究结果说明：执行器误差与姿态禁入区', '',
        f'**找到了强化学习相对直接目标PID的明确优势：在本次姿态禁入区任务的512个独立测试场景中，DQN路线选择加PID的三个种子完成率为{lo:.2%}～{hi:.2%}，直接目标PID为{float(direct["success_rate"]):.2%}。解析路线规划加PID为100%，因此没有证明学习优于解析规划。**', '',
        '执行器一阶滞后、纯延迟和三轴迁移也全部保留。它们与禁入区是不同任务，分别采用各自预先声明的训练、验证和独立测试数据，不能混成一个实验的排名。', '',
        '## 一、这次依次做了什么', '',
        '第一步，分别给单轴执行器加入一阶滞后和纯延迟，比较调参后的PID、传统模型补偿、直接SAC、PID加SAC补偿。先用开发集和验证集选参数、奖励与检查点，再冻结控制器进行独立测试。', '',
        '第二步，把已训练的单轴控制器逐轴用于带完整刚体耦合的三轴模型，并加入重新整定的三轴PID作比较。这一步是迁移验证，没有另训三轴联合SAC。', '',
        '第三步，另立“姿态禁入区”任务。目标不仅是转到指定角度，还必须全程避开禁止朝向。用DQN选择转动方向，随后由与传统对照完全相同的PID完成反馈控制。', '',
        '## 二、先看一阶滞后和纯延迟的结果', '',
        '一阶滞后：指令改变后，实际力矩逐渐靠近指令。纯延迟：指令先等待一段时间，再原样执行。每回合的时间常数或延迟在训练范围0.05～0.30秒内抽取，回合内固定；测试时控制器不知道真实值。', '',
        '验收要求是从第8秒起直到第22秒，每个采样点都满足角度误差不超过1°、角速度不超过0.5°/秒。所有控制器使用同一批初态、同一力矩限制和同一运行时长。', '',
        '直接SAC在开发阶段没有达到95%的验证通过门槛，只保留开发候选作诊断。PID加SAC补偿达到开发门槛，随后完成三个全新种子的正式训练；所有种子均保留，不根据测试成绩删选。', '']
    for kind, title in [('lag', '一阶滞后'), ('delay', '纯延迟')]:
        lines += [f'### {title}：训练范围内的独立测试', '']
        selected = [row for row in rows12 if row['kind'] == kind and row['region'] == 'in_domain']
        lines += table(['方法', '完成数', '完成率', '平均实际力矩平方积分', '全程RMS误差/°'], [
            [method12(row), f'{row["passes"]}/{row["count"]}', pct(row['success_rate']),
             f'{float(row["mean_actual_cost"]):.6f}', f'{float(row["mean_rms_error_deg"]):.3f}'] for row in selected]) + ['']
    supported12 = [row for row in a12['comparisons'] if row['cost_advantage_supported']]
    if not supported12:
        lines += ['**这轮单轴实验没有得到满足预定可靠性要求的SAC代价优势。** 补偿方法即使在某些场景响应更快，也要同时看力矩代价和持续停稳，不能只挑较好的一项写“全面更优”。', '']
    else:
        lines += ['部分比较满足代价优势判据，具体比较及统计区间见12号研究报告；未达标组合仍完整保留。', '']
    lines += ['这里的代价是0～22秒实际力矩平方积分，单位为(N·m)²·s。它衡量施加力矩的强度与持续时间，不是实测电能。任务失败时较小的代价不能解释为节能。', '',
        '另外，独立诊断发现这轮SAC奖励存在一种不足：某些提前失效的固定动作，其回报高于长时间不完成任务。这说明奖励可能给出不恰当的激励，不能据本轮直接SAC失败推断“SAC不适合卫星控制”。诊断只是构造反例，没有证明某个模型确实主动利用了它。原模型和原结果保留，未在测试后更换奖励。', '',
        '[完整单轴结果、训练范围外结果与图表](../12_执行器滞后与纯延迟对比研究/06_结果与说明/研究结果说明.md)；[奖励设计诊断](../12_执行器滞后与纯延迟对比研究/05_独立核验/奖励设计诊断.md)。', '',
        '## 三、三轴做到了哪一步', '',
        '三轴模型保留欧拉刚体动力学中的陀螺耦合项，采用四元数表示姿态，惯量为[1.2, 1.0, 0.8] kg·m²。不能把它看成三份完全独立的单轴仿真。', '',
        '下表是训练参数范围内的独立迁移结果。每个方法使用128个新初态与6个执行器参数，共768回合；不同参数下复用同一初态。', '']
    data13 = []
    order13 = {'retuned': 0, 'pid_transfer': 1, 'compensated_transfer': 2, 'direct': 3, 'residual': 4}
    for row in sorted((r for r in rows13 if r['region'] == 'in_domain'),
                      key=lambda r: (r['kind'] != 'lag', order13[r['method']], r['controller'])):
        spec = specs13[row['controller']]
        label = {'retuned': '重新整定三轴PID', 'pid_transfer': '单轴PID逐轴迁移',
                 'compensated_transfer': '单轴传统补偿逐轴迁移',
                 'direct': '单轴直接SAC逐轴迁移', 'residual': '单轴PID加SAC逐轴迁移'}.get(row['method'], row['method'])
        if spec.get('seed') is not None:
            label += f'（种子{spec["seed"]}）'
        elif 'seed' in row['controller']:
            label += '（种子' + re.search(r'seed(\d+)', row['controller']).group(1) + '）'
        if row['diagnostic_only'] == 'True':
            label += '，仅开发诊断'
        data13.append([{'lag': '一阶滞后', 'delay': '纯延迟'}[row['kind']], label,
                       f'{row["passes"]}/{row["count"]}', f'{float(row["mean_actual_cost"]):.6f}'])
    lines += table(['执行器', '方法', '完成数', '平均实际力矩平方积分'], data13) + ['',
        '三轴迁移没有表现出跨种子的稳定优势。一阶滞后中，补偿种子202完成768/768，重新整定三轴PID为766/768，但该学习模型的平均代价约为PID的4.45倍，另外两个学习种子的通过率也不一致。纯延迟中，重新整定PID完成768/768，三个学习补偿模型分别只完成119、3、0回合。不能只挑种子202宣称强化学习更好。', '',
        '迁移条件同时改变了惯量、多轴姿态几何与耦合，并使用整体姿态角和角速度向量范数验收。通过率下降不能单独归因于“耦合”；也不能据此推断三轴联合训练的效果。', '',
        '[三轴迁移研究报告及全部测试范围](../13_执行器误差下的三轴耦合迁移验证/研究报告.md)。', '',
        '## 四、在哪个任务中找到了学习方法的优势', '',
        '设卫星有一段禁止朝向，例如某载荷需要避开太阳方向。我们把它简化为单轴角度圆上的固定禁入弧[160°, 200°]，任务要求转到目标，同时全程不进入这段区域。这里是简化模型，没有使用真实太阳星历或真实载荷。', '',
        '普通目标PID根据当前角度误差决定转动，却没有选择安全路线的功能。例如从约−76°转到129°，按角度圆上的较短方向转动会经过−180°，恰好落在禁入区；选择经过0°的另一条路线才能安全到达。单纯调整PID增益不能给它增加路线规划功能。', '',
        'DQN适合这里的离散选择：每回合只选一次顺时针或逆时针，然后把路线交给同一个固定PID。网络没有在测试时继续训练，也没有获得正确方向标签；它根据训练环境返回的整条路线奖励学习选择。这个问题是单步决策，不是复杂的连续强化学习控制。', '',
        '三个对照分别是：直接目标PID、解析路线规划加PID、DQN路线选择加PID。底层PID参数完全相同；搜索允许积分项，最终选择的积分增益为0，所以实际执行的是PD形式。禁入区任务采用理想执行器，力矩上限为2 N·m，12秒完成并持续验收到22秒；与前两项执行器研究分别报告。', '',
        '### 自然测试分布：主要结果', '']
    lines += table(['方法', '完成数', '完成率', '碰撞次数'], [
        [names14[row['method']], f'{row["successes"]}/{row["count"]}', pct(row['success_rate']), row['collisions']] for row in natural])
    lines += ['', '三个DQN种子相对直接目标PID的配对完成率增益如下。区间以独立场景为单位计算，并对三次比较作Bonferroni校正。', '']
    lines += table(['种子', '完成率提高/百分点', '校正后98.333%区间/百分点'], [
        [row['seed'], f'{100 * row["difference"]:.2f}', f'[{100 * row["lower"]:.2f}, {100 * row["upper"]:.2f}]']
        for row in intervals14['intervals']])
    lines += ['', '三个区间均高于0，支持“在本次测试分布下，增加学习路线选择提高了直接目标PID的任务完成率”。它不是学习反馈胜过PID的证据；两者底层反馈是同一个PID。', '',
        '### 压力测试：保留没有成功的场景', '']
    lines += table(['方法', '绕行集完成数/128', '边缘集完成数/128'], [
        [names14[method], next(r['successes'] for r in rows14 if r['method'] == method and r['region'] == 'detour'),
         next(r['successes'] for r in rows14 if r['method'] == method and r['region'] == 'edge')]
        for method in names14])
    lines += ['', '绕行集专门抽取需要绕行的情况；边缘集专门抽取靠近允许采样范围边界的情况。它们分别有128个新场景，不能代替自然分布作为主要结果。DQN仍有碰撞，解析规划在这三类测试中均全部成功。', '',
        '**因此，这项研究证明了学习路线选择相对直接目标PID的作用，没有证明强化学习优于解析路线规划。** 对一个固定禁入弧，传统解析解本来就简单；实际选型时应优先考虑它的准确性与实现成本。', '',
        '![禁入区任务的完整成功率对比](图表/图1_禁入区各方法完整成功率.png)', '',
        '![同一初态的路线对比](图表/图2_禁入区配对代表轨迹.png)', '',
        '代表轨迹是在测试完成后为解释机制选取的：按明确、可复现的规则，从自然测试中选择“直接PID碰撞、解析规划和三个DQN均成功”的场景中编号最小的一例。它用于解释机制，总体效果由上面的完整测试统计说明。', '',
        '[禁入区完整报告](../14_姿态禁入区下的学习路线选择/06_报告/研究报告.md)。', '',
        '## 五、答辩可以怎样讲', '',
        '可以这样说：**我们先在相同时间、力矩和停稳要求下研究执行器滞后与延迟，发现本轮SAC方案没有获得预定的控制代价优势。随后转向含禁入区的路线选择任务，让DQN负责选择转动方向，PID负责底层反馈。独立测试表明，加入学习路线选择后，完成率从84.77%提高到98.44%～100%。解析路线规划同样能解决该任务且更可靠，因此本研究展示的是学习模块在路线决策上的作用及局限。**', '',
        '不宜写成“强化学习全面超过PID”“已经证明优于所有传统方法”“实现零风险避障”或“三轴联合强化学习已完成”。没有实测电能，也不写“节电多少”。', '',
        '这个版本的研究主线可以表述为：**卫星姿态控制中传统反馈与学习决策的适用条件比较——执行器误差控制与姿态禁入区路线选择。** 这是研究内容的描述，不是新算法名称。', '',
        '## 六、实际交付与复现', '',
        f'单轴执行器研究共完成{a12["completed_runs"]}次训练，其中正式训练{a12["formal_runs"]}次，实际交互{a12["actual_training_steps"]:,}步；独立测试{a12["test_records"]:,}回合，使用256个独立初态并跨参数、控制器复用。', '',
        f'三轴迁移研究有{len(registry13["controllers"])}个控制器条目、{len(rows13)}组结果，共{sum(int(row["count"]) for row in rows13):,}回合，使用128个独立初态；没有新增三轴学习训练。', '',
        f'路线选择研究完成开发1次、正式3次训练，共{resource14["total_route_decisions"]:,}次路线决策、{resource14["total_gradient_updates"]:,}次梯度更新。训练环境缓存来自20,000个训练初态、两个方向共40,000条实际闭环仿真，实际传播9,885,217个物理区间。7万次决策包括对缓存状态动作的重复访问，不能写成7万条不同物理轨迹。独立测试共有512＋128＋128＝768个场景，五个方法共3,840条轨迹。', '',
        '每项研究都保存方案与数据划分、传统控制搜索记录、学习模型和训练元数据、冻结清单、原始轨迹、回合指标、核验记录与报告。单轴执行器与路线任务逐段独立核验全部响应；三轴对全部轨迹重算指标、代价与限制，独立动力学求解仅覆盖代表工况，两者范围不同。', '',
        '汇总重建：在本目录运行 `python -X utf8 -B 生成研究总览.py`。图表重建运行 `python -X utf8 -B 绘制禁入区结果.py`。两个脚本只读取研究结果，不训练、不改变控制器。详细研究入口分别为12号、13号、14号目录内的运行脚本。', '',
        '原始报告与既有答辩材料保留，本说明作为新增研究附加材料。需要写论文或更新答辩时，应连同负面结果、传统规划对照和结论范围一起采用。', '']
    markdown = '\n'.join(lines)
    (HERE / '先读这份说明.md').write_text(markdown, encoding='utf-8')
    (HERE / '研究结果总览.html').write_text(render(lines), encoding='utf-8')
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'sources_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(SOURCES))},
        'single_axis': a12,
        'three_axis': {'controller_count': len(registry13['controllers']), 'output_groups': len(rows13),
                       'test_records': sum(int(row['count']) for row in rows13), 'joint_training': False},
        'route_primary': natural,
        'scope': 'learning route choice improves direct-target PID; no superiority over analytic planning',
    }
    (HERE / '汇总数据来源.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(markdown[:700])


if __name__ == '__main__':
    main()
