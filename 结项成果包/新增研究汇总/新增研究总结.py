"""从已冻结结果生成当前说明；不训练，不更换控制器，不把阴性结果删掉。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = []


def read(relative):
    path = ROOT / relative
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    INPUTS.append(path)
    return result


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def generate_current(render):
    INPUTS.clear()
    a15 = read('15_连续控制训练修正与多算法研究/06_结果与说明/结果摘要.json')
    a16 = read('16_柔性附体振动抑制连续控制/07_训练修正与复核/08_最终报告/最终结果.json')
    a17 = read('17_周期与姿态相关扰动补偿/结果摘要.json')
    a18 = read('18_执行器约束下的连续姿态跟踪/06_研究报告/AI增量核验结论.json')
    audit18 = read('18_执行器约束下的连续姿态跟踪/05_独立核验/verification.json')
    # Study 19 writes this only after its formal tests and audit are complete.
    a19 = read('19_周期扰动下的三轴联合学习控制/结果摘要.json')
    assert a15['independent_verification_passed'] and not a15['practical_advantage']
    assert a16['full_physics_audit_passed']
    assert a17['primary_supported'] and a17['primary_conclusion_replicated']
    assert not a17['full_original_audit_passed'], 'Do not erase the recorded strict state-check failure'
    assert audit18['passed'] and not a18['learning_increment_beyond_identical_feedforward_base']
    assert a19['verification']['passed'] and a19['independent_closed_loop_verification']['passed']
    assert a19['independent_closed_loop_verification']['primary_conclusion_replicated']
    assert a19['joint_network'] and a19['real_Euler_coupling'] and not a19['single_axis_transfer']
    assert not a19['primary_supported']

    methods16 = a16['methods']
    pid17 = a17['results']['pid']
    ai17 = [a17['results'][f'TD3_{s}'] for s in (17111, 17222, 17333)]
    rms_reductions = [100 * (1 - x['mean_steady_rms_deg'] / pid17['mean_steady_rms_deg']) for x in ai17]
    cost_reductions = [100 * (1 - x['mean_actual_cost'] / pid17['mean_actual_cost']) for x in ai17]
    r_range = f'{min(rms_reductions):.1f}%～{max(rms_reductions):.1f}%'
    c_range = f'{min(cost_reductions):.1f}%～{max(cost_reductions):.1f}%'
    post = a16['post_hoc_pid_verification']

    lines = [
        '# 本轮研究结果：强化学习在哪些条件下对PID有帮助', '',
        '**已经找到两个有实际意义、相对PID有收益的仿真场景：柔性附件振动抑制和周期扰动补偿。但没有得到“强化学习全面优于传统控制”的结论。**', '',
        '本页更新于2026年10月18日。先修正原训练问题，再研究三个实际场景，最后单独进行三轴联合训练。失败尝试、三个正式训练种子、传统方法和独立测试均保留。', '',
        '## 先看你最关心的结论', '',
    ]
    lines += table(['研究', '学习算法实际做什么', '独立结果', '可以怎样下结论'], [
        ['执行器一阶滞后、纯延迟：训练修正', 'SAC、TD3、PPO连续输出力矩补偿', '12次补偿开发训练均选回训练前的零补偿；未进入正式训练', '这套训练配置仍未解决问题，不能算AI成功，也不能据此给算法排名'],
        ['场景一：柔性附件振动', 'PPO连续修正PID力矩，同时抑制主星和附件的相对振动', f'三个模型分别640/640通过；加强PID复核{post["passes"]}/{post["count"]}通过', '在本模型与测试范围内，相对所比较PID提高了任务成功率；优化LQR同样全通过且代价更低'],
        ['场景二：周期与姿态相关扰动', 'TD3根据响应和扰动相位连续补偿力矩', f'PID及三个TD3均128/128通过；稳态误差降低{r_range}，力矩代价降低{c_range}', '学习补偿确实比单独PID更精确、用力更少；扰动观测器仍更好'],
        ['场景三：受执行器约束的姿态跟踪', 'PPO在同一个带前馈和预览的PID上连续补偿', '相对同一前馈PID的误差改善区间均跨0，代价增加', '未证明稳定的学习增量；不能把前馈本身的收益算给AI'],
    ])
    lines += ['', '## 一、为什么先重做训练，而不是继续只用SAC', '',
        '此前只选SAC的理由是适用连续力矩、项目已有实现；这不能证明SAC最好。15号研究同时检查SAC、TD3、PPO，修正失效提前结束可能造成的奖励问题，统一运行时长，并保留全部检查点。', '',
        f'15号研究实际完成{a15["completed_runs"]}次训练、{a15["actual_training_steps"]:,}交互步、{a15["test_records"]:,}回合独立测试。三种直接控制的理想执行器检查未达标，12次学习补偿均选回零步检查点。它说明这一配置没有学出有效增量，不能把退回PID后的成绩说成训练成功。', '',
        '随后的不同任务分别在开发集上比较候选算法：柔性附件中PPO有效，周期扰动中TD3有效。算法是否合适需要看训练曲线、验证集、多个正式种子和独立测试，不存在本项目已经证明的“唯一最好算法”。', '',
        '[训练修正结果](../15_连续控制训练修正与多算法研究/06_结果与说明/结果总览.html) · [所有开发训练曲线](../15_连续控制训练修正与多算法研究/06_结果与说明/全部开发训练曲线.png)', '',
        '## 二、场景一：卫星停下了，帆板还在晃', '',
        '真实卫星可能带太阳帆板、天线等柔性附件。主星转向或刹车时，附件会被激起振动；只管主星朝向，可能还留下附件晃动。这里用两个由弹性和阻尼连接的转动惯量描述这个现象，回合之间改变振动频率与阻尼。它是简化双惯量模型，假设理想力矩源、无测量噪声且附件振动可测，未用实物标定。', '',
        '验收同时要求主星角度、主星转速、附件相对变形和相对转速，从第8秒起持续达标直到第22秒。PPO在保守PID上连续补偿力矩；不同方法获得同样的可测状态，不能读取真实模态参数。', '',
    ]
    rows16 = []
    for key, label in [('PID', '原成功率优先整定PID'), ('PID_safe', '学习补偿使用的同一保守PID'),
                       ('ZVD_PID', 'ZVD输入整形加PID'), ('LQR_refined', '扩大参数搜索后的LQR'),
                       ('PID_PPO_111', 'PID＋PPO，种子111'), ('PID_PPO_222', 'PID＋PPO，种子222'),
                       ('PID_PPO_333', 'PID＋PPO，种子333')]:
        x = methods16[key]
        rows16.append([label, f'{x["passes"]}/{x["count"]}', f'{100*x["success"]:.2f}%', f'{x["cost"]:.6f}'])
    rows16.insert(2, ['事后扩大搜索的PID（单列复核）', f'{post["passes"]}/{post["count"]}', f'{100*post["success"]:.2f}%', '见原复核表'])
    lines += table(['方法', '任务通过数', '任务通过率', '平均实际力矩平方积分'], rows16)
    lines += ['', f'三个PPO都通过640/640，较单列加强PID的{100*post["success"]:.2f}%高{100*(1-post["success"]):.2f}个百分点。对每个初态的多种模态工况进行成组统计后，优势仍成立。加强PID是看到前一轮结果后增加的传统参考，参数选择没有读取独立测试响应；它需要明确标为事后证据，仍值得用另一批新测试确认。', '',
        '**边界：充分搜索的LQR也达到640/640，平均代价0.005564，比三个PPO都小。这里证明的是PPO相对所比较PID的成功率收益，不能写成超过所有传统控制方法。**', '',
        f'本次修正版有6份完整训练记录、{a16["training_steps"]:,}步；主测试{a16["held_out_episodes"]:,}回合，加加强PID复核共{a16["total_test_episodes_including_post_hoc"]:,}回合，全部状态转移和指标独立核验。更早的失败与中断训练保留，未将缺少精确步数的中断记录凑入该统计。', '',
        '[柔性附件最终报告](../16_柔性附体振动抑制连续控制/07_训练修正与复核/08_最终报告/研究报告.html) · [通俗讲解](../../答辩讲解包/08_柔性附体振动抑制与学习补偿/本步讲解.md)', '',
        '## 三、场景二：不是随机抖一下，而是反复出现的扰动力矩', '',
        '有些外力随时间、相位或卫星姿态呈现规律。这里将扰动力矩设为两个周期项加姿态和角速度相关项，各项幅度在每回合变化。工程动机来自周期性外力和姿态相关外力；本实验的正弦和非线性项是受控仿真假设，不是某颗实际卫星的辨识模型。', '',
        'TD3输出连续力矩补偿，PID承担基础反馈。可测相位、频率给所有适用的对照方法；真实扰动力矩不直接提供。各方法在同一128个新初态上运行32秒，从第8秒起到结束持续检查误差与转速。', '',
    ]
    rows17 = []
    for key, label in [('pid', '充分整定PID'), ('TD3_17111', 'PID＋TD3，种子17111'),
                       ('TD3_17222', 'PID＋TD3，种子17222'), ('TD3_17333', 'PID＋TD3，种子17333'),
                       ('observer', 'PID＋扰动观测器'), ('harmonic', 'PID＋谐波补偿')]:
        x = a17['results'][key]
        rows17.append([label, f'{x["passes"]}/{x["count"]}', f'{x["mean_steady_rms_deg"]:.6f}', f'{x["mean_actual_cost"]:.6f}'])
    lines += table(['方法', '任务通过数', '8～32秒稳态RMS误差/°', '0～32秒实际力矩平方积分'], rows17)
    lines += ['', f'PID的稳态RMS约0.02962°，三个TD3为0.01690°～0.02188°，降低{r_range}；实际力矩平方积分降低{c_range}。三个正式模型都改善，多重比较校正后的配对区间均支持误差下降。绝对误差减少约0.0077°～0.0127°，其实际意义仍取决于具体卫星的精度要求。', '',
        '**边界：扰动观测器的误差与代价均低于TD3。学习有收益，但这组已知结构的扰动也能被传统补偿有效处理。**', '',
        '数值核验须分清范围：原来用同一个混合单位阈值检查全部状态的严格核验没有通过，失败记录保留。随后用另一高精度积分器对全部768回合重新做闭环测试，重现了全部验收决策和三组主要统计结论；稳态RMS差异最大约0.000000157°。因此可以说“主要指标与结论独立重现”，不能说“原严格全状态核验全部通过”。', '',
        f'实际训练{a17["training_runs"]}次、{a17["interaction_steps"]:,}步，正式模型3份，独立测试{a17["test_episodes"]}回合。', '',
        '[周期扰动研究报告](../17_周期与姿态相关扰动补偿/06_报告/研究报告.html) · [通俗讲解](../../答辩讲解包/09_周期与姿态相关扰动补偿/本步讲解.md)', '',
        '## 四、场景三：电机跟不上扫描指令', '',
        '姿态扫描需要持续跟踪变化目标；执行器的最大力矩与力矩变化速度受限。所有适用方法获得同样的3秒目标预览。比较单独PID、带模型前馈和预览的PID、MPC，以及在同一前馈PID上加学习补偿。', '',
        '单独PID的平均跟踪RMS为3.9381°，加入前馈后变成1.0198°。再加三个PPO模型后为0.9526°、0.9843°、0.9931°，表面平均值又好了一点，但校正后的改善区间都包含0，力矩代价还增加约3.6%～5.4%。因此目前不能确认这点进一步改善稳定存在；主要收益来自前馈，不能都归功于强化学习。', '',
        '修正版三个种子均选到了训练后的检查点，所以它与“模型选回零步”的问题不同。这里是确实训练了，却尚未证明同一基础控制器之外的可靠收益。', '',
        'MPC求解器的38400次优化均成功，但该固定时域和权重配置仍有10/128回合物理失败。这说明这套MPC参数还不够好，不能据此宣布AI超过充分优化的MPC。', '',
        f'该研究实际6次训练、2,359,296步；{audit18["episodes"]}回合、{audit18["physical_intervals"]:,}个物理区间全量核验通过。', '',
        '[受限跟踪最终报告](../18_执行器约束下的连续姿态跟踪/06_研究报告/研究报告.html) · [通俗讲解](../../答辩讲解包/10_执行器约束下的连续姿态跟踪/本步讲解.md)', '',
        '## 五、三轴联合训练的结果', '',
    ]
    # Kept as data from study 19, rather than selecting a favourable seed here.
    lines += [line.replace('## 19. 周期扰动下的三轴联合学习控制', '### 完整三轴联合测试') for line in a19['aggregate_markdown_lines']]
    lines += ['', '## 六、现在适合用于答辩的说法', '',
        '“我们先建立统一的卫星姿态控制评价流程，再检查执行器误差下的训练不足，并在三个工程场景中研究学习补偿。柔性附件中PPO提高了相对PID的任务成功率；周期扰动中TD3同时降低了相对PID的稳态误差和力矩代价；受限跟踪中未证明相对同一前馈PID的稳定增量。LQR、扰动观测器等传统方法仍有竞争力，因此我们的结论是学习补偿在特定模型和条件下有效，而非强化学习全面取代传统控制。”', '',
        '这些是连续控制实验：网络每个采样时刻给出连续补偿力矩，不是简单选择顺时针或逆时针。测试期间不更新网络参数；不同工况下动作变化属于固定策略对状态作出不同反应，不能说模型自动在线自训练。', '',
        '这里的控制代价都是各实验固定完整时窗内的实际力矩平方积分；三轴则把三个轴的平方积分相加。它不是电池电量、燃料消耗或实测能耗，不能直接写“省电多少”。不同任务的时长和验收要求不同，不把跨任务的代价数值直接相互排名。', '',
        '全部结果只覆盖本次简化仿真、参数范围、整定与训练预算。多个正式种子改善和独立测试能加强证据，但仍不能替代实物验证或证明全局最优。最适合作为主要结果的是上面的两项相对PID收益，同时如实报告更强传统方法与无优势场景。', '',
        '## 七、文件怎么找', '',
        '[答辩讲解包：按研究顺序阅读](../../答辩讲解包/按顺序阅读.html)；[新增研究说明](../../答辩讲解包/本轮新增研究说明.md)。讲解包每一步外层是精华解释、图表和用稿，内层“支撑代码与数据”用来源路径和SHA-256指向原工程的模型、日志、测试轨迹和核验记录。', '',
        '[12与13号历史阶段说明](12与13阶段说明.html)保留原执行器与三轴迁移的阴性结果。那一阶段没有比较三种算法，也没有进行三轴联合训练；这些限制只描述当时阶段，不表示后续工作没有完成。已删除用户否定的路线二选一研究，不再列入本轮贡献。', '',
        '本页重建命令：在本目录运行 `python -X utf8 -B 生成研究总览.py`。仅从固定结果读取并核对关键状态，不训练、不更换控制器、不根据独立测试重新调参。', '',
    ]
    markdown = '\n'.join(lines)
    (HERE / '先读这份说明.md').write_text(markdown, encoding='utf-8')
    (HERE / '研究结果总览.html').write_text(render(lines).replace('后续研究结果说明', '本轮连续学习补偿研究结果'), encoding='utf-8')
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '15 training repair; three practical scenarios 16-18; genuine jointly trained coupled-three-axis study 19',
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
        'claims': {
            'flexible_appendage_PPO_success_better_than_compared_PID': True,
            'periodic_disturbance_TD3_RMS_and_effort_better_than_PID': True,
            'rate_constrained_tracking_learning_increment_supported': False,
            'universal_RL_superiority': False,
            'study_17_original_full_state_audit_passed': False,
            'study_17_primary_conclusion_independently_replicated': True,
        },
        'study_19': a19,
        'periodic_RMS_reduction_percent': rms_reductions,
        'periodic_effort_reduction_percent': cost_reductions,
        'historical_sources': '12与13阶段数据来源.json',
        'not_hardware_validated': True,
    }
    (HERE / '汇总数据来源.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已生成当前总览，保留12/13历史阶段说明。正结果2项，受限跟踪无稳定学习增量；三轴见冻结结果。')
