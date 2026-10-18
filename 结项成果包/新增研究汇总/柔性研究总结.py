"""Final flexible-appendage research overview; reads fixed results only."""
from pathlib import Path
import hashlib
import json
import re
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE=HERE.parent/'16_柔性附体振动抑制连续控制/07_训练修正与复核'
SUP=BASE/'10_补充实验与论文复核'


def generate_current(render=None):
    import markdown
    result=json.loads((SUP/'结果摘要.json').read_text(encoding='utf-8'))
    original=json.loads((BASE/'08_最终报告/最终结果.json').read_text(encoding='utf-8'))
    assert result['verification']['passed'] and result['new_test_episodes']==17280
    # Preserve the earlier presentation as a separately titled historical snapshot.
    replacements={
        '../../答辩讲解包/08_柔性附体振动抑制与学习补偿/本步讲解.md':'../16_柔性附体振动抑制连续控制/07_训练修正与复核/08_最终报告/研究报告.html',
        '../../答辩讲解包/09_周期与姿态相关扰动补偿/本步讲解.md':'../17_周期与姿态相关扰动补偿/06_报告/研究报告.html',
        '../../答辩讲解包/10_执行器约束下的连续姿态跟踪/本步讲解.md':'../18_执行器约束下的连续姿态跟踪/06_研究报告/研究报告.html',
        '../../答辩讲解包/11_周期扰动下的三轴联合学习控制/本步讲解.md':'../19_周期扰动下的三轴联合学习控制',
        '../../答辩讲解包/本轮新增研究说明.md':'../../答辩讲解包/本次整理说明.md',
    }
    for source,dest in [('先读这份说明.md','15至19阶段历史汇总.md'),('研究结果总览.html','15至19阶段历史汇总.html'),('汇总数据来源.json','15至19阶段历史数据来源.json')]:
        target=HERE/dest
        if not target.exists():
            text=(HERE/source).read_text(encoding='utf-8')
            for old,new in replacements.items(): text=text.replace(old,new)
            if target.suffix=='.md':
                text='> 历史阶段记录：当前最终主实验及论文结构以[柔性研究总览](研究结果总览.html)为准。\n\n'+text
            elif target.suffix=='.html':
                text=text.replace('<body>','<body><p><strong>历史阶段记录：</strong>当前最终主实验及论文结构以<a href="研究结果总览.html">柔性研究总览</a>为准。</p>',1)
            target.write_text(text,encoding='utf-8')
    lines=['# 最终研究：卫星姿态控制与柔性附件振动抑制','',
      '最终主实验已确定为柔性附件。答辩讲解包已按论文逻辑重组为8步；惯量变化、一阶执行器滞后和纯延迟压缩为前期探索，只以SAC为代表算法。旧第10、11步从讲解包移除，原实验记录保留。','',
      '## 先看最终发现','',
      '**冻结已有三个PPO补偿模型和六种传统控制器后，新增17,280回合独立测试。三个PPO在三组各640场景中均全部完成联合任务；相对补充搜索PID与同一PID基底的成功率改善均得到成组配对统计支持。**','',
      '优化LQR也全部完成，力矩平方积分比三个PPO都低。因此本研究证明的是学习补偿相对所比较PID的收益，不能概括为超过所有传统控制。PPO333残余振动比LQR小，另外两个种子不具备相同优势，不能只挑最好模型。','',
      '## 任务到底是什么','',
      '不仅让卫星本体对准，还要让柔性附件停稳。用本体和附件由弹性、阻尼连接的双惯量单模态模型描述，控制力矩作用于本体。每回合22秒、采样0.05秒、力矩上限±0.5 N·m。从8秒到22秒的每个采样点，本体角度≤1°、角速度≤0.5°/s、附件相对角度≤0.1°、相对角速度≤0.5°/s必须同时满足。','',
      'PPO每步连续修正既有反馈的力矩，不是选择方向；其基础反馈积分增益为0，实际为PD。测试不更新网络权重，没有在线自训练。','',
      '## 三组新测试','',
      '|测试组|条件|补充搜索PID通过数|同一PID基底通过数|PPO111/222/333各自通过数|优化LQR通过数|',
      '|---|---|---:|---:|---:|---:|']
    groups=[('A_original_confirmation','A 原分布新样本','附件初始无相对振动'),('B_initial_vibration','B 初始小振动','相对初角±0.05°、相对初速±0.2°/s'),('C_vibration_and_noise','C 初振与小噪声','四状态假定小高斯测量噪声')]
    for group,label,condition in groups:
        m=result['results'][group]
        assert all(m['PID_PPO_'+str(seed)]['passes']==640 for seed in [111,222,333])
        lines.append(f'|{label}|{condition}|{m["PID_strengthened"]["passes"]}/640|{m["PID_safe"]["passes"]}/640|640/640|{m["LQR_refined"]["passes"]}/640|')
    lines += ['', '三组各128个全新初态，每个初态在五个频率0.4—0.8 Hz上复用，九种方法共享场景。初态、模型、参数、源码和噪声都先冻结。三组分别抽样，组间差值不能全部归因于噪声或初振。C组噪声只进入测量，验收按真值；噪声数值是假定工况，未按实物标定。','',
      '## 统计确认和传统方法的边界','',
      '|组别|相对补充搜索PID成功率改善|99.7222%同时校正区间|', '|---|---:|---:|']
    for group,label,_ in groups:
        row=next(x for x in result['primary_statistics'][group] if x['learner']=='PID_PPO_111' and x['reference']=='PID_strengthened')
        lines.append(f'|{label}|{100*row["mean_difference"]:.3f}个百分点|[{100*row["lower"]:.3f}, {100*row["upper"]:.3f}]个百分点|')
    lines += ['', '上述成功率三个种子相同，区间也相同。主要18项比较按128初态组配对bootstrap，保留五频率相关性，执行200,000次，并作Bonferroni校正。代价和残余振动的108项比较另作一族。','',
      '|组别|LQR代价|PPO111代价|PPO222代价|PPO333代价|','|---|---:|---:|---:|---:|']
    for group,label,_ in groups:
        m=result['results'][group]
        lines.append('|'+label+'|'+'|'.join(f'{m[n]["cost"]:.6f}' for n in ['LQR_refined','PID_PPO_111','PID_PPO_222','PID_PPO_333'])+'|')
    lines += ['', '代价为0—22秒完整时域实际力矩平方积分，单位N²·m²·s，不是电能。LQR三组同样全部通过，且代价显著更低。PPO和LQR均利用附件测量；普通PID结构只反馈本体状态，不能把结构和状态利用程度的收益全部算成学习算法优势。','',
      '## 新产出与核验','',
      f'原柔性研究有6份完整训练记录、{original["training_steps"]:,}交互步、6,400回合测试；本次不重新训练或调参，新增17,280回合，柔性两批合计23,680回合。新增7,603,200次状态转移全量独立核验通过，最大单步误差约2.22×10⁻¹⁶，记录指令整段重放最大误差约1.56×10⁻¹⁴。','',
      '新补充实验保留冻结方案、27组全轨迹（真值、测量、指令、策略动作）、主次配对统计、五项关键测试、四张论文图和完整研究报告。独立核验建立了另一物理推进方法，策略复算仍用原策略实现；没有硬件或三轴柔性验证。','',
      '## 入口','',
      '- [按研究顺序阅读：新8步答辩讲解包](../../答辩讲解包/按顺序阅读.html)',
      '- [论文正文初稿](../../答辩讲解包/论文正文初稿.md)与[论文大纲和证据对照](../../答辩讲解包/论文大纲与证据对照.md)',
      '- [本次柔性补充完整报告](../16_柔性附体振动抑制连续控制/07_训练修正与复核/10_补充实验与论文复核/05_报告/研究报告.html)',
      '- [机器结果摘要](../16_柔性附体振动抑制连续控制/07_训练修正与复核/10_补充实验与论文复核/结果摘要.json)',
      '- [原柔性正式报告](../16_柔性附体振动抑制连续控制/07_训练修正与复核/08_最终报告/研究报告.html)',
      '- [前期多场景历史汇总](15至19阶段历史汇总.html)，只供研究过程追溯，不是当前论文主线。','',
      '本页重建：`python -X utf8 -B 生成研究总览.py`。只读取固定结果，不训练、不选模、不调整控制器。','']
    text='\n'.join(lines)
    (HERE/'先读这份说明.md').write_text(text,encoding='utf-8')
    content=markdown.markdown(text,extensions=['tables','fenced_code'])
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最终柔性附件研究总览</title><style>body{font:17px/1.9 "Microsoft YaHei",system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#18334a;background:#fbfcfe}h1{font-size:31px}h2{color:#1d5278;font-size:23px;margin-top:35px}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{border:1px solid #cdd9e3;padding:9px}th{background:#eaf1f7}a{color:#165f99}img{max-width:100%}</style></head><body>'+content+'</body></html>'
    (HERE/'研究结果总览.html').write_text(page,encoding='utf-8')
    inputs=[BASE/'08_最终报告/最终结果.json',SUP/'结果摘要.json',SUP/'01_方案与冻结/freeze.json',SUP/'03_独立核验/verification.json',SUP/'04_统计/paired_statistics.json']
    manifest=dict(version='final_flexible_appendage_20261018',sources_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},new_test_episodes=17280,original_flexible_test_episodes=6400,all_flexible_test_episodes=23680,all_primary_group_claims=result['claims'],scope='Flexible attachment is final main experiment; lag/delay/inertia are brief prelude; older multi-scene results retained separately')
    (HERE/'汇总数据来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    generate_current()
