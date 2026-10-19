"""从已冻结的研究结果生成结题报告图08—11；不训练、不重新测试。

运行：python 结题报告配图/06_绘图代码与数据/柔性结果/生成柔性结果图.py
依赖：numpy、matplotlib。优先使用本目录原样复制的制图输入，支持整包移动。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / '结题报告配图' / '05_完整图表版'
SOURCE = ROOT / '结项成果包/16_柔性附体振动抑制连续控制/07_训练修正与复核/10_补充实验与论文复核'
INPUTS = {
    'summary.json': '02_新独立测试/summary.json',
    'paired_statistics.json': '04_统计/paired_statistics.json',
    'verification.json': '03_独立核验/verification.json',
    'freeze.json': '01_方案与冻结/freeze.json',
    'supplement.py': '代码/supplement.py',
}
GROUPS = {
    'A_original_confirmation': 'A组：原分布全新样本',
    'B_initial_vibration': 'B组：增加初始小振动',
    'C_vibration_and_noise': 'C组：初始小振动与测量噪声',
}
LABELS = {
    'PID': '原整定PD',
    'PID_safe': '同一PD基底',
    'PID_strengthened': '补充搜索PD',
    'ZVD_PID': 'ZVD＋PID',
    'ZVD_PID_refined': '优化ZVD＋PD',
    'LQR_refined': '优化LQR',
    'PID_PPO_111': 'PD＋PPO（111）',
    'PID_PPO_222': 'PD＋PPO（222）',
    'PID_PPO_333': 'PD＋PPO（333）',
}
COLORS = ['#788798', '#96A3AF', '#586F85', '#B69463', '#907B58', '#16756E', '#3776B5', '#6C61A7', '#3D8A72']
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'svg.fonttype': 'path', 'savefig.facecolor': 'white',
})


def read(name):
    return json.loads((HERE / '制图输入' / name).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        fig.savefig(OUT / (name + suffix), dpi=260, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def axes_style(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#BCC5CF')
    ax.grid(axis='x', color='#E3E8EE', lw=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)


def success(summary):
    fig, axes = plt.subplots(3, 1, figsize=(9.7, 12.8))
    fig.subplots_adjust(left=.23, right=.80, top=.935, bottom=.14, hspace=.47)
    fig.suptitle('柔性附件联合任务：三组测试、全部九种控制器', y=.99, fontsize=17, fontweight='bold')
    for ax, (g, title) in zip(axes, GROUPS.items()):
        rows = [summary[g][m] for m in LABELS]
        p = np.array([r['passes'] / r['count'] * 100 for r in rows])
        f = np.array([r['physical_failures'] / r['count'] * 100 for r in rows])
        n = 100 - p - f
        y = np.arange(9)
        ax.barh(y, p, color=COLORS, height=.65)
        ax.barh(y, n, left=p, color='#E3E7EC', height=.65)
        ax.barh(y, f, left=p+n, color='#D87A68', height=.65)
        ax.set_yticks(y, list(LABELS.values()))
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel('回合占比（%）')
        ax.set_title(title, loc='left', pad=13, fontweight='bold')
        for yi, r in zip(y, rows):
            ax.text(102, yi, f"{r['passes']}/{r['count']}  ·  {r['success']*100:.2f}%", va='center', fontsize=10)
        ax.text(102, -1.0, '通过回合 · 成功率', fontsize=10, color='#536273')
        axes_style(ax)
    handles = [Patch(facecolor='#3776B5', label='持续达标且未越界'),
               Patch(facecolor='#E3E7EC', label='未持续达标，未越界'),
               Patch(facecolor='#D87A68', label='模型范围越界')]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(.51, .042), frameon=False, fontsize=10)
    fig.text(.03, .002, '验收窗口8—22 s；每方法每组640回合。PD名称对应积分增益为0的已选PID参数。来源：summary.json。', fontsize=9, color='#536273')
    save(fig, '图08_三组测试全部控制器成功率')


def costs(summary):
    fig, axes = plt.subplots(3, 1, figsize=(9.7, 12.8))
    fig.subplots_adjust(left=.23, right=.82, top=.935, bottom=.16, hspace=.47)
    fig.suptitle('完整0—22 s时窗内的平均力矩平方积分', y=.99, fontsize=17, fontweight='bold')
    for ax, (g, title) in zip(axes, GROUPS.items()):
        y = np.arange(9)
        for i, (name, color) in enumerate(zip(LABELS, COLORS)):
            r = summary[g][name]
            ax.barh(i, r['cost']-.003, left=.003, height=.64, color=color,
                    edgecolor='#384A5D', linewidth=.4, hatch='///' if r['passes'] < r['count'] else '')
            ax.text(.205, i, f"{r['cost']:.6f}", va='center', fontsize=10)
        ax.set_yticks(y, list(LABELS.values()))
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlim(.003, .18)
        ax.xaxis.set_major_locator(FixedLocator([.003, .01, .03, .1]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.set_xlabel('平均力矩平方积分（N²·m²·s，对数坐标）')
        ax.set_title(title, loc='left', pad=13, fontweight='bold')
        ax.text(.205, -1.0, '精确数值', fontsize=10, color='#536273')
        axes_style(ax)
    fig.legend(handles=[Patch(facecolor='#D9E3ED', edgecolor='#384A5D', hatch='///', label='存在未成功回合：不能据此单独宣称效率优势')],
               loc='lower center', bbox_to_anchor=(.51, .065), frameon=False, fontsize=10)
    fig.text(.03, .026, '优化LQR与三个PPO均全部通过；本批各组LQR代价均更低。该积分不是实测电能。', fontsize=10, color='#536273')
    fig.text(.03, .003, '来源：summary.json的cost字段。发生越界后停止施力产生的较低积分，不代表更高控制效率。', fontsize=9, color='#536273')
    save(fig, '图09_三组测试控制代价')


def intervals(stats):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.8), sharex=True)
    fig.subplots_adjust(left=.15, right=.985, top=.87, bottom=.12, wspace=.27, hspace=.63)
    fig.suptitle('PPO补偿相对两种PD参照的成功率差与配对区间', y=.98, fontsize=17, fontweight='bold')
    fig.text(.5, .925, '三组 × 三个独立训练种子 × 两种参照＝18项主要比较；区间全部位于0右侧', ha='center', fontsize=11, color='#536273')
    for ri, (g, group_label) in enumerate(GROUPS.items()):
        for ci, ref in enumerate(['PID_strengthened', 'PID_safe']):
            ax = axes[ri, ci]
            rs = [r for r in stats['comparisons'][g] if r['family']=='primary_success' and r['reference']==ref]
            assert len(rs)==3
            for yi, r in enumerate(rs):
                m, lo, hi = (100*r[x] for x in ('mean_difference', 'lower', 'upper'))
                ax.errorbar(m, yi, xerr=[[m-lo], [hi-m]], fmt='o', ms=7,
                            capsize=5, color=COLORS[6+yi], lw=2)
                ax.text(m, yi+.21, f'{m:.3f} [{lo:.3f}, {hi:.3f}]', ha='center', fontsize=9)
            ax.axvline(0, lw=1.2, color='#A05348', ls='--')
            ax.set_xlim(-1, 29)
            ax.set_ylim(2.65, -.55)
            ax.set_yticks(range(3), ['PPO 111', 'PPO 222', 'PPO 333'])
            ax.set_title(group_label+'\n相对'+LABELS[ref], fontsize=11, loc='left', pad=10, fontweight='bold')
            ax.set_xticks(range(0, 30, 5))
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_xlabel('成功率差（PPO－参照，百分点）', fontsize=10)
            axes_style(ax)
    fig.text(.03, .048, f"按{stats['clusters_per_group']}个独立初态成组配对重采样；每个初态保留5个频率；bootstrap {stats['bootstrap_repeats']:,}次。", fontsize=10, color='#536273')
    fig.text(.03, .022, f"Bonferroni校正18项主要比较后的双侧区间：{stats['primary_confidence']*100:.4f}%。来源：paired_statistics.json原始区间。", fontsize=10, color='#536273')
    save(fig, '图10_成功率差与配对区间')


def box(ax, x, y, w, h, face='#EFF4FA', edge='#D6DFE9'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.009,rounding_size=.012',
                               transform=ax.transAxes, fc=face, ec=edge, linewidth=.8))


def verification(v):
    checks = v['checks']
    local = max(r['max_state_transition_residual'] for r in checks)
    whole = max(r['max_full_recorded_command_replay_error'] for r in checks)
    policy = max(r['max_policy_action_error'] for r in checks)
    fig, ax = plt.subplots(figsize=(11, 9.4))
    fig.subplots_adjust(left=.035, right=.965, top=.925, bottom=.06)
    fig.suptitle('独立数值核验：覆盖范围、数值差异与检查结果', y=.985, fontsize=17, fontweight='bold')
    ax.axis('off')
    for i, (num, label) in enumerate([(f"{v['episodes']:,}", '测试回合'), (f"{v['transitions']:,}", '状态转移'),
                                     (str(len(checks)), '控制器×测试组'), ('通过' if v['passed'] else '未通过', '总体核验')]):
        x=.008+i*.25
        box(ax, x, .838, .233, .13)
        ax.text(x+.1165, .912, num, transform=ax.transAxes, ha='center', fontsize=22, color='#225B87', fontweight='bold')
        ax.text(x+.1165, .864, label, transform=ax.transAxes, ha='center', fontsize=11)
    ax.text(.01, .791, '数值差异（分开列示，不将不同单位混在一个坐标轴）', transform=ax.transAxes, fontsize=13, fontweight='bold')
    card_specs = [(.008, '独立单步物理推进', f'{local:.3e}', '各状态分量原始数值差异的最大值\n角度为rad；角速度为rad/s'),
                  (.343, '记录指令整段重放', f'{whole:.3e}', '各状态分量原始数值差异的最大值\n角度为rad；角速度为rad/s'),
                  (.678, '策略动作复算', f'{policy:g}', '归一化策略动作差（无量纲）\n不等同于力矩误差（N·m）')]
    for x, title, value, note in card_specs:
        box(ax, x, .546, .306, .205, face='#F7F9FC')
        ax.text(x+.153, .711, title, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
        ax.text(x+.153, .649, value, transform=ax.transAxes, ha='center', fontsize=21, color='#225B87', fontweight='bold')
        ax.text(x+.153, .58, note, transform=ax.transAxes, ha='center', va='center', fontsize=9.5, linespacing=1.5, color='#536273')
    ax.text(.01, .493, '检查项目与记录状态', transform=ax.transAxes, fontsize=13, fontweight='bold')
    statuses = [
        ('测量值＝真值＋共享冻结噪声', f"{sum(r['every_measurement_equals_truth_plus_shared_noise'] for r in checks)}/{len(checks)}组一致", '逐元素相等检查'),
        ('按真实状态判定持续达标', f"{sum(r['true_state_acceptance'] for r in checks)}/{len(checks)}组通过", '独立重算验收、越界与指标'),
        ('命令力矩与力矩平方积分', '复算检查通过', '由核验源码断言及总体passed记录支持'),
        ('全部控制器×测试组覆盖', '全部27组已检查' if v['all_27_controller_group_results_checked'] else '覆盖不完整', '9种控制器×3组，全部覆盖'),
    ]
    for i, (label, status, note) in enumerate(statuses):
        y=.419-i*.077
        box(ax, .008, y-.032, .976, .065, face='#F0F6F4' if i%2==0 else '#F8FAFC')
        ax.text(.026, y, label, transform=ax.transAxes, va='center', fontsize=10.5)
        ax.text(.52, y, status, transform=ax.transAxes, va='center', fontsize=11, color='#16756E', fontweight='bold')
        ax.text(.744, y, note, transform=ax.transAxes, va='center', fontsize=8.3, color='#536273')
    ax.text(.01, .081, '状态推进由独立SciPy状态空间矩阵指数核验；策略复算复用原策略实现。', transform=ax.transAxes, fontsize=10, color='#536273')
    ax.text(.01, .045, '这些记录验证本次仿真代码与存档的一致性；独立数值核验不替代模型适用性检验或真实硬件验证。', transform=ax.transAxes, fontsize=9.5, color='#536273')
    ax.text(.01, .009, '来源：verification.json及supplement.py中verify、independent_metrics；原始字段和来源SHA-256随制图资料保存。', transform=ax.transAxes, fontsize=9, color='#536273')
    save(fig, '图11_独立数值核验结果')


def main():
    input_dir = HERE / '制图输入'
    input_dir.mkdir(exist_ok=True)
    sources=[]
    for name, rel in INPUTS.items():
        origin, target = SOURCE / rel, input_dir / name
        if not target.exists():
            shutil.copyfile(origin, target)
        if origin.exists() and sha(origin) != sha(target):
            raise ValueError(f'制图输入与研究来源不同：{name}')
        sources.append({'source': origin.relative_to(ROOT).as_posix(), 'local_copy': target.relative_to(HERE).as_posix(),
                        'sha256': sha(target), 'bytes': target.stat().st_size})
    summary, stats, v, frozen = (read(n) for n in ['summary.json', 'paired_statistics.json', 'verification.json', 'freeze.json'])
    assert len(summary)==3 and len(v['checks'])==27 and v['episodes']==17280 and v['transitions']==7603200
    assert stats['primary_comparison_count']==18 and stats['clusters_per_group']==128
    for controller in frozen['controllers']:
        if 'PD' in LABELS[controller['name']]:
            assert controller['settings']['ki']==0
    rows=[]
    for g in GROUPS:
        assert len(summary[g])==9
        for name in LABELS:
            r=summary[g][name]
            assert r['count']==640 and abs(r['passes']/640-r['success'])<1e-12
            rows.append(dict(group=g,group_label=GROUPS[g],source_name=name,label=LABELS[name],**{k:r[k] for k in
                             ['count','passes','success','physical_failures','cost','residual_relative_rms_deg']}))
    with (HERE/'图08_09_完整制图数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    primary=[dict(group=g,**r) for g in GROUPS for r in stats['comparisons'][g] if r['family']=='primary_success']
    (HERE/'图10_18项主要比较原值.json').write_text(json.dumps(primary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'名称映射.json').write_text(json.dumps({'labels':LABELS,'controllers':frozen['controllers'],
        'note':'保留原始源码键；积分增益为0的所选PID标为PD；PPO标号为训练随机种子。'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    success(summary);costs(summary);intervals(stats);verification(v)
    derived=dict(episodes=v['episodes'],transitions=v['transitions'],controller_groups=len(v['checks']),passed=v['passed'],
        all_27_controller_group_results_checked=v['all_27_controller_group_results_checked'],
        max_state_transition_residual=max(r['max_state_transition_residual'] for r in v['checks']),
        max_full_recorded_command_replay_error=max(r['max_full_recorded_command_replay_error'] for r in v['checks']),
        max_policy_action_error=max(r['max_policy_action_error'] for r in v['checks']),
        measurement_consistent_groups=sum(r['every_measurement_equals_truth_plus_shared_noise'] for r in v['checks']),
        true_state_acceptance_groups=sum(r['true_state_acceptance'] for r in v['checks']),
        units=dict(state_errors='源核验代码对各状态分量原始数值的最大值：角度rad、角速度rad/s；不是同量纲范数',
                   max_policy_action_error='归一化策略动作，无量纲；命令力矩另有assert_allclose，未保存力矩误差最大值'),
        cost_check='verify调用independent_metrics复算cost，assert_allclose(atol=3e-12)，总体passed=true；未保存独立代价差最大值')
    (HERE/'图11_核验指标原值.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    outputs=[dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.glob('图*.*'))
             if p.suffix in ('.png','.svg') and any(p.name.startswith(f'图{i:02}_') for i in range(8,12))]
    (HERE/'来源与导出清单.json').write_text(json.dumps({'scope':'仅重绘冻结的研究结果，未新增训练或测试。',
        'inputs':sources,'outputs':outputs,'plots':{
            '08':'summary.json：passes/count/success/physical_failures，全部27种方法组组合',
            '09':'summary.json：cost，完整0—22s平均实际命令力矩平方积分；未全成功者斜线',
            '10':'paired_statistics.json：primary_success原始mean_difference/lower/upper乘100；全部18比较',
            '11':'verification.json的覆盖/最大值/状态，以及supplement.py verify和independent_metrics范围'}},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'generated_png_svg':8,'source_files':len(sources),'chart_rows':27,'paired_comparisons':len(primary)},ensure_ascii=False))


if __name__=='__main__':
    main()
