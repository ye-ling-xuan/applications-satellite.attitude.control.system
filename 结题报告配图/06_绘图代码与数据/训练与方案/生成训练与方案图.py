"""Read existing records and render report figures; never run an experiment."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
OUT = ROOT / '05_完整图表版'
HERE = Path(__file__).resolve().parent
BASE = PROJECT / '结项成果包/16_柔性附体振动抑制连续控制/07_训练修正与复核'
SUP = BASE / '10_补充实验与论文复核'
SOURCES = {}
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 12, 'axes.titlesize': 15, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})


def source(path):
    SOURCES[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(source(path).read_text(encoding='utf-8'))


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def card(ax, x, y, w, h, title, text, color='#edf4fa'):
    ax.add_patch(FancyBboxPatch((x,y), w,h,boxstyle='round,pad=0.014',
                              facecolor=color, edgecolor='#c5d4df', transform=ax.transAxes))
    ax.text(x+.025, y+h-.055, title, transform=ax.transAxes, ha='left',va='top',
            fontsize=14, weight='bold', color='#183e5a')
    ax.text(x+.025, y+h-.20, text, transform=ax.transAxes, ha='left',va='top',
            fontsize=12, linespacing=1.8, color='#233a4c')


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    rows=[]
    for stage,algo,seed in [('development','SAC',16202),('development','TD3',16201),
                            ('development','PPO',16203),('formal','PPO',111),
                            ('formal','PPO',222),('formal','PPO',333)]:
        path=BASE / f'03_学习训练/{stage}/{algo}/seed_{seed}/metadata.json'
        meta=read(path)
        assert meta['status']=='complete'
        rows.append({'stage':stage,'algorithm':algo,'seed':seed,
                     'steps':meta['actual_steps'],'best_steps':meta['best']['steps'],
                     'snapshots':meta['snapshots'],'best':meta['best']})
    old=read(BASE/'08_最终报告/最终结果.json')
    summary=read(SUP/'结果摘要.json')
    config=read(BASE/'01_实验方案/执行配置.json')
    protocol=read(SUP/'01_方案与冻结/预声明.json')
    total=sum(r['steps'] for r in rows)
    assert total==old['training_steps']==4718592
    assert len(rows)==6 and summary['new_test_episodes']==17280
    counts=[old['total_test_episodes_including_post_hoc'],summary['new_test_episodes']]

    fig,axes=plt.subplots(2,1,figsize=(11,8),gridspec_kw={'height_ratios':[2.2,1]})
    labels=[('开发 '+r['algorithm'] if r['stage']=='development' else '正式 PPO')+f" / {r['seed']}" for r in rows]
    y=np.arange(6)
    axes[0].barh(y,[r['steps']/1e6 for r in rows],color=['#7892ad']*3+['#dc8615']*3)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,1.32)
    axes[0].set_xlabel('实际环境交互步数（百万步）')
    for i,r in enumerate(rows): axes[0].text(r['steps']/1e6+.02,i,f"{r['steps']:,}",va='center')
    axes[0].set_title(f'6份完整训练记录：3份开发＋3份正式；合计 {total:,} 步',loc='left')
    axes[1].barh([0,1],counts,color=['#7892ad','#208c86'])
    axes[1].set_yticks([0,1],['原柔性研究测试（含追加PID复核）','新增冻结模型测试']);axes[1].invert_yaxis()
    axes[1].set_xlim(0,21500);axes[1].set_xlabel('实际测试回合数（不等于独立初态数）')
    for i,c in enumerate(counts):axes[1].text(c+200,i,f'{c:,}',va='center')
    fig.suptitle('柔性附件阶段的训练与测试产出',fontsize=18,weight='bold')
    fig.text(.02,.015,'新增测试未重新训练或调参；17,280＝3组×9控制器×640场景。原研究和新测试分别报告。\n配套文书：2篇学习综述、论文初稿、八步答辩讲解包及离线阅读页。',fontsize=11)
    fig.tight_layout(rect=[0,.055,1,.95]);save(fig,'图01_柔性训练与测试数量')

    fig,axes=plt.subplots(2,1,figsize=(11,9),sharey=True)
    colors=['#247cb7','#d47b12','#26865e']
    curves=[]
    for ax,stage,title in zip(axes,['development','formal'],['开发比较：每种算法524,288步','正式PPO：每个模型1,048,576步']):
        for color,r in zip(colors,[r for r in rows if r['stage']==stage]):
            steps=[s['steps'] for s in r['snapshots']]
            rates=[s['validation']['success']*100 for s in r['snapshots']]
            ax.plot(np.array(steps)/1000,rates,'o-',color=color,label=f"{r['algorithm']} / 种子{r['seed']}",ms=4)
            ax.plot(r['best_steps']/1000,r['best']['validation']['success']*100,'D',color=color,
                    markerfacecolor='white',ms=9,mew=2)
            if stage=='formal':
                ax.annotate(f"{rates[-1]:.1f}%",(steps[-1]/1000,rates[-1]),xytext=(7,0),
                            textcoords='offset points',va='center',fontsize=10,color=color)
            curves.append({'stage':stage,'algorithm':r['algorithm'],'seed':r['seed'],
                           'steps':steps,'validation_success_percent':rates,'selected_steps':r['best_steps']})
        if stage=='formal':
            ax.text(.98,.98,'选中步数：111 → 262,144；222 → 393,216；333 → 327,680',
                    transform=ax.transAxes,ha='right',va='top',fontsize=10,color='#233a4c',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        ax.set_title(title,loc='left');ax.set_ylim(0,115);ax.set_yticks(range(0,101,20));ax.set_ylabel('验证成功率（%）')
        ax.set_xlabel('实际环境交互步数（千步）');ax.grid(alpha=.18);ax.legend(loc='lower left',ncol=3,fontsize=10)
    fig.suptitle('全部开发与正式训练的验证过程',fontsize=18,weight='bold')
    fig.text(.02,.015,'菱形为按预设验证规则选中的检查点。每次验证320场景＝64初态×5频率；曲线是验证表现，不是测试成功率。\n三个正式末模型均退化，最终测试使用选中模型；测试结果未用于选择检查点。',fontsize=11)
    fig.tight_layout(rect=[0,.065,1,.96]);save(fig,'图04_算法开发与正式训练验证曲线')

    fig=plt.figure(figsize=(11,11));grid=fig.add_gridspec(3,1,height_ratios=[1.5,.72,1.5],hspace=.22)
    ax=fig.add_subplot(grid[0]);ax.axis('off')
    card(ax,.02,.05,.45,.86,'柔性对象与参数范围',
         '本体惯量：0.7 kg·m²；附件：0.3 kg·m²\n模态频率：0.4～0.8 Hz\n阻尼比：0.01～0.04\n频率、阻尼在回合内保持不变')
    tol=config['tolerance_deg_or_deg_s']
    card(ax,.53,.05,.45,.86,'四项持续验收（真实状态）',
         f'本体角误差：|θ| ≤ {tol[0]:g}°\n本体角速度：|ω| ≤ {tol[1]:g}°/s\n附件相对角度：|q| ≤ {tol[2]:g}°\n附件相对角速度：|v| ≤ {tol[3]:g}°/s', '#eef6ef')
    ax=fig.add_subplot(grid[1]);ax.axvspan(0,8,color='#e2ebf4');ax.axvspan(8,22,color='#d5eee4')
    ax.set_xlim(0,22);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_xticks([0,8,22],['0秒','8秒','22秒'])
    ax.text(4,.6,'机动及减振',ha='center');ax.text(15,.6,'每个采样点四项均达标',ha='center',weight='bold')
    ax.set_title('采样0.05秒；每回合440步；控制力矩限幅±0.5 N·m',fontsize=13)
    ax=fig.add_subplot(grid[2]);ax.axis('off')
    card(ax,.02,.18,.45,.74,'三组独立抽取初态',
         'A：原分布全新样本\nB：增加初始小振动\nC：初始小振动＋假定高斯测量噪声\n不同控制器复用相同场景和规定噪声')
    card(ax,.53,.18,.45,.74,'测试规模与分组统计',
         '每组128初态 × 5种频率＝640场景\n3组 × 9控制器 × 640＝17,280回合\n统计按128初态成组，复用5频率\n18项主要比较；配对重采样200,000次', '#fff3df')
    fig.suptitle('柔性附件实验设置与持续停稳判据',fontsize=18,weight='bold',y=.97)
    fig.text(.05,.035,'这是测试设计图，不是性能结果。组间初态独立抽取，不将组间差异作单因素因果解释。\n控制代价为完整22秒的力矩平方积分，单位(N·m)²·s；不等于实际电能。',fontsize=11)
    fig.subplots_adjust(top=.93,bottom=.09);save(fig,'图05_参数范围与测试验收设计')

    copies=[]
    for previous,current in [('图1_本体与柔性附件双惯量模型.png','图02_双惯量柔性附件模型.png'),
                             ('图2_PD反馈与PPO连续力矩补偿结构.png','图03_PD与PPO补偿控制结构.png')]:
        original=ROOT/'01_正文用图'/previous
        source(original);shutil.copy2(original,OUT/current)
        copies.append({'source':original.relative_to(PROJECT).as_posix(),'file':current})
    for previous,current in [('图1_本体与柔性附件双惯量模型.svg','图02_双惯量柔性附件模型.svg'),
                             ('图2_PD反馈与PPO连续力矩补偿结构.svg','图03_PD与PPO补偿控制结构.svg')]:
        original=ROOT/'03_SVG原图'/previous
        source(original);shutil.copy2(original,OUT/current)
        copies.append({'source':original.relative_to(PROJECT).as_posix(),'file':current})
    data={'training_runs':rows,'training_total_steps':total,'test_counts':counts,
          'configuration':config,'new_test_protocol':protocol,'training_curves':curves,'copied_figures':copies}
    (HERE/'制图数值.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'来源与哈希.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Built figures 01, 04, 05; copied existing diagrams as 02, 03. No experiment was run.')


if __name__=='__main__':
    main()
