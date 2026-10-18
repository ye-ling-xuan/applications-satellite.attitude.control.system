"""Read original validation metadata and draw all registered checkpoints; no training."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve()
PROJECT = next(p for p in HERE.parents if (p/'结项成果包').is_dir() and (p/'答辩讲解包').is_dir())
STUDY = PROJECT/'结项成果包/16_柔性附体振动抑制连续控制/07_训练修正与复核'
STEP = HERE.parents[1]

for candidate in (Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf')):
    if candidate.is_file():
        font_manager.fontManager.addfont(str(candidate))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(candidate)).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.0), sharey=True)
colours = ['#276aab', '#d4832b', '#3d8758']
sets = [
    [('development', 'SAC', 16202), ('development', 'TD3', 16201), ('development', 'PPO', 16203)],
    [('formal', 'PPO', 111), ('formal', 'PPO', 222), ('formal', 'PPO', 333)],
]
for ax, records in zip(axes, sets):
    for colour, (stage, algo, seed) in zip(colours, records):
        path = STUDY/f'03_学习训练/{stage}/{algo}/seed_{seed}/metadata.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        snaps = data['snapshots']
        ax.plot([s['steps']/1000 for s in snaps], [100*s['validation']['success'] for s in snaps],
                '-o', color=colour, markersize=3.3, linewidth=1.7, label=f'{algo} / 种子 {seed}')
        chosen = data['best']
        ax.scatter(chosen['steps']/1000, 100*chosen['validation']['success'], marker='D',
                   s=57, facecolor='white', edgecolor=colour, linewidth=1.6, zorder=5)
    ax.set_xlabel('实际环境交互步数（千步）')
    ax.set_ylim(0, 106)
    ax.grid(alpha=.20)
    ax.legend(fontsize=9, loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('固定验证集成功率（%）')
axes[0].set_title('开发比较：各524,288步；SAC/TD3最佳仍为零步')
axes[1].set_title('正式PPO：完整训练保留；后段存在明显退化')
fig.suptitle('选用验证最佳检查点，并不表示训练稳定收敛', fontsize=15, weight='bold', y=.99)
fig.text(.5, .012, '菱形：按预设规则选中的检查点。数据仅来自metadata.json；未用独立测试结果选择模型。',
         ha='center', fontsize=9, color='#4d6073')
fig.tight_layout(rect=(0, .045, 1, .94))
target = STEP/'图表/训练检查点与末模型退化.png'
target.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(target, dpi=160, facecolor='white')
plt.close(fig)
print(target)
