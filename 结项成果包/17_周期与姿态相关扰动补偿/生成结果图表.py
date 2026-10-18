"""Plot frozen held-out results without changing any controller or model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import disturbance_study as s
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

verification=s.read_json(s.STUDY/'05_独立核验'/'verification.json')
if not verification['passed']:raise RuntimeError('verification required before final figures')
summary=s.read_json(s.STUDY/'04_独立测试'/'summary.json')
results=summary['results'];names=list(results)
labels={'pid':'整定PID','observer':'扰动观测器＋PID','harmonic':'谐波前馈＋PID'}
for name in names:
    if name not in labels:labels[name]='TD3＋PID\n种子'+name.rsplit('_',1)[-1]
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
out=s.STUDY/'06_报告'/'图表';out.mkdir(parents=True,exist_ok=True)
colors=['#64748b','#16a34a','#0d9488']+['#2563eb']*(len(names)-3)
fig,axes=plt.subplots(1,2,figsize=(15,5.3),layout='constrained')
rms=[results[n]['mean_steady_rms_deg'] for n in names]
cost=[results[n]['mean_actual_cost'] for n in names]
positions=np.arange(len(names))
for ax,values,title,ylabel in ((axes[0],rms,'8–32秒平均角度RMS','角度RMS（°，对数坐标）'),
                               (axes[1],cost,'全回合实际力矩平方积分','力矩平方积分（N²·m²·s）')):
    bars=ax.bar(positions,values,color=colors,width=.68)
    ax.set_xticks(positions,[labels[n] for n in names],fontsize=9)
    ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value*1.05,f'{value:.4f}',ha='center',va='bottom',fontsize=9)
axes[0].set_yscale('log');axes[0].set_ylim(min(rms)*.5,max(rms)*2.2)
axes[1].set_ylim(0,max(cost)*1.2)
fig.suptitle('独立测试：学习补偿与完整传统对照（128个相同场景）',fontsize=15)
fig.savefig(out/'独立测试指标对比.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
times=np.arange(s.STEPS+1)*s.DT
for name,color in zip(names,colors):
    trace=np.load(s.STUDY/'04_独立测试'/f'{name}_traces.npz')
    axes[0].plot(times,np.degrees(trace['theta'][:,0]),label=labels[name].replace('\n',' '),color=color,lw=1.3,alpha=.9)
    axes[1].plot(times,trace['tau'][:,0],label=labels[name].replace('\n',' '),color=color,lw=1.2,alpha=.9)
axes[0].set_xlim(s.DEADLINE,s.DURATION);axes[0].set_ylabel('角度误差（°）')
axes[1].set_ylabel('实际执行力矩（N·m）');axes[1].set_xlabel('时间（s）')
for ax in axes:ax.grid(alpha=.2)
axes[0].legend(ncol=3,fontsize=9)
fig.suptitle('固定展示独立测试场景0：持续扰动下的精密指向',fontsize=15)
fig.savefig(out/'场景0指向与实际力矩.png',dpi=160);plt.close(fig)
print(out)
