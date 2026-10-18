"""Figures from frozen jointly trained three-axis test responses."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import joint_study as s
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
verification=s.read_json(s.STUDY/'05_独立核验'/'verification.json')
if not verification['passed']:raise RuntimeError('complete verification first')
results=s.read_json(s.STUDY/'04_独立测试'/'summary.json')['results'];names=list(results)
labels={'pid':'整定四元数PID','observer':'扰动观测器＋PID','harmonic':'谐波前馈＋PID'}
for name in names:
    if name not in labels:labels[name]='联合TD3＋PID\n种子'+name.rsplit('_',1)[-1]
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
out=s.STUDY/'06_报告'/'图表';out.mkdir(parents=True,exist_ok=True)
colors=['#64748b','#16a34a','#0d9488']+['#2563eb']*(len(names)-3)
fig,axes=plt.subplots(1,2,figsize=(15,5.4),layout='constrained')
for ax,field,title,ylabel in ((axes[0],'mean_steady_rms_deg','8–32秒总姿态角RMS','角度RMS（°，对数坐标）'),(axes[1],'mean_actual_cost','全回合三轴力矩平方积分之和','实际控制代价（N²·m²·s）')):
    values=[results[n][field] for n in names];bars=ax.bar(np.arange(len(names)),values,color=colors,width=.68)
    ax.set_xticks(np.arange(len(names)),[labels[n] for n in names],fontsize=9);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v*1.05,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
    if field=='mean_steady_rms_deg':ax.set_yscale('log');ax.set_ylim(min(values)*.5,max(values)*2.2)
    else:ax.set_ylim(0,max(values)*1.2)
fig.suptitle('真实三轴耦合、联合网络：128个相同独立测试场景',fontsize=15);fig.savefig(out/'三轴联合控制独立测试对比.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True);times=np.arange(s.STEPS+1)*s.DT
for name,color in zip(names,colors):
    trace=np.load(s.STUDY/'04_独立测试'/f'{name}_traces.npz');angle=np.degrees(s.quaternion_angle(trace['quaternion'][:,0,:]));rate=np.degrees(np.linalg.norm(trace['omega'][:,0,:],axis=-1))
    axes[0].plot(times,angle,label=labels[name].replace('\n',' '),color=color,lw=1.2);axes[1].plot(times,rate,color=color,lw=1.2)
axes[0].set_ylabel('总姿态角误差（°）');axes[1].set_ylabel('角速度范数（°/s）');axes[1].set_xlabel('时间（s）')
axes[0].legend(ncol=3,fontsize=9)
for ax in axes:ax.grid(alpha=.2);ax.set_xlim(s.DEADLINE,s.DURATION)
fig.suptitle('固定独立场景0：三轴持续指向响应',fontsize=15);fig.savefig(out/'场景0三轴持续指向.png',dpi=160);plt.close(fig)
print(out)
