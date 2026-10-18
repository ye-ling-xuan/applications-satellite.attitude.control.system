"""Readable, audited delivery; adds no fitting or model selection."""
from pathlib import Path
import html
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'代码'))
import study as b


def stricter_interval(reference,candidate):
    # Six conservative claims: three seeds against PID and three against the
    # identical feedforward PID base.  The original primary intervals remain.
    pairs=np.array([[a['rms_error_deg'],c['rms_error_deg']] for a,c in zip(reference,candidate)])
    assert all(a['scenario_id']==c['scenario_id'] for a,c in zip(reference,candidate))
    rng=np.random.default_rng(202610181811)
    boot=np.empty((20000,2))
    for start in range(0,20000,1000):
        ix=rng.integers(0,len(pairs),(1000,len(pairs)));boot[start:start+1000]=pairs[ix].mean(axis=1)
    alpha=.05/6
    low,high=np.quantile(boot[:,0]-boot[:,1],[alpha/2,1-alpha/2])
    return dict(confidence=1-alpha,difference_interval=[float(low),float(high)],reference_mean=float(pairs[:,0].mean()),
                candidate_mean=float(pairs[:,1].mean()),relative_reduction=float(1-pairs[:,1].mean()/pairs[:,0].mean()))


def main():
    audit=b.read_json(ROOT/'05_独立核验'/'verification.json')
    if not audit['passed']:raise RuntimeError('audit must pass first')
    frozen=b.read_json(ROOT/'04_独立测试'/'freeze.json')
    responses={entry['name']:b.read_json(ROOT/'04_独立测试'/entry['name']/'results.json') for entry in frozen['controllers']}
    formal=[e for e in frozen['controllers'] if e.get('revision') and not e.get('diagnostic')]
    intervals={entry['name']:{name:stricter_interval(responses[name]['rows'],responses[entry['name']]['rows'])
               for name in ('PID','PID_feedforward_preview')} for entry in formal}
    contribution=bool(len(formal)==3 and all(e['selected_steps']>0 and intervals[e['name']]['PID_feedforward_preview']['difference_interval'][0]>0
                      and responses[e['name']]['summary']['physical_failures']<=responses['PID_feedforward_preview']['summary']['physical_failures'] for e in formal))
    output=ROOT/'06_研究报告';output.mkdir(exist_ok=True)
    b.write_json(output/'AI增量核验结论.json',dict(learning_increment_beyond_identical_feedforward_base=contribution,
                 more_conservative_six_claim_intervals=intervals,explanation='Six simultaneous intervals use Bonferroni 99.1667%; stricter than the original three primary 98.3333% intervals.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    labels=['PID','PID＋前馈和预览','MPC（SLSQP）']+[f'PPO＋同一前馈PID\n种子{e["seed"]}' for e in formal]
    names=['PID','PID_feedforward_preview','MPC_SLSQP']+[e['name'] for e in formal]
    means=[responses[name]['summary']['mean_rms_error_deg'] for name in names]
    costs=[responses[name]['summary']['mean_actual_cost'] for name in names]
    fig,axs=plt.subplots(1,2,figsize=(13,4.7))
    colors=['#8296aa','#557db1','#657a65']+['#d08445']*len(formal)
    for ax,data,title,unit in [(axs[0],means,'持续扫描的跟踪误差','角度RMS误差 / °'),(axs[1],costs,'实际力矩平方积分','N²·m²·s')]:
        bars=ax.bar(np.arange(len(names)),data,color=colors)
        ax.set_xticks(np.arange(len(names)),labels,rotation=20,ha='right',fontsize=9)
        ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for bar,value in zip(bars,data):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.3f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('128条相同独立扫描轨迹；所有方法使用相同幅值和变化速率限制',fontsize=12)
    fig.tight_layout();fig.savefig(output/'控制器对比.png',dpi=180);plt.close(fig)
    # A declared deterministic illustration: median improvement, never the
    # largest favourable example. All per-episode records remain accessible.
    if formal:
        chosen=formal[0]['name'];base=responses['PID_feedforward_preview']['rows'];ai=responses[chosen]['rows']
        delta=np.array([a['rms_error_deg']-c['rms_error_deg'] for a,c in zip(base,ai)])
        sample=int(np.argsort(delta)[len(delta)//2]);records=responses[chosen]['scenes'];ref=b.Reference(records)
        time=np.arange(b.N_STEPS+1)*b.DT;target=ref.at(np.broadcast_to(time,(len(records),len(time))))[0][sample]
        fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
        axs[0].plot(time,np.rad2deg(target),'k--',label='扫描指令',lw=1.8)
        for name,label,color in [('PID','PID','#8296aa'),('PID_feedforward_preview','同一前馈PID基底','#557db1'),(chosen,'加入PPO连续补偿','#d08445'),('MPC_SLSQP','MPC','#657a65')]:
            trace=np.load(ROOT/'04_独立测试'/name/'traces.npz')
            axs[0].plot(time,np.rad2deg(trace['theta'][:,sample]),label=label,color=color)
            axs[1].plot(time,trace['torque'][:,sample],label=label,color=color)
        axs[0].set_ylabel('姿态角 / °');axs[1].set_ylabel('实际力矩 / N·m');axs[1].set_xlabel('时间 / s')
        axs[0].set_title(f'按PPO相对前馈基底误差改善排序的中位数样例：{records[sample]["scenario_id"]}')
        for ax in axs:ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=9)
        fig.tight_layout();fig.savefig(output/'中位数改善样例.png',dpi=180);plt.close(fig)
    rows=[]
    for name in names:
        stats=responses[name]['summary']
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{stats["mean_rms_error_deg"]:.4f}</td><td>{stats["mean_actual_cost"]:.5f}</td><td>{stats["physical_failures"]}/{stats["count"]}</td></tr>')
    conclusion='三个正式种子均在独立测试中，相对同一前馈PID基底降低跟踪误差。' if contribution else '未证明三个正式种子均相对同一前馈PID基底获得稳定学习增量。'
    details=[]
    for entry in formal:
        ci=intervals[entry['name']]['PID_feedforward_preview'];stats=responses[entry['name']]['summary'];baseline=responses['PID_feedforward_preview']['summary']
        effort=stats['mean_actual_cost']/baseline['mean_actual_cost']-1
        details.append(f'<li>种子{entry["seed"]}：相对同一前馈PID的RMS误差降低{100*ci["relative_reduction"]:.2f}%；误差差值99.1667%区间[{ci["difference_interval"][0]:.5f}, {ci["difference_interval"][1]:.5f}]°；力矩平方积分变化{effort*100:+.2f}%。</li>')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>执行器约束下的连续姿态跟踪</title>
<style>body{{max-width:1100px;margin:36px auto;padding:0 22px;font:17px/1.85 "Microsoft YaHei",sans-serif;color:#243746}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #ccd6df;padding:9px;text-align:left}}th{{background:#edf3f7}}img{{width:100%;margin:15px 0}}.finding{{padding:16px;background:#edf4f9;border-left:5px solid #557db1}}code{{background:#f0f3f5}}</style>
<h1>执行器约束下的连续姿态跟踪</h1><p class="finding">{conclusion}</p>
<p>卫星持续跟随随时间变化的扫描指令。执行器力矩既不能超过±0.25 N·m，每秒也最多变化0.08 N·m；突然改变指令不会让真实力矩瞬间改变。这是连续控制任务，AI每0.1秒输出一次连续力矩补偿。</p>
<p>所有方法都能获得相同的姿态、角速度、实际力矩和未来3秒已知扫描指令。AI基本反馈使用已经整定的“PID＋模型前馈和预览”，关闭AI时严格恢复这一基底。</p>
<table><tr><th>方法</th><th>平均角度RMS误差 / °</th><th>平均力矩平方积分 / N²·m²·s</th><th>物理失败</th></tr>{''.join(rows)}</table>
<img src="控制器对比.png" alt="控制器对比"><ul>{''.join(details)}</ul>
<p>RMS越低表示持续跟踪越准确；力矩平方积分越低表示本研究定义的力矩使用代价越低。它不是实测电能。跟踪误差改善和代价上升可以同时发生，不能称为所有指标全面优越。</p>
<h2>训练做了什么改进</h2><p>原TD3、PPO开发均未在固定综合验证目标上超越零补偿，原始尝试完整保留。后续追加声明采用较小补偿、低学习率、较长rollout，并利用物理反号对称性约束PPO演员：所有姿态、角速度和力矩同时反号时，输出补偿也反号。此结构从训练起保持一致，没有测试时修改动作，也没有PID动作标签。</p>
<p>AI是否增加作用，应看它相对同一前馈PID基底的比较。只比较普通PID，会把模型前馈和预览带来的收益也混算到AI上。本页同时报告两种比较。</p>
<h2>范围和局限</h2><p>这是单轴、理想传感器反馈、固定执行器约束的仿真。传统PID、前馈和预览经过开发与验证整定。MPC求解的是固定3秒预测时域和固定权重的凸二次规划；完整保存求解残差，但尚未进行全面的预测时域、终端代价和权重优化。因此结果只能比较本次实现，不能声称AI优于充分优化的所有MPC。</p>
<p>测试为128条独立扫描轨迹，每条30秒；模型冻结后才打开测试响应。{audit["episodes"]}回合、{audit["physical_intervals"]}个区间完成全量物理、指令和指标回放，并以独立ODE积分检查代表区间。</p>
<p><a href="研究报告.md">完整报告</a> · <a href="开发过程说明.md">开发过程说明</a> · <a href="AI增量核验结论.json">学习增量与区间</a> · <a href="../05_独立核验/verification.json">全量核验</a></p>
<img src="中位数改善样例.png" alt="中位数改善样例"></html>'''
    (output/'研究报告.html').write_text(body,encoding='utf-8')
    (output/'读者说明.md').write_text('# 最重要的结论\n\n'+conclusion+'\n\n'
        +'\n'.join(f'- 种子{e["seed"]}：相对相同前馈PID，RMS误差降低{100*intervals[e["name"]]["PID_feedforward_preview"]["relative_reduction"]:.2f}%。' for e in formal)
        +'\n\n不能把前馈本身的收益算作AI贡献。MPC仅代表本次固定时域、固定权重实现，未充分优化终端代价。力矩平方积分不是电能。详见同目录研究报告.html。\n',encoding='utf-8')
    print(json.dumps(dict(learning_increment=contribution,intervals=intervals),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
