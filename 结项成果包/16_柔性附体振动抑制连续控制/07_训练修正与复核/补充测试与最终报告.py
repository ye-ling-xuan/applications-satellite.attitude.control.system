"""Freeze supplementary references, audit their tests, report the full family."""
from pathlib import Path
import html
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'代码'))
import flexible_study as study


def full_metrics(states,commands,failed):
    # Independent expressions for all sample-window requirements and effort.
    tail=states[:,160:,:]
    tolerance=np.array([np.pi/180.,np.pi/360.,np.pi/1800.,np.pi/360.])
    return dict(success=np.all(np.abs(tail)<=tolerance,axis=(1,2))&~failed,
                failed=failed,cost=.05*np.einsum('nt,nt->n',commands,commands),
                rms_angle_deg=np.sqrt(np.mean(states[:,:,0]**2,axis=1))*180/np.pi,
                peak_relative_deg=np.max(np.abs(states[:,:,2]),axis=1)*180/np.pi,
                residual_relative_rms_deg=np.sqrt(np.mean(tail[:,:,2]**2,axis=1))*180/np.pi)


def supplementary_tests():
    directory=HERE/'04_补充独立测试'
    selected=study.read(HERE/'02_补充传统优化'/'selected.json')
    registry=[dict(name=name,settings=record['settings']) for name,record in selected.items()]
    frozen=dict(controllers=registry,selected_sha256=study.digest(HERE/'02_补充传统优化'/'selected.json'),
                test_scenarios_sha256=study.digest(HERE/'01_实验方案'/'test.json'),
                production_source_hashes=study.sources(),
                evaluator_sha256=study.digest(Path(__file__)),
                core_freeze_sha256=study.digest(HERE/'04_独立测试'/'freeze.json'))
    path=directory/'freeze.json'
    if path.exists() and study.read(path)!=frozen:raise RuntimeError('supplementary freeze differs')
    study.write(path,frozen)
    data=study.scenes('test')
    f,g=study.independent_transition_matrices(data)
    manifest=[]
    for controller in registry:
        name=controller['name']
        metric,states,commands=study.simulate(data,controller['settings'],record=True)
        folder=directory/name
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'responses.npz',state=states,command=commands,**metric)
        failed=np.zeros(len(data),bool)
        maximum=0.
        plant=study.Plant(len(data))
        plant.reset([s['theta'] for s in data],[s['omega'] for s in data],
                    [s['frequency'] for s in data],[s['damping'] for s in data])
        control=study.Controller(controller['settings'],len(data))
        control.reset(plant)
        for k in range(study.STEPS):
            plant.state[:]=states[:,k]
            plant.count[:]=k
            plant.failed[:]=failed
            np.testing.assert_allclose(control.command(plant),commands[:,k],rtol=0,atol=2e-12)
            predicted=np.einsum('nij,nj->ni',f,states[:,k])+g*commands[:,k,None]
            predicted[failed]=states[failed,k]
            residual=float(np.max(np.abs(predicted-states[:,k+1])))
            maximum=max(maximum,residual)
            if residual>2e-11:raise RuntimeError('independent supplementary physical replay differs')
            next_state=states[:,k+1]
            failed|=((np.abs(next_state[:,0])>np.deg2rad(170.))|
                     (np.abs(next_state[:,1])>np.deg2rad(300.))|
                     (np.abs(next_state[:,2])>np.deg2rad(10.)))
        recomputed=full_metrics(states,commands,failed)
        for key in metric:np.testing.assert_allclose(metric[key],recomputed[key],rtol=0,atol=2e-13)
        result=study.summary(recomputed)
        study.write(folder/'metrics.json',result)
        manifest.append(dict(name=name,episodes=len(data),transitions=len(data)*study.STEPS,
                             max_state_residual=maximum,responses_sha256=study.digest(folder/'responses.npz'),
                             metrics_sha256=study.digest(folder/'metrics.json')))
        print(json.dumps(dict(event='supplementary_test',name=name,**result),ensure_ascii=False),flush=True)
    output=dict(passed=True,controllers=manifest,episodes=sum(x['episodes'] for x in manifest),
                freeze_sha256=study.digest(path),full_physics_replay=True)
    study.write(directory/'verification.json',output)
    return output


def interval(a,b,groups,seed):
    difference=(np.asarray(a,float)-np.asarray(b,float)).reshape(5,groups).mean(axis=0)
    rng=np.random.default_rng(seed)
    means=np.concatenate([difference[rng.integers(0,groups,(1000,groups))].mean(axis=1) for _ in range(20)])
    low,high=np.quantile(means,[.05/36/2,1-.05/36/2])
    return dict(mean_difference=float(difference.mean()),lower=float(low),upper=float(high),
                confidence=1-.05/36,repeats=20000,resampling_unit='independent initial-state group')


def generate():
    core=study.read(HERE/'05_独立核验'/'verification.json')
    extra=study.read(HERE/'04_补充独立测试'/'verification.json')
    if not core['passed'] or not extra['passed']:raise RuntimeError('independent audits required')
    registry=study.read(HERE/'04_独立测试'/'freeze.json')['controllers']
    refined=study.read(HERE/'04_补充独立测试'/'freeze.json')['controllers']
    methods={}
    arrays={}
    for controller in registry+refined:
        name=controller['name']
        folder=HERE/('04_补充独立测试' if name.endswith('_refined') else '04_独立测试')/name
        methods[name]=study.read(folder/'metrics.json')
        with np.load(folder/'responses.npz') as archive:
            arrays[name]={key:archive[key].copy() for key in ('success','failed','cost')}
    learners=[s for s in registry if 'seed' in s]
    baselines=['PID','PID_safe','ZVD_PID','LQR','ZVD_PID_refined','LQR_refined']
    comparisons=[]
    for learner in learners:
        for i,baseline in enumerate(baselines):
            a,b=arrays[learner['name']],arrays[baseline]
            comparisons.append(dict(controller=learner['name'],seed=learner['seed'],baseline=baseline,
                selected_steps=learner['selected_steps'],
                learned_checkpoint=learner['selected_steps']>0,
                no_more_physical_failures=int(a['failed'].sum())<=int(b['failed'].sum()),
                both_all_pass=bool(a['success'].all() and b['success'].all()),
                success=interval(a['success'],b['success'],study.COUNTS['test'],16300+i),
                cost=interval(a['cost'],b['cost'],study.COUNTS['test'],16400+i),
                mean_cost_reduction_fraction=float(1-a['cost'].mean()/b['cost'].mean())))
    claims={}
    for baseline in baselines:
        records=[r for r in comparisons if r['baseline']==baseline]
        claims[baseline]=dict(
            success_advantage=len(records)==3 and all(r['learned_checkpoint'] and r['no_more_physical_failures'] and r['success']['lower']>0 for r in records),
            cost_advantage=len(records)==3 and all(r['learned_checkpoint'] and r['both_all_pass'] and r['cost']['upper']<0 for r in records))
    folder=HERE/'08_最终报告'
    study.write(folder/'全部对比与置信区间.json',dict(family_comparisons=36,comparisons=comparisons,claims=claims))
    labels={'PID':'原PID（通过数优先）','PID_safe':'保守PID（物理失败优先）',
            'ZVD_PID':'ZVD输入整形＋PID','LQR':'LQR（初始网格）',
            'ZVD_PID_refined':'ZVD输入整形＋PID（补充优化）','LQR_refined':'LQR（补充优化）'}
    for learner in learners:labels[learner['name']]=f'PID＋PPO（种子{learner["seed"]}）'
    order=baselines+[s['name'] for s in learners]
    training=[]
    for metadata in sorted((HERE/'03_学习训练').rglob('metadata.json')):
        record=study.read(metadata)
        if record.get('stage') in ('development','formal'):
            training.append(dict(algorithm=record['algorithm'],stage=record['stage'],seed=record['seed'],
                                 actual_steps=record['actual_steps'],selected_steps=record['best']['steps'],
                                 gradient_updates=record.get('gradient_updates'),status=record['status']))
    output=dict(methods=methods,claims=claims,formal_models=learners,training=training,
                training_steps=sum(r['actual_steps'] for r in training),
                held_out_episodes=core['total_episodes']+extra['episodes'],
                independent_initial_states=128,frequencies_per_initial=5,
                full_physics_audit_passed=True)
    study.write(folder/'最终结果.json',output)
    lines=['# 柔性附体振动抑制：最终报告','',
           '这次比较研究本体转向时的附体振动，不是路线选择。学习模型每0.05s输出连续力矩补偿，传统PID仍承担基本反馈。所有方法都有相同的本体和附体测量信息，真实频率与阻尼没有输入策略。','',
           '同样要求：8s后直到22s，本体误差≤1°、角速度≤0.5°/s、附体相对变形≤0.1°、相对速度≤0.5°/s；命令力矩≤0.5N·m。','',
           '| 方法 | 持续停稳/640 | 物理失败 | 平均力矩平方积分 | 平均残余变形均方根/° |',
           '|---|---:|---:|---:|---:|']
    rows=[]
    for name in order:
        result=methods[name]
        cells=[labels[name],f'{result["passes"]}/640',str(result['physical_failures']),
               f'{result["cost"]:.7f}',f'{result["residual_relative_rms_deg"]:.5f}']
        lines.append('| '+' | '.join(cells)+' |')
        rows.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in cells)+'</tr>')
    lines+=['','控制代价是实际力矩平方积分，不是电能。失败提前吸收可能降低它，因此没有全部达标的控制器，不能仅靠表里的较小代价宣称更省。','',
            '每种方法640回合来自128个独立初态×五个频率。置信区间按128个初态组重采样，不把640回合全当独立样本；36项同时比较使用Bonferroni校正。','']
    claim_text=[]
    for baseline in baselines:
        text=f'相对{labels[baseline]}：三种子一致的完成率优势={claims[baseline]["success_advantage"]}；双方全部达标条件下的代价优势={claims[baseline]["cost_advantage"]}。'
        lines.append('- '+text)
        claim_text.append('<li>'+html.escape(text)+'</li>')
    lines+=['',f'本轮完整开发及正式训练共{output["training_steps"]:,}交互步，{len(training)}次从零训练。三种子所选检查点步数分别为'+
            '、'.join(str(s['selected_steps']) for s in learners)+'。训练末模型可能退化，所以按照预先固定规则选择验证集最佳检查点，独立测试没有参与选择。',
            '',f'正式及补充独立测试共{output["held_out_episodes"]:,}回合，全部物理状态转移用独立状态空间矩阵指数重建，指令和指标同时核验通过。',
            '', '第一次开发的未成功结果留在上一级；第一次SAC中止的精确总步数没有保存，不能编造一个精确值，也不计入这里可精确核算的本轮训练总步数。',
            '', '局限：双惯量单柔性模态近似、理想力矩源、无测量噪声、假设可测附体振动；没有实物验证。优于PID不等于优于全部传统控制，补充优化的ZVD和LQR结果必须同时展示。']
    make_plots(folder,order,labels,methods,learners)
    lines+=['','![完整比较](完整比较.png)','','![固定输入初态示例](固定初态响应.png)','']
    (folder/'研究报告.md').write_text('\n'.join(lines),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>柔性附体连续控制研究</title>
<style>body{max-width:1200px;margin:36px auto;padding:0 22px;font:17px/1.8 system-ui;color:#17304a;background:#f9fbfd}h1{font-size:30px}table{border-collapse:collapse;background:white;width:100%;font-size:15px}td,th{border:1px solid #ccd6df;padding:10px;text-align:left}th{background:#e9f0f6}img{max-width:100%;background:white;margin:18px 0}.box{padding:16px;background:#e8f1fa;border-left:5px solid #356fa3}a{color:#125da5}</style>'''
    page+='<h1>本体停稳与柔性附体振动抑制</h1><p>学习模型每0.05s输出连续力矩补偿。所有方法同时满足本体和附体的稳定要求，并接受相同力矩约束。</p>'
    page+='<p class="box">这里的任务是连续控制。最终结论同时检验三个从零训练的PPO模型，并保留充分搜索的PID、ZVD输入整形和LQR对照。</p>'
    page+='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['方法','持续停稳/640','物理失败','平均力矩平方积分','平均残余变形RMS/°'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    page+='<p>先看持续停稳和物理失败，再比较代价。力矩平方积分不是电能；失败样本的低代价不代表控制更好。</p><ul>'+''.join(claim_text)+'</ul>'
    page+='<img src="完整比较.png" alt="所有方法的完成率和控制代价"><img src="固定初态响应.png" alt="按固定输入规则选取的轨迹示例">'
    page+=f'<p>{output["training_steps"]:,}交互步；{output["held_out_episodes"]:,}回合独立测试；全量物理状态转移独立核验通过。640回合由128个独立初态在五个频率上重复，置信区间按初态组计算并校正36项比较。</p>'
    page+='<p>模型局限：单柔性模态、理想执行器、无噪声、需要附体振动测量，没有硬件验证。完整参数、选择记录和不成功尝试均保留。</p><p><a href="研究报告.md">详细报告</a> · <a href="全部对比与置信区间.json">全部配对统计</a> · <a href="../修正预声明.md">训练修正预声明</a> · <a href="../补充对照预声明.md">补充传统对照预声明</a></p></html>'
    (folder/'研究报告.html').write_text(page,encoding='utf-8')
    return output


def make_plots(folder,order,labels,methods,learners):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    colors=['#6888a5' if name not in [s['name'] for s in learners] else '#d78a36' for name in order]
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    y=np.arange(len(order))
    axes[0].barh(y,[100*methods[x]['success'] for x in order],color=colors)
    axes[0].set(yticks=y,yticklabels=[labels[x] for x in order],xlabel='持续停稳率/%',xlim=(0,105))
    axes[0].invert_yaxis()
    for i,name in enumerate(order):axes[0].text(100*methods[name]['success']+.5,i,f'{methods[name]["success"]:.1%}',va='center',fontsize=9)
    bars=axes[1].barh(y,[methods[x]['cost'] for x in order],color=colors)
    axes[1].set(yticks=y,yticklabels=[labels[x] for x in order],xlabel='平均实际力矩平方积分/N²·m²·s')
    axes[1].invert_yaxis()
    for bar,name in zip(bars,order):
        if methods[name]['success']<1.:bar.set_hatch('//')
    axes[1].set_title('斜线：未全部达标，低代价不可直接解释为优势',fontsize=11)
    fig.savefig(folder/'完整比较.png',dpi=180)
    plt.close(fig)
    # Input-only selection rule, fixed independently of response quality.
    initial=study.read(HERE/'01_实验方案'/'test.json')
    index=max(range(len(initial)),key=lambda i:abs(initial[i]['theta']))
    show=['PID_safe','ZVD_PID_refined','LQR_refined']+[s['name'] for s in learners[:1]]
    fig,axes=plt.subplots(5,1,figsize=(12,13),sharex=True,layout='constrained')
    clock=np.arange(study.STEPS+1)*study.DT
    for name in show:
        folder_in=HERE/('04_补充独立测试' if name.endswith('_refined') else '04_独立测试')/name
        with np.load(folder_in/'responses.npz') as archive:
            for component in range(4):axes[component].plot(clock,np.rad2deg(archive['state'][index,:,component]),label=labels[name],lw=1.7)
            axes[4].step(clock[:-1],archive['command'][index],where='post',label=labels[name],lw=1.4)
    for axis,tolerance,label in zip(axes[:4],[1.,.5,.1,.5],['本体角度/°','本体角速度/(°/s)','相对变形/°','相对速度/(°/s)']):
        axis.axhspan(-tolerance,tolerance,color='#91bfa7',alpha=.18)
        axis.set_ylabel(label)
    axes[4].set_ylabel('已发力矩/N·m')
    for axis in axes:
        axis.axvline(8.,color='#5e6b75',ls='--',lw=1)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('时间/s')
    axes[0].legend(ncol=2,fontsize=9)
    axes[0].set_title(f'固定输入规则示例：测试初角绝对值最大的一例，f=0.4Hz；初角{np.rad2deg(initial[index]["theta"]):.2f}°')
    fig.savefig(folder/'固定初态响应.png',dpi=170)
    plt.close(fig)
    study.write(folder/'示例选择规则.json',dict(rule='largest absolute initial angle, first frequency 0.4Hz; response outcomes unused',sample_id=index))


def main():
    parser=__import__('argparse').ArgumentParser()
    parser.add_argument('command',choices=['test','report','all'])
    option=parser.parse_args().command
    if option in ('test','all'):supplementary_tests()
    if option in ('report','all'):generate()


if __name__=='__main__':main()
