"""Prospectively frozen confirmation of existing flexible-body controllers.

No training or tuning commands are provided. Truth and sensor observations are separate.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import datetime
import html
import json
import sys

import numpy as np
# Initialize numerical libraries before importing Torch/SB3 (Windows OMP order).
from scipy.linalg import expm
_ = np.linalg.solve(np.eye(4), np.ones(4))
_ = expm(np.zeros((5, 5)))

HERE = Path(__file__).resolve().parents[1]
OLD = HERE.parent
sys.path.insert(0, str(OLD/'代码'))
import flexible_study as s

GROUPS = ('A_original_confirmation', 'B_initial_vibration', 'C_vibration_and_noise')
SEEDS = dict(zip(GROUPS, (2026101816101, 2026101816102, 2026101816103)))
NOISE_SEED = 2026101816104
N, REPEATS = 128, 200000
NOISE_STD = np.deg2rad([.01, .02, .005, .02])
NAMES = ('PID', 'PID_safe', 'PID_strengthened', 'ZVD_PID', 'ZVD_PID_refined', 'LQR_refined',
         'PID_PPO_111', 'PID_PPO_222', 'PID_PPO_333')
LABELS = dict(PID='原PID', PID_safe='同一PID基底', PID_strengthened='补充搜索PID',
              ZVD_PID='原ZVD输入整形＋PID', ZVD_PID_refined='优化ZVD输入整形＋PID',
              LQR_refined='优化LQR', **{f'PID_PPO_{x}':f'PPO补偿＋PID（种子{x}）' for x in (111,222,333)})
GROUP_LABELS = dict(zip(GROUPS, ('A 原分布独立确认', 'B 加入初始附体振动', 'C 初始附体振动＋测量噪声')))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, value): s.write(p, value)
def sha(p): return s.digest(p)

def registry():
    base = read(OLD/'02_传统控制/selected.json')
    refined = read(OLD/'02_补充传统优化/selected.json')
    strengthened = read(OLD/'09_事后加强PID复核/selected.json')
    items = [dict(name=name, settings=base[name]['settings']) for name in ('PID','PID_safe')]
    items += [dict(name='PID_strengthened', settings=strengthened['settings']),
              dict(name='ZVD_PID', settings=base['ZVD_PID']['settings'])]
    items += [dict(name=name, settings=refined[name]['settings']) for name in ('ZVD_PID_refined','LQR_refined')]
    for seed in (111,222,333):
        folder = OLD/f'03_学习训练/formal/PPO/seed_{seed}'
        meta = read(folder/'metadata.json')
        if meta['status'] != 'complete': raise RuntimeError('incomplete old model')
        items.append(dict(name=f'PID_PPO_{seed}', settings=base['PID_safe']['settings'], seed=seed,
                          model=str((folder/'best.zip').relative_to(OLD)), selected_steps=meta['best']['steps'],
                          old_training_steps=meta['actual_steps']))
    assert tuple(i['name'] for i in items)==NAMES
    return items

def old_inputs():
    paths = [OLD/'代码/flexible_study.py', OLD/'代码/test_flexible.py',
             OLD/'02_传统控制/selected.json', OLD/'02_补充传统优化/selected.json',
             OLD/'09_事后加强PID复核/selected.json', OLD/'01_实验方案/执行配置.json']
    for split in ('development','validation','test'): paths.append(OLD/f'01_实验方案/{split}.json')
    for seed in (111,222,333):
        paths += [OLD/f'03_学习训练/formal/PPO/seed_{seed}/{x}' for x in ('metadata.json','best.zip')]
    return {p.relative_to(OLD).as_posix():sha(p) for p in paths}

def source_hashes():
    return {p.relative_to(HERE).as_posix():sha(p) for p in sorted((HERE/'代码').glob('*.py'))}

def protocol():
    return dict(version=1,task='Frozen-controller prospective flexible appendage confirmation and stress tests',
        new_training_steps=0,new_parameter_searches=0,old_results_seen=True,
        selection='all nine previously fixed controllers; no test-driven selection or changes',
        groups=GROUPS,initial_seed=SEEDS,noise_seed=NOISE_SEED,initial_states_per_group=N,
        frequencies_Hz=s.FREQUENCIES,frequency_and_damping_constant_within_episode=True,
        dt_s=s.DT,horizon_s=s.HORIZON,deadline_s=s.DEADLINE,inertias=[s.JB,s.JF],command_limit_Nm=s.LIMIT,
        initial_distributions=dict(theta_deg=[-30,30],omega_deg_s=[-2,2],damping=[.01,.04],
            A_q_deg=0,A_qdot_deg_s=0,BC_q_deg=[-.05,.05],BC_qdot_deg_s=[-.2,.2]),
        observation_noise=dict(group='C only',distribution='independent zero-mean Gaussian per scene, component and sample',
            component_order=['theta','omega','relative_angle','relative_rate'],
            std_deg_or_deg_s=[.01,.02,.005,.02],same_sample_for_all_methods=True,
            noise_at_initial_sample=True,noise_in_true_physics_or_acceptance=False,
            assumed_sensor_scenario=True,calibrated_to_real_sensor=False),
        acceptance=dict(all_samples_between_s=[8,22],true_state_tolerances_deg_or_deg_s=[1,.5,.1,.5],
            failure_bounds_deg_or_deg_s=[170,300,10],absorption='failure freezes state, zero subsequent torque'),
        methods=NAMES,policy_inputs='measured four states, own integral, previous sent command, elapsed time; no frequency/damping',
        primary=dict(metric='PPO minus reference success fraction',references=['PID_strengthened','PID_safe'],
            training_seeds=[111,222,333],comparisons=18,familywise_alpha=.05,
            interval='two-sided paired cluster percentile bootstrap; simultaneous Bonferroni',
            clusters='128 independent initial states; each cluster contains all five frequencies',
            repetitions=REPEATS,bootstrap_seed=2026101816190,
            superiority='each of three seeds has lower CI>0 against BOTH references, evaluated separately for each group',
            zero_physical_failure_claim='reported separately, not silently inferred from success fraction'),
        secondary=dict(metrics=['cost','residual_relative_rms_deg'],references=list(NAMES[:6]),comparisons=108,
            familywise_alpha=.05,interval='same paired clusters; separate simultaneous Bonferroni family',
            repetitions=REPEATS,bootstrap_seed=2026101816190,
            cost_advantage_requires='both methods succeed in all 640 episodes and cost-difference upper CI<0',
            other_metrics='full-angle RMS, peak relative angle and frequency summaries descriptive'),
        total_episodes=3*128*5*9,formal_results_unchanged=True,scope='single-axis two-inertia one-mode numerical model; ideal torque source; measured flexure assumed; no flight/hardware validation')

def freeze():
    path=HERE/'01_方案与冻结/freeze.json'
    if path.exists(): assert_frozen(); return read(path)
    folder=path.parent; folder.mkdir(parents=True,exist_ok=True)
    write(folder/'预声明.json',protocol())
    for group in GROUPS:
        rng=np.random.default_rng(SEEDS[group])
        theta=np.deg2rad(rng.uniform(-30,30,N));omega=np.deg2rad(rng.uniform(-2,2,N));z=rng.uniform(.01,.04,N)
        q=np.zeros(N) if group==GROUPS[0] else np.deg2rad(rng.uniform(-.05,.05,N))
        qdot=np.zeros(N) if group==GROUPS[0] else np.deg2rad(rng.uniform(-.2,.2,N))
        initial=[dict(id=i,theta=float(theta[i]),omega=float(omega[i]),q=float(q[i]),qdot=float(qdot[i]),damping=float(z[i])) for i in range(N)]
        write(folder/f'{group}.json',initial)
    noise=np.random.default_rng(NOISE_SEED).normal(0.,NOISE_STD,(N*5,s.STEPS+1,4))
    np.savez_compressed(folder/'C_共享测量噪声.npz',noise=noise)
    files={p.relative_to(HERE).as_posix():sha(p) for p in sorted(folder.iterdir()) if p.name!='freeze.json'}
    value=dict(frozen_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               protocol_sha256=sha(folder/'预声明.json'),source_sha256=source_hashes(),old_inputs_sha256=old_inputs(),
               scenario_and_noise_sha256=files,controllers=registry(),test_not_started_when_frozen=True)
    write(path,value)
    return value

def assert_frozen():
    f=read(HERE/'01_方案与冻结/freeze.json')
    if source_hashes()!=f['source_sha256'] or old_inputs()!=f['old_inputs_sha256']:
        raise RuntimeError('frozen source/model/old parameter mismatch')
    if registry()!=f['controllers'] or read(HERE/'01_方案与冻结/预声明.json')!=json.loads(json.dumps(protocol())):
        raise RuntimeError('registry or protocol changed')
    for p,h in f['scenario_and_noise_sha256'].items():
        if sha(HERE/p)!=h: raise RuntimeError('scenario/noise mismatch')
    return f

def scenes(group):
    return [dict(**x,frequency=f) for f in s.FREQUENCIES for x in read(HERE/f'01_方案与冻结/{group}.json')]

def initial_state(data):
    return np.asarray([[x['theta'],x['omega'],x.get('q',0.),x.get('qdot',0.)] for x in data])

def noise_for(group):
    if group==GROUPS[2]:
        with np.load(HERE/'01_方案与冻结/C_共享测量噪声.npz') as a:return a['noise']
    return np.zeros((N*5,s.STEPS+1,4))

def load(item):return s.model_load('PPO',OLD/item['model']) if 'model' in item else None

def rollout(data,item,noise,model=None):
    n=len(data);plant=s.Plant(n)
    plant.reset([r['theta'] for r in data],[r['omega'] for r in data],[r['frequency'] for r in data],[r['damping'] for r in data])
    plant.state[:]=initial_state(data)
    view=SimpleNamespace(state=plant.state+noise[:,0],failed=plant.failed.copy(),count=plant.count.copy())
    c=s.Controller(item['settings'],n);c.reset(view)
    states=np.empty((n,s.STEPS+1,4));measurement=np.empty_like(states);commands=np.empty((n,s.STEPS));actions=np.zeros_like(commands)
    failures=np.zeros((n,s.STEPS+1),bool);states[:,0]=plant.state;measurement[:,0]=view.state
    for k in range(s.STEPS):
        view.state=states[:,k]+noise[:,k];view.failed=plant.failed.copy();view.count=plant.count.copy()
        if model is not None:
            action=model.predict(c.observation(view),deterministic=True)[0];actions[:,k]=np.asarray(action).reshape(n)
        else:action=None
        out=plant.step(c.command(view,action));states[:,k+1]=plant.state;commands[:,k]=out['command']
        failures[:,k+1]=plant.failed;measurement[:,k+1]=plant.state+noise[:,k+1]
    metrics=s.episode_metrics(states,commands,plant.failed)
    return dict(state=states,measurement=measurement,command=commands,policy_action=actions,failed_trace=failures,**metrics)

def run_tests():
    f=assert_frozen();out=HERE/'02_新独立测试'
    if (out/'manifest.json').exists():raise RuntimeError('test already complete; do not overwrite')
    write(out/'test_start.json',dict(started_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=sha(HERE/'01_方案与冻结/freeze.json')))
    entries=[];summary={}
    for group in GROUPS:
        data=scenes(group);noise=noise_for(group);summary[group]={}
        for item in f['controllers']:
            target=out/group/item['name'];target.mkdir(parents=True,exist_ok=True)
            if (target/'responses.npz').exists():raise RuntimeError('refusing partial overwrite')
            response=rollout(data,item,noise,load(item));np.savez_compressed(target/'responses.npz',**response)
            metrics={k:response[k] for k in ('success','failed','cost','rms_angle_deg','peak_relative_deg','residual_relative_rms_deg')}
            sm=s.summary(metrics);sm['frequency_results']={str(freq):s.summary({k:v[j*N:(j+1)*N] for k,v in metrics.items()}) for j,freq in enumerate(s.FREQUENCIES)}
            write(target/'metrics.json',sm);summary[group][item['name']]=sm
            entries.append(dict(group=group,name=item['name'],responses_sha256=sha(target/'responses.npz'),metrics_sha256=sha(target/'metrics.json')))
            print(json.dumps(dict(group=group,name=item['name'],passes=sm['passes'],physical_failures=sm['physical_failures'],cost=sm['cost'])),flush=True)
    assert_frozen();write(out/'summary.json',summary)
    manifest=dict(episodes=17280,controllers=9,groups=3,transitions=17280*s.STEPS,freeze_sha256=sha(HERE/'01_方案与冻结/freeze.json'),entries=entries,summary_sha256=sha(out/'summary.json'))
    write(out/'manifest.json',manifest);return manifest

def matrices(data):
    fs=[];gs=[]
    for r in data:
        wn=2*np.pi*r['frequency'];inv=1/.7+1/.3;k=wn*wn/inv;c=2*r['damping']*wn/inv
        a=np.zeros((5,5));a[:4,:4]=[[0,1,0,0],[0,0,k/.7,c/.7],[0,0,0,1],[0,0,-k*inv,-c*inv]]
        a[:4,4]=[0,1/.7,0,-1/.7];e=expm(a*.05);fs.append(e[:4,:4]);gs.append(e[:4,4])
    return np.asarray(fs),np.asarray(gs)

def physical_failure(x):
    return (np.abs(x[:,0])>np.pi*170/180)|(np.abs(x[:,1])>np.pi*300/180)|(np.abs(x[:,2])>np.pi*10/180)

def independent_metrics(states,commands,failed):
    t=states[:,160:,:]
    return dict(success=np.all(np.abs(t)<=np.deg2rad([1,.5,.1,.5]),axis=(1,2))&~failed,
                failed=failed,cost=np.einsum('nt,nt->n',commands,commands)*.05,
                rms_angle_deg=np.sqrt(np.mean(states[:,:,0]**2,axis=1))*180/np.pi,
                peak_relative_deg=np.max(np.abs(states[:,:,2]),axis=1)*180/np.pi,
                residual_relative_rms_deg=np.sqrt(np.mean(t[:,:,2]**2,axis=1))*180/np.pi)

def verify():
    f=assert_frozen();out=HERE/'02_新独立测试';manifest=read(out/'manifest.json')
    if sha(out/'summary.json')!=manifest['summary_sha256']:raise RuntimeError('summary hash')
    checks=[];entries={(x['group'],x['name']):x for x in manifest['entries']}
    for group in GROUPS:
        data=scenes(group);noise=noise_for(group);fm,gm=matrices(data);n=len(data)
        for item in f['controllers']:
            folder=out/group/item['name'];entry=entries[group,item['name']]
            if sha(folder/'responses.npz')!=entry['responses_sha256'] or sha(folder/'metrics.json')!=entry['metrics_sha256']:raise RuntimeError('response hash')
            with np.load(folder/'responses.npz') as a: response={k:a[k] for k in a.files}
            state,measurement,commands=response['state'],response['measurement'],response['command']
            np.testing.assert_array_equal(state[:,0],initial_state(data))
            np.testing.assert_array_equal(measurement,state+noise)
            view=SimpleNamespace(state=measurement[:,0],failed=np.zeros(n,bool),count=np.zeros(n,int))
            control=s.Controller(item['settings'],n);control.reset(view);model=load(item)
            failed=np.zeros(n,bool);whole=initial_state(data);localerr=0.;wholeerr=0.;actionerr=0.
            for k in range(s.STEPS):
                view.state=measurement[:,k];view.failed=failed.copy();view.count[:]=k
                action=None if model is None else model.predict(control.observation(view),deterministic=True)[0]
                action_flat=np.zeros(n) if action is None else np.asarray(action).reshape(n)
                actionerr=max(actionerr,float(np.max(np.abs(action_flat-response['policy_action'][:,k]))))
                u=control.command(view,action);np.testing.assert_allclose(u,commands[:,k],rtol=0,atol=2e-12)
                expected=np.einsum('nij,nj->ni',fm,state[:,k])+gm*commands[:,k,None]
                expected[failed]=state[failed,k]
                localerr=max(localerr,float(np.max(np.abs(expected-state[:,k+1]))))
                independent=np.einsum('nij,nj->ni',fm,whole)+gm*commands[:,k,None]
                independent[failed]=whole[failed];whole=independent
                wholeerr=max(wholeerr,float(np.max(np.abs(whole-state[:,k+1]))))
                failed|=physical_failure(state[:,k+1]);np.testing.assert_array_equal(failed,response['failed_trace'][:,k+1])
            if localerr>2e-11 or wholeerr>2e-10 or actionerr>2e-12:raise RuntimeError('independent state/action replay')
            metrics=independent_metrics(state,commands,failed)
            for key,x in metrics.items():
                if x.dtype==bool:np.testing.assert_array_equal(x,response[key])
                else:np.testing.assert_allclose(x,response[key],rtol=0,atol=3e-12)
            published=read(folder/'metrics.json');sm=s.summary(metrics)
            for key in sm:
                if abs(sm[key]-published[key])>3e-12:raise RuntimeError('published summary')
            check=dict(group=group,name=item['name'],episodes=n,max_state_transition_residual=localerr,
                       max_full_recorded_command_replay_error=wholeerr,max_policy_action_error=actionerr,
                       every_measurement_equals_truth_plus_shared_noise=True,true_state_acceptance=True)
            checks.append(check);print(json.dumps(check),flush=True)
    value=dict(passed=True,episodes=17280,transitions=17280*s.STEPS,all_27_controller_group_results_checked=True,
               scope='independent SciPy state-space matrix exponential for every transition and complete recorded-command trajectory; deterministic controllers/actions replayed from recorded measurements; all noise and truth-based metrics reconstructed; not a new training or an independent implementation of PPO',
               matrix_exponential_implementation='scipy.linalg.expm, original second-order physical state equations',
               freeze_sha256=sha(HERE/'01_方案与冻结/freeze.json'),manifest_sha256=sha(out/'manifest.json'),checks=checks)
    write(HERE/'03_独立核验/verification.json',value);return value

def statistics():
    assert_frozen();v=read(HERE/'03_独立核验/verification.json')
    if not v['passed']:raise RuntimeError('verify first')
    result={};claims={};confidence_primary=1-.05/18;confidence_secondary=1-.05/108
    for gi,group in enumerate(GROUPS):
        arrays={}
        for name in NAMES:
            with np.load(HERE/'02_新独立测试'/group/name/'responses.npz') as a:
                arrays[name]={k:a[k].astype(float).reshape(5,N).mean(axis=0) for k in ('success','cost','residual_relative_rms_deg')}
        descriptions=[];differences=[]
        for name in NAMES[6:]:
            for ref in ('PID_strengthened','PID_safe'):
                descriptions.append(dict(family='primary_success',learner=name,reference=ref,metric='success'))
                differences.append(arrays[name]['success']-arrays[ref]['success'])
            for ref in NAMES[:6]:
                for metric in ('cost','residual_relative_rms_deg'):
                    descriptions.append(dict(family='secondary',learner=name,reference=ref,metric=metric))
                    differences.append(arrays[name][metric]-arrays[ref][metric])
        d=np.asarray(differences);draws=np.empty((REPEATS,len(d)));rng=np.random.default_rng(2026101816190+gi)
        for start in range(0,REPEATS,1000):
            idx=rng.integers(0,N,(1000,N));weights=np.zeros((1000,N))
            np.add.at(weights,(np.repeat(np.arange(1000),N),idx.reshape(-1)),1./N)
            draws[start:start+1000]=weights@d.T
        rows=[]
        for j,desc in enumerate(descriptions):
            conf=confidence_primary if desc['family']=='primary_success' else confidence_secondary
            lo,hi=np.quantile(draws[:,j],[(1-conf)/2,1-(1-conf)/2])
            row=dict(**desc,mean_difference=float(d[j].mean()),lower=float(lo),upper=float(hi),confidence=conf,
                     difference_direction='learner minus reference',cluster_count=N,frequencies_per_cluster=5)
            if desc['family']=='primary_success':row['superiority_supported']=bool(lo>0.)
            else:
                allpass=bool(np.all(arrays[desc['learner']]['success']==1)&np.all(arrays[desc['reference']]['success']==1))
                row['both_all_success']=allpass;row['conditional_advantage_supported']=bool(hi<0. and allpass)
            rows.append(row)
        result[group]=rows
        primary=[row for row in rows if row['family']=='primary_success']
        claims[group]=dict(all_three_seeds_beat_both_PID_references=all(row['superiority_supported'] for row in primary),
            by_reference={ref:all(row['superiority_supported'] for row in primary if row['reference']==ref) for ref in ('PID_strengthened','PID_safe')})
    value=dict(primary_comparison_count=18,secondary_comparison_count=108,bootstrap_repeats=REPEATS,
               clusters_per_group=N,primary_confidence=confidence_primary,secondary_confidence=confidence_secondary,
               comparisons=result,claims=claims,scope='separate prespecified familywise corrections; groups have independent initial-state samples')
    write(HERE/'04_统计/paired_statistics.json',value);return value

def plots(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    out=HERE/'05_报告/图表';out.mkdir(parents=True,exist_ok=True)
    colors=['#64748b']*3+['#0d9488']*2+['#16a34a']+['#d97706']*3
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained');group=GROUPS[0];y=np.arange(9)
    for ax,field,xlabel in ((axes[0],'success','持续停稳成功率（%）'),(axes[1],'cost','平均实际力矩平方积分（N²·m²·s）')):
        values=[summaries[group][name][field]*(100 if field=='success' else 1) for name in NAMES]
        bars=ax.barh(y,values,color=colors);ax.set_yticks(y,[LABELS[n] for n in NAMES]);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        if field=='success':
            ax.set_xlim(0,110)
            for i,value in enumerate(values):ax.text(value+.5,i,f'{value:.1f}%',va='center',fontsize=9)
        else:
            for bar,name in zip(bars,NAMES):
                if summaries[group][name]['passes']<640:bar.set_hatch('//')
            ax.set_title('斜线表示未全部达标，不能直接声称代价优势',fontsize=11)
    fig.suptitle('全新保留样本确认：128初态×5频率，冻结模型与参数',fontsize=15)
    fig.savefig(out/'原分布独立确认.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,7),layout='constrained')
    for ax,group in zip(axes,GROUPS):
        values=[summaries[group][n]['success']*100 for n in NAMES];ax.barh(y,values,color=colors)
        ax.set_yticks(y,[LABELS[n] for n in NAMES],fontsize=9);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('持续停稳成功率（%）');ax.set_title(GROUP_LABELS[group]);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(values):ax.text(value+.4,i,f'{value:.1f}',va='center',fontsize=8)
    fig.suptitle('初始振动与测量噪声的影响：全部控制器，不筛选种子',fontsize=15)
    fig.savefig(out/'初始振动与噪声.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5.7),layout='constrained',sharey=True)
    shown=('PID_safe','PID_strengthened','ZVD_PID_refined','LQR_refined')+NAMES[6:]
    for ax,group in zip(axes,GROUPS):
        for name in shown:
            values=[summaries[group][name]['frequency_results'][str(f)]['success']*100 for f in s.FREQUENCIES]
            ax.plot(s.FREQUENCIES,values,'o-',label=LABELS[name],lw=1.5,ms=4)
        ax.set_title(GROUP_LABELS[group]);ax.set_xlabel('固定模态频率（Hz）');ax.set_ylim(0,104);ax.grid(alpha=.2)
    axes[0].set_ylabel('持续停稳成功率（%）');axes[1].legend(fontsize=8,loc='lower right')
    fig.savefig(out/'频率分组.png',dpi=170);plt.close(fig)
    stats=read(HERE/'04_统计/paired_statistics.json');fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for ax,ref in zip(axes,('PID_strengthened','PID_safe')):
        for gi,group in enumerate(GROUPS):
            rows=[x for x in stats['comparisons'][group] if x['family']=='primary_success' and x['reference']==ref]
            for j,row in enumerate(rows):
                x=gi*4+j;mean=row['mean_difference']*100;lo=row['lower']*100;hi=row['upper']*100
                ax.errorbar(x,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',capsize=4,color=['#b45309','#2563eb','#7c3aed'][j])
        ax.axhline(0,color='#64748b',ls='--');ax.set_xticks([1,5,9],[GROUP_LABELS[g] for g in GROUPS],fontsize=9);ax.set_title('相对'+LABELS[ref]);ax.set_ylabel('成功率差（百分点）；99.722%同时校正区间');ax.grid(axis='y',alpha=.2)
    fig.suptitle('全部三个正式种子：每组从左到右111、222、333；按128初态聚类',fontsize=14)
    fig.savefig(out/'三个种子与配对区间.png',dpi=170);plt.close(fig)

def report():
    f=assert_frozen();summaries=read(HERE/'02_新独立测试/summary.json');v=read(HERE/'03_独立核验/verification.json');st=read(HERE/'04_统计/paired_statistics.json')
    if not v['passed']:raise RuntimeError('verify first')
    plots(summaries)
    lines=['# 柔性附体研究：全新样本确认、初始振动与测量噪声','',
        '本补充实验使用此前已经固定的三个PPO模型和六种传统控制器，不进行新训练或调参。它解决原先补充搜索PID只在旧测试样本复核的问题，并检查初始附体振动和测量噪声对结果的影响。全部方案、源码、模型、传统参数、初态和共享噪声先冻结，再执行新测试。','',
        '## 1. 任务和工程含义','',
        '卫星本体转向会带动柔性附件振动。这里用本体与附体通过弹性、阻尼连接的双惯量单模态模型近似，控制力矩作用于本体。每0.05秒更新一次指令，每回合22秒；要求从8秒开始一直到22秒，本体角度≤1°、本体角速度≤0.5°/s、附体相对角度≤0.1°、相对角速度≤0.5°/s，力矩限制±0.5 N·m。四项条件必须同时持续满足。','',
        '附体初始小振动可对应前一次机动或展开后尚未完全衰减的振动；测量噪声用于检查小幅传感误差。噪声标准差是假定的仿真工况，没有经过具体硬件标定。真实状态决定动力学、失败和验收；控制器只读取加噪测量。','',
        '## 2. 预先固定的实验条件','',
        '三组各128个全新独立初态，每个初态测试0.4、0.5、0.6、0.7、0.8 Hz五个频率。频率和阻尼在一回合内固定。每组每方法640回合，共9方法×3组×640＝17,280回合。三组初态独立抽样，因此组间成功率差不能全部归因于新增因素；它们各自估计相应分布下的表现，A组是主要独立确认，B/C是鲁棒性补充。','',
        '- A：沿用原初态分布，本体初角±30°、初始角速度±2°/s、阻尼比0.01–0.04，附体初始相对角度和速度为0。',
        '- B：其他同A，附体初始相对角度均匀分布±0.05°，相对角速度均匀分布±0.2°/s。',
        '- C：其他同B，四个状态各叠加独立零均值高斯白噪声，标准差依次为[0.01°、0.02°/s、0.005°、0.02°/s]。初始采样也加噪；每个场景在所有方法间使用同一份噪声。','',
        '九种方法是：原PID、学习补偿使用的同一PID基底、从3825个历史候选中补充选择的PID、原ZVD输入整形＋PID、优化ZVD输入整形＋PID、优化LQR，以及PPO种子111、222、333。补充搜索PID在旧研究中属于事后复核，在本次新样本中参数事先冻结，故A组可以提供新的前瞻独立确认。','',
        '## 3. 统计方法','',
        '主要检验为三份PPO各自相对补充搜索PID、同一PID基底的成功率差，共3组×3种子×2参照＝18项。按128个独立初态聚类重采样，每次保留该初态全部五个频率，200,000次配对bootstrap；对18项做Bonferroni校正，报告99.7222%双侧区间。仅当三个种子相对两个参照的区间下界全部大于0，才称该组主要优势确认成立。','',
        '次级指标是实际力矩平方积分和8–22秒附体残余相对角度RMS。与六个传统参照的108项配对比较另作为一族做Bonferroni校正，区间置信水平99.9537%。代价优势只在双方640回合全部成功时判断，失败吸收后力矩变零导致的低代价不能当作优势。全回合本体角度RMS、峰值相对角度和频率分组仅作描述。力矩平方积分不是电能。','']
    for group in GROUPS:
        lines += ['## '+GROUP_LABELS[group]+'：结果','',
            '|方法|持续停稳/640|物理越界回合|平均力矩平方积分|平均残余相对角度RMS（°）|','|---|---:|---:|---:|---:|']
        for name in NAMES:
            x=summaries[group][name];lines.append(f'|{LABELS[name]}|{x["passes"]}/640|{x["physical_failures"]}|{x["cost"]:.7f}|{x["residual_relative_rms_deg"]:.6f}|')
        claim=st['claims'][group]
        lines += ['',f'三个PPO种子相对两个PID参照的主要优势确认：{"成立" if claim["all_three_seeds_beat_both_PID_references"] else "不成立"}。', '',
            '|PPO种子|参照|成功率差（百分点）|同时校正区间（百分点）|下界>0|','|---|---|---:|---:|---|']
        for row in st['comparisons'][group]:
            if row['family']=='primary_success':
                lines.append(f'|{row["learner"].split("_")[-1]}|{LABELS[row["reference"]]}|{row["mean_difference"]*100:.3f}|[{row["lower"]*100:.3f}, {row["upper"]*100:.3f}]|{"是" if row["superiority_supported"] else "否"}|')
        lines += ['']
    lines += ['## 7. 论文图与结果解释','',
        '![原分布新样本确认](图表/原分布独立确认.png)','',
        '![初振与噪声](图表/初始振动与噪声.png)','',
        '![频率分组](图表/频率分组.png)','',
        '![三个种子及配对区间](图表/三个种子与配对区间.png)','',
        '主要结论只涉及这几组分布及冻结控制器。PPO相对PID的结果与相对ZVD、LQR的结果必须同时报告。LQR使用本体和附体的完整测量状态；这说明充分利用柔性状态的传统控制具有竞争力。不能从“优于某组PID”推出“优于所有传统控制”。三个学习种子均展示，不以最好一个代替整体。','',
        '越过10°相对角度表示超出本研究线性近似有效范围，不代表已经证明真实附件损坏。未进行具体飞行卫星、执行器动态、传感器校准、完整柔性多模态或硬件验证。含噪组使用未在含噪条件下重新训练的模型和未经含噪重整定的控制器，研究的是现有设计直接使用时的鲁棒性，不能作为双方充分针对噪声优化后的最优能力比较。','',
        '## 8. 独立核验与材料位置','',
        f'全量17,280回合、{17280*s.STEPS:,}个状态转移通过独立SciPy状态空间矩阵指数核验。最大单步状态差异为{max(c["max_state_transition_residual"] for c in v["checks"]):.3e}，按记录指令从初态逐步重放最大差异为{max(c["max_full_recorded_command_replay_error"] for c in v["checks"]):.3e}，全部策略动作复算差异为{max(c["max_policy_action_error"] for c in v["checks"]):.3e}。每个测量值都与真实状态＋冻结共享噪声逐元素一致；四项验收、越界、代价与误差指标另行重算。','',
        '该核验使用独立的物理状态空间矩阵指数，但使用冻结的原控制器和PPO实现复算动作，不声称重新独立实现了强化学习算法。所有输出文件均有哈希清单。新关键测试另检查零噪声与原实现一致、非零初始振动正确保留、噪声只进入测量及真值验收、共享噪声与随机样本独立。','',
        '- `01_方案与冻结/`：测试前冻结方案、源码/模型/参数哈希、全新初态和共享噪声。',
        '- `02_新独立测试/`：27组完整真值、测量、策略动作、指令和逐回合指标。',
        '- `03_独立核验/verification.json`：全量状态/动作/噪声/指标核验。',
        '- `04_统计/paired_statistics.json`：18项主比较与108项次比较。',
        '- `结果摘要.json`：供论文和总览读取的固定结构摘要。','']
    text='\n'.join(lines)+'\n';out=HERE/'05_报告';out.mkdir(parents=True,exist_ok=True)
    (out/'研究报告.md').write_text(text,encoding='utf-8')
    blocks=[];table=False;listing=False
    for line in lines:
        if line.startswith('|'):
            if not table:blocks.append('<div class="table"><table>');table=True
            if set(line)<=set('|-: '):continue
            tag='th' if '方法|' in line or 'PPO种子|' in line else 'td'
            blocks.append('<tr>'+''.join(f'<{tag}>{html.escape(x.strip())}</{tag}>' for x in line.strip('|').split('|'))+'</tr>');continue
        if table:blocks.append('</table></div>');table=False
        if line.startswith('- '):
            if not listing:blocks.append('<ul>');listing=True
            blocks.append('<li>'+html.escape(line[2:])+'</li>');continue
        if listing:blocks.append('</ul>');listing=False
        if not line:continue
        if line.startswith('# '):blocks.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):blocks.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('!['):
            alt,src=line[2:].split('](');blocks.append('<img src="'+html.escape(src[:-1],quote=True)+'" alt="'+html.escape(alt,quote=True)+'">')
        else:blocks.append('<p>'+html.escape(line)+'</p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>柔性附体补充独立确认</title><style>body{font:17px/1.85 "Microsoft YaHei",system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#fbfcfe;color:#172b43}h1{font-size:30px}h2{font-size:23px;margin-top:38px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #ccd6e0;padding:9px}th{background:#e8eef7}.table{overflow:auto}img{max-width:100%;height:auto;margin:18px 0}</style><body>'+''.join(blocks)+'</body></html>'
    (out/'研究报告.html').write_text(page,encoding='utf-8')
    summary=dict(study='16_supplement_prospective_confirmation',new_training_runs=0,new_training_steps=0,
        new_test_episodes=17280,new_test_initial_states=384,initial_states_per_group=128,frequencies_per_initial_state=5,
        controllers=9,groups=3,old_models_changed=False,old_parameters_changed=False,
        formal_models=[{k:x[k] for k in ('name','seed','selected_steps','old_training_steps')} for x in f['controllers'] if 'model' in x],
        results=summaries,claims=st['claims'],primary_statistics={g:[x for x in rows if x['family']=='primary_success'] for g,rows in st['comparisons'].items()},
        verification=dict(passed=v['passed'],episodes=v['episodes'],transitions=v['transitions'],scope=v['scope'],
            max_state_transition_residual=max(c['max_state_transition_residual'] for c in v['checks']),
            max_full_recorded_command_replay_error=max(c['max_full_recorded_command_replay_error'] for c in v['checks']),
            max_policy_action_error=max(c['max_policy_action_error'] for c in v['checks'])),
        protocol_sha256=sha(HERE/'01_方案与冻结/预声明.json'),freeze_sha256=sha(HERE/'01_方案与冻结/freeze.json'),
        report_markdown='05_报告/研究报告.md',report_html='05_报告/研究报告.html',
        statistics='04_统计/paired_statistics.json',figures=['05_报告/图表/'+x+'.png' for x in ('原分布独立确认','初始振动与噪声','频率分组','三个种子与配对区间')],
        limitations=['single-axis two-inertia one-mode model','ideal torque source','flexure measured','noise level is assumed, not hardware calibrated','groups have independent initial draws','no parameter/weight updates on new scenarios','LQR and ZVD comparisons retained','torque-square integral is not electric energy'])
    write(HERE/'结果摘要.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','test','verify','statistics','report'])
    args=parser.parse_args();fn={'freeze':freeze,'test':run_tests,'verify':verify,'statistics':statistics,'report':report}[args.command]
    value=fn();print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
