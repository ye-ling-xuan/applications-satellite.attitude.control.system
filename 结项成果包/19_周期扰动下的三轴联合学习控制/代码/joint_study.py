"""Joint three-axis continuous TD3 compensation using fresh independent scenes."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from physics import Plant,DT,DURATION,DEADLINE,LAG,INERTIA,LIMIT,STEPS,quaternion_angle

STUDY=Path(__file__).resolve().parents[1]
COUNTS={'development':24,'validation':48,'test':128}
SEEDS={'development':202610181901,'validation':202610181902,'test':202610181903}
FORMAL_SEEDS=(19111,19222,19333)
DEV_STEPS,FORMAL_STEPS,CHECK_EVERY,N_ENVS=262144,524288,32768,32
ANGLE_TOL,RATE_TOL=2.,1.

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def source_hashes():return {p.relative_to(STUDY).as_posix():sha(p) for p in sorted((STUDY/'代码').glob('*.py'))}
def random_scenes(rng,n):
    fields={k:rng.uniform(lo,hi,(n,3)) for k,lo,hi in (
        ('euler_deg',-12.,12.),('omega_deg_s',-1.,1.),('phase_rad',-np.pi,np.pi),
        ('frequency_rad_s',.35,.65),('amplitude_1_Nm',.025,.045),('amplitude_2_Nm',.005,.015),
        ('attitude_coefficient_Nm',.015,.025),('rate_coefficient_Nm',.003,.007))}
    return [dict(sample_id=i,**{k:v[i].tolist() for k,v in fields.items()}) for i in range(n)]
def scenes(split):return read_json(STUDY/'01_方案与场景'/f'{split}.json')
def design():
    c=dict(task='Joint three-axis quaternion attitude stabilization with unknown continuous disturbance',
        dt=DT,duration=DURATION,deadline=DEADLINE,actuator_lag=LAG,inertia=INERTIA.tolist(),limit_per_axis=LIMIT,
        coupled_Euler_dynamics=True,joint_network=True,single_axis_transfer=False,
        acceptance=dict(angle_deg=ANGLE_TOL,rate_norm_deg_s=RATE_TOL,interval=[8.,32.]),
        disturbance='per-axis A1sin(phi)+A2sin(2phi+.4)+ctheta*sin(2*rotation_vector_component)+crate*tanh(omega/.05)',
        observed=['body rotation-vector error','body rate','measured phase/frequency','own PID integral and past sent command','time'],
        hidden=['true coefficients','disturbance','actual actuator torque'],counts=COUNTS,split_seeds=SEEDS,
        baselines=['quaternion PID with known gyroscopic feedforward','disturbance observer plus PID','measured-phase harmonic feedforward plus PID'],
        algorithm='TD3 based on separate single-axis development; no universal-best claim',
        development=dict(configs={'A':[128,128],'B':[256,128]},steps_each=DEV_STEPS),
        formal=dict(seeds=FORMAL_SEEDS,steps_each=FORMAL_STEPS),
        training=dict(environments=N_ENVS,batch_size=256,gradient_steps=4,gamma=.999,learning_rate=.0003,
            action_noise=.15,residual_limit_Nm=.12,steady_reward_angle_scale_deg=.03),
        validate_every=CHECK_EVERY,selection='fewest failures,most passes,smallest steady attitude RMS,then actual torque-square cost',
        gate='zero physical failures and >=95% validation passes; both development attempts retained',
        primary='PID and each seed pass every heldout; each Bonferroni98.333% paired RMS-improvement interval above0; each mean cost<=110%PID',
        heldout_used_to_select=False,weights_frozen_during_test=True,
        practical_scope='Engineering illustrative periodic/attitude-dependent disturbances, not a flight satellite or complete orbit/reaction-wheel model')
    c=json.loads(json.dumps(c));path=STUDY/'01_方案与场景'/'预声明.json'
    if path.exists() and read_json(path)!=c:raise RuntimeError('protocol changed')
    if not path.exists():write_json(path,c)
    for split,seed in SEEDS.items():
        data=random_scenes(np.random.default_rng(seed),COUNTS[split]);path=STUDY/'01_方案与场景'/f'{split}.json'
        if path.exists() and read_json(path)!=data:raise RuntimeError('split changed')
        if not path.exists():write_json(path,data)
    return c

class Controller:
    def __init__(self,p,family='pid',harmonic=None):self.p=p;self.family=family;self.harmonic=harmonic
    def reset(self,n,obs):
        def expand(k,d=0.):return np.broadcast_to(np.asarray(self.p.get(k,d),float),(n,)).copy()[:,None]
        self.kp=expand('kp')*INERTIA;self.kd=expand('kd')*INERTIA;self.ki=expand('ki')*INERTIA
        self.filter_time=expand('filter_time',.1);self.predict=expand('predict',0.)
        for k in ('integral','tau','previous_u','previous_w','estimate','derivative'):setattr(self,k,np.zeros((n,3)))
        self.have=np.zeros((n,1),bool)
    def act(self,obs,elapsed=None,residual=None):
        error,omega=obs[:,:3],obs[:,3:6];proposed=np.clip(self.integral+error*DT,-1.,1.)
        raw=self.kp*error-self.kd*omega+self.ki*proposed+np.cross(omega,INERTIA*omega)
        if self.family=='observer':
            e=np.exp(-DT/LAG);mean=self.previous_u+(self.tau-self.previous_u)*LAG*(1-e)/DT
            gyro=.5*(np.cross(omega,INERTIA*omega)+np.cross(self.previous_w,INERTIA*self.previous_w))
            measured=INERTIA*(omega-self.previous_w)/DT+gyro-mean
            updated=self.estimate+DT/(self.filter_time+DT)*(measured-self.estimate)
            self.derivative=np.where(self.have,.5*self.derivative+.5*(updated-self.estimate)/DT,self.derivative)
            self.estimate=np.where(self.have,updated,self.estimate);raw-=self.estimate+self.predict*LAG*self.derivative
        elif self.family=='harmonic':
            phase=np.arctan2(obs[:,6:9],obs[:,9:12]);freq=obs[:,12:15];c=np.asarray(self.harmonic)
            feed=c[None,:,0]*np.sin(phase)+c[None,:,1]*np.cos(phase)+c[None,:,2]*np.sin(2*phase)+c[None,:,3]*np.cos(2*phase)
            deriv=freq*(c[None,:,0]*np.cos(phase)-c[None,:,1]*np.sin(phase)+2*c[None,:,2]*np.cos(2*phase)-2*c[None,:,3]*np.sin(2*phase));raw-=feed+LAG*deriv
        if residual is not None:raw+=residual
        u=np.clip(raw,-LIMIT,LIMIT);self.integral=np.where((raw-u)*self.ki*(proposed-self.integral)<=1e-15,proposed,self.integral)
        self.tau=self.previous_u+(self.tau-self.previous_u)*np.exp(-DT/LAG);self.previous_u=u.copy();self.previous_w=omega.copy();self.have[:]=True
        return u

def summary(rows):
    return dict(count=len(rows),passes=sum(r['success'] for r in rows),physical_failures=sum(r['failed'] for r in rows),
        success_rate=sum(r['success'] for r in rows)/len(rows),mean_steady_rms_deg=float(np.mean([r['steady_rms_deg'] for r in rows])),
        mean_actual_cost=float(np.mean([r['actual_cost'] for r in rows])),mean_full_rms_deg=float(np.mean([r['full_rms_deg'] for r in rows])))
def rank(r):return r['physical_failures'],-r['passes'],r['mean_steady_rms_deg'],r['mean_actual_cost']
def evaluate(data,controller,keep_traces=False,calibrate=False):
    plant=Plant(data);n=len(data);controller.reset(n,plant.measured());cost=np.zeros(n);steady=np.zeros(n);full=np.zeros(n);success=np.ones(n,bool)
    estimated_tau=np.zeros((n,3));previous_w=plant.omega.copy();cx=[];cy=[];trace=None
    if keep_traces:
        trace={k:np.empty((STEPS+1,n,width)) for k,width in [('quaternion',4),('omega',3),('tau',3),('phase',3)]}
        trace.update(sent_command=np.empty((STEPS,n,3)),command=np.empty((STEPS,n,3)),actual_cost_interval=np.empty((STEPS,n)),failed=np.empty((STEPS+1,n),bool))
        for k in ('quaternion','omega','tau','phase','failed'):trace[k][0]=getattr(plant,k)
    for k in range(STEPS):
        sent=controller.act(plant.measured(),np.full(n,k*DT));out=plant.step(sent)
        angle=np.degrees(quaternion_angle(plant.quaternion));rate=np.degrees(np.linalg.norm(plant.omega,axis=1));cost+=out['actual_cost'];full+=angle**2*DT
        if (k+1)*DT>=DEADLINE-1e-10:
            success&=(angle<=ANGLE_TOL)&(rate<=RATE_TOL)&~plant.failed;steady+=angle**2*DT
            if calibrate:
                phase=plant.phase-.5*DT*plant.frequency;cx.append(np.stack((np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)),axis=-1))
                avg=sent+(estimated_tau-sent)*LAG*(1-np.exp(-DT/LAG))/DT;gyro=.5*(np.cross(plant.omega,INERTIA*plant.omega)+np.cross(previous_w,INERTIA*previous_w));cy.append(INERTIA*(plant.omega-previous_w)/DT+gyro-avg)
        estimated_tau=sent+(estimated_tau-sent)*np.exp(-DT/LAG);previous_w=plant.omega.copy()
        if trace is not None:
            for key in ('quaternion','omega','tau','phase','failed'):trace[key][k+1]=getattr(plant,key)
            trace['sent_command'][k]=sent;trace['command'][k]=out['command'];trace['actual_cost_interval'][k]=out['actual_cost']
    rows=[dict(r,success=bool(success[i] and not plant.failed[i]),failed=bool(plant.failed[i]),steady_rms_deg=float(np.sqrt(steady[i]/(DURATION-DEADLINE+DT))),full_rms_deg=float(np.sqrt(full[i]/DURATION)),actual_cost=float(cost[i])) for i,r in enumerate(data)]
    return dict(rows=rows,traces=trace,calibration_x=np.concatenate(cx) if cx else None,calibration_y=np.concatenate(cy) if cy else None)

def tune():
    design();before=source_hashes();dev=scenes('development');val=scenes('validation');rng=np.random.default_rng(19004)
    candidates=[dict(kp=float(kp),kd=float(kd),ki=ki) for kp in np.geomspace(.5,100.,15) for kd in np.geomspace(.5,60.,15) for ki in (0.,.2,1.)]
    candidates.extend(dict(kp=float(np.exp(rng.uniform(np.log(.5),np.log(150.)))),kd=float(np.exp(rng.uniform(np.log(.5),np.log(100.)))),ki=float(np.exp(rng.uniform(np.log(.01),np.log(5.))))) for _ in range(180))
    candidates.extend([dict(kp=50.,kd=20.,ki=0.),dict(kp=35.984283650057584,kd=20.,ki=0.)])
    def search(family,parameters,harmonic=None):
        scores=[]
        for offset in range(0,len(parameters),32):
            chunk=parameters[offset:offset+32];packed={k:np.repeat([r.get(k,0.) for r in chunk],len(dev)) for k in set().union(*(r.keys() for r in chunk))};output=evaluate(dev*len(chunk),Controller(packed,family,harmonic))['rows'];scores.extend(summary(output[i*len(dev):(i+1)*len(dev)]) for i in range(len(chunk)))
        indexes=sorted(range(len(parameters)),key=lambda i:rank(scores[i]))[:24];validation=[summary(evaluate(val,Controller(parameters[i],family,harmonic))['rows']) for i in indexes];j=min(range(len(indexes)),key=lambda j:rank(validation[j]));best=indexes[j]
        result=dict(parameters=parameters[best],development=scores[best],validation=validation[j],harmonic=harmonic,development_candidates=len(parameters),validation_candidates=len(indexes));write_json(STUDY/'02_传统控制'/f'{family}_search.json',dict(candidates=parameters,development=scores,validation=[dict(index=i,summary=x) for i,x in zip(indexes,validation)],selected=result,source_sha256=before,test_used=False));print(json.dumps(dict(event='baseline_selected',family=family,**result)),flush=True);return result
    pid=search('pid',candidates);cal=evaluate(dev,Controller(pid['parameters']),calibrate=True);coeff=[np.linalg.lstsq(cal['calibration_x'][:,axis,:],cal['calibration_y'][:,axis],rcond=None)[0].tolist() for axis in range(3)]
    prior=read_json(STUDY/'02_传统控制'/'pid_search.json');leaders=sorted(range(len(candidates)),key=lambda i:rank(prior['development'][i]))[:8];base=[candidates[i] for i in leaders]+[dict(kp=35.984283650057584,kd=20.,ki=0.),dict(kp=25.,kd=10.,ki=0.)]
    observer=search('observer',[dict(p,filter_time=t,predict=v) for p in base for t in (.025,.075,.15,.3) for v in (0.,.5,1.)]);harmonic=search('harmonic',base,coeff)
    if source_hashes()!=before:raise RuntimeError('source changed during tuning')
    result=dict(pid=pid,observer=observer,harmonic=harmonic,source_sha256=before,test_used=False);write_json(STUDY/'02_传统控制'/'selected.json',result);return result

def imports():
    import torch;torch.set_num_threads(1)
    from stable_baselines3 import TD3
    from stable_baselines3.common.vec_env import VecEnv
    from gymnasium import spaces
    return torch,TD3,VecEnv,spaces
def features(obs,c,previous,elapsed):return np.column_stack((obs[:,:3]/np.radians(12.),obs[:,3:6]/np.radians(5.),obs[:,6:12],obs[:,12:15]/.5,c.integral,previous/LIMIT,np.asarray(elapsed)/DURATION)).astype(np.float32)
def make_environment(p,seed,n=N_ENVS):
    _,_,VecEnv,spaces=imports()
    class Environment(VecEnv):
        def __init__(self):
            self.rng=np.random.default_rng(seed);self.plant=Plant(random_scenes(self.rng,n));self.controller=Controller(p);self.controller.reset(n,self.plant.measured());self.previous=np.zeros((n,3));self.pending=None;self.returns=np.zeros(n)
            super().__init__(n,spaces.Box(-np.inf,np.inf,shape=(22,),dtype=np.float32),spaces.Box(-1.,1.,shape=(3,),dtype=np.float32))
        def _obs(self):return features(self.plant.measured(),self.controller,self.previous,self.plant.count*DT)
        def reset(self):self.plant.reset(random_scenes(self.rng,n));self.controller.reset(n,self.plant.measured());self.previous[:]=0.;self.returns[:]=0.;return self._obs()
        def step_async(self,actions):self.pending=np.asarray(actions,float).reshape(n,3)
        def step_wait(self):
            u=self.controller.act(self.plant.measured(),residual=.12*np.clip(self.pending,-1.,1.));self.plant.step(u);self.previous=u;angle=np.degrees(quaternion_angle(self.plant.quaternion));rate=np.degrees(np.linalg.norm(self.plant.omega,axis=1));scale=np.where(self.plant.count*DT>=DEADLINE,.03,3.)
            reward=-(np.minimum((angle/scale)**2,300.)+.1*np.minimum((rate/3.)**2,300.)+.002*np.sum((u/LIMIT)**2,axis=1))*DT;reward=np.where(self.plant.failed,-30.,reward);done=self.plant.count>=STEPS;self.returns+=reward;terminal=self._obs();infos=[{} for _ in range(n)]
            for i in np.flatnonzero(done):infos[i]=dict(terminal_observation=terminal[i].copy(),episode=dict(r=float(self.returns[i]),l=STEPS));infos[i]['TimeLimit.truncated']=False
            if np.any(done):
                idx=np.flatnonzero(done);self.plant.reset(random_scenes(self.rng,len(idx)),done);fresh=Controller(p);fresh.reset(len(idx),self.plant.measured()[idx])
                for key in ('integral','tau','previous_u','previous_w','estimate','derivative','have'):getattr(self.controller,key)[idx]=getattr(fresh,key)
                self.previous[idx]=0.;self.returns[idx]=0.
            return self._obs(),reward.astype(np.float32),done,infos
        def close(self):pass
        def get_attr(self,name,indices=None):return [getattr(self,name,None) for _ in self._get_indices(indices)]
        def set_attr(self,name,value,indices=None):setattr(self,name,value)
        def env_method(self,name,*args,indices=None,**kwargs):return [getattr(self,name)(*args,**kwargs) for _ in self._get_indices(indices)]
        def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
    return Environment()
class LearnedController(Controller):
    def __init__(self,p,model):super().__init__(p);self.model=model
    def reset(self,n,obs):super().reset(n,obs);self.previous=np.zeros((n,3))
    def act(self,obs,elapsed=None,residual=None):
        a,_=self.model.predict(features(obs,self,self.previous,elapsed),deterministic=True);u=super().act(obs,elapsed,.12*np.asarray(a).reshape(-1,3));self.previous=u.copy();return u
def train(config,seed,steps,stage):
    torch,TD3,_,_=imports()
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    p=read_json(STUDY/'02_传统控制'/'selected.json')['pid']['parameters'];out=STUDY/'03_训练'/stage/config/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True);meta=out/'metadata.json'
    if meta.exists() and read_json(meta).get('status')=='complete':return read_json(meta)
    before=source_hashes();env=make_environment(p,seed);arch={'A':[128,128],'B':[256,128]}[config]
    model=TD3('MlpPolicy',env,learning_rate=3e-4,buffer_size=400000,learning_starts=4096,batch_size=256,gamma=.999,tau=.005,train_freq=(1,'step'),gradient_steps=4,action_noise=NormalActionNoise(np.zeros(3),np.full(3,.15)),target_policy_noise=.15,target_noise_clip=.3,policy_kwargs=dict(net_arch=arch),seed=seed,device='cpu',verbose=0)
    with torch.no_grad():model.actor.mu[-2].weight.mul_(.01);model.actor.mu[-2].bias.zero_()
    val=scenes('validation');best=None;records=[]
    class Validation(BaseCallback):
        def _on_step(self):
            nonlocal best
            if self.num_timesteps%CHECK_EVERY==0 or self.num_timesteps>=steps:
                row=dict(steps=int(self.num_timesteps),**summary(evaluate(val,LearnedController(p,self.model))['rows']));records.append(row)
                if best is None or rank(row)<rank(best):best=row;self.model.save(out/'best.zip')
                write_json(meta,dict(status='running',config=config,seed=seed,actual_steps=int(self.num_timesteps),source_sha256=before,best=best,checkpoints=records));print(json.dumps(dict(event='validation',stage=stage,config=config,seed=seed,**row)),flush=True)
            return True
    started=time.perf_counter();model.learn(total_timesteps=steps,callback=Validation());model.save(out/'final.zip');env.close()
    if source_hashes()!=before:raise RuntimeError('training source changed')
    result=dict(status='complete',config=config,seed=seed,requested_steps=steps,actual_steps=int(model.num_timesteps),source_sha256=before,best=best,checkpoints=records,model_sha256=sha(out/'best.zip'),wall_time_s=time.perf_counter()-started,parameters=p,test_used=False);write_json(meta,result);return result
def development():
    r=[train(c,19011+i,DEV_STEPS,'development') for i,c in enumerate(('A','B'))];b=min(r,key=lambda r:rank(r['best']));v=b['best'];result=dict(config=b['config'],seed=b['seed'],validation=v,gate=v['physical_failures']==0 and v['success_rate']>=.95,both_completed=True,test_used=False);write_json(STUDY/'03_训练'/'development_decision.json',result);return result
def formal():
    c=read_json(STUDY/'03_训练'/'development_decision.json')
    return [train(c['config'],seed,FORMAL_STEPS,'formal') for seed in FORMAL_SEEDS] if c['gate'] else dict(skipped=True,reason='development failed')
def registry():
    _,TD3,_,_=imports();b=read_json(STUDY/'02_传统控制'/'selected.json');result={k:Controller(b[k]['parameters'],k,b[k].get('harmonic')) for k in ('pid','observer','harmonic')};c=read_json(STUDY/'03_训练'/'development_decision.json')
    if c['gate']:
        for seed in FORMAL_SEEDS:
            path=STUDY/'03_训练'/'formal'/c['config']/f'seed_{seed}';m=read_json(path/'metadata.json')
            if m['status']!='complete' or m['source_sha256']!=source_hashes():raise RuntimeError('unfrozen model')
            result[f'TD3_{seed}']=LearnedController(b['pid']['parameters'],TD3.load(path/'best.zip',device='cpu'))
    return result
def independent_test():
    controllers=registry();data=scenes('test');folder=STUDY/'04_独立测试';freeze=dict(source_sha256=source_hashes(),models={p.relative_to(STUDY).as_posix():sha(p) for p in (STUDY/'03_训练'/'formal').rglob('best.zip')},scenario_sha256=sha(STUDY/'01_方案与场景'/'test.json'),baseline_sha256=sha(STUDY/'02_传统控制'/'selected.json'),controllers=list(controllers),weights_frozen=True);path=folder/'freeze.json'
    if path.exists() and read_json(path)!=freeze:raise RuntimeError('test freeze changed')
    if not path.exists():write_json(path,freeze)
    results={}
    for name,controller in controllers.items():
        o=evaluate(data,controller,keep_traces=True);write_json(folder/f'{name}_episodes.json',o['rows']);np.savez_compressed(folder/f'{name}_traces.npz',**o['traces']);results[name]=summary(o['rows']);print(json.dumps(dict(event='test',controller=name,**results[name])),flush=True)
    before=read_json(folder/'pid_episodes.json');comparisons={};confidence=1.-.05/3
    for name in controllers:
        if name in ('pid','observer','harmonic'):continue
        after=read_json(folder/f'{name}_episodes.json');a=np.array([r['steady_rms_deg'] for r in before]);b=np.array([r['steady_rms_deg'] for r in after]);idx=np.random.default_rng(190999).integers(0,len(a),(20000,len(a)));ci=np.quantile((a[idx]-b[idx]).mean(axis=1),[(1-confidence)/2,1-(1-confidence)/2]).tolist();ratio=results[name]['mean_actual_cost']/results['pid']['mean_actual_cost'];allpass=all(r['success'] for r in before+after);comparisons[name]=dict(all_pass=allpass,improvement_interval_deg=ci,confidence=confidence,cost_ratio=ratio,primary_supported=bool(allpass and ci[0]>0 and ratio<=1.1))
    files={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.json','.npz') and p.name!='summary.json'};r=dict(results=results,comparisons=comparisons,files_sha256=files);write_json(folder/'summary.json',r);return r
def verify():
    folder=STUDY/'04_独立测试';f=read_json(folder/'freeze.json');r=read_json(folder/'summary.json')
    if source_hashes()!=f['source_sha256']:raise RuntimeError('source mismatch')
    for p,h in f['models'].items():
        if sha(STUDY/p)!=h:raise RuntimeError('model mismatch')
    for p,h in r['files_sha256'].items():
        if sha(folder/p)!=h:raise RuntimeError('response mismatch')
    data=scenes('test');controllers=registry();largest=0.;norm_error=0.;cost_error=0.;tail=np.arange(STEPS+1)*DT>=DEADLINE-1e-10
    for name,c in controllers.items():
        rows=read_json(folder/f'{name}_episodes.json');t=np.load(folder/f'{name}_traces.npz');replay=evaluate(data,c,keep_traces=True)
        for k in ('quaternion','omega','tau','phase','sent_command','command','actual_cost_interval'):
            e=float(np.max(np.abs(t[k]-replay['traces'][k])));largest=max(largest,e)
            if e>1e-11:raise RuntimeError('trace/controller mismatch')
        q=t['quaternion'];qn=np.linalg.norm(q,axis=-1);norm_error=max(norm_error,float(np.max(np.abs(qn-1.))));angle=np.degrees(2*np.arccos(np.clip(np.abs(q[:,:,0])/qn,0.,1.)));rate=np.degrees(np.linalg.norm(t['omega'],axis=-1));rms=np.sqrt(np.mean(angle[tail]**2,axis=0));passed=np.all(angle[tail]<=ANGLE_TOL,axis=0)&np.all(rate[tail]<=RATE_TOL,axis=0)&~np.any(t['failed'],axis=0)
        old=t['tau'][:-1];u=t['command'];d=old-u;e=np.exp(-DT/LAG);integ=np.sum(u*u*DT+2*u*d*LAG*(1-e)+d*d*LAG*.5*(1-e*e),axis=-1);integ=np.where(t['failed'][:-1],0.,integ);cost_error=max(cost_error,float(np.max(np.abs(integ-t['actual_cost_interval']))))
        for i,row in enumerate(rows):
            if passed[i]!=row['success'] or abs(rms[i]-row['steady_rms_deg'])>2e-7 or abs(t['actual_cost_interval'][:,i].sum()-row['actual_cost'])>1e-10:raise RuntimeError('independent metric mismatch')
    if norm_error>1e-12 or cost_error>1e-12:raise RuntimeError('quaternion norm or cost mismatch')
    value=dict(passed=True,episodes=len(data)*len(controllers),controllers=len(controllers),all_traces_and_commands_replayed=True,all_metrics_independently_reconstructed=True,max_replay_error=largest,max_quaternion_norm_error=norm_error,max_independent_interval_cost_error=cost_error,physical_numerical_checks='test_physics.py: independent32s DOP853,conservation,convergence',scope='all frozen traces/actions/metrics and pretraining independent numerical tests');write_json(STUDY/'05_独立核验'/'verification.json',value);return value
def report():
    v=read_json(STUDY/'05_独立核验'/'verification.json');r=read_json(STUDY/'04_独立测试'/'summary.json')
    if not v['passed']:raise RuntimeError('verify first')
    supported=bool(r['comparisons']) and all(c['primary_supported'] for c in r['comparisons'].values());lines=['# 周期扰动下的三轴联合学习控制','','真实Euler陀螺耦合＋四元数姿态＋滞后执行器。一个联合TD3网络输出3维连续补偿；三个正式模型从零训练，没有逐轴套用单轴模型。','','|方法|128场景成功数|8–32秒总姿态角RMS（°）|三轴实际力矩平方积分之和|','|---|---:|---:|---:|']
    for name,x in r['results'].items():lines.append(f"|{name}|{x['passes']}/{x['count']}|{x['mean_steady_rms_deg']:.6f}|{x['mean_actual_cost']:.6f}|")
    lines+=['',f"三种子预声明主结论全部成立：{'是' if supported else '否'}。",'',f'开发两个配置各{DEV_STEPS:,}步，正式三个新种子各{FORMAL_STEPS:,}步。所有方法共用信息与限制，测试时不更新权重。', '', '相对PID与相对DOB/谐波的结论须分别报告，控制代价不是电能。本模型不对应具体飞行卫星，未含完整轨道、柔性帆板或反作用轮。', '',f"{v['episodes']}回合全部轨迹/指令/指标重放；物理另经独立DOP853整段重放、守恒和步长收敛测试。"]
    out=STUDY/'06_报告';out.mkdir(parents=True,exist_ok=True);text='\n'.join(lines)+'\n';(out/'研究报告.md').write_text(text,encoding='utf-8')
    import html
    (out/'研究报告.html').write_text('<!doctype html><meta charset=utf-8><style>body{font:17px/1.8 system-ui;max-width:1100px;margin:40px auto}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>',encoding='utf-8');return dict(primary_supported=supported)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['design','tune','development','formal','seed','test','verify','report']);p.add_argument('--seed',type=int,default=FORMAL_SEEDS[0]);a=p.parse_args()
    if a.command=='seed':c=read_json(STUDY/'03_训练'/'development_decision.json');result=train(c['config'],a.seed,FORMAL_STEPS,'formal')
    else:result=globals()[{'test':'independent_test'}.get(a.command,a.command)]()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=='__main__':main()
