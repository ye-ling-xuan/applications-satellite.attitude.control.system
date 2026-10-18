"""Continuous residual torque control for measurable-phase unknown disturbances.

All scenario allocation, budgets, objectives and traditional alternatives are
declared before held-out responses.  This study tests precision disturbance
rejection rather than discrete route selection.  No labels or true disturbance
parameters are supplied to learned or traditional deployed controllers.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
DT, DURATION, DEADLINE, LAG, INERTIA, LIMIT = .05, 32., 8., .15, 1., .5
STEPS = round(DURATION/DT)
ANGLE_TOL, RATE_TOL = 2., 1.
FORMAL_SEEDS = (17111, 17222, 17333)
DEV_STEPS, FORMAL_STEPS, CHECK_EVERY = 98304, 196608, 24576
COUNTS = {'development':32, 'validation':48, 'test':128}
SPLIT_SEEDS = {'development':202610181701, 'validation':202610181702, 'test':202610181703}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    return {p.relative_to(STUDY).as_posix():sha(p) for p in sorted((STUDY/'代码').glob('*.py'))}


def random_scenes(rng, n):
    return [dict(sample_id=i, angle_deg=float(a), omega_deg_s=float(w), phase_rad=float(p),
                 frequency_rad_s=float(f), amplitude_1_Nm=float(x), amplitude_2_Nm=float(y),
                 attitude_coefficient_Nm=float(c), rate_coefficient_Nm=float(v))
            for i,(a,w,p,f,x,y,c,v) in enumerate(zip(
                rng.uniform(-12.,12.,n),rng.uniform(-1.,1.,n),rng.uniform(-np.pi,np.pi,n),
                rng.uniform(.35,.65,n),rng.uniform(.025,.045,n),rng.uniform(.005,.015,n),
                rng.uniform(.015,.025,n),rng.uniform(.003,.007,n)))]


def design():
    settings = dict(dt_s=DT,duration_s=DURATION,deadline_s=DEADLINE,inertia=INERTIA,
        actuator_time_constant_s=LAG,command_limit_Nm=LIMIT,
        acceptance=dict(angle_deg=ANGLE_TOL,angular_velocity_deg_s=RATE_TOL,
                        sustained_interval_s=[DEADLINE,DURATION]),
        physical_failure=dict(angle_deg=90.,angular_velocity_deg_s=60.,frozen_until_horizon=True),
        disturbance='A1*sin(phi)+A2*sin(2*phi+0.4)+c_theta*sin(2*theta)+c_rate*tanh(omega/0.05)',
        phase='phi_dot=frequency; phase and frequency are measured, disturbance coefficients hidden',
        coefficient_ranges=dict(A1=[.025,.045],A2=[.005,.015],c_theta=[.015,.025],c_rate=[.003,.007]),
        practical_scope='Illustrative periodic mechanical and attitude-dependent disturbance rejection; no claim that these numbers describe a flight satellite or a complete reaction wheel model.',
        algorithms=['SAC','TD3'],development_steps=DEV_STEPS,formal_steps=FORMAL_STEPS,
        learning=dict(network=[64,64],gamma=.999,learning_rate=.0003,batch_size=128,
            parallel_environments=8,residual_limit_Nm=.12,steady_reward_angle_scale_deg=.03,
            early_reward_angle_scale_deg=2.,SAC_entropy_initial=.01,TD3_action_noise_std=.15),
        formal_seeds=FORMAL_SEEDS,validation_every=CHECK_EVERY,counts=COUNTS,split_seeds=SPLIT_SEEDS,
        baselines=['independently tuned PID','disturbance observer plus PID','measured-phase harmonic regression plus PID'],
        selection='minimize physical failures, maximize sustained passes, minimize steady-state RMS angle error; then actual torque-square integral',
        primary='all three fresh learned seeds and PID pass all held-out scenarios; paired RMS reduction confidence lower bounds >0 and mean actual torque-square cost not over 110% PID',
        correction=dict(alpha=.05,comparisons=3,confidence=1.-.05/3),
        independent_test_used_to_choose=False,weights_frozen_during_test=True,
        traditional_comparison='Full observer and harmonic alternatives are reported separately; no superiority to all traditional methods is implied.',
        zero_residual_is_PID=True,interaction_budget=2*DEV_STEPS+3*FORMAL_STEPS)
    path=STUDY/'01_方案与场景'/'预声明.json'
    normalized=json.loads(json.dumps(settings))
    if path.exists() and read_json(path)!=normalized:
        raise RuntimeError('predeclared protocol changed')
    if not path.exists(): write_json(path,normalized)
    for split,seed in SPLIT_SEEDS.items():
        data=random_scenes(np.random.default_rng(seed),COUNTS[split])
        path=STUDY/'01_方案与场景'/f'{split}.json'
        if path.exists() and read_json(path)!=data: raise RuntimeError('scenario split changed')
        if not path.exists(): write_json(path,data)
    return normalized


def scenes(split):
    return read_json(STUDY/'01_方案与场景'/f'{split}.json')


def actual_cost(old,command,h=DT):
    e=np.exp(-h/LAG)
    delta=old-command
    return command**2*h+2*command*delta*LAG*(1-e)+delta**2*LAG*.5*(1-e*e)


class Plant:
    def __init__(self,data):
        self.n=len(data)
        self.reset(data)
    def reset(self,data,mask=None):
        if mask is None:
            mask=np.ones(self.n,bool)
            self.theta=np.zeros(self.n);self.omega=np.zeros(self.n);self.tau=np.zeros(self.n)
            self.phase=np.zeros(self.n);self.frequency=np.zeros(self.n);self.count=np.zeros(self.n,int)
            self.failed=np.zeros(self.n,bool)
            self.coefficients=np.zeros((self.n,4))
        index=np.flatnonzero(mask)
        if len(data)!=len(index): raise ValueError('reset subset count differs')
        self.theta[index]=np.radians([r['angle_deg'] for r in data])
        self.omega[index]=np.radians([r['omega_deg_s'] for r in data])
        self.phase[index]=[r['phase_rad'] for r in data]
        self.frequency[index]=[r['frequency_rad_s'] for r in data]
        self.coefficients[index]=[[r[k] for k in ('amplitude_1_Nm','amplitude_2_Nm','attitude_coefficient_Nm','rate_coefficient_Nm')] for r in data]
        self.tau[index]=0.;self.count[index]=0;self.failed[index]=False
    def measured(self):
        return np.column_stack((-self.theta,self.omega,np.sin(self.phase),np.cos(self.phase),self.frequency))
    def _disturbance(self,theta,omega,phase):
        c=self.coefficients
        return c[:,0]*np.sin(phase)+c[:,1]*np.sin(2*phase+.4)+c[:,2]*np.sin(2*theta)+c[:,3]*np.tanh(omega/.05)
    def step(self,commands):
        commands=np.asarray(commands,float)
        if commands.shape!=(self.n,) or np.any(~np.isfinite(commands)): raise ValueError('invalid commands')
        active=(self.count<STEPS)&~self.failed
        u=np.where(active,np.clip(commands,-LIMIT,LIMIT),0.)
        old=self.tau.copy()
        cost=np.where(active,actual_cost(old,u),0.)
        x=np.column_stack((self.theta,self.omega))
        initial=x.copy()
        def rhs(state,offset):
            torque=u+(old-u)*np.exp(-offset/LAG)
            disturbance=self._disturbance(state[:,0],state[:,1],self.phase+self.frequency*offset)
            return np.column_stack((state[:,1],(torque+disturbance)/INERTIA))
        # Two RK4 substeps resolve the state-dependent forcing; actuator stage
        # torque is analytic, and physical control cost is interval exact.
        for j in range(2):
            h=DT/2;offset=j*h
            a=rhs(x,offset);b=rhs(x+h*a/2,offset+h/2)
            c=rhs(x+h*b/2,offset+h/2);d=rhs(x+h*c,offset+h)
            x+=h*(a+2*b+2*c+d)/6
        x=np.where(active[:,None],x,initial)
        self.theta,self.omega=x.T
        tau_end=u+(old-u)*np.exp(-DT/LAG)
        self.tau=np.where(active,tau_end,0.)
        new_failure=active&((np.abs(self.theta)>np.radians(90.))|(np.abs(self.omega)>np.radians(60.)))
        self.failed|=new_failure
        self.phase+=self.frequency*DT
        self.count=np.minimum(self.count+1,STEPS)
        return dict(command=u,actual_cost=cost,command_cost=u*u*DT,
                    tau_start=np.where(active,old,0.),tau_end=self.tau.copy(),new_failure=new_failure)


class Controller:
    def __init__(self,parameters,family='pid',harmonic=None):
        self.p=parameters;self.family=family;self.harmonic=harmonic
    def reset(self,n,obs):
        def expand(key,default=0.): return np.broadcast_to(np.asarray(self.p.get(key,default),float),(n,)).copy()
        self.kp=expand('kp');self.kd=expand('kd');self.ki=expand('ki')
        self.filter_time=expand('filter_time',.1);self.predict=expand('predict',0.)
        self.integral=np.zeros(n);self.tau=np.zeros(n);self.previous_u=np.zeros(n)
        self.previous_w=np.zeros(n);self.have=np.zeros(n,bool);self.estimate=np.zeros(n)
        self.derivative=np.zeros(n);self.previous_estimate=np.zeros(n)
    def _feedback(self,obs):
        proposed=np.clip(self.integral+obs[:,0]*DT,-1.,1.)
        return self.kp*obs[:,0]-self.kd*obs[:,1]+self.ki*proposed
    def _commit(self,obs,raw,u):
        proposed=np.clip(self.integral+obs[:,0]*DT,-1.,1.)
        allow=(raw-u)*self.ki*(proposed-self.integral)<=1e-15
        self.integral=np.where(allow,proposed,self.integral)
    def act(self,obs,elapsed=None,residual=None):
        raw=self._feedback(obs)
        if self.family=='observer':
            e=np.exp(-DT/LAG)
            mean_act=self.previous_u+(self.tau-self.previous_u)*LAG*(1-e)/DT
            measured=INERTIA*(obs[:,1]-self.previous_w)/DT-mean_act
            alpha=DT/(self.filter_time+DT)
            updated=self.estimate+alpha*(measured-self.estimate)
            derivative=(updated-self.estimate)/DT
            self.estimate=np.where(self.have,updated,self.estimate)
            self.derivative=np.where(self.have,.5*self.derivative+.5*derivative,self.derivative)
            raw-=self.estimate+self.predict*LAG*self.derivative
        elif self.family=='harmonic':
            phase=np.arctan2(obs[:,2],obs[:,3]);freq=obs[:,4]
            c=np.asarray(self.harmonic,float)
            feed=c[0]*np.sin(phase)+c[1]*np.cos(phase)+c[2]*np.sin(2*phase)+c[3]*np.cos(2*phase)
            derivative=freq*(c[0]*np.cos(phase)-c[1]*np.sin(phase)+2*c[2]*np.cos(2*phase)-2*c[3]*np.sin(2*phase))
            raw-=feed+LAG*derivative
        if residual is not None: raw+=np.asarray(residual,float)
        u=np.clip(raw,-LIMIT,LIMIT)
        self._commit(obs,raw,u)
        self.tau=self.previous_u+(self.tau-self.previous_u)*np.exp(-DT/LAG)
        self.previous_u=u.copy();self.previous_w=obs[:,1].copy();self.have[:]=True
        return u


def summary(rows):
    passed=sum(r['success'] for r in rows)
    return dict(count=len(rows),passes=passed,physical_failures=sum(r['failed'] for r in rows),
                success_rate=passed/len(rows),mean_steady_rms_deg=float(np.mean([r['steady_rms_deg'] for r in rows])),
                mean_actual_cost=float(np.mean([r['actual_cost'] for r in rows])),
                mean_full_rms_deg=float(np.mean([r['full_rms_deg'] for r in rows])))


def rank(value):
    return (value['physical_failures'],-value['passes'],value['mean_steady_rms_deg'],value['mean_actual_cost'])


def evaluate(data,controller,keep_traces=False):
    plant=Plant(data);obs=plant.measured();controller.reset(len(data),obs)
    n=len(data);cost=np.zeros(n);full=np.zeros(n);steady=np.zeros(n)
    successful=np.ones(n,bool);failure=np.full(n,np.nan)
    traces=None
    if keep_traces:
        traces={key:np.empty((STEPS+1,n)) for key in ('theta','omega','tau','phase')}
        traces.update({key:np.empty((STEPS,n)) for key in ('sent_command','command','actual_cost_interval')})
        traces['failed']=np.empty((STEPS+1,n),bool)
        for key in ('theta','omega','tau','phase','failed'):traces[key][0]=getattr(plant,key)
    observations=[];inferred=[]
    previous_w=plant.omega.copy();estimated_tau=np.zeros(n)
    for k in range(STEPS):
        sent=controller.act(obs,np.full(n,k*DT));step=plant.step(sent)
        error=np.degrees(plant.theta);rate=np.degrees(plant.omega)
        full+=error**2*DT;cost+=step['actual_cost']
        if (k+1)*DT>=DEADLINE-1e-10:
            steady+=error**2*DT
            successful&=(np.abs(error)<=ANGLE_TOL)&(np.abs(rate)<=RATE_TOL)&~plant.failed
        failure=np.where(step['new_failure'],(k+1)*DT,failure)
        # Harmonic calibration uses measured acceleration and a known-actuator
        # observer driven solely by commands, never true disturbance labels.
        if keep_traces:
            for key in ('theta','omega','tau','phase','failed'):traces[key][k+1]=getattr(plant,key)
            traces['sent_command'][k]=sent;traces['command'][k]=step['command'];traces['actual_cost_interval'][k]=step['actual_cost']
        if k>=round(DEADLINE/DT):
            mid_phase=plant.phase-.5*DT*plant.frequency
            observations.append(np.column_stack((np.sin(mid_phase),np.cos(mid_phase),np.sin(2*mid_phase),np.cos(2*mid_phase))))
            avg_act=sent+(estimated_tau-sent)*LAG*(1-np.exp(-DT/LAG))/DT
            inferred.append(INERTIA*(plant.omega-previous_w)/DT-avg_act)
        estimated_tau=sent+(estimated_tau-sent)*np.exp(-DT/LAG)
        previous_w=plant.omega.copy();obs=plant.measured()
    steady_duration=DURATION-DEADLINE+DT
    rows=[dict(r,success=bool(successful[i] and not plant.failed[i]),failed=bool(plant.failed[i]),
               actual_cost=float(cost[i]),steady_rms_deg=float(np.sqrt(steady[i]/steady_duration)),
               full_rms_deg=float(np.sqrt(full[i]/DURATION)),
               failure_time_s=None if np.isnan(failure[i]) else float(failure[i])) for i,r in enumerate(data)]
    return dict(rows=rows,traces=traces,calibration_x=np.concatenate(observations),calibration_y=np.concatenate(inferred))


def tune():
    design();before=source_hashes();rng=np.random.default_rng(17004)
    dev=scenes('development');val=scenes('validation')
    candidates=[]
    for kp in np.geomspace(.5,300.,18):
        for kd in np.geomspace(.5,150.,18):
            for ki in (0.,.2,1.,5.): candidates.append(dict(kp=float(kp),kd=float(kd),ki=ki))
    for _ in range(300): candidates.append(dict(kp=float(np.exp(rng.uniform(np.log(.5),np.log(300.)))),
        kd=float(np.exp(rng.uniform(np.log(.5),np.log(150.)))),ki=float(np.exp(rng.uniform(np.log(.01),np.log(20.))))))
    # Retain the earlier narrower grid when extending a search: a wider but
    # coarser grid must not silently discard already available good gains.
    for kp in np.geomspace(.5,50.,15):
        for kd in np.geomspace(.5,20.,15):
            for ki in (0.,.2,1.):candidates.append(dict(kp=float(kp),kd=float(kd),ki=ki))
    def search(family,parameters,harmonic=None):
        scores=[]
        for offset in range(0,len(parameters),32):
            chunk=parameters[offset:offset+32]
            packed={key:np.repeat([r.get(key,0.) for r in chunk],len(dev)) for key in set().union(*(r.keys() for r in chunk))}
            output=evaluate(dev*len(chunk),Controller(packed,family,harmonic))['rows']
            scores.extend(summary(output[i*len(dev):(i+1)*len(dev)]) for i in range(len(chunk)))
        indexes=sorted(range(len(parameters)),key=lambda i:rank(scores[i]))[:24]
        validation=[]
        for i in indexes: validation.append(summary(evaluate(val,Controller(parameters[i],family,harmonic))['rows']))
        j=min(range(len(indexes)),key=lambda j:rank(validation[j]));best=indexes[j]
        result=dict(parameters=parameters[best],development=scores[best],validation=validation[j],
                    development_candidates=len(parameters),validation_candidates=len(indexes),harmonic=harmonic)
        write_json(STUDY/'02_传统控制'/f'{family}_search.json',dict(candidates=parameters,development=scores,
            validation=[dict(index=i,summary=s) for i,s in zip(indexes,validation)],selected=result,source_sha256=before,
            heldout_read=False))
        print(json.dumps(dict(event='baseline_selected',family=family,**result),ensure_ascii=False),flush=True)
        return result
    pid=search('pid',candidates)
    calibration=evaluate(dev,Controller(pid['parameters']))
    coefficients=np.linalg.lstsq(calibration['calibration_x'],calibration['calibration_y'],rcond=None)[0].tolist()
    leaders=read_json(STUDY/'02_传统控制'/'pid_search.json')
    indexes=sorted(range(len(candidates)),key=lambda i:rank(leaders['development'][i]))[:12]
    observer=[];harmonic=[]
    for index in indexes:
        base=candidates[index]
        for factor in (.5,1.,1.5):
            shifted=dict(base,kp=base['kp']*factor,kd=base['kd']*factor)
            harmonic.append(shifted)
            for filter_time in (.025,.075,.15,.3,.6):
                for predict in (0.,.5,1.): observer.append(dict(shifted,filter_time=filter_time,predict=predict))
    dob=search('observer',observer)
    harmonic=search('harmonic',harmonic,coefficients)
    if source_hashes()!=before: raise RuntimeError('source changed during traditional tuning')
    result=dict(pid=pid,observer=dob,harmonic=harmonic,source_sha256=before,heldout_used=False)
    write_json(STUDY/'02_传统控制'/'selected.json',result)
    return result


def _imports():
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC,TD3
    from stable_baselines3.common.vec_env import VecEnv
    from gymnasium import spaces
    return torch,SAC,TD3,VecEnv,spaces


def features(obs,controller,previous,elapsed):
    return np.column_stack((obs[:,0]/np.radians(12.),obs[:,1]/np.radians(5.),obs[:,2],obs[:,3],
                            obs[:,4]/.5,controller.integral,previous/LIMIT,np.asarray(elapsed)/DURATION)).astype(np.float32)


def make_environment(parameters,seed,n=8):
    _,_,_,VecEnv,spaces=_imports()
    class Environment(VecEnv):
        def __init__(self):
            self.rng=np.random.default_rng(seed);self.plant=Plant(random_scenes(self.rng,n))
            self.controller=Controller(parameters);self.controller.reset(n,self.plant.measured())
            self.previous=np.zeros(n);self.pending=None;self.returns=np.zeros(n)
            super().__init__(n,spaces.Box(-np.inf,np.inf,shape=(8,),dtype=np.float32),spaces.Box(-1.,1.,shape=(1,),dtype=np.float32))
        def _obs(self):return features(self.plant.measured(),self.controller,self.previous,self.plant.count*DT)
        def reset(self):
            self.plant.reset(random_scenes(self.rng,n));self.controller.reset(n,self.plant.measured());self.previous[:]=0.;self.returns[:]=0.
            return self._obs()
        def step_async(self,actions):self.pending=np.asarray(actions,float).reshape(n)
        def step_wait(self):
            obs=self.plant.measured();u=self.controller.act(obs,residual=.12*np.clip(self.pending,-1.,1.))
            step=self.plant.step(u);self.previous=u
            angle=np.degrees(self.plant.theta);rate=np.degrees(self.plant.omega)
            precision=np.where(self.plant.count*DT>=DEADLINE,.03,2.)
            reward=-(np.minimum((angle/precision)**2,100.)+.1*np.minimum((rate/2.)**2,100.)+.002*(u/LIMIT)**2)*DT
            reward=np.where(self.plant.failed,-20.,reward)
            done=self.plant.count>=STEPS;self.returns+=reward
            terminal=self._obs();infos=[{} for _ in range(n)]
            for i in np.flatnonzero(done):infos[i]={'terminal_observation':terminal[i].copy(),'TimeLimit.truncated':False,
                'episode':dict(r=float(self.returns[i]),l=STEPS)}
            if np.any(done):
                index=np.flatnonzero(done);self.plant.reset(random_scenes(self.rng,len(index)),done)
                # All controller internal arrays are reset only for completed rows.
                fresh=Controller(parameters);fresh.reset(len(index),self.plant.measured()[index])
                for key in ('integral','tau','previous_u','previous_w','have','estimate','derivative','previous_estimate'):
                    getattr(self.controller,key)[index]=getattr(fresh,key)
                self.previous[index]=0.;self.returns[index]=0.
            return self._obs(),reward.astype(np.float32),done,infos
        def close(self):pass
        def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name,None) for _ in self._get_indices(indices)]
        def set_attr(self,attr_name,value,indices=None):setattr(self,attr_name,value)
        def env_method(self,method_name,*args,indices=None,**kwargs):return [getattr(self,method_name)(*args,**kwargs) for _ in self._get_indices(indices)]
        def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
    return Environment()


class LearnedController(Controller):
    def __init__(self,parameters,model):super().__init__(parameters);self.model=model
    def reset(self,n,obs):super().reset(n,obs);self.previous=np.zeros(n)
    def act(self,obs,elapsed=None,residual=None):
        action,_=self.model.predict(features(obs,self,self.previous,elapsed),deterministic=True)
        command=super().act(obs,elapsed,.12*np.asarray(action).reshape(-1))
        self.previous=command.copy()
        return command


def train(algorithm,seed,steps,stage):
    torch,SAC,TD3,_,_=_imports()
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    selection=read_json(STUDY/'02_传统控制'/'selected.json');parameters=selection['pid']['parameters']
    output=STUDY/'03_训练'/stage/algorithm/f'seed_{seed}';output.mkdir(parents=True,exist_ok=True)
    metadata=output/'metadata.json'
    if metadata.exists() and read_json(metadata).get('status')=='complete':return read_json(metadata)
    before=source_hashes();torch.manual_seed(seed);environment=make_environment(parameters,seed)
    cls={'SAC':SAC,'TD3':TD3}[algorithm]
    options=dict(learning_rate=3e-4,buffer_size=250000,learning_starts=2048,batch_size=128,
                 gamma=.999,tau=.005,train_freq=(1,'step'),gradient_steps=1,seed=seed,
                 policy_kwargs=dict(net_arch=[64,64]),device='cpu',verbose=0)
    if algorithm=='SAC':options.update(ent_coef='auto_0.01',policy_kwargs=dict(net_arch=[64,64],log_std_init=-2.))
    else:options.update(action_noise=NormalActionNoise(np.zeros(1),np.full(1,.15)),target_policy_noise=.15,target_noise_clip=.3)
    model=cls('MlpPolicy',environment,**options)
    with torch.no_grad():
        if algorithm=='SAC':model.actor.mu.weight.mul_(.01);model.actor.mu.bias.zero_()
        else:model.actor.mu[-2].weight.mul_(.01);model.actor.mu[-2].bias.zero_()
    validation=scenes('validation');records=[];best=None
    class Validation(BaseCallback):
        def _on_step(self):
            nonlocal best
            if self.num_timesteps%CHECK_EVERY==0 or self.num_timesteps>=steps:
                stats=summary(evaluate(validation,LearnedController(parameters,self.model))['rows'])
                row=dict(steps=int(self.num_timesteps),**stats);records.append(row)
                if best is None or rank(row)<rank(best):best=row;self.model.save(output/'best.zip')
                print(json.dumps(dict(event='validation',stage=stage,algorithm=algorithm,seed=seed,**row)),flush=True)
                write_json(metadata,dict(status='running',algorithm=algorithm,seed=seed,requested_steps=steps,
                    actual_steps=int(self.num_timesteps),source_sha256=before,best=best,checkpoints=records))
            return True
    started=time.perf_counter();model.learn(total_timesteps=steps,callback=Validation())
    model.save(output/'final.zip');environment.close()
    if source_hashes()!=before:raise RuntimeError('source changed during training')
    value=dict(status='complete',algorithm=algorithm,seed=seed,requested_steps=steps,actual_steps=int(model.num_timesteps),
               source_sha256=before,best=best,checkpoints=records,model_sha256=sha(output/'best.zip'),
               wall_time_s=time.perf_counter()-started,parameters=parameters,heldout_used=False)
    write_json(metadata,value);return value


def development():
    records=[train(a,17011+i,DEV_STEPS,'development') for i,a in enumerate(('SAC','TD3'))]
    best=min(records,key=lambda r:rank(r['best']))
    decision=dict(algorithm=best['algorithm'],seed=best['seed'],validation=best['best'],
        gate=best['best']['physical_failures']==0 and best['best']['success_rate']>=.95,
        both_algorithms_completed=True,test_used=False)
    write_json(STUDY/'03_训练'/'development_decision.json',decision)
    return decision


def formal():
    decision=read_json(STUDY/'03_训练'/'development_decision.json')
    if not decision['gate']:return dict(skipped=True,reason='development gate not met')
    return [train(decision['algorithm'],seed,FORMAL_STEPS,'formal') for seed in FORMAL_SEEDS]


def _controller_registry():
    _,SAC,TD3,_,_=_imports()
    baseline=read_json(STUDY/'02_传统控制'/'selected.json');result={}
    for family in ('pid','observer','harmonic'):
        item=baseline[family];result[family]=Controller(item['parameters'],family,item.get('harmonic'))
    decision=read_json(STUDY/'03_训练'/'development_decision.json')
    algorithm=decision['algorithm'];cls={'SAC':SAC,'TD3':TD3}[algorithm]
    if decision['gate']:
        for seed in FORMAL_SEEDS:
            path=STUDY/'03_训练'/'formal'/algorithm/f'seed_{seed}'
            metadata=read_json(path/'metadata.json')
            if metadata['status']!='complete' or metadata['source_sha256']!=source_hashes():raise RuntimeError('model not frozen with current sources')
            result[f'{algorithm}_{seed}']=LearnedController(baseline['pid']['parameters'],cls.load(path/'best.zip',device='cpu'))
    return result


def independent_test():
    design();registry=_controller_registry();directory=STUDY/'04_独立测试'
    models={p.relative_to(STUDY).as_posix():sha(p) for p in (STUDY/'03_训练'/'formal').rglob('best.zip')}
    freeze=dict(source_sha256=source_hashes(),models=models,scenario_sha256=sha(STUDY/'01_方案与场景'/'test.json'),
                baseline_sha256=sha(STUDY/'02_传统控制'/'selected.json'),controllers=list(registry),weights_frozen=True)
    path=directory/'freeze.json'
    if path.exists() and read_json(path)!=freeze:raise RuntimeError('test freeze changed')
    if not path.exists():write_json(path,freeze)
    data=scenes('test');all_results={}
    for name,controller in registry.items():
        output=evaluate(data,controller,keep_traces=True)
        write_json(directory/f'{name}_episodes.json',output['rows'])
        np.savez_compressed(directory/f'{name}_traces.npz',**output['traces'])
        all_results[name]=summary(output['rows'])
        print(json.dumps(dict(event='independent_test',controller=name,**all_results[name])),flush=True)
    comparisons={}
    confidence=1.-.05/3
    for name in registry:
        if name in ('pid','observer','harmonic'):continue
        before=read_json(directory/'pid_episodes.json');after=read_json(directory/f'{name}_episodes.json')
        a=np.array([r['steady_rms_deg'] for r in before]);b=np.array([r['steady_rms_deg'] for r in after])
        rng=np.random.default_rng(170999);sample=rng.integers(0,len(a),(20000,len(a)))
        improvement=(a[sample]-b[sample]).mean(axis=1)
        interval=np.quantile(improvement,[(1-confidence)/2,1-(1-confidence)/2]).tolist()
        cost_ratio=all_results[name]['mean_actual_cost']/all_results['pid']['mean_actual_cost']
        all_pass=all(r['success'] for r in before+after)
        comparisons[name]=dict(all_pass=all_pass,mean_paired_rms_improvement_deg=float(np.mean(a-b)),
            improvement_interval_deg=interval,confidence=confidence,cost_ratio=cost_ratio,
            primary_supported=bool(all_pass and interval[0]>0. and cost_ratio<=1.1))
    hashes={p.name:sha(p) for p in directory.iterdir() if p.suffix in ('.json','.npz') and p.name not in ('summary.json',)}
    write_json(directory/'summary.json',dict(results=all_results,comparisons=comparisons,files_sha256=hashes))
    return all_results


def verify():
    directory=STUDY/'04_独立测试';freeze=read_json(directory/'freeze.json');report=read_json(directory/'summary.json')
    if freeze['source_sha256']!=source_hashes():raise RuntimeError('source hash mismatch')
    for relative,h in freeze['models'].items():
        if sha(STUDY/relative)!=h:raise RuntimeError('model hash mismatch')
    for filename,h in report['files_sha256'].items():
        if sha(directory/filename)!=h:raise RuntimeError('response hash mismatch')
    registry=_controller_registry();data=scenes('test');largest=0.;physics_error=0.
    for name,controller in registry.items():
        stored=read_json(directory/f'{name}_episodes.json');traces=np.load(directory/f'{name}_traces.npz')
        replay=evaluate(data,controller,keep_traces=True)
        for key in ('theta','omega','tau','phase','sent_command','command','actual_cost_interval'):
            error=float(np.max(np.abs(traces[key]-replay['traces'][key])));largest=max(largest,error)
            if error>1e-11:raise RuntimeError('controller/physics replay mismatch')
        for i,row in enumerate(stored):
            angle=np.degrees(traces['theta'][:,i]);rate=np.degrees(traces['omega'][:,i]);tail=np.arange(STEPS+1)*DT>=DEADLINE-1e-10
            success=bool(np.all(np.abs(angle[tail])<=ANGLE_TOL)&np.all(np.abs(rate[tail])<=RATE_TOL)&~np.any(traces['failed'][:,i]))
            rms=float(np.sqrt(np.mean(angle[tail]**2)))
            cost=float(traces['actual_cost_interval'][:,i].sum())
            if success!=row['success'] or abs(rms-row['steady_rms_deg'])>1e-10 or abs(cost-row['actual_cost'])>1e-11:
                raise RuntimeError('independent metric reconstruction mismatch')
        # Independent adaptive ODE integration on two fixed scenes per method,
        # with recorded held commands; does not reuse the plant RK4 routine.
        from scipy.integrate import solve_ivp
        for i in (0,len(data)//2):
            r=data[i];state=np.array([np.radians(r['angle_deg']),np.radians(r['omega_deg_s']),0.])
            for k in range(STEPS):
                u=traces['command'][k,i]
                def rhs(t,x):
                    phase=r['phase_rad']+r['frequency_rad_s']*t
                    disturb=r['amplitude_1_Nm']*math.sin(phase)+r['amplitude_2_Nm']*math.sin(2*phase+.4)+r['attitude_coefficient_Nm']*math.sin(2*x[0])+r['rate_coefficient_Nm']*math.tanh(x[1]/.05)
                    return [x[1],(x[2]+disturb)/INERTIA,(u-x[2])/LAG]
                state=solve_ivp(rhs,(k*DT,(k+1)*DT),state,rtol=1e-10,atol=1e-12).y[:,-1]
                reference=np.array([traces['theta'][k+1,i],traces['omega'][k+1,i],traces['tau'][k+1,i]])
                physics_error=max(physics_error,float(np.max(np.abs(state-reference))))
                if np.any(traces['failed'][:,i]):break
    if physics_error>2e-7:raise RuntimeError('independent ODE comparison exceeds tolerance')
    value=dict(passed=True,controllers=len(registry),episodes=len(registry)*len(data),
        all_command_and_physics_replays=True,all_metrics_independently_reconstructed=True,
        independent_adaptive_ODE_episodes=2*len(registry),max_replay_error=largest,max_independent_ODE_error=physics_error,
        scope='All response hashes, commands and metrics; independent ODE comparison on two preselected scenes per controller.')
    write_json(STUDY/'05_独立核验'/'verification.json',value);return value


def report():
    verification=read_json(STUDY/'05_独立核验'/'verification.json')
    if not verification['passed']:raise RuntimeError('verification not passed')
    data=read_json(STUDY/'04_独立测试'/'summary.json');decision=read_json(STUDY/'03_训练'/'development_decision.json')
    lines=['# 周期与姿态相关扰动下的连续学习补偿','',
        '研究问题：当外力矩随可测相位与姿态连续变化时，学习补偿能否在相同执行器、相同PID基础反馈和相同指向要求下减少指向误差？',
        '', '控制器输出始终是连续力矩。模型看不到真实扰动力矩和随机系数；相位与频率均视为可测量。此处是工程示意模型，数值不代表某颗实际卫星。',
        '', '传统对照包括重新整定PID、扰动观测器加PID、根据开发集测量响应拟合的谐波前馈加PID。谐波拟合使用已发送指令与角速度变化，未读取真实扰动标签。',
        '', '| 方法 | 独立测试成功数 | 8–32秒角度RMS（°） | 实际力矩平方积分 |', '|---|---:|---:|---:|']
    for name,r in data['results'].items():lines.append(f"| {name} | {r['passes']}/{r['count']} | {r['mean_steady_rms_deg']:.6f} | {r['mean_actual_cost']:.6f} |")
    supported=bool(data['comparisons']) and all(r['primary_supported'] for r in data['comparisons'].values())
    lines+=['',f"预声明的三种子主结论是否全部成立：{'是' if supported else '否'}。",'',
        '主结论要求所有测试持续停稳成功、三种子的误差改善区间分别高于零，且平均实际控制代价均不超过PID的110%。控制代价是力矩平方积分，不是实测电能。',
        '', '与扰动观测器和谐波补偿的结果须单独比较，不能把相对单独PID的收益说成优于全部传统控制。',
        '', f"开发阶段比较SAC和TD3，每种{DEV_STEPS:,}步；选中{decision['algorithm']}。正式模型使用三个新随机种子，每种{FORMAL_STEPS:,}步；测试期间不更新网络。",'',
        f"核验：{verification['episodes']}回合全部重放与指标重算通过；另以独立自适应ODE求解器检查{verification['independent_adaptive_ODE_episodes']}回合。",'',
        '所有失败、训练过程与传统对照保留。新场景的初态和扰动参数独立于此前研究；测试响应不用于模型或参数选择。']
    out=STUDY/'06_报告';out.mkdir(parents=True,exist_ok=True)
    (out/'研究报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import html
    (out/'研究报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>周期扰动连续补偿研究</title><style>body{font:17px/1.75 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n'.join(lines))+'</pre></html>',encoding='utf-8')
    return dict(primary_supported=supported,report=str(out/'研究报告.md'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['design','tune','development','formal','test','verify','report','all','seed'])
    parser.add_argument('--seed',type=int,default=FORMAL_SEEDS[0]);args=parser.parse_args()
    if args.command=='seed':
        d=read_json(STUDY/'03_训练'/'development_decision.json');result=train(d['algorithm'],args.seed,FORMAL_STEPS,'formal')
    elif args.command=='all':
        design();tune();development();formal();independent_test();verify();result=report()
    else:result=globals()[{'test':'independent_test'}.get(args.command,args.command)]()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
