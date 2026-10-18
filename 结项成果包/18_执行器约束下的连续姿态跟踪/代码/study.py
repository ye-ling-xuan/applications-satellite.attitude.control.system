"""Predeclared single-axis continuous tracking with torque and slew limits."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
DT, DURATION, INERTIA, LIMIT, SLEW = .1, 30., 1., .25, .08
N_STEPS, N_ENVS = 300, 8
PREVIEW = np.array([.5, 1., 1.5, 2., 2.5, 3.])
RESIDUAL = .15
DEVELOPMENT_STEPS, FORMAL_STEPS, CHECKPOINT = 131072, 262144, 32768
ALGORITHMS = ('TD3', 'PPO')
DEVELOPMENT_SEED, FORMAL_SEEDS = 18101, (111, 222, 333)
GAMMA = .995
FAIL_ANGLE, FAIL_RATE = np.deg2rad(45.), np.deg2rad(45.)
FEATURE_DIM = 20


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    return sha256(__file__)


def relative(path):
    return Path(path).relative_to(STUDY).as_posix()


def draw_scenes(rng, n, split):
    return [dict(scenario_id=f'{split}_{i:05d}', sample_id=i, split=split,
                 a1_deg=float(rng.uniform(4., 8.)), period1_s=float(rng.uniform(6., 9.)),
                 a2_deg=float(rng.uniform(1., 3.)), period2_s=float(rng.uniform(7., 12.)),
                 phase1=float(rng.uniform(-np.pi, np.pi)), phase2=float(rng.uniform(-np.pi, np.pi)))
            for i in range(n)]


def declaration():
    return dict(dt=DT, duration=DURATION, inertia=INERTIA, torque_limit=LIMIT, torque_slew_limit=SLEW,
                reference='two independent sinusoidal components: A1 U[4,8]deg,T1 U[6,9]s; A2 U[1,3]deg,T2 U[7,12]s; phases U[-pi,pi]',
                all_methods_measure=['angle','angular velocity','actual torque','current reference and derivatives','known future reference through 3 s'],
                initial_state='angle and rate equal the moving reference; actual torque equals clipped reference angular acceleration times inertia',
                development=dict(algorithms=ALGORITHMS, seed=DEVELOPMENT_SEED, steps=DEVELOPMENT_STEPS),
                formal=dict(selected_algorithms=1, seeds=FORMAL_SEEDS, steps=FORMAL_STEPS),
                checkpoint=CHECKPOINT, n_envs=N_ENVS, gamma=GAMMA, preview=PREVIEW.tolist(), residual_limit=RESIDUAL,
                feature_order=['current error/10deg','current rate error/15deg_s','actual torque/limit','current reference acceleration*J/limit',
                               'six future reference angle minus current angle /10deg','six future reference rate minus current rate /15deg_s',
                               'PID raw output/limit','PID integral/2','previous sent command/limit','elapsed/duration'],
                reward='-dt*(min((angle_error/2deg)^2,100)+0.1*min((rate_error/5deg_s)^2,100)+0.01*actual_tau_square_integral/(dt*limit^2)+0.001*((tau_end-tau_start)/(slew*dt))^2); failed absorbing cost201 until 300 intervals',
                selection='fewest physical failures; lowest mean normalized tracking objective; earliest checkpoint',
                models=dict(TD3=dict(net_arch=[128,128], learning_rate=.0003, batch_size=256, buffer_size=200000,
                                     learning_starts=2048, gradient_steps=1, train_freq=1, exploration_std=.08,
                                     target_policy_noise=.1,target_noise_clip=.2,policy_delay=2),
                            PPO=dict(net_arch=[128,128],learning_rate=.0003,n_steps=128,batch_size=256,n_epochs=10,log_std_init=-2.5)),
                split_seeds=dict(development=202610181801,validation=202610181802,test=202610181803),
                split_counts=dict(development=48,validation=64,test=128),
                primary='paired tracking RMS error reduction vs PID for all three formal seeds; no more physical failures',
                confidence=1.-.05/3, bootstrap_resamples=20000, no_test_tuning=True,
                mpc=dict(horizon_steps=30,iterations=120,rho=3000.,q_angle=1.,q_rate=.1,r_torque=.01,r_slew=.001,
                         method='ADMM constrained quadratic program; fixed 120 warm-started iterations; actual constraint residual retained'))


def prepare():
    data = json.loads(json.dumps(declaration()))
    path = STUDY/'01_实验方案'/'预声明.json'
    if path.exists() and read_json(path) != data:
        raise RuntimeError('declaration changed')
    write_json(path, data)
    for split, n in data['split_counts'].items():
        scenes = draw_scenes(np.random.default_rng(data['split_seeds'][split]), n, split)
        path = STUDY/'01_实验方案'/f'{split}.json'
        if path.exists() and read_json(path) != scenes:
            raise RuntimeError('scene split changed')
        write_json(path, scenes)


def scenes(split):
    return read_json(STUDY/'01_实验方案'/f'{split}.json')


class Reference:
    def __init__(self, records):
        self.records = records
        self.n = len(records)
        self.a = np.deg2rad(np.array([[s['a1_deg'],s['a2_deg']] for s in records]))
        self.w = 2*np.pi/np.array([[s['period1_s'],s['period2_s']] for s in records])
        self.phase = np.array([[s['phase1'],s['phase2']] for s in records])

    def at(self, elapsed):
        t = np.asarray(elapsed, float)
        if t.ndim == 0:
            t = np.full(self.n, t)
        if t.shape[0] != self.n:
            raise ValueError('reference elapsed batch differs')
        shape = (self.n,)+(1,)*(t.ndim-1)+(2,)
        a,w,phase = self.a.reshape(shape),self.w.reshape(shape),self.phase.reshape(shape)
        argument = t[...,None]*w+phase
        return np.sum(a*np.sin(argument),axis=-1),np.sum(a*w*np.cos(argument),axis=-1),np.sum(-a*w*w*np.sin(argument),axis=-1)


class Plant:
    """Exact propagation for a linear torque ramp in every sample interval."""
    def __init__(self, n):
        self.n = n
        self.theta=np.zeros(n);self.omega=np.zeros(n);self.torque=np.zeros(n)
        self.failed=np.zeros(n,bool);self.count=np.zeros(n,dtype=int)

    @property
    def elapsed(self):
        return self.count*DT

    def reset(self, ref, mask=None):
        mask=np.ones(self.n,bool) if mask is None else np.asarray(mask,bool)
        t,w,a=ref.at(np.zeros(self.n))
        self.theta[mask]=t[mask];self.omega[mask]=w[mask];self.torque[mask]=np.clip(INERTIA*a[mask],-LIMIT,LIMIT)
        self.failed[mask]=False;self.count[mask]=0

    def step(self, command):
        command=np.clip(np.asarray(command,float).reshape(self.n),-LIMIT,LIMIT)
        active=~self.failed&(self.count<N_STEPS)
        old=self.torque.copy()
        end=old+np.clip(command-old,-SLEW*DT,SLEW*DT)
        delta=end-old
        self.theta[active]+=DT*self.omega[active]+DT**2*(old[active]/2+delta[active]/6)/INERTIA
        self.omega[active]+=DT*(old[active]+end[active])/(2*INERTIA)
        self.torque[active]=end[active]
        new=active&((np.abs(self.theta)>FAIL_ANGLE)|(np.abs(self.omega)>FAIL_RATE))
        cost=np.where(active,DT*(old*old+old*end+end*end)/3,0.)
        self.failed|=new
        self.count=np.minimum(self.count+1,N_STEPS)
        return dict(command=command,tau_start=old,tau_end=self.torque.copy(),actual_cost=cost,
                    delta_torque=np.where(active,delta,0.),new_failure=new,failed=self.failed.copy(),done=self.count==N_STEPS)


def measurement(plant, ref):
    target,rate,_=ref.at(plant.elapsed)
    return np.column_stack((target-plant.theta,rate-plant.omega,plant.torque))


class PID:
    def __init__(self, parameters):
        self.parameters=dict(parameters)

    def reset(self,n):
        self.integral=np.zeros(n)

    def propose(self,plant,ref):
        p=self.parameters
        target,rate,acc=ref.at(plant.elapsed+p.get('preview_s',0.))
        error=target-plant.theta
        rate_error=rate-plant.omega
        integral=np.clip(self.integral+DT*error,-2.,2.)
        return p['kp']*error+p['kd']*rate_error+p.get('ki',0.)*integral+p.get('feedforward',0.)*INERTIA*acc

    def commit(self,plant,ref,command,raw):
        target,_,_=ref.at(plant.elapsed+self.parameters.get('preview_s',0.))
        delta=DT*(target-plant.theta)
        candidate=np.clip(self.integral+delta,-2.,2.)
        gain=self.parameters.get('ki',0.)
        reachable=plant.torque+np.clip(command-plant.torque,-SLEW*DT,SLEW*DT)
        accept=(raw-reachable)*gain*(candidate-self.integral)<=1e-15
        self.integral=np.where(accept,candidate,self.integral)

    def act(self,plant,ref):
        raw=self.propose(plant,ref)
        command=np.clip(raw,-LIMIT,LIMIT)
        self.commit(plant,ref,command,raw)
        return command


class MPC:
    """Constrained finite-horizon tracking QP, solved by fixed-iteration ADMM."""
    def __init__(self):
        from scipy.linalg import cho_factor
        self.h=30
        h=self.h
        self.A=np.array([[1.,DT],[0.,1.]])
        self.b0=np.array([DT**2/(3*INERTIA),DT/(2*INERTIA)])
        self.b1=np.array([DT**2/(6*INERTIA),DT/(2*INERTIA)])
        self.F=np.zeros((2*h,2));self.G=np.zeros((2*h,h));self.c=np.zeros(2*h)
        # q[k] is torque at the END of interval k. Previous torque acts as
        # a known affine input in the first interval, and q[k-1] thereafter.
        for k in range(h):
            ak=np.linalg.matrix_power(self.A,k+1)
            self.F[2*k:2*k+2]=ak
            self.c[2*k:2*k+2]=np.linalg.matrix_power(self.A,k)@self.b0
            for j in range(k+1):
                value=np.linalg.matrix_power(self.A,k-j)@self.b1
                if j<k:value+=np.linalg.matrix_power(self.A,k-j-1)@self.b0
                self.G[2*k:2*k+2,j]=value
        weights=np.tile([1/np.deg2rad(2.)**2,.1/np.deg2rad(5.)**2],h)
        self.WG=weights[:,None]*self.G
        self.D=np.eye(h)-np.eye(h,k=-1)
        self.C=np.vstack([np.eye(h),self.D])
        self.rho=3000.
        self.H=2*(self.G.T@self.WG+.01/LIMIT**2*np.eye(h)+.001/(SLEW*DT)**2*(self.D.T@self.D))
        self.factor=cho_factor(self.H+self.rho*self.C.T@self.C,lower=True,check_finite=False)
        self.slew_weight=.001/(SLEW*DT)**2

    def reset(self,n):
        self.q=np.zeros((n,self.h));self.z=np.zeros((n,2*self.h));self.y=np.zeros((n,2*self.h))
        self.max_solver_residual=0.

    def act(self,plant,ref):
        from scipy.linalg import cho_solve
        target,rate,_=ref.at(plant.elapsed[:,None]+DT*np.arange(1,self.h+1))
        desired=np.stack([target,rate],axis=-1).reshape(plant.n,2*self.h)
        free=np.column_stack([plant.theta,plant.omega])@self.F.T+plant.torque[:,None]*self.c
        g=2*(free-desired)@self.WG
        g[:,0]-=2*self.slew_weight*plant.torque
        low=np.full((plant.n,2*self.h),-LIMIT);high=-low
        low[:,self.h:]=-SLEW*DT;high[:,self.h:]=SLEW*DT
        low[:,self.h]+=plant.torque;high[:,self.h]+=plant.torque
        self.q=np.column_stack([self.q[:,1:],self.q[:,-1]])
        self.z=np.clip(self.q@self.C.T,low,high)
        self.y*=.5
        for _ in range(120):
            rhs=-g+self.rho*(self.z-self.y)@self.C
            self.q=cho_solve(self.factor,rhs.T,check_finite=False).T
            cq=self.q@self.C.T
            self.z=np.clip(cq+self.y,low,high)
            self.y+=cq-self.z
        self.max_solver_residual=max(self.max_solver_residual,float(np.max(np.abs(cq-self.z))))
        return np.clip(self.q[:,0],-LIMIT,LIMIT)


def normalized_interval_cost(error,rate_error,step):
    cost=(np.minimum((error/np.deg2rad(2.))**2,100.)+.1*np.minimum((rate_error/np.deg2rad(5.))**2,100.)
          +.01*step['actual_cost']/(DT*LIMIT**2)+.001*(step['delta_torque']/(SLEW*DT))**2)
    return np.where(step['failed'],201.,cost)


def evaluate(records,controller,keep_traces=False):
    ref=Reference(records);plant=Plant(len(records));plant.reset(ref);controller.reset(len(records))
    n=len(records)
    error2=np.zeros(n);rate2=np.zeros(n);cost=np.zeros(n);objective=np.zeros(n);max_error=np.zeros(n)
    max_torque=np.abs(plant.torque);max_slew=np.zeros(n)
    trace=None
    if keep_traces:
        trace={name:np.empty((N_STEPS+1,n)) for name in ('theta','omega','torque','error','rate_error')}
        trace.update({name:np.empty((N_STEPS,n)) for name in ('command','actual_cost','objective_interval')})
        trace['failed']=np.empty((N_STEPS+1,n),bool)
    for k in range(N_STEPS+1):
        target,rate,_=ref.at(plant.elapsed)
        error=target-plant.theta;rate_error=rate-plant.omega
        if trace is not None:
            for name,value in [('theta',plant.theta),('omega',plant.omega),('torque',plant.torque),('error',error),('rate_error',rate_error),('failed',plant.failed)]:
                trace[name][k]=value
        max_error=np.maximum(max_error,np.abs(error))
        if k==N_STEPS:break
        command=controller.act(plant,ref)
        if np.any(~np.isfinite(command)) or np.any(np.abs(command)>LIMIT+1e-9):raise ValueError('invalid command')
        step=plant.step(command)
        next_target,next_rate,_=ref.at(plant.elapsed)
        next_error=next_target-plant.theta;next_rate_error=next_rate-plant.omega
        interval=DT*normalized_interval_cost(next_error,next_rate_error,step)
        error2+=DT*(error**2+next_error**2)/2
        rate2+=DT*(rate_error**2+next_rate_error**2)/2
        cost+=step['actual_cost'];objective+=interval
        max_torque=np.maximum(max_torque,np.abs(plant.torque));max_slew=np.maximum(max_slew,np.abs(step['delta_torque'])/DT)
        if trace is not None:
            trace['command'][k]=command;trace['actual_cost'][k]=step['actual_cost'];trace['objective_interval'][k]=interval
    rows=[dict(scene,failed=bool(plant.failed[i]),complete=bool(plant.count[i]==N_STEPS),rms_error_deg=float(np.rad2deg(np.sqrt(error2[i]/DURATION))),
               rms_rate_error_deg_s=float(np.rad2deg(np.sqrt(rate2[i]/DURATION))),max_error_deg=float(np.rad2deg(max_error[i])),
               actual_cost=float(cost[i]),objective=float(objective[i]),max_torque=float(max_torque[i]),max_slew=float(max_slew[i])) for i,scene in enumerate(records)]
    return dict(rows=rows,traces=trace,mpc_max_solver_residual=getattr(controller,'max_solver_residual',None))


def summary(rows):
    return dict(count=len(rows),physical_failures=sum(r['failed'] for r in rows),
                mean_rms_error_deg=float(np.mean([r['rms_error_deg'] for r in rows])),mean_objective=float(np.mean([r['objective'] for r in rows])),
                mean_actual_cost=float(np.mean([r['actual_cost'] for r in rows])),max_error_deg=max(r['max_error_deg'] for r in rows),
                maximum_torque=max(r['max_torque'] for r in rows),maximum_slew=max(r['max_slew'] for r in rows))


def rank(stats):
    return (-stats['physical_failures'],-stats['mean_objective'])


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def tune():
    if (STUDY/'04_独立测试'/'freeze.json').exists():raise RuntimeError('test freeze already exists')
    out=STUDY/'02_传统控制';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(202610181804);dev=scenes('development');val=scenes('validation')
    selected={};all_search=[]
    for family in ('PID','PID_feedforward_preview'):
        candidates=[]
        for _ in range(500):
            wn=np.exp(rng.uniform(np.log(.3),np.log(5.)))
            zeta=np.exp(rng.uniform(np.log(.5),np.log(2.)))
            pars=dict(kp=float(wn**2),kd=float(2*zeta*wn),ki=float(rng.choice([0.,0.,.01,.03,.1,.3])))
            if family!='PID':pars.update(feedforward=float(rng.choice([.5,.8,1.,1.2])),preview_s=float(rng.choice([0.,.1,.2,.3,.4,.5,.6])))
            result=evaluate(dev,PID(pars));stats=summary(result['rows'])
            candidates.append(dict(parameters=pars,development=stats));all_search.append(dict(family=family,**candidates[-1]))
        candidates.sort(key=lambda x:rank(x['development']),reverse=True)
        finalists=[]
        for candidate in candidates[:40]:
            result=evaluate(val,PID(candidate['parameters']));stats=summary(result['rows'])
            finalists.append(dict(candidate,validation=stats))
        chosen=max(finalists,key=lambda x:rank(x['validation']))
        selected[family]=chosen
        write_json(out/f'{family}.json',dict(chosen,source_sha256=source_hash(),test_responses_used=False,finalists=finalists))
        print(json.dumps(dict(event='traditional_selection',family=family,**chosen),ensure_ascii=False),flush=True)
    write_json(out/'search.json',dict(source_sha256=source_hash(),attempts=all_search))
    return selected


class PolicyInterface:
    def __init__(self,parameters):
        self.pid=PID(parameters)

    def reset(self,n):
        self.pid.reset(n);self.previous_command=np.zeros(n)

    def features(self,plant,ref):
        target,rate,acc=ref.at(plant.elapsed)
        future,future_rate,_=ref.at(plant.elapsed[:,None]+PREVIEW)
        return np.column_stack([(target-plant.theta)/np.deg2rad(10.),(rate-plant.omega)/np.deg2rad(15.),plant.torque/LIMIT,
                                INERTIA*acc/LIMIT,(future-plant.theta[:,None])/np.deg2rad(10.),
                                (future_rate-plant.omega[:,None])/np.deg2rad(15.),self.pid.propose(plant,ref)/LIMIT,
                                self.pid.integral/2,self.previous_command/LIMIT,plant.elapsed/DURATION]).astype(np.float32)

    def command(self,plant,ref,action):
        raw=self.pid.propose(plant,ref)+RESIDUAL*np.clip(np.asarray(action,float).reshape(plant.n),-1.,1.)
        command=np.clip(raw,-LIMIT,LIMIT)
        self.pid.commit(plant,ref,command,raw);self.previous_command=command.copy()
        return command


class LearnedController:
    def __init__(self,model,parameters):
        self.model=model;self.interface=PolicyInterface(parameters)

    def reset(self,n):self.interface.reset(n)

    def act(self,plant,ref):
        actions,_=self.model.predict(self.interface.features(plant,ref),deterministic=True)
        return self.interface.command(plant,ref,actions)


def make_env(seed,parameters,directory):
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import VecEnv
    class TrackingEnv(VecEnv):
        render_mode=None
        def __init__(self):
            self.run_seed=int(seed);self.rngs=[np.random.default_rng(np.random.SeedSequence([seed,rank])) for rank in range(N_ENVS)]
            self.records=draw_scenes(np.random.default_rng(0),N_ENVS,'placeholder')
            self.ref=Reference(self.records);self.plant=Plant(N_ENVS);self.interface=PolicyInterface(parameters)
            self.total_steps=0;self.actions=None;self.returns=np.zeros(N_ENVS);self.episode_ids=np.zeros(N_ENVS,int)
            self.log=(directory/'training_episodes.jsonl').open('x',encoding='utf-8')
            super().__init__(N_ENVS,Box(-np.inf,np.inf,(FEATURE_DIM,),dtype=np.float32),Box(-1.,1.,(1,),dtype=np.float32))
        def seed(self,value=None):
            if value is not None and value!=self.run_seed:raise ValueError('fixed seed')
            return [int(np.random.SeedSequence([self.run_seed,i]).generate_state(1)[0]) for i in range(N_ENVS)]
        def reset_rows(self,mask,first=False):
            for rank in np.flatnonzero(mask):
                scene=draw_scenes(self.rngs[rank],1,'training')[0]
                scene['scenario_id']=f'train_{self.run_seed}_{rank}_{self.episode_ids[rank]}'
                self.episode_ids[rank]+=1;self.records[rank]=scene
            self.ref=Reference(self.records);self.plant.reset(self.ref,mask)
            if first:self.interface.reset(N_ENVS)
            else:self.interface.pid.integral[mask]=0.;self.interface.previous_command[mask]=0.
            self.returns[mask]=0.
        def reset(self):
            self.reset_rows(np.ones(N_ENVS,bool),True);self._reset_seeds();self._reset_options()
            return self.interface.features(self.plant,self.ref)
        def step_async(self,actions):self.actions=np.asarray(actions).copy()
        def step_wait(self):
            command=self.interface.command(self.plant,self.ref,self.actions);self.actions=None
            step=self.plant.step(command);self.total_steps+=N_ENVS
            target,rate,_=self.ref.at(self.plant.elapsed)
            reward=(-DT*normalized_interval_cost(target-self.plant.theta,rate-self.plant.omega,step)).astype(np.float32)
            self.returns+=reward;features=self.interface.features(self.plant,self.ref)
            done=step['done'];infos=[]
            for rank in range(N_ENVS):
                info={'TimeLimit.truncated':False}
                if done[rank]:
                    episode=dict(r=float(self.returns[rank]),l=N_STEPS,failed=bool(self.plant.failed[rank]))
                    info.update(episode=episode,terminal_observation=features[rank].copy())
                    self.log.write(json.dumps(dict(scene=self.records[rank],**episode),allow_nan=False)+'\n')
                infos.append(info)
            if np.any(done):
                self.reset_rows(done);features=self.interface.features(self.plant,self.ref);self.log.flush()
            return features,reward,done,infos
        def close(self):self.log.close()
        def get_attr(self,name,indices=None):return [getattr(self,name) for _ in self._get_indices(indices)]
        def set_attr(self,*args,**kwargs):raise AttributeError('fixed study')
        def env_method(self,*args,**kwargs):raise AttributeError('fixed study')
        def env_is_wrapped(self,cls,indices=None):return [False for _ in self._get_indices(indices)]
    return TrackingEnv()


def build_model(algorithm,env,seed,steps):
    import torch
    import stable_baselines3 as sb3
    from stable_baselines3.common.noise import NormalActionNoise
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    kwargs=dict(seed=int(seed),device='cpu',verbose=0,learning_rate=3e-4,gamma=GAMMA,policy_kwargs=dict(net_arch=[128,128]))
    if algorithm=='TD3':
        kwargs.update(batch_size=256,buffer_size=min(200000,steps),learning_starts=2048,train_freq=(1,'step'),gradient_steps=1,
                      action_noise=NormalActionNoise(np.zeros(1),.08*np.ones(1)),target_policy_noise=.1,target_noise_clip=.2,policy_delay=2)
    elif algorithm=='PPO':
        kwargs.update(n_steps=128,batch_size=256,n_epochs=10)
        kwargs['policy_kwargs']['log_std_init']=-2.5
    else:raise ValueError(algorithm)
    model=getattr(sb3,algorithm)('MlpPolicy',env,**kwargs)
    with torch.no_grad():
        if algorithm=='TD3':
            final=[m for m in model.actor.mu.modules() if isinstance(m,torch.nn.Linear)][-1]
            final.weight.zero_();final.bias.zero_();model.actor_target.load_state_dict(model.actor.state_dict())
        else:model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    return model


def train(algorithm,seed,stage):
    if (STUDY/'04_独立测试'/'freeze.json').exists():raise RuntimeError('test freeze prevents fitting')
    if algorithm not in ALGORITHMS or stage not in ('development','formal'):raise ValueError('unknown run')
    if stage=='development' and seed!=DEVELOPMENT_SEED:raise ValueError('seed')
    if stage=='formal':
        decision=read_json(STUDY/'03_训练记录'/'decision.json')
        if not decision['formal_eligible'] or algorithm!=decision['selected']['algorithm'] or seed not in FORMAL_SEEDS:raise ValueError('formal selection')
    steps=DEVELOPMENT_STEPS if stage=='development' else FORMAL_STEPS
    directory=STUDY/'03_训练记录'/stage/algorithm/f'seed_{seed}'
    existing=directory/'training_metadata.json'
    if existing.exists():
        old=read_json(existing)
        if old['status']=='complete' and old['source_sha256']==source_hash():return old
        raise RuntimeError('existing attempt retained; no overwrite')
    directory.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,directory/'source_snapshot.py')
    parameters=read_json(STUDY/'02_传统控制'/'PID_feedforward_preview.json')['parameters']
    metadata=dict(status='running',algorithm=algorithm,seed=seed,stage=stage,requested_steps=steps,actual_steps=0,
                  source_sha256=source_hash(),base_parameters=parameters,checkpoints=[],best=None,started_utc=datetime.now(timezone.utc).isoformat(),
                  replay_buffer_saved=False,test_responses_used=False,trained_checkpoint_improved_initial=False)
    write_json(existing,metadata)
    from stable_baselines3.common.logger import configure
    import torch,stable_baselines3,gymnasium
    metadata['versions']=dict(torch=torch.__version__,stable_baselines3=stable_baselines3.__version__,gymnasium=gymnasium.__version__,numpy=np.__version__)
    env=make_env(seed,parameters,directory);model=build_model(algorithm,env,seed,steps)
    model.set_logger(configure(str(directory/'optimizer_logs'),['csv']))
    start=time.monotonic();val=scenes('validation')
    try:
        for checkpoint in range(0,steps+1,CHECKPOINT):
            if checkpoint:
                model.learn(CHECKPOINT,reset_num_timesteps=checkpoint==CHECKPOINT,log_interval=1)
                if model.num_timesteps!=checkpoint or env.total_steps!=checkpoint:raise RuntimeError('step count differs')
            path=directory/'checkpoints'/f'steps_{checkpoint:06d}';path.mkdir(parents=True,exist_ok=False)
            model.save(path/'model.zip')
            result=evaluate(val,LearnedController(model,parameters));stats=summary(result['rows'])
            write_csv(path/'validation.csv',result['rows'])
            item=dict(steps=checkpoint,model_path=relative(path/'model.zip'),model_sha256=sha256(path/'model.zip'),validation=stats,optimizer_updates=int(model._n_updates))
            metadata['checkpoints'].append(item)
            if checkpoint==0:metadata['initial']=item
            if metadata['best'] is None or (*rank(stats),-checkpoint)>(*rank(metadata['best']['validation']),-metadata['best']['steps']):
                shutil.copy2(path/'model.zip',directory/'best.zip');metadata['best']=dict(item,checkpoint_model_path=item['model_path'],model_path=relative(directory/'best.zip'))
            metadata['actual_steps']=int(model.num_timesteps);metadata['optimizer_updates']=int(model._n_updates)
            metadata['trained_checkpoint_improved_initial']=metadata['best']['steps']>0
            metadata['elapsed_seconds']=time.monotonic()-start
            if source_hash()!=metadata['source_sha256']:raise RuntimeError('production source changed')
            write_json(existing,metadata)
            print(json.dumps(dict(event='tracking_checkpoint',algorithm=algorithm,seed=seed,stage=stage,steps=checkpoint,validation=stats,elapsed_seconds=metadata['elapsed_seconds']),ensure_ascii=False),flush=True)
        model.save(directory/'final.zip');metadata['status']='complete'
    except BaseException as error:
        metadata['status']='incomplete';metadata['error']=repr(error);raise
    finally:
        metadata['actual_steps']=int(model.num_timesteps);metadata['source_sha256_after']=source_hash();metadata['finished_utc']=datetime.now(timezone.utc).isoformat()
        metadata['elapsed_seconds']=time.monotonic()-start;write_json(existing,metadata);env.close();model.logger.close()
    return metadata


def decide():
    runs=[]
    for algorithm in ALGORITHMS:
        path=STUDY/'03_训练记录'/'development'/algorithm/f'seed_{DEVELOPMENT_SEED}'/'training_metadata.json'
        data=read_json(path)
        if data['status']!='complete' or data['source_sha256']!=source_hash():raise RuntimeError('incomplete development')
        runs.append(dict(algorithm=algorithm,metadata_path=relative(path),best=data['best']))
    eligible=[r for r in runs if r['best']['steps']>0 and r['best']['validation']['physical_failures']==0]
    selected=max(eligible,key=lambda x:rank(x['best']['validation'])) if eligible else None
    decision=dict(formal_eligible=selected is not None,selected=selected,runs=runs,test_responses_used=False,source_sha256=source_hash())
    write_json(STUDY/'03_训练记录'/'decision.json',decision)
    return decision


def freeze():
    from stable_baselines3 import PPO,TD3
    del PPO,TD3
    controllers=[]
    for family in ('PID','PID_feedforward_preview'):
        path=STUDY/'02_传统控制'/f'{family}.json';data=read_json(path)
        controllers.append(dict(name=family,family=family,parameters=data['parameters'],metadata_path=relative(path),metadata_sha256=sha256(path)))
    controllers.append(dict(name='MPC',family='MPC'))
    decision=read_json(STUDY/'03_训练记录'/'decision.json')
    if decision['formal_eligible']:
        algorithm=decision['selected']['algorithm']
        for seed in FORMAL_SEEDS:
            path=STUDY/'03_训练记录'/'formal'/algorithm/f'seed_{seed}'/'training_metadata.json';data=read_json(path)
            if data['status']!='complete' or data['actual_steps']!=FORMAL_STEPS or data['source_sha256']!=source_hash():raise RuntimeError('formal incomplete')
            best=data['best']
            controllers.append(dict(name=f'{algorithm}_seed_{seed}',family='learned',algorithm=algorithm,seed=seed,
                                    model_path=best['model_path'],model_sha256=best['model_sha256'],selected_steps=best['steps'],
                                    parameters=data['base_parameters'],metadata_path=relative(path),metadata_sha256=sha256(path)))
    else:
        for run in decision['runs']:
            data=read_json(STUDY/run['metadata_path']);best=data['best']
            controllers.append(dict(name=f'{run["algorithm"]}_development_only',family='learned',algorithm=run['algorithm'],seed=DEVELOPMENT_SEED,
                                    model_path=best['model_path'],model_sha256=best['model_sha256'],selected_steps=best['steps'],parameters=data['base_parameters'],diagnostic=True))
    data=dict(controllers=controllers,source_sha256=source_hash(),declaration_sha256=sha256(STUDY/'01_实验方案'/'预声明.json'),
              test_scenarios_sha256=sha256(STUDY/'01_实验方案'/'test.json'),created_utc=datetime.now(timezone.utc).isoformat())
    path=STUDY/'04_独立测试'/'freeze.json'
    if path.exists():raise RuntimeError('freeze already exists')
    write_json(path,data);return data


def controller_from_entry(entry):
    if entry['family']=='MPC':return MPC()
    if entry['family'] in ('PID','PID_feedforward_preview'):return PID(entry['parameters'])
    import stable_baselines3 as sb3
    if sha256(STUDY/entry['model_path'])!=entry['model_sha256']:raise RuntimeError('model changed')
    return LearnedController(getattr(sb3,entry['algorithm']).load(STUDY/entry['model_path'],device='cpu'),entry['parameters'])


def test():
    import torch
    torch.set_num_threads(1)
    frozen=read_json(STUDY/'04_独立测试'/'freeze.json')
    if frozen['source_sha256']!=source_hash():raise RuntimeError('source changed')
    records=scenes('test');outputs=[]
    for entry in frozen['controllers']:
        path=STUDY/'04_独立测试'/entry['name'];path.mkdir(parents=True,exist_ok=False)
        result=evaluate(records,controller_from_entry(entry),True)
        np.savez_compressed(path/'traces.npz',**result['traces'])
        write_csv(path/'episode_metrics.csv',result['rows'])
        write_json(path/'results.json',dict(rows=result['rows'],summary=summary(result['rows']),scenes=records,mpc_max_solver_residual=result['mpc_max_solver_residual']))
        outputs.append(dict(name=entry['name'],results_path=relative(path/'results.json'),results_sha256=sha256(path/'results.json'),
                            trace_path=relative(path/'traces.npz'),trace_sha256=sha256(path/'traces.npz')))
        print(json.dumps(dict(event='tracking_test',name=entry['name'],summary=summary(result['rows'])),ensure_ascii=False),flush=True)
    write_json(STUDY/'04_独立测试'/'manifest.json',dict(freeze_sha256=sha256(STUDY/'04_独立测试'/'freeze.json'),outputs=outputs))


def verify():
    """Replay all saved states/actions independently; integrate ODE spot cases."""
    from scipy.integrate import solve_ivp
    import torch
    torch.set_num_threads(1)
    frozen=read_json(STUDY/'04_独立测试'/'freeze.json');manifest=read_json(STUDY/'04_独立测试'/'manifest.json')
    checks=[];max_state_error=0.;max_ode_error=0.;episodes=0
    if manifest['freeze_sha256']!=sha256(STUDY/'04_独立测试'/'freeze.json'):raise RuntimeError('freeze changed')
    if frozen['source_sha256']!=source_hash():raise RuntimeError('source changed')
    if sha256(STUDY/'01_实验方案'/'test.json')!=frozen['test_scenarios_sha256']:raise RuntimeError('test scenes changed')
    entries={e['name']:e for e in frozen['controllers']}
    if set(e['name'] for e in manifest['outputs'])!=set(entries):raise RuntimeError('controller coverage differs')
    for output in manifest['outputs']:
        name=output['name'];entry=entries[name]
        if sha256(STUDY/output['results_path'])!=output['results_sha256'] or sha256(STUDY/output['trace_path'])!=output['trace_sha256']:raise RuntimeError('response changed')
        data=read_json(STUDY/output['results_path']);records=scenes('test')
        if data['scenes']!=records or len(data['rows'])!=len(records):raise RuntimeError('scene coverage differs')
        trace=dict(np.load(STUDY/output['trace_path']))
        n=len(records);ref=Reference(records);theta=trace['theta'][0].copy();omega=trace['omega'][0].copy();torque=trace['torque'][0].copy()
        failed=np.zeros(n,bool);area=np.zeros(n);velarea=np.zeros(n);cost=np.zeros(n);objective=np.zeros(n);maxerr=np.zeros(n);maxslew=np.zeros(n);maxtorque=np.abs(torque)
        controller=controller_from_entry(entry);controller.reset(n)
        dummy=Plant(n);dummy.reset(ref)
        for k in range(N_STEPS):
            dummy.theta=theta.copy();dummy.omega=omega.copy();dummy.torque=torque.copy();dummy.count[:]=k;dummy.failed=failed.copy()
            reproduced=controller.act(dummy,ref)
            if not np.allclose(reproduced,trace['command'][k],atol=2e-7,rtol=1e-6):raise RuntimeError('deployed command replay differs: '+name)
            command=np.clip(trace['command'][k],-LIMIT,LIMIT)
            newtau=np.minimum(np.maximum(command,torque-SLEW*DT),torque+SLEW*DT)
            active=~failed
            # Independently express the ramp using acceleration and jerk.
            acceleration=torque/INERTIA;jerk=(newtau-torque)/(DT*INERTIA)
            nexttheta=theta+DT*omega+DT**2*acceleration/2+DT**3*jerk/6
            nextomega=omega+DT*acceleration+DT**2*jerk/2
            nexttheta=np.where(active,nexttheta,theta);nextomega=np.where(active,nextomega,omega);newtau=np.where(active,newtau,torque)
            interval=np.where(active,DT*(torque**2+torque*(newtau-torque)+(newtau-torque)**2/3),0.)
            if k in (0,73,149,299):
                for i in range(min(4,n)):
                    if active[i]:
                        sol=solve_ivp(lambda t,y:[y[1],(torque[i]+(newtau[i]-torque[i])*t/DT)/INERTIA],(0,DT),[theta[i],omega[i]],rtol=1e-11,atol=1e-12)
                        max_ode_error=max(max_ode_error,float(np.max(np.abs(sol.y[:,-1]-[nexttheta[i],nextomega[i]]))))
            err=max(float(np.max(np.abs(nexttheta-trace['theta'][k+1]))),float(np.max(np.abs(nextomega-trace['omega'][k+1]))),float(np.max(np.abs(newtau-trace['torque'][k+1]))))
            max_state_error=max(max_state_error,err)
            if err>2e-9 or not np.allclose(interval,trace['actual_cost'][k],atol=1e-11):raise RuntimeError('physical replay differs: '+name)
            failed|=(np.abs(nexttheta)>FAIL_ANGLE)|(np.abs(nextomega)>FAIL_RATE)
            if not np.array_equal(failed,trace['failed'][k+1]):raise RuntimeError('failure flags differ')
            target0,rate0,_=ref.at(k*DT);target1,rate1,_=ref.at((k+1)*DT)
            e0,e1=target0-theta,target1-nexttheta;w0,w1=rate0-omega,rate1-nextomega
            area+=DT*(e0*e0+e1*e1)/2;velarea+=DT*(w0*w0+w1*w1)/2;cost+=interval
            maxerr=np.maximum(maxerr,np.maximum(np.abs(e0),np.abs(e1)));maxslew=np.maximum(maxslew,np.abs(newtau-torque)/DT);maxtorque=np.maximum(maxtorque,np.abs(newtau))
            ordinary=np.minimum((e1/np.deg2rad(2.))**2,100.)+.1*np.minimum((w1/np.deg2rad(5.))**2,100.)+.01*interval/(DT*LIMIT**2)+.001*((newtau-torque)/(DT*SLEW))**2
            objective+=DT*np.where(failed,201.,ordinary)
            theta,omega,torque=nexttheta,nextomega,newtau
        independent=dict(rms_error_deg=np.rad2deg(np.sqrt(area/DURATION)),rms_rate_error_deg_s=np.rad2deg(np.sqrt(velarea/DURATION)),
                         actual_cost=cost,objective=objective,max_error_deg=np.rad2deg(maxerr),max_slew=maxslew,max_torque=maxtorque)
        for i,row in enumerate(data['rows']):
            if row['scenario_id']!=records[i]['scenario_id'] or row['failed']!=bool(failed[i]) or not row['complete']:raise RuntimeError('row coverage or acceptance differs')
            for field,values in independent.items():
                if not np.isclose(row[field],values[i],atol=2e-8,rtol=2e-7):raise RuntimeError('metric differs '+name+' '+field)
        if np.max(maxtorque)>LIMIT+1e-10 or np.max(maxslew)>SLEW+1e-10:raise RuntimeError('actuator constraint violation')
        episodes+=n;checks.append(dict(name=name,episodes=n,all_interval_physics_and_command_replay=True,metrics_recomputed=True))
    if max_ode_error>2e-9:raise RuntimeError('ODE spot check differs')
    result=dict(passed=True,episodes=episodes,physical_intervals=episodes*N_STEPS,max_all_interval_state_error=max_state_error,
                max_independent_solve_ivp_error=max_ode_error,checks=checks,limitations='reference function and controller code reused for action replay; physics and metric formulas independently reconstructed; no hardware validation')
    write_json(STUDY/'05_独立核验'/'verification.json',result);return result


def paired(reference,candidate):
    first={r['scenario_id']:r for r in reference};second={r['scenario_id']:r for r in candidate}
    if first.keys()!=second.keys():raise RuntimeError('paired scenarios differ')
    pairs=np.array([[first[k]['rms_error_deg'],second[k]['rms_error_deg']] for k in first])
    rng=np.random.default_rng(202610181809);boot=np.empty((20000,2))
    for offset in range(0,20000,1000):
        indices=rng.integers(0,len(pairs),(1000,len(pairs)));boot[offset:offset+1000]=pairs[indices].mean(axis=1)
    alpha=.05/3
    return dict(reference_mean=float(pairs[:,0].mean()),candidate_mean=float(pairs[:,1].mean()),
                absolute_reduction=float(np.mean(pairs[:,0]-pairs[:,1])),relative_reduction=float(1-pairs[:,1].mean()/pairs[:,0].mean()),
                difference_interval=np.quantile(boot[:,0]-boot[:,1],[alpha/2,1-alpha/2]).tolist(),confidence=1-alpha,
                candidate_failures=sum(r['failed'] for r in candidate),reference_failures=sum(r['failed'] for r in reference))


def report():
    audit=read_json(STUDY/'05_独立核验'/'verification.json')
    if not audit['passed']:raise RuntimeError('audit failed')
    frozen=read_json(STUDY/'04_独立测试'/'freeze.json');responses={e['name']:read_json(STUDY/'04_独立测试'/e['name']/'results.json') for e in frozen['controllers']}
    intervals={};formal=[]
    for entry in frozen['controllers']:
        if entry['family']=='learned':
            intervals[entry['name']]={name:paired(responses[name]['rows'],responses[entry['name']]['rows']) for name in ('PID','PID_feedforward_preview','MPC')}
            if not entry.get('diagnostic'):formal.append(entry)
    advantage=bool(len(formal)==3 and all(e['selected_steps']>0 and intervals[e['name']]['PID']['difference_interval'][0]>0 and intervals[e['name']]['PID']['candidate_failures']<=intervals[e['name']]['PID']['reference_failures'] for e in formal))
    output=STUDY/'06_研究报告';output.mkdir(parents=True,exist_ok=True)
    write_json(output/'paired_intervals.json',dict(advantage_vs_PID=advantage,comparisons=intervals))
    lines=['# 执行器约束下的连续姿态跟踪结果','',
           '本任务要求卫星持续跟随随时间变化的扫描指令。所有方法使用相同的物理模型、测量、未来3秒参考指令、力矩幅值和力矩变化速率约束。AI输出连续力矩补偿，基本反馈由同一PID承担。','',
           '**'+('三个新随机种子均获得相对充分整定PID的可重复跟踪误差改善。' if advantage else '本次预算未证明三个新随机种子均优于充分整定PID。')+'**','',
           '| 方法 | 跟踪角度RMS误差（°） | 平均力矩平方积分（N²·m²·s） | 物理失败 |','|---|---:|---:|---:|']
    for name,data in responses.items():
        s=data['summary'];lines.append(f'| {name} | {s["mean_rms_error_deg"]:.4f} | {s["mean_actual_cost"]:.5f} | {s["physical_failures"]}/{s["count"]} |')
    lines+=['','每个方法测试128条独立轨迹，每条30秒。三个正式种子使用同一测试集配对比较；误差差值置信区间采用98.333%置信水平。正的差值表示AI误差较小。','',
            '| AI模型 | 相对PID的平均误差降低 | 配对误差差值区间（°） |','|---|---:|---|']
    for name,comp in intervals.items():
        x=comp['PID'];lines.append(f'| {name} | {x["relative_reduction"]*100:.2f}% | [{x["difference_interval"][0]:.5f}, {x["difference_interval"][1]:.5f}] |')
    lines+=['','传统PID加模型前馈、预览和模型预测控制的结果同样完整保留。相对PID的改进不能推导为优于所有传统控制。实际力矩是可测量反馈，所有方法均可使用；AI测试时不更新参数。力矩平方积分不是实测电能。','',
            f'全量回放核验通过：{audit["episodes"]}回合、{audit["physical_intervals"]}个物理区间；另以独立solve_ivp核验代表区间。核验的局限见05_独立核验/verification.json。']
    (output/'研究报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import html
    body=html.escape('\n'.join(lines))
    (output/'研究报告.html').write_text('<!doctype html><meta charset="utf-8"><title>执行器约束连续跟踪研究</title><style>body{max-width:1100px;margin:40px auto;font:17px/1.8 sans-serif;padding:20px}pre{white-space:pre-wrap}</style><pre>'+body+'</pre>',encoding='utf-8')
    return dict(advantage_vs_PID=advantage,summary={name:data['summary'] for name,data in responses.items()},audit=audit)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','tune','train','decide','freeze','test','verify','report','all']);parser.add_argument('--algorithm',choices=ALGORITHMS,default='TD3');parser.add_argument('--seed',type=int,default=DEVELOPMENT_SEED);parser.add_argument('--stage',choices=['development','formal'],default='development');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='tune':tune()
    elif args.command=='train':train(args.algorithm,args.seed,args.stage)
    elif args.command=='decide':print(json.dumps(decide(),ensure_ascii=False),flush=True)
    elif args.command=='freeze':freeze()
    elif args.command=='test':test()
    elif args.command=='verify':print(json.dumps(verify(),ensure_ascii=False),flush=True)
    elif args.command=='report':print(json.dumps(report(),ensure_ascii=False),flush=True)
    elif args.command=='all':
        prepare();tune()
        for algorithm in ALGORITHMS:train(algorithm,DEVELOPMENT_SEED,'development')
        decision=decide()
        if decision['formal_eligible']:
            for seed in FORMAL_SEEDS:train(decision['selected']['algorithm'],seed,'formal')
        freeze();test();verify();report()


if __name__=='__main__':main()
