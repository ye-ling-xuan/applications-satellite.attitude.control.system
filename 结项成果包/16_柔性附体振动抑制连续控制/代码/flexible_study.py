"""Predeclared continuous flexible-appendage experiment; SI units internally."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT, HORIZON, DEADLINE, LIMIT = .05, 22., 8., .5
JB, JF = .7, .3
RHO = JF/(JB+JF)
STEPS = round(HORIZON/DT)
FREQUENCIES = (.4, .5, .6, .7, .8)
COUNTS = {'development': 48, 'validation': 64, 'test': 128}
SPLITS = {'development': 202610181601, 'validation': 202610181602, 'test': 202610181603}
TOL = np.deg2rad([1., .5, .1, .5])
FORMAL_SEEDS = (111, 222, 333)
DEV_STEPS, FORMAL_STEPS, VALIDATE_EVERY = 262144, 524288, 32768


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def sources():
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT/'代码').glob('*.py'))}


def design():
    config = dict(dt_s=DT, horizon_s=HORIZON, deadline_s=DEADLINE, command_limit_Nm=LIMIT,
                  hub_inertia=JB, appendage_inertia=JF, frequency_training_Hz=[.4,.8],
                  damping_training=[.01,.04], evaluation_frequencies_Hz=FREQUENCIES,
                  tolerance_deg_or_deg_s=[1.,.5,.1,.5], fail_relative_angle_deg=10.,
                  fail_hub_angle_deg=170., fail_hub_rate_deg_s=300.,
                  counts=COUNTS, split_seeds=SPLITS, formal_seeds=FORMAL_SEEDS,
                  development_algorithms=['TD3','SAC'], development_steps=DEV_STEPS,
                  formal_steps=FORMAL_STEPS, validation_every=VALIDATE_EVERY,
                  bonferroni_comparisons=18, bootstrap_repeats=20000,
                  primary_split='test', true_parameters_in_policy=False,
                  visible_states=['hub_angle','hub_rate','relative_angle','relative_rate',
                                  'own_integral','previous_sent_command','elapsed_time'])
    # Tuple serialization and comparison are deliberate and stable.
    config = json.loads(json.dumps(config))
    path = ROOT/'01_实验方案'/'执行配置.json'
    if path.exists() and read(path) != config:
        raise RuntimeError('predeclared protocol differs')
    write(path, config)
    for split, seed in SPLITS.items():
        rng = np.random.default_rng(seed)
        values = [dict(id=i, theta=float(theta), omega=float(omega), damping=float(zeta))
                  for i, (theta, omega, zeta) in enumerate(zip(
                      np.deg2rad(rng.uniform(-30.,30.,COUNTS[split])),
                      np.deg2rad(rng.uniform(-2.,2.,COUNTS[split])),
                      rng.uniform(.01,.04,COUNTS[split])))]
        path = ROOT/'01_实验方案'/f'{split}.json'
        if path.exists() and read(path) != values:
            raise RuntimeError('scenario split differs')
        write(path, values)
    return config


def scenes(split):
    return [dict(**state, frequency=f) for f in FREQUENCIES
            for state in read(ROOT/'01_实验方案'/f'{split}.json')]


class Plant:
    """Exact zero-order-hold propagation via rigid/modal coordinate separation."""
    def __init__(self, n):
        self.n = n
        self.state = np.zeros((n,4))
        self.frequency = np.full(n,.6)
        self.damping = np.full(n,.025)
        self.failed = np.zeros(n,dtype=bool)
        self.count = np.zeros(n,dtype=np.int64)

    def reset(self, theta, omega, frequency, damping, mask=None):
        mask = np.ones(self.n,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
        count = np.count_nonzero(mask)
        values = [np.broadcast_to(np.asarray(x,dtype=float),(count,)).copy()
                  for x in (theta,omega,frequency,damping)]
        if not all(np.all(np.isfinite(x)) for x in values):
            raise ValueError('finite state and modal parameters required')
        if np.any(values[2] <= 0.) or np.any((values[3] <= 0.)|(values[3] >= 1.)):
            raise ValueError('positive frequency and damping in (0,1) required')
        self.state[mask] = np.column_stack((values[0],values[1],np.zeros((count,2))))
        self.frequency[mask],self.damping[mask] = values[2],values[3]
        self.failed[mask],self.count[mask] = False,0

    def step(self, command):
        command = np.asarray(command,dtype=float)
        if command.shape != (self.n,) or not np.all(np.isfinite(command)):
            raise ValueError('finite n-vector command required')
        active = ~self.failed & (self.count < STEPS)
        sent = np.where(active,np.clip(command,-LIMIT,LIMIT),0.)
        x = self.state[active]
        u = sent[active]
        if x.shape[0]:
            wn = 2*np.pi*self.frequency[active]
            zeta = self.damping[active]
            wd = wn*np.sqrt(1-zeta*zeta)
            decay = np.exp(-zeta*wn*DT)
            cosine,sine = np.cos(wd*DT),np.sin(wd*DT)
            equilibrium = -u/(JB*wn*wn)
            relative = x[:,2]-equilibrium
            q = equilibrium+decay*(relative*cosine+(x[:,3]+zeta*wn*relative)/wd*sine)
            v = decay*(x[:,3]*cosine-(zeta*wn*x[:,3]+wn*wn*relative)/wd*sine)
            mean_angle = x[:,0]+RHO*x[:,2]
            mean_rate = x[:,1]+RHO*x[:,3]
            mean_angle += mean_rate*DT+.5*u*DT*DT/(JB+JF)
            mean_rate += u*DT/(JB+JF)
            self.state[active] = np.column_stack((mean_angle-RHO*q,mean_rate-RHO*v,q,v))
        new_failure = active & ((np.abs(self.state[:,0]) > np.deg2rad(170.))
                                |(np.abs(self.state[:,1]) > np.deg2rad(300.))
                                |(np.abs(self.state[:,2]) > np.deg2rad(10.)))
        self.failed |= new_failure
        self.count = np.minimum(self.count+1,STEPS)
        return dict(command=sent,cost=sent*sent*DT,failed=self.failed.copy(),
                    new_failure=new_failure,done=self.count>=STEPS)


def shaper_target(initial, elapsed):
    zeta,f = .025,.6
    k = math.exp(-zeta*math.pi/math.sqrt(1-zeta*zeta))
    weights = np.array([1.,2*k,k*k])/(1+k)**2
    times = np.arange(3)/(2*f*math.sqrt(1-zeta*zeta))
    # A shaped step from the measured initial angle to the target zero.
    remaining = np.sum(weights[None,:]*(elapsed[:,None] < times[None,:]-1e-12),axis=1)
    return initial*remaining


class Controller:
    def __init__(self, settings, n):
        self.settings = settings
        self.n = n
        self.integral = np.zeros(n)
        self.previous = np.zeros(n)
        self.initial = np.zeros(n)

    def reset(self, plant, mask=None):
        mask = np.ones(self.n,dtype=bool) if mask is None else mask
        self.integral[mask],self.previous[mask] = 0.,0.
        self.initial[mask] = plant.state[mask,0]

    def observation(self,plant):
        x = plant.state
        return np.column_stack((-x[:,0]/np.deg2rad(30.),x[:,1]/np.deg2rad(10.),
                                x[:,2]/np.deg2rad(1.),x[:,3]/np.deg2rad(5.),
                                self.integral/np.deg2rad(60.),self.previous/LIMIT,
                                plant.count/STEPS)).astype(np.float32)

    def command(self, plant, action=None):
        s = self.settings
        x = plant.state
        if s['method'] == 'LQR':
            raw = -np.sum(np.asarray(s['gain'])*x,axis=1)
        else:
            reference = shaper_target(self.initial,plant.count*DT) if s['method']=='ZVD_PID' else 0.
            error = reference-x[:,0]
            kp,kd,ki = (np.asarray(s[name]) for name in ('kp','kd','ki'))
            proposed = np.clip(self.integral+error*DT,-np.deg2rad(120.),np.deg2rad(120.))
            raw = kp*error-kd*x[:,1]+ki*proposed
            blocked = (np.abs(raw)>LIMIT)&(raw*error>0.)
            self.integral = np.where(blocked,self.integral,proposed)
            raw = kp*error-kd*x[:,1]+ki*self.integral
        if action is not None:
            raw = raw+.25*np.asarray(action).reshape(self.n)
        self.previous = np.where(plant.failed,0.,np.clip(raw,-LIMIT,LIMIT))
        return self.previous.copy()


def episode_metrics(states, commands, failed):
    inside = np.all(np.abs(states[:,round(DEADLINE/DT):,:])<=TOL,axis=(1,2))
    success = inside & ~failed
    return dict(success=success,failed=np.asarray(failed),
                cost=np.sum(commands*commands*DT,axis=1),
                rms_angle_deg=np.rad2deg(np.sqrt(np.mean(states[:,:,0]**2,axis=1))),
                peak_relative_deg=np.rad2deg(np.max(np.abs(states[:,:,2]),axis=1)),
                residual_relative_rms_deg=np.rad2deg(np.sqrt(np.mean(
                    states[:,round(DEADLINE/DT):,2]**2,axis=1))))


def simulate(data, settings, model=None, record=False):
    n = len(data)
    plant = Plant(n)
    plant.reset([s['theta'] for s in data],[s['omega'] for s in data],
                [s['frequency'] for s in data],[s['damping'] for s in data])
    control = Controller(settings,n)
    control.reset(plant)
    states = np.empty((n,STEPS+1,4))
    commands = np.empty((n,STEPS))
    states[:,0] = plant.state
    for step in range(STEPS):
        action = None if model is None else model.predict(control.observation(plant),deterministic=True)[0]
        result = plant.step(control.command(plant,action))
        states[:,step+1],commands[:,step] = plant.state,result['command']
    metrics = episode_metrics(states,commands,plant.failed)
    return (metrics,states,commands) if record else metrics


def summary(metrics):
    result = {k: float(np.mean(v)) for k,v in metrics.items()}
    result.update(count=len(metrics['success']),passes=int(np.sum(metrics['success'])),
                  physical_failures=int(np.sum(metrics['failed'])))
    # A failed rollout cannot improve selection merely by spending less effort.
    result['selection_cost'] = float(np.mean(np.where(metrics['success'],metrics['cost'],
                                                     np.maximum(metrics['cost'],LIMIT**2*HORIZON))))
    return result


def tuning():
    from scipy.linalg import solve_continuous_are
    rng = np.random.default_rng(16020)
    dev,val = scenes('development'),scenes('validation')
    selected = {}
    for method in ('PID','ZVD_PID','LQR'):
        candidates = []
        if method != 'LQR':
            # Broad range includes slow smooth loops and active/brisk responses.
            for _ in range(1200):
                frequency = np.exp(rng.uniform(np.log(.15),np.log(3.)))
                damping = rng.uniform(.35,2.2)
                candidates.append(dict(method=method,kp=float(frequency**2),
                                       kd=float(2*damping*frequency),
                                       ki=0. if rng.random()<.4 else float(np.exp(rng.uniform(-7.,-1.)))))
        else:
            wn,z = 2*np.pi*.6,.025
            inv = 1/JB+1/JF
            k,c = wn*wn/inv,2*z*wn/inv
            a = np.array([[0,1,0,0],[0,0,k/JB,c/JB],[0,0,0,1],[0,0,-wn*wn,-2*z*wn]],dtype=float)
            b = np.array([[0],[1/JB],[0],[-1/JB]])
            for qtheta in (.1,1.,10.,100.):
                for qrelative in (1.,10.,100.,1000.):
                    for r in (.1,1.,10.):
                        q = np.diag([qtheta,.3*qtheta,qrelative,.1*qrelative])
                        p = solve_continuous_are(a,b,q,np.array([[r]]))
                        gain = (b.T@p/r).reshape(4).tolist()
                        candidates.append(dict(method=method,gain=gain,q_diag=np.diag(q).tolist(),r=r))
        logs = []
        # Chunked vectorization avoids storing 1200 complete trajectory blocks.
        for first in range(0,len(candidates),24):
            chunk = candidates[first:first+24]
            settings = dict(method=method)
            if method=='LQR':
                settings['gain'] = np.repeat(np.asarray([s['gain'] for s in chunk]),len(dev),axis=0)
            else:
                settings.update({key:np.repeat([s[key] for s in chunk],len(dev)) for key in ('kp','kd','ki')})
            metrics = simulate(dev*len(chunk),settings)
            for j,candidate in enumerate(chunk):
                record = summary({key:value[j*len(dev):(j+1)*len(dev)] for key,value in metrics.items()})
                logs.append(dict(index=first+j,settings=candidate,development=record))
        ranked = sorted(logs,key=lambda s:(-s['development']['passes'],s['development']['selection_cost'],s['index']))[:32]
        for record in ranked:
            record['validation'] = summary(simulate(val,record['settings']))
        best = min(ranked,key=lambda s:(-s['validation']['passes'],s['validation']['selection_cost'],s['index']))
        selected[method] = best
        write(ROOT/'02_传统控制'/f'{method}_search.json',dict(all_development=logs,validation=ranked,selected=best))
        print(json.dumps(dict(event='baseline_selected',method=method,**best['validation']),ensure_ascii=False),flush=True)
    write(ROOT/'02_传统控制'/'selected.json',selected)
    return selected


def reward(plant, result):
    """Bounded living cost; every failed interval costs more than any live one."""
    x = plant.state
    smooth = (2*(x[:,0]/np.deg2rad(30.))**2+.05*(x[:,1]/np.deg2rad(10.))**2
              +.5*(x[:,2]/np.deg2rad(1.))**2+.05*(x[:,3]/np.deg2rad(5.))**2
              +.15*(result['command']/LIMIT)**2)
    violation = (plant.count*DT>=DEADLINE)&~np.all(np.abs(x)<=TOL,axis=1)
    living = np.minimum(smooth,30.)+10.*violation
    return -DT*np.where(plant.failed,50.,living)


def vector_environment(seed,settings):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv

    class FlexibleVecEnv(VecEnv):
        def __init__(self):
            self.rng = np.random.default_rng(seed)
            self.plant = Plant(8)
            self.control = Controller(settings,8)
            self.actions = np.zeros(8)
            super().__init__(8,gym.spaces.Box(-np.inf,np.inf,(7,),np.float32),
                             gym.spaces.Box(-1.,1.,(1,),np.float32))

        def _reset(self,mask):
            n = np.count_nonzero(mask)
            self.plant.reset(np.deg2rad(self.rng.uniform(-30.,30.,n)),
                             np.deg2rad(self.rng.uniform(-2.,2.,n)),
                             self.rng.uniform(.4,.8,n),self.rng.uniform(.01,.04,n),mask)
            self.control.reset(self.plant,mask)

        def reset(self):
            self._reset(np.ones(8,dtype=bool))
            return self.control.observation(self.plant)

        def step_async(self,actions):
            self.actions = np.asarray(actions).reshape(8)

        def step_wait(self):
            result = self.plant.step(self.control.command(self.plant,self.actions))
            rewards = reward(self.plant,result)
            done = result['done']
            obs = self.control.observation(self.plant)
            info = [dict(failed=bool(x)) for x in self.plant.failed]
            for i in np.flatnonzero(done):
                info[i]['terminal_observation'] = obs[i].copy()
                # A fixed finite-horizon episodic objective, not an unobserved
                # continuing task that bootstraps past its declared horizon.
                info[i]['TimeLimit.truncated'] = False
            if np.any(done):
                self._reset(done)
                obs = self.control.observation(self.plant)
            return obs,rewards.astype(np.float32),done,info

        def close(self):
            pass

        def get_attr(self,attr_name,indices=None):
            indices = range(8) if indices is None else [indices] if isinstance(indices,int) else indices
            return [None if attr_name=='render_mode' else getattr(self,attr_name) for _ in indices]

        def set_attr(self,attr_name,value,indices=None):
            setattr(self,attr_name,value)

        def env_method(self,method_name,*args,indices=None,**kwargs):
            return [getattr(self,method_name)(*args,**kwargs)]

        def env_is_wrapped(self,wrapper_class,indices=None):
            return [False]*8

    return FlexibleVecEnv()


def train(algorithm,seed,steps,stage):
    import torch
    from stable_baselines3 import SAC,TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    torch.set_num_threads(1)
    folder = ROOT/'03_学习训练'/stage/algorithm/f'seed_{seed}'
    metadata = folder/'metadata.json'
    if metadata.exists():
        old = read(metadata)
        if old.get('status')=='complete' and old['source_hashes']==sources() and old['actual_steps']==steps:
            return old
        raise RuntimeError('refusing to overwrite a partial or different training run: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    settings = read(ROOT/'02_传统控制'/'selected.json')['PID']['settings']
    env = vector_environment(seed,settings)
    common = dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',learning_rate=3e-4,
                  buffer_size=100000,learning_starts=2048,batch_size=128,gamma=.999,
                  train_freq=1,gradient_steps=1,policy_kwargs=dict(net_arch=[64,64]),verbose=0)
    if algorithm=='TD3':
        model = TD3(**common,action_noise=NormalActionNoise(np.zeros(1),np.full(1,.1)))
        torch.nn.init.zeros_(model.actor.mu[-2].weight)
        torch.nn.init.zeros_(model.actor.mu[-2].bias)
        model.actor_target.load_state_dict(model.actor.state_dict())
    elif algorithm=='SAC':
        model = SAC(**common,ent_coef='auto_0.01')
        torch.nn.init.zeros_(model.actor.mu.weight)
        torch.nn.init.zeros_(model.actor.mu.bias)
        torch.nn.init.zeros_(model.actor.log_std.weight)
        torch.nn.init.constant_(model.actor.log_std.bias,-2.)
    else:
        raise ValueError(algorithm)
    data = scenes('validation')
    snapshots = []
    started = time.time()
    source = sources()
    state = dict(status='running',algorithm=algorithm,seed=seed,stage=stage,requested_steps=steps,
                 source_hashes=source,baseline_sha256=digest(ROOT/'02_传统控制'/'selected.json'),
                 snapshots=snapshots,actual_steps=0)

    def validate(actual):
        score = summary(simulate(data,settings,model))
        snapshot = dict(steps=actual,validation=score)
        snapshots.append(snapshot)
        rank = lambda value:(-value['validation']['passes'],value['validation']['selection_cost'],value['steps'])
        if rank(snapshot)==min(map(rank,snapshots)):
            model.save(folder/'best.zip')
            state['best']=snapshot
        state['actual_steps']=actual
        state['elapsed_s']=time.time()-started
        write(metadata,state)
        print(json.dumps(dict(event='validation',algorithm=algorithm,seed=seed,**snapshot),ensure_ascii=False),flush=True)

    class Callback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % VALIDATE_EVERY==0:
                validate(self.num_timesteps)
            return True

    validate(0)
    model.learn(total_timesteps=steps,callback=Callback())
    if snapshots[-1]['steps']!=model.num_timesteps:
        validate(model.num_timesteps)
    if sources()!=source:
        raise RuntimeError('production source changed during training')
    model.save(folder/'final.zip')
    state.update(status='complete',actual_steps=model.num_timesteps,
                 gradient_updates=int(model._n_updates),
                 best_model_sha256=digest(folder/'best.zip'),elapsed_s=time.time()-started)
    write(metadata,state)
    env.close()
    return state


def development():
    records = [train(algorithm,seed,DEV_STEPS,'development') for algorithm,seed in [('TD3',16101),('SAC',16102)]]
    candidates = [r for r in records if r['best']['steps']>0 and
                  r['best']['validation']['success']>=.95 and r['best']['validation']['physical_failures']==0]
    selected = min(candidates,key=lambda r:(-r['best']['validation']['passes'],
                                          r['best']['validation']['selection_cost'],r['best']['steps'])) if candidates else None
    decision = dict(eligible=bool(candidates),selected_algorithm=None if selected is None else selected['algorithm'],
                    candidates=records,rule='validation only; held-out test outcomes unused')
    write(ROOT/'03_学习训练'/'development_decision.json',decision)
    return decision


def formal():
    decision = read(ROOT/'03_学习训练'/'development_decision.json')
    if decision['eligible']:
        for seed in FORMAL_SEEDS:
            train(decision['selected_algorithm'],seed,FORMAL_STEPS,'formal')


def model_load(algorithm,path):
    import torch
    from stable_baselines3 import SAC,TD3
    torch.set_num_threads(1)
    return {'SAC':SAC,'TD3':TD3}[algorithm].load(path,device='cpu')


def freeze_and_test():
    path = ROOT/'04_独立测试'/'freeze.json'
    baselines = read(ROOT/'02_传统控制'/'selected.json')
    decision = read(ROOT/'03_学习训练'/'development_decision.json')
    registry = [dict(name=method,settings=value['settings']) for method,value in baselines.items()]
    if decision['eligible']:
        algorithm = decision['selected_algorithm']
        for seed in FORMAL_SEEDS:
            folder = ROOT/'03_学习训练'/'formal'/algorithm/f'seed_{seed}'
            metadata = read(folder/'metadata.json')
            if metadata['status']!='complete' or metadata['source_hashes']!=sources():
                raise RuntimeError('formal training incomplete or source differs')
            registry.append(dict(name=f'PID_{algorithm}_{seed}',settings=baselines['PID']['settings'],
                                 model=str((folder/'best.zip').relative_to(ROOT)),
                                 model_sha256=digest(folder/'best.zip'),algorithm=algorithm,seed=seed,
                                 selected_steps=metadata['best']['steps']))
    else:
        # Explicit diagnostic: not treated as a three-seed research conclusion.
        for record in decision['candidates']:
            algorithm,seed = record['algorithm'],record['seed']
            model = ROOT/'03_学习训练'/'development'/algorithm/f'seed_{seed}'/'best.zip'
            registry.append(dict(name=f'DIAGNOSTIC_PID_{algorithm}',settings=baselines['PID']['settings'],
                                 model=str(model.relative_to(ROOT)),model_sha256=digest(model),
                                 algorithm=algorithm,diagnostic=True))
    value = dict(source_hashes=sources(),test_scenarios_sha256=digest(ROOT/'01_实验方案'/'test.json'),
                 baseline_sha256=digest(ROOT/'02_传统控制'/'selected.json'),controllers=registry)
    if path.exists() and read(path)!=value:
        raise RuntimeError('frozen controller registry differs')
    write(path,value)
    data = scenes('test')
    results = {}
    manifest = []
    for item in registry:
        model = model_load(item['algorithm'],ROOT/item['model']) if 'model' in item else None
        metrics,states,commands = simulate(data,item['settings'],model,record=True)
        folder = ROOT/'04_独立测试'/item['name']
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'responses.npz',state=states,command=commands,
                            **{k:v for k,v in metrics.items()})
        result = summary(metrics)
        results[item['name']] = result
        write(folder/'metrics.json',result)
        manifest.append(dict(name=item['name'],responses_sha256=digest(folder/'responses.npz'),
                             metrics_sha256=digest(folder/'metrics.json')))
        print(json.dumps(dict(event='held_out_test',name=item['name'],**result),ensure_ascii=False),flush=True)
    write(ROOT/'04_独立测试'/'summary.json',results)
    write(ROOT/'04_独立测试'/'manifest.json',dict(freeze_sha256=digest(path),controllers=manifest,
                                                total_episodes=len(data)*len(registry)))


def independent_transition_matrices(data):
    """Original physical state-space exponential; no modal analytic formulas."""
    n = len(data)
    augmented = np.zeros((n,5,5))
    for i,item in enumerate(data):
        wn,z = 2*np.pi*item['frequency'],item['damping']
        inv = 1/JB+1/JF
        stiffness,damping = wn*wn/inv,2*z*wn/inv
        augmented[i,:4,:4] = [[0,1,0,0],[0,0,stiffness/JB,damping/JB],
                              [0,0,0,1],[0,0,-stiffness*inv,-damping*inv]]
        augmented[i,:4,4] = [0,1/JB,0,-1/JB]
    a = augmented*DT
    matrix_norm = max(1.,float(np.max(np.sum(np.abs(a),axis=2))))
    scale = max(0,math.ceil(math.log2(matrix_norm/.25)))
    a = a/(2**scale)
    eye = np.broadcast_to(np.eye(5),(n,5,5)).copy()
    result,term = eye.copy(),eye.copy()
    for k in range(1,25):
        term = term@a/k
        result += term
    for _ in range(scale):
        result = result@result
    return result[:,:4,:4],result[:,:4,4]


def verify():
    frozen = read(ROOT/'04_独立测试'/'freeze.json')
    manifest = read(ROOT/'04_独立测试'/'manifest.json')
    if frozen['source_hashes']!=sources() or manifest['freeze_sha256']!=digest(ROOT/'04_独立测试'/'freeze.json'):
        raise RuntimeError('source or freeze hash mismatch')
    if frozen['test_scenarios_sha256']!=digest(ROOT/'01_实验方案'/'test.json'):
        raise RuntimeError('test scenario hash mismatch')
    if frozen['baseline_sha256']!=digest(ROOT/'02_传统控制'/'selected.json'):
        raise RuntimeError('baseline hash mismatch')
    data = scenes('test')
    f,g = independent_transition_matrices(data)
    checks = []
    for item,entry in zip(frozen['controllers'],manifest['controllers']):
        if item['name']!=entry['name']:
            raise RuntimeError('controller coverage mismatch')
        path = ROOT/'04_独立测试'/item['name']/'responses.npz'
        if digest(path)!=entry['responses_sha256']:
            raise RuntimeError('trace file hash mismatch')
        if digest(path.parent/'metrics.json')!=entry['metrics_sha256']:
            raise RuntimeError('metric file hash mismatch')
        if 'model' in item and digest(ROOT/item['model'])!=item['model_sha256']:
            raise RuntimeError('model hash mismatch')
        with np.load(path) as archive:
            states,commands = archive['state'],archive['command']
            if states.shape!=(len(data),STEPS+1,4) or commands.shape!=(len(data),STEPS):
                raise RuntimeError('trace dimensions wrong')
            expected_initial = np.asarray([[s['theta'],s['omega'],0.,0.] for s in data])
            np.testing.assert_array_equal(states[:,0],expected_initial)
            failed = np.zeros(len(data),dtype=bool)
            max_residual = 0.
            # Independently rebuild commands (including actions) from saved
            # pre-action states. Predictor sees no true modal parameters.
            model = model_load(item['algorithm'],ROOT/item['model']) if 'model' in item else None
            plant = Plant(len(data))
            plant.reset([s['theta'] for s in data],[s['omega'] for s in data],
                        [s['frequency'] for s in data],[s['damping'] for s in data])
            control = Controller(item['settings'],len(data))
            control.reset(plant)
            for step in range(STEPS):
                plant.state[:] = states[:,step]
                plant.failed[:] = failed
                plant.count[:] = step
                action = None if model is None else model.predict(control.observation(plant),deterministic=True)[0]
                reconstructed = control.command(plant,action)
                np.testing.assert_allclose(commands[:,step],reconstructed,rtol=0.,atol=2e-12)
                expected = np.einsum('nij,nj->ni',f,states[:,step])+g*commands[:,step,None]
                expected[failed] = states[failed,step]
                residual = float(np.max(np.abs(expected-states[:,step+1])))
                max_residual = max(max_residual,residual)
                if residual>2e-11:
                    raise RuntimeError('independent physical replay failed')
                next_state = states[:,step+1]
                failed |= ((np.abs(next_state[:,0])>np.deg2rad(170.))|
                           (np.abs(next_state[:,1])>np.deg2rad(300.))|
                           (np.abs(next_state[:,2])>np.deg2rad(10.)))
            # Separate metric expressions: do not call episode_metrics here.
            tail = states[:,int(round(8./.05)):,:]
            bound = np.array([np.pi/180.,np.pi/360.,np.pi/1800.,np.pi/360.])
            within = np.all(np.abs(tail)<=bound,axis=2)
            metrics = dict(success=np.all(within,axis=1)&~failed,failed=failed,
                           cost=.05*np.einsum('nt,nt->n',commands,commands),
                           rms_angle_deg=np.sqrt(np.einsum('nt,nt->n',states[:,:,0],states[:,:,0])/(STEPS+1))*180./np.pi,
                           peak_relative_deg=np.max(np.abs(states[:,:,2]),axis=1)*180./np.pi,
                           residual_relative_rms_deg=np.sqrt(np.sum(tail[:,:,2]*tail[:,:,2],axis=1)/tail.shape[1])*180./np.pi)
            for key,value in metrics.items():
                np.testing.assert_allclose(archive[key],value,rtol=0.,atol=1e-13)
            expected_summary = summary(metrics)
            saved_summary = read(path.parent/'metrics.json')
            for key,value in expected_summary.items():
                if not np.isclose(saved_summary[key],value,rtol=2e-13,atol=2e-13):
                    raise RuntimeError('summary differs from independent recomputation')
            checks.append(dict(name=item['name'],episodes=len(data),transitions=len(data)*STEPS,
                               max_state_residual=max_residual))
    if len(checks)!=len(frozen['controllers']) or len(checks)!=len(manifest['controllers']):
        raise RuntimeError('missing controller traces')
    result = dict(passed=True,full_trace_physics_replayed=True,
                  total_episodes=sum(s['episodes'] for s in checks),checks=checks,
                  source_hashes=sources(),manifest_sha256=digest(ROOT/'04_独立测试'/'manifest.json'))
    write(ROOT/'05_独立核验'/'verification.json',result)
    return result


def paired_intervals(a,b,groups,seed):
    delta = (np.asarray(a,dtype=float)-np.asarray(b,dtype=float)).reshape(len(FREQUENCIES),groups).mean(axis=0)
    rng = np.random.default_rng(seed)
    # Batching avoids a large temporary bootstrap matrix.
    means = np.concatenate([delta[rng.integers(0,groups,(1000,groups))].mean(axis=1) for _ in range(20)])
    alpha = .05/18
    low,high = np.quantile(means,[alpha/2,1-alpha/2])
    return dict(mean_difference=float(delta.mean()),lower=float(low),upper=float(high),
                confidence=1-alpha,resampling_unit='independent initial-state group',repeats=20000)


def report():
    verification = read(ROOT/'05_独立核验'/'verification.json')
    if not verification['passed']:
        raise RuntimeError('independent verification has not passed')
    frozen = read(ROOT/'04_独立测试'/'freeze.json')
    results = read(ROOT/'04_独立测试'/'summary.json')
    contrasts = []
    for item in frozen['controllers']:
        if not item['name'].startswith('PID_'):
            continue
        with np.load(ROOT/'04_独立测试'/item['name']/'responses.npz') as learned:
            for i,baseline in enumerate(('PID','ZVD_PID','LQR')):
                with np.load(ROOT/'04_独立测试'/baseline/'responses.npz') as other:
                    intervals = {key:paired_intervals(learned[key],other[key],COUNTS['test'],16040+i)
                                 for key in ('success','cost')}
                    both_all_pass = bool(np.all(learned['success']) and np.all(other['success']))
                    contrasts.append(dict(controller=item['name'],baseline=baseline,
                                          both_all_pass=both_all_pass,intervals=intervals,
                                          cost_reduction_fraction=float(1-learned['cost'].mean()/other['cost'].mean())))
    folder = ROOT/'06_报告'
    write(folder/'paired_intervals.json',contrasts)
    decision = read(ROOT/'03_学习训练'/'development_decision.json')
    lines = ['# 柔性附体振动抑制：独立测试结果','',
             '任务是同时让卫星本体按时停稳、让附体不再明显振动。学习模块每拍输出连续力矩补偿；真实模态频率和阻尼没有输入策略。',
             '', '| 方法 | 持续停稳 | 物理失败 | 平均力矩平方积分 | 峰值相对变形均值/° |',
             '|---|---:|---:|---:|---:|']
    for name,result in results.items():
        lines.append(f'| {name} | {result["passes"]}/{result["count"]} | {result["physical_failures"]} | {result["cost"]:.6f} | {result["peak_relative_deg"]:.3f} |')
    lines += ['', '控制代价为实际力矩平方的时间积分，不是电能。每种方法640回合测试由128个独立初态在五个频率上重复构成，统计按128个初态组抽样。', '']
    for baseline in ('PID','ZVD_PID','LQR'):
        comparisons = [c for c in contrasts if c['baseline']==baseline]
        success_claim = len(comparisons)==3 and all(c['intervals']['success']['lower']>0 and
                        results[c['controller']]['physical_failures']<=results[baseline]['physical_failures'] for c in comparisons)
        cost_claim = len(comparisons)==3 and all(c['both_all_pass'] and c['intervals']['cost']['upper']<0 for c in comparisons)
        lines.append(f'- 相对{baseline}：三种子一致的完成率优势={success_claim}；在双方全部达标前提下的代价优势={cost_claim}。')
    if not decision['eligible']:
        lines += ['', '本次没有开发候选达到正式训练门槛，学习结果只作开发问题诊断。没有正式三种子证据，不能据此判断该算法的能力上限。']
    lines += ['',f'独立核验通过：{verification["total_episodes"]}回合的每个物理状态转移均用独立状态空间矩阵指数重建，同时重算指令、停稳判定及代价。',
              '', '局限：单主要柔性模态、理想执行器、无测量噪声、假设可测附体变形；没有真实硬件或高阶模态验证。']
    text = '\n'.join(lines)+'\n'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'研究报告.md').write_text(text,encoding='utf-8')
    import html
    (folder/'研究报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>柔性附体研究结果</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.8 system-ui;color:#172b4d}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre></html>',encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['design','tune','development','formal','test','verify','report','all'])
    args = parser.parse_args()
    design()
    if args.command=='all':
        tuning();development();formal();freeze_and_test();verify();report()
    elif args.command=='tune': tuning()
    elif args.command=='development': development()
    elif args.command=='formal': formal()
    elif args.command=='test': freeze_and_test()
    elif args.command=='verify': verify()
    elif args.command=='report': report()


if __name__=='__main__':
    main()
