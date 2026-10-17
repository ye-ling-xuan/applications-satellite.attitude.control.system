"""Independent coupled three-axis validation of frozen single-axis controllers.

This package does not train a three-axis policy. Single-axis models are applied
axis by axis to BODY target rotation-vector components, keeping their original
history, torque scale, integral rules, and residual base PID unchanged.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PACKAGE = STUDY.parent
SINGLE = PACKAGE/'12_执行器滞后与纯延迟对比研究'
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(SINGLE/'代码'))
from multi_algorithm_benchmark.envs import quaternion_error, euler_to_quaternion

DT = .05
DURATION = 22.
DEADLINE = 8.
LIMIT = .5
J = np.array([1.2, 1., .8])
KINDS = ('lag', 'delay')
PARAMETERS = (.05, .10, .15, .20, .25, .30)
OUT = {'lag': (.40, .50), 'delay': (.35, .40, .50)}
COUNTS = {'development': 32, 'validation': 32, 'test': 128}
SEEDS = {'development': 202610171301, 'validation': 202610171302, 'test': 202610171303}
RK_SUBSTEPS = 2
ANGLE_TOL = np.deg2rad(1.)
RATE_TOL = np.deg2rad(.5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_hashes():
    sources = list((STUDY/'代码').glob('*.py'))
    sources += list((SINGLE/'代码'/'actuator_study').glob('*.py'))
    sources += [PACKAGE/'multi_algorithm_benchmark'/'envs.py']
    sources += list((STUDY/'01_实验方案').glob('*.json'))
    return {p.relative_to(PACKAGE).as_posix(): digest(p) for p in sorted(sources)}


def design():
    declaration = dict(study='三轴耦合迁移验证', joint_three_axis_training=False,
        inertia_kg_m2=J.tolist(), dt_s=DT, horizon_s=DURATION, deadline_s=DEADLINE,
        per_axis_command_limit_Nm=LIMIT, acceptance_attitude_error_deg=1.,
        acceptance_rate_vector_norm_deg_s=.5, quaternion='scalar-first body-to-inertial',
        error='existing quaternion_error: shortest target rotation vector in body frame',
        dynamics='J*omega_dot=tau_actual-cross(omega,J*omega); q_dot=.5*q*[0,omega]',
        rk4_substeps=RK_SUBSTEPS, lag_torque='analytic exponential evaluated at every RK stage',
        actual_cost='exact continuous torque-square integral, sum of three axes',
        initial_actual_torque=0., command_prehistory=0.,
        parameter_constant_per_episode=True, parameter_shared_by_three_actuators=True,
        in_domain_s=PARAMETERS, out_of_domain_s=OUT, counts=COUNTS, independent_seeds=SEEDS,
        initial_euler_xyz_deg=[-40.,40.], initial_body_rate_each_axis_deg_s=[-3.,3.],
        physical_failure_attitude_deg=170., physical_failure_rate_norm_deg_s=300.,
        selection='maximize validation successes then minimize mean full-horizon actual torque-square cost',
        transfer='12 selected PID/model compensation and every available formal seed; development models diagnostic only',
        residual_base='original per-run single-axis training base; never replace with retuned three-axis gains',
        source12_read_only=True, independent_dynamics_verification='predeclared representative cases only, not all trajectories',
        metric_verification='all saved test trajectories, costs and limits recomputed',
        no_continuous_or_infinite_time_stability_claim=True)
    declaration = json.loads(json.dumps(declaration))
    path = STUDY/'01_实验方案'/'执行配置.json'
    if path.exists() and read_json(path) != declaration:
        raise RuntimeError('predeclared three-axis configuration changed')
    write_json(path, declaration)
    for split, count in COUNTS.items():
        rng = np.random.default_rng(SEEDS[split])
        angles = rng.uniform(-40.,40.,(count,3))
        rates = rng.uniform(-3.,3.,(count,3))
        states = [dict(sample_id=i, euler_deg=a.tolist(), omega_deg_s=w.tolist())
                  for i,(a,w) in enumerate(zip(angles,rates))]
        path = STUDY/'01_实验方案'/f'{split}_initial_states.json'
        if path.exists() and read_json(path) != states:
            raise RuntimeError('three-axis scenario split changed')
        write_json(path, states)
    return declaration


def scenes(kind, split, region='in_domain'):
    parameters = PARAMETERS if region == 'in_domain' else OUT[kind] if region == 'out_of_domain' else (0.,)
    states = read_json(STUDY/'01_实验方案'/f'{split}_initial_states.json')
    return [dict(kind=kind, split=split, region=region, parameter_s=p,
                 scenario_id=f'{kind}_{region}_{split}_{p:.2f}_{s["sample_id"]:03d}', **s)
            for p in parameters for s in states]


def derivative(q, omega, torque):
    """Complete Euler coupling, with scalar-first quaternion kinematics."""
    w0,w1,w2 = omega.T
    gyroscopic = np.column_stack(((J[2]-J[1])*w1*w2,
                                  (J[0]-J[2])*w2*w0, (J[1]-J[0])*w0*w1))
    dq = np.empty_like(q)
    s,x,y,z = q.T
    dq[:,0] = -.5*(x*w0+y*w1+z*w2)
    dq[:,1] = .5*(s*w0+y*w2-z*w1)
    dq[:,2] = .5*(s*w1+z*w0-x*w2)
    dq[:,3] = .5*(s*w2+x*w1-y*w0)
    return dq, (torque-gyroscopic)/J


def rigid_interval(q, omega, tau_start, command, parameter_s, kind, dt=DT, substeps=RK_SUBSTEPS):
    """RK4 rigid dynamics driven by analytic time-varying actuator torque.

    In delay mode command is the already delayed actual torque. Returns
    q_end, omega_end, tau_end, exact per-axis integral(tau_actual**2 dt).
    """
    q,omega,a,u = [np.asarray(v,float).copy() for v in (q,omega,tau_start,command)]
    parameter = np.broadcast_to(np.asarray(parameter_s,float),(len(q),))
    if kind not in KINDS or substeps < 1 or dt <= 0. or np.any(parameter < 0.):
        raise ValueError('invalid integration settings')
    if (q.shape!=(len(q),4) or omega.shape!=(len(q),3) or a.shape!=omega.shape
            or u.shape!=omega.shape or not all(np.all(np.isfinite(v)) for v in (q,omega,a,u,parameter))):
        raise ValueError('invalid rigid-body states')
    dynamic = (parameter > 0.) if kind=='lag' else np.zeros(len(q),bool)
    def torque(t):
        result = u.copy()
        result[dynamic] += (a[dynamic]-u[dynamic])*np.exp(-t/parameter[dynamic,None])
        return result
    h = dt/substeps
    for k in range(substeps):
        t = k*h
        t0,tm,t1 = torque(t),torque(t+.5*h),torque(t+h)
        a_q,a_w = derivative(q,omega,t0)
        b_q,b_w = derivative(q+.5*h*a_q,omega+.5*h*a_w,tm)
        c_q,c_w = derivative(q+.5*h*b_q,omega+.5*h*b_w,tm)
        d_q,d_w = derivative(q+h*c_q,omega+h*c_w,t1)
        q += h*(a_q+2*b_q+2*c_q+d_q)/6
        omega += h*(a_w+2*b_w+2*c_w+d_w)/6
        q /= np.linalg.norm(q,axis=1,keepdims=True)
    cost = u*u*dt
    if np.any(dynamic):
        p = parameter[dynamic,None]
        e = np.exp(-dt/p)
        b = a[dynamic]-u[dynamic]
        cost[dynamic] = (u[dynamic]**2*dt+2*u[dynamic]*b*p*(-np.expm1(-dt/p))
                         +b*b*.5*p*(-np.expm1(-2*dt/p)))
    return q,omega,torque(dt),np.maximum(cost,0.)


class Plant:
    def __init__(self, kind, ss):
        self.kind,self.n = kind,len(ss)
        self.parameter = np.array([s['parameter_s'] for s in ss])
        self.q = euler_to_quaternion(np.radians([s['euler_deg'] for s in ss]))
        self.omega = np.radians([s['omega_deg_s'] for s in ss])
        self.actual = np.zeros((self.n,3))
        self.failed = np.zeros(self.n,bool)
        self.k = 0
        self.delay_steps = np.rint(self.parameter/DT).astype(int)
        if kind=='delay' and not np.allclose(self.delay_steps*DT,self.parameter,atol=1e-12):
            raise ValueError('integer sample delay required')
        self.queue = np.zeros((self.n,max(1,int(np.max(self.delay_steps))),3))

    def observation(self):
        return np.column_stack((quaternion_error(self.q),self.omega))

    def step(self, commands):
        if self.k >= round(DURATION/DT):
            raise RuntimeError('plant complete')
        command = np.asarray(commands,float)
        if command.shape!=(self.n,3) or not np.all(np.isfinite(command)):
            raise ValueError('commands must be finite (n,3)')
        command = np.where(self.failed[:,None],0.,np.clip(command,-LIMIT,LIMIT))
        active = ~self.failed
        actual_command = command.copy()
        if self.kind=='delay':
            delayed = self.delay_steps > 0
            rows = np.flatnonzero(delayed)
            actual_command[rows] = self.queue[rows,self.delay_steps[rows]-1]
            self.queue[:,1:] = self.queue[:,:-1]
            self.queue[:,0] = command
        tau_start = self.actual.copy() if self.kind=='lag' else actual_command.copy()
        tau_start[(self.kind=='lag') & (self.parameter==0.)] = command[(self.kind=='lag') & (self.parameter==0.)]
        cost = np.zeros((self.n,3))
        if np.any(active):
            q,w,a,c = rigid_interval(self.q[active],self.omega[active],self.actual[active],
                                     actual_command[active],self.parameter[active],self.kind)
            self.q[active],self.omega[active],self.actual[active],cost[active] = q,w,a,c
        self.actual[~active] = 0.
        tau_start[~active] = 0.
        self.failed |= ((np.linalg.norm(quaternion_error(self.q),axis=1)>np.deg2rad(170.))
                        |(np.linalg.norm(self.omega,axis=1)>np.deg2rad(300.)))
        self.k += 1
        return dict(command=command,tau_start=tau_start,tau_end=self.actual.copy(),
                    actual_cost_axis=cost,command_cost_axis=command**2*DT)


class GeometricPID:
    """Common or inertia-scaled gains, independently tuned for the 3D task."""
    def __init__(self, parameters, candidates=None):
        self.parameters = parameters
        self.candidates = candidates or [parameters]

    def reset(self,n,obs):
        if n % len(self.candidates):
            raise ValueError('candidate-major scene count indivisible')
        repeat = n//len(self.candidates)
        self.gains = {}
        for key in ('kp','kd','ki'):
            values = [c.get(key,0.)*(J if c.get('inertia_scaled',True) else np.ones(3)) for c in self.candidates]
            self.gains[key] = np.repeat(np.array(values),repeat,axis=0)
        self.integral = np.zeros((n,3))

    def act(self,obs,elapsed):
        e,w = obs[:,:3],obs[:,3:]
        proposed = np.clip(self.integral+e*DT,-2.,2.)
        raw = self.gains['kp']*e-self.gains['kd']*w+self.gains['ki']*proposed
        command = np.clip(raw,-LIMIT,LIMIT)
        accept = (raw-command)*self.gains['ki']*(proposed-self.integral)<=1e-15
        self.integral = np.where(accept,proposed,self.integral)
        return command


class AxisTransfer:
    """Three independent deployments of the ORIGINAL single-axis interface."""
    def __init__(self,kind,method,parameters=None,model_path=None,base_parameters=None):
        from actuator_study.baselines import PIDController,ModelCompensatedController
        self.controllers=[]
        if method in ('pid_transfer','compensated_transfer'):
            self.controllers=[PIDController(parameters) if method=='pid_transfer'
                              else ModelCompensatedController(kind,parameters) for _ in range(3)]
        else:
            import torch
            from stable_baselines3 import SAC
            from actuator_study.learning import SACController
            torch.set_num_threads(1)
            model = SAC.load(model_path,device='cpu')
            self.controllers=[SACController(model,method,base_parameters) for _ in range(3)]

    def reset(self,n,obs):
        for axis,controller in enumerate(self.controllers):
            controller.reset(n,obs[:,[axis,axis+3]])

    def act(self,obs,elapsed):
        return np.column_stack([controller.act(obs[:,[axis,axis+3]],elapsed)
                                for axis,controller in enumerate(self.controllers)])


def summarize(rows):
    passed=[r for r in rows if r['success']]
    return dict(count=len(rows),passes=len(passed),success_rate=len(passed)/len(rows),
        physical_failures=sum(r['failed'] for r in rows),
        mean_actual_cost=float(np.mean([r['actual_cost'] for r in rows])),
        mean_command_cost=float(np.mean([r['command_cost'] for r in rows])),
        max_settling_time_s=max((r['settling_time_s'] for r in passed),default=None))


def evaluate(kind,ss,controller,keep_traces=False):
    plant=Plant(kind,ss)
    n=len(ss)
    obs=plant.observation()
    controller.reset(n,obs)
    angle=np.linalg.norm(obs[:,:3],axis=1)
    rate=np.linalg.norm(obs[:,3:],axis=1)
    last_bad=np.where((angle>ANGLE_TOL)|(rate>RATE_TOL),0,-1)
    success=np.ones(n,bool)
    cost,ucost=np.zeros(n),np.zeros(n)
    max_angle,max_rate=angle.copy(),rate.copy()
    traces=None
    steps=round(DURATION/DT)
    if keep_traces:
        traces=dict(time=np.arange(steps+1)*DT,q=np.empty((steps+1,n,4)),omega=np.empty((steps+1,n,3)),
                    failed=np.empty((steps+1,n),bool),
                    **{key:np.empty((steps,n,3)) for key in ('command','tau_start','tau_end','actual_cost_axis','command_cost_axis')})
        traces['q'][0],traces['omega'][0],traces['failed'][0]=plant.q,plant.omega,plant.failed
    for k in range(steps):
        command=np.asarray(controller.act(obs,np.full(n,k*DT)),float)
        if command.shape!=(n,3) or not np.all(np.isfinite(command)) or np.max(np.abs(command))>LIMIT+1e-10:
            raise ValueError('invalid/unclipped controller output')
        step=plant.step(command)
        obs=plant.observation()
        angle=np.linalg.norm(obs[:,:3],axis=1)
        rate=np.linalg.norm(obs[:,3:],axis=1)
        outside=(angle>ANGLE_TOL)|(rate>RATE_TOL)|plant.failed
        last_bad=np.where(outside,k+1,last_bad)
        if (k+1)*DT>=DEADLINE-1e-10:
            success &= ~outside
        cost+=np.sum(step['actual_cost_axis'],axis=1)
        ucost+=np.sum(step['command_cost_axis'],axis=1)
        max_angle=np.maximum(max_angle,angle)
        max_rate=np.maximum(max_rate,rate)
        if traces is not None:
            traces['q'][k+1],traces['omega'][k+1],traces['failed'][k+1]=plant.q,plant.omega,plant.failed
            for key in step:
                traces[key][k]=step[key]
    rows=[dict(s,success=bool(success[i] and not plant.failed[i]),failed=bool(plant.failed[i]),
        actual_cost=float(cost[i]),command_cost=float(ucost[i]),
        settling_time_s=float(max(0,last_bad[i]+1)*DT) if success[i] and not plant.failed[i] else None,
        max_error_deg=float(np.degrees(max_angle[i])),max_rate_norm_deg_s=float(np.degrees(max_rate[i])),
        final_error_deg=float(np.degrees(angle[i])),final_rate_norm_deg_s=float(np.degrees(rate[i])))
        for i,s in enumerate(ss)]
    return dict(rows=rows,traces=traces)


def _rank(row):
    return (-row['passes'],row['mean_actual_cost'])


def tune(kind,batch_size=12):
    if (STUDY/'04_独立测试'/'freeze.json').exists():
        raise RuntimeError('test freeze exists: tuning prohibited')
    start=time.monotonic()
    frozen=input_hashes()
    rng=np.random.default_rng(202610171399+KINDS.index(kind))
    candidates=[dict(kp=float(kp),kd=float(kd),ki=0.,inertia_scaled=scaled)
                for kp in np.geomspace(.1,8.,17) for kd in np.geomspace(.35,8.,17) for scaled in (True,False)]
    candidates += [dict(kp=float(np.exp(rng.uniform(np.log(.15),np.log(6.)))),
                        kd=float(np.exp(rng.uniform(np.log(.4),np.log(7.)))),
                        ki=float(np.exp(rng.uniform(np.log(.0001),np.log(.1)))),inertia_scaled=True)
                   for _ in range(72)]
    directory=STUDY/'02_三轴PID整定'/kind
    all_scores=[]
    def stage(cs,split,name):
        ss=scenes(kind,split)
        scores=[]
        for offset in range(0,len(cs),batch_size):
            chunk=cs[offset:offset+batch_size]
            result=evaluate(kind,ss*len(chunk),GeometricPID(chunk[0],chunk))
            for i,c in enumerate(chunk):
                stats=summarize(result['rows'][i*len(ss):(i+1)*len(ss)])
                scores.append(dict(candidate_id=f'{kind}_{name}_{offset+i:05d}',split=split,stage=name,**stats))
            print(f'3D PID {kind} {name}: {offset+len(chunk)}/{len(cs)} elapsed={time.monotonic()-start:.1f}s',flush=True)
        write_csv(directory/f'{name}_{split}.csv',[dict(**s,**c) for s,c in zip(scores,cs)])
        all_scores.extend([dict(**s,parameters=c) for s,c in zip(scores,cs)])
        return scores
    broad=stage(candidates,'development','broad')
    indexes=set(sorted(range(len(candidates)),key=lambda i:_rank(broad[i]))[:24])
    for scaled in (True,False):
        ii=[i for i,c in enumerate(candidates) if c['inertia_scaled']==scaled]
        indexes.update(sorted(ii,key=lambda i:_rank(broad[i]))[:4])
    ii=[i for i,c in enumerate(candidates) if c['ki']>0.]
    indexes.update(sorted(ii,key=lambda i:_rank(broad[i]))[:4])
    cs=[candidates[i] for i in sorted(indexes)]
    stats=stage(cs,'validation','shortlist')
    for number,sigma in ((1,.15),(2,.06)):
        leaders=sorted(range(len(cs)),key=lambda i:_rank(stats[i]))[:6]
        additions=[]
        for i in leaders:
            for j in range(12):
                c=dict(cs[i]);c['kp']*=float(np.exp(rng.normal(0.,sigma)));c['kd']*=float(np.exp(rng.normal(0.,sigma)))
                if j%4==0:c['ki']=0.
                elif c['ki']>0.:c['ki']*=float(np.exp(rng.normal(0.,sigma)))
                elif j%4==1:c['ki']=float(np.exp(rng.uniform(np.log(.0001),np.log(.02))))
                additions.append(c)
        more=stage(additions,'validation',f'refine{number}')
        cs.extend(additions);stats.extend(more)
    best=min(range(len(cs)),key=lambda i:_rank(stats[i]))
    if input_hashes()!=frozen:
        raise RuntimeError('inputs/source changed during three-axis PID tuning')
    result=dict(kind=kind,parameters=cs[best],validation=stats[best],test_responses_used=False,
        source_and_input_hashes=frozen,wall_time_s=time.monotonic()-start,
        controller='PD' if cs[best]['ki']==0. else 'PID',
        selection_rule='maximum validation successes then minimum mean actual cost',
        broad_candidates=len(candidates),validation_candidates=len(cs),no_global_optimum_claim=True)
    write_json(directory/'all_attempts.json',all_scores)
    write_json(directory/'selected.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def controller_specs(include_models=True):
    specs=[]
    for kind in KINDS:
        selected=read_json(STUDY/'02_三轴PID整定'/kind/'selected.json')
        single=read_json(SINGLE/'02_传统控制'/f'selected_{kind}.json')
        specs.extend([dict(id=f'{kind}_retuned_3D_PID',kind=kind,method='retuned',stage='traditional',parameters=selected['parameters']),
            dict(id=f'{kind}_single_PID_transfer',kind=kind,method='pid_transfer',stage='traditional',parameters=single['pid']['parameters']),
            dict(id=f'{kind}_single_compensated_transfer',kind=kind,method='compensated_transfer',stage='traditional',parameters=single['model_compensated']['parameters'])])
    if include_models:
        # The single-axis freeze is authoritative: preserve every formal seed
        # and its predeclared development diagnostics without new model choice.
        single_freeze=read_json(SINGLE/'04_独立测试'/'freeze.json')
        from actuator_study.delivery import check_artifacts
        check_artifacts(single_freeze['artifacts'])
        for entry in single_freeze['controllers']:
            if 'model_path' not in entry:
                continue
            model_path=SINGLE/entry['model_path']
            if digest(model_path)!=entry['model_sha256']:
                raise RuntimeError('single-axis frozen model checksum differs')
            metadata_path=SINGLE/entry['metadata_path']
            metadata=read_json(metadata_path)
            if entry['base_parameters']!=metadata['base_parameters']:
                raise RuntimeError('residual base differs from original training')
            specs.append(dict(id=entry['id']+'_three_axis_transfer',kind=entry['kind'],
                method=entry['family'],stage=entry['stage'],seed=entry['seed'],
                reward_name=entry['reward_name'],run_status=metadata['status'],
                model_path=str(model_path),model_sha256=entry['model_sha256'],
                base_parameters=entry['base_parameters'],metadata_path=str(metadata_path),
                single_axis_controller_id=entry['id'],source_model_best_steps=entry['selected_steps'],
                diagnostic_only=(entry['stage']!='formal')))
    return specs


def freeze(include_models=True):
    if include_models and not (SINGLE/'04_独立测试'/'freeze.json').exists():
        raise RuntimeError('single-axis final controller freeze is not ready; finish baseline validation first')
    specs=controller_specs(include_models)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=input_hashes(),
        controllers=specs,joint_three_axis_training=False,include_models=include_models,
        single_axis_freeze_sha256=digest(SINGLE/'04_独立测试'/'freeze.json') if include_models else None,
        test_initial_sha256=digest(STUDY/'01_实验方案'/'test_initial_states.json'),
        selection_finished_before_test=True,development_models_diagnostic_only=True)
    path=STUDY/'04_独立测试'/'freeze.json'
    if path.exists():
        old=read_json(path)
        manifest['created_utc']=old['created_utc']
        if old!=manifest:raise RuntimeError('frozen test manifest changed')
    write_json(path,manifest)
    return manifest


def make_controller(spec):
    if spec['method']=='retuned':return GeometricPID(spec['parameters'])
    return AxisTransfer(spec['kind'],spec['method'],spec.get('parameters'),spec.get('model_path'),spec.get('base_parameters'))


def run_test():
    freeze_path=STUDY/'04_独立测试'/'freeze.json'
    manifest=read_json(freeze_path)
    freeze_sha=digest(freeze_path)
    if manifest.get('single_axis_freeze_sha256') and digest(SINGLE/'04_独立测试'/'freeze.json')!=manifest['single_axis_freeze_sha256']:
        raise RuntimeError('single-axis controller freeze changed')
    if input_hashes()!=manifest['source_and_input_hashes']:raise RuntimeError('frozen inputs changed')
    totals=[]
    for spec in manifest['controllers']:
        if spec.get('model_path') and digest(spec['model_path'])!=spec['model_sha256']:
            raise RuntimeError('frozen model changed')
        for region in ('in_domain','ideal','out_of_domain'):
            directory=STUDY/'04_独立测试'/spec['id']/region
            if (directory/'complete.json').exists():
                completion=read_json(directory/'complete.json')
                if completion['freeze_sha256']!=freeze_sha:
                    raise RuntimeError('saved result belongs to another freeze')
                for name,value in completion['artifact_sha256'].items():
                    if digest(directory/name)!=value:raise RuntimeError('saved output artifact changed')
                totals.append(read_json(directory/'summary.json'));continue
            if (directory/'metrics.csv').exists():
                raise RuntimeError('incomplete result directory retained; review artifacts before resuming')
            result=evaluate(spec['kind'],scenes(spec['kind'],'test',region),make_controller(spec),True)
            directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'traces.npz',**result['traces'])
            write_csv(directory/'metrics.csv',result['rows'])
            stats=dict(controller=spec['id'],kind=spec['kind'],method=spec['method'],stage=spec['stage'],
                       region=region,diagnostic_only=spec.get('diagnostic_only',False),**summarize(result['rows']))
            write_json(directory/'summary.json',stats)
            write_json(directory/'complete.json',dict(complete=True,freeze_sha256=freeze_sha,
                controller=spec['id'],region=region,
                artifact_sha256={name:digest(directory/name) for name in ('traces.npz','metrics.csv','summary.json')}))
            totals.append(stats)
            print(json.dumps(stats,ensure_ascii=False),flush=True)
    write_csv(STUDY/'04_独立测试'/'all_summaries.csv',totals)
    return totals


def verify_saved():
    """Recompute ALL metrics/costs/limits; representative ODE checks are separate."""
    from scipy.spatial.transform import Rotation
    freeze_path=STUDY/'04_独立测试'/'freeze.json'
    manifest=read_json(freeze_path)
    if manifest.get('single_axis_freeze_sha256') and digest(SINGLE/'04_独立测试'/'freeze.json')!=manifest['single_axis_freeze_sha256']:
        raise RuntimeError('single-axis controller freeze changed')
    expected={STUDY/'04_独立测试'/spec['id']/region/'metrics.csv'
              for spec in manifest['controllers'] for region in ('in_domain','ideal','out_of_domain')}
    actual_paths=set((STUDY/'04_独立测试').glob('*/*/metrics.csv'))
    if not expected or expected!=actual_paths:
        raise RuntimeError('test outputs do not cover exactly every frozen controller and region')
    checks=[]
    for path in sorted(expected):
        completion=read_json(path.with_name('complete.json'))
        assert completion['complete'] and completion['freeze_sha256']==digest(freeze_path)
        for name,value in completion['artifact_sha256'].items():
            assert digest(path.parent/name)==value,(path,name,'artifact hash')
        with np.load(path.with_name('traces.npz')) as archive:
            data={key:archive[key] for key in archive.files}
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        q,w=data['q'],data['omega']
        # Independent scipy rotation magnitude, not production quaternion_error.
        angle=Rotation.from_quat(q[...,[1,2,3,0]].reshape(-1,4)).magnitude().reshape(q.shape[:2])
        rate=np.sqrt(np.sum(w*w,axis=2))
        outside=(angle>ANGLE_TOL)|(rate>RATE_TOL)|data['failed']
        accept=np.all(~outside[np.arange(len(angle))*DT>=DEADLINE-1e-10],axis=0)
        actual=np.zeros(len(rows));command=np.sum(data['command']**2,axis=(0,2))*DT
        kind=rows[0]['kind'];params=np.array([float(r['parameter_s']) for r in rows])
        if kind=='delay':
            delayed=np.zeros_like(data['command'])
            for i,p in enumerate(params):
                m=round(p/DT)
                delayed[m:,i]=data['command'][:len(delayed)-m,i] if m else data['command'][:,i]
            delayed=np.where(data['failed'][:-1,:,None],0.,delayed)
            assert np.max(np.abs(delayed-data['tau_start']))<1e-11
            actual=np.sum(delayed*delayed,axis=(0,2))*DT
        else:
            # Independent Gaussian quadrature of each continuous exponential;
            # 20 nodes resolves even T=.05 over a .05-second interval.
            nodes,weights=np.polynomial.legendre.leggauss(20)
            for node,weight in zip(nodes,weights):
                t=.5*DT*(node+1)
                dynamic=params>0.
                torque=data['command'].copy()
                torque[:,dynamic]+=(data['tau_start'][:,dynamic]-data['command'][:,dynamic])*np.exp(-t/params[dynamic,None])
                torque=np.where(data['failed'][:-1,:,None],0.,torque)
                actual+=.5*DT*weight*np.sum(torque*torque,axis=(0,2))
        last_bad=np.max(np.where(outside,np.arange(len(angle))[:,None],-1),axis=0)
        for i,row in enumerate(rows):
            assert accept[i]==(row['success']=='True'),(path,i,'success')
            assert bool(data['failed'][-1,i])==(row['failed']=='True')
            assert abs(actual[i]-float(row['actual_cost']))<2e-9,(path,i,'actual cost')
            assert abs(command[i]-float(row['command_cost']))<2e-9,(path,i,'command cost')
            if accept[i]:assert abs(max(0,last_bad[i]+1)*DT-float(row['settling_time_s']))<1e-10
            for key,value in (('max_error_deg',np.degrees(np.max(angle[:,i]))),
                              ('max_rate_norm_deg_s',np.degrees(np.max(rate[:,i]))),
                              ('final_error_deg',np.degrees(angle[-1,i])),
                              ('final_rate_norm_deg_s',np.degrees(rate[-1,i]))):
                assert abs(value-float(row[key]))<1e-8,(path,i,key)
        assert np.max(np.abs(data['command']))<=LIMIT+1e-12
        assert np.max(np.abs(data['command_cost_axis']-data['command']**2*DT))<1e-12
        if kind=='delay':
            assert np.max(np.abs(data['tau_end']-delayed))<1e-11
        else:
            predicted_end=data['command'].copy()
            dynamic=params>0.
            predicted_end[:,dynamic]+=(data['tau_start'][:,dynamic]-data['command'][:,dynamic])*np.exp(-DT/params[dynamic,None])
            predicted_end=np.where(data['failed'][:-1,:,None],0.,predicted_end)
            assert np.max(np.abs(data['tau_end']-predicted_end))<1e-11
            expected_start=np.zeros_like(data['tau_start'])
            expected_start[1:]=data['tau_end'][:-1]
            expected_start[:,~dynamic]=data['command'][:,~dynamic]
            expected_start=np.where(data['failed'][:-1,:,None],0.,expected_start)
            assert np.max(np.abs(data['tau_start']-expected_start))<1e-11
        assert np.max(np.abs(np.linalg.norm(q,axis=2)-1.))<1e-10
        assert np.all(data['actual_cost_axis']>=-1e-12)
        assert np.max(np.abs(np.sum(data['actual_cost_axis'],axis=(0,2))-actual))<2e-9
        checks.append(dict(metrics_path=str(path.relative_to(STUDY)),count=len(rows),passed=True,
                           actual_cost_independent='20-node Gaussian quadrature' if kind=='lag' else 'reconstructed delayed commands'))
    report=dict(all_saved_trajectories_metric_cost_limit_checks=checks,
                expected_output_groups=len(expected),verified_output_groups=len(checks),
                independent_full_dynamics_scope='representative cases only; see unit-test verification, not all saved responses',
                source_and_input_hashes_unchanged=input_hashes()==read_json(STUDY/'04_独立测试'/'freeze.json')['source_and_input_hashes'])
    assert report['source_and_input_hashes_unchanged']
    write_json(STUDY/'05_独立核验'/'all_saved_verification.json',report)
    return report


def report():
    lines=['# 执行器误差下的三轴耦合迁移验证','',
        '本研究把12号研究已冻结的单轴控制器逐轴部署到完整耦合三轴刚体；不进行三轴联合强化学习训练。',
        '刚体惯量为[1.2,1.0,0.8] kg·m²，四元数采用标量在前的本体系到惯性系约定；动力学保留ω×Jω项。',
        '采样0.05秒，运行22秒，从第8秒起整体姿态误差≤1°且角速度向量范数≤0.5°/秒才通过；每轴指令限制±0.5 Nm。',
        '',
        '三轴RK4每采样间隔分成2个子步。滞后实际力矩在每个RK阶段按连续指数解析值计算，不能将区间端点力矩当作区间恒力矩。实际力矩平方积分按每轴解析积分后求和；纯延迟严格执行u[k-m]。',
        '',
        '开发32、验证32、测试128个独立初态，分别与全部预定执行器参数交叉。初态欧拉角各轴±40°、本体角速度各轴±3°/秒。一个回合的三轴执行器共用同一固定T或d。',
        '',
        '三轴PID重新整定，搜索公共及按惯量缩放的比例/微分增益，并允许积分。单轴PID与模型补偿直接逐轴迁移，不重新调节其参数；其名义惯量仍为单轴训练的1 kg·m²，这种模型失配属于迁移条件。',
        'SAC各轴输入为既有quaternion_error返回的目标旋转向量分量及对应角速度[e_i,ω_i]；纯单轴时e_i=-θ，与训练符号一致。每轴保留独立历史、积分及命令队列。残差SAC保留原训练基础PID，不替换成新整定三轴PID。',
        '',
        '## 验证集选择结果','',
        '|执行器|控制器|通过数|平均实际力矩代价|最大停稳时间|','|---|---|---:|---:|---:|']
    for kind in KINDS:
        path=STUDY/'02_三轴PID整定'/kind/'selected.json'
        if path.exists():
            selected=read_json(path);stats=selected['validation']
            lines.append(f'|{kind}|重新整定三轴{selected["controller"]}|{stats["passes"]}/{stats["count"]}|{stats["mean_actual_cost"]:.8f}|{stats["max_settling_time_s"]}|')
    freeze_path=STUDY/'04_独立测试'/'freeze.json'
    if not freeze_path.exists():
        lines += ['', '三轴传统基线已准备；最终测试清单尚未冻结，等待12号研究全部正式模型及开发诊断模型就绪。当前不构成正式三轴迁移测试结论。']
    else:
        manifest=read_json(freeze_path)
        lines += ['',f'最终冻结清单包含{len(manifest["controllers"])}个控制器条目。开发模型仅作为诊断，正式种子逐一保留，不以测试结果筛选种子。',
                  '', '## 独立测试结果','',
                  '|控制器|范围|通过数|物理失败|平均实际力矩代价|','|---|---|---:|---:|---:|']
        for path in sorted((STUDY/'04_独立测试').glob('*/*/summary.json')):
            stats=read_json(path)
            lines.append(f'|{stats["controller"]}|{stats["region"]}|{stats["passes"]}/{stats["count"]}|{stats["physical_failures"]}|{stats["mean_actual_cost"]:.8f}|')
    lines += ['', '## 核验与结论边界','',
        'test_transfer_physics.py和test_transfer_independent.py采用独立solve_ivp/四元数实现核对预定代表情况。它们仅代表所列物理工况，不能声称全部响应都完成独立动力学求解核验。',
        'verify命令对全部已保存测试轨迹独立重算整体误差、角速度范数、成功/失败、停稳时间、实际及指令代价、延迟指令和力矩限制；滞后代价另用20节点高斯求积核验。',
        '这是一项固定参数范围、有限验收窗口下的迁移实验。结果不能代替三轴联合训练比较，也不证明连续时间、无限时域稳定性或真实航天器可部署性。',
        '若重新整定三轴PID表现更好，应如实报告；若迁移SAC失败，只能说明本次单轴模型在这些耦合条件下未成功迁移，不能推断强化学习不能控制三轴。',
        '', '命令：python -X utf8 -B transfer_study.py design / tune / freeze / test / verify / report。12号研究源码及模型只读。','']
    path=STUDY/'研究报告.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('design','tune','freeze','test','verify','report'))
    parser.add_argument('--kind',choices=KINDS)
    parser.add_argument('--batch-size',type=int,default=12)
    parser.add_argument('--traditional-only',action='store_true')
    args=parser.parse_args()
    if args.command=='design':design()
    elif args.command=='tune':
        for kind in (args.kind,) if args.kind else KINDS:tune(kind,args.batch_size)
    elif args.command=='freeze':freeze(not args.traditional_only)
    elif args.command=='test':run_test()
    elif args.command=='verify':verify_saved()
    elif args.command=='report':report()


if __name__=='__main__':main()
