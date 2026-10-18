"""Declared extra development: symmetry-constrained low-gain residual PPO."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
import numpy as np
import study as b

ROOT=b.STUDY
STEPS=524288
CHECKPOINT=32768
SEED=18202
RESIDUAL=.05


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+Path(b.__file__).read_bytes()).hexdigest()


class Interface(b.PolicyInterface):
    def features(self,plant,ref):
        result=super().features(plant,ref)
        result[:,-1]=0.
        return result

    def command(self,plant,ref,action):
        raw=self.pid.propose(plant,ref)+RESIDUAL*np.clip(np.asarray(action,float).reshape(plant.n),-1.,1.)
        command=np.clip(raw,-b.LIMIT,b.LIMIT)
        self.pid.commit(plant,ref,command,raw);self.previous_command=command.copy()
        return command


class Controller(b.LearnedController):
    def __init__(self,model,parameters):
        self.model=model;self.interface=Interface(parameters)


class ExactMPC(b.MPC):
    """SLSQP solution of the same convex torque trajectory quadratic program."""
    def __init__(self):
        # SLSQP does not need ADMM's Cholesky factor.  Avoid initializing
        # scipy.linalg's second OpenMP runtime after Torch in this Windows
        # installation; all predictive matrices below use NumPy only.
        self.h=30
        h=self.h;self.A=np.array([[1.,b.DT],[0.,1.]])
        self.b0=np.array([b.DT**2/(3*b.INERTIA),b.DT/(2*b.INERTIA)])
        self.b1=np.array([b.DT**2/(6*b.INERTIA),b.DT/(2*b.INERTIA)])
        self.F=np.zeros((2*h,2));self.G=np.zeros((2*h,h));self.c=np.zeros(2*h)
        powers=[np.linalg.matrix_power(self.A,k) for k in range(h+1)]
        for k in range(h):
            self.F[2*k:2*k+2]=powers[k+1];self.c[2*k:2*k+2]=powers[k]@self.b0
            for j in range(k+1):
                value=powers[k-j]@self.b1
                if j<k:value+=powers[k-j-1]@self.b0
                self.G[2*k:2*k+2,j]=value
        weights=np.tile([1/np.deg2rad(2.)**2,.1/np.deg2rad(5.)**2],h)
        self.WG=weights[:,None]*self.G;self.D=np.eye(h)-np.eye(h,k=-1);self.C=np.vstack([np.eye(h),self.D])
        self.slew_weight=.001/(b.SLEW*b.DT)**2
        self.H=2*(self.G.T@self.WG+.01/b.LIMIT**2*np.eye(h)+self.slew_weight*self.D.T@self.D)

    def reset(self,n):
        super().reset(n);self.optimization_failures=0;self.max_solver_residual=0.;self.maximum_iterations=0

    def act(self,plant,ref):
        from scipy.optimize import minimize,LinearConstraint
        target,rate,_=ref.at(plant.elapsed[:,None]+b.DT*np.arange(1,self.h+1))
        desired=np.stack([target,rate],axis=-1).reshape(plant.n,2*self.h)
        free=np.column_stack([plant.theta,plant.omega])@self.F.T+plant.torque[:,None]*self.c
        gradients=2*(free-desired)@self.WG
        gradients[:,0]-=2*self.slew_weight*plant.torque
        for row in range(plant.n):
            low=np.r_[np.full(self.h,-b.LIMIT),np.full(self.h,-b.SLEW*b.DT)]
            high=np.r_[np.full(self.h,b.LIMIT),np.full(self.h,b.SLEW*b.DT)]
            low[self.h]+=plant.torque[row];high[self.h]+=plant.torque[row]
            start=np.r_[self.q[row,1:],self.q[row,-1]]
            previous=plant.torque[row]
            for k in range(self.h):
                start[k]=np.clip(start[k],max(-b.LIMIT,previous-b.SLEW*b.DT),min(b.LIMIT,previous+b.SLEW*b.DT));previous=start[k]
            # Dimensionless decision torques and a common objective scale
            # improve SLSQP conditioning without changing the convex optimum.
            g=gradients[row]*b.LIMIT/100.;H=self.H*b.LIMIT**2/100.
            result=minimize(lambda x:.5*x@H@x+g@x,start/b.LIMIT,jac=lambda x:H@x+g,
                            constraints=[LinearConstraint(self.C,low/b.LIMIT,high/b.LIMIT)],method='SLSQP',
                            options=dict(maxiter=120,ftol=1e-9,disp=False))
            solution=result.x*b.LIMIT;cq=self.C@solution
            violation=float(max(np.max(low-cq),np.max(cq-high),0.))
            if not result.success or violation>1e-7:self.optimization_failures+=1
            self.maximum_iterations=max(self.maximum_iterations,int(result.nit))
            self.max_solver_residual=max(self.max_solver_residual,violation)
            self.q[row]=solution if violation<=1e-7 else start
        return np.clip(self.q[:,0],-b.LIMIT,b.LIMIT)


def declare():
    data=dict(source_sha256=source_hash(),base_source_sha256=b.source_hash(),algorithm='PPO',seed=SEED,
              steps=STEPS,formal_seeds=b.FORMAL_SEEDS,formal_steps=STEPS,checkpoint=CHECKPOINT,
              residual_limit=RESIDUAL,learning_rate=3e-5,n_steps=256,n_epochs=5,gae_lambda=.98,target_kl=.01,
              actor='Tanh MLP 128x128; all actor linear biases fixed zero; final weights zero initialized',
              input='same as initial development, except final elapsed feature replaced by zero for odd physical symmetry',
              test_responses_used=False,selection='unchanged: fewest physical failures, lowest mean_objective, earliest checkpoint',
              mpc='same convex QP and 30-step preview, SLSQP maxiter120 ftol1e-9; constraint residual and optimizer failures retained')
    path=ROOT/'01_实验方案'/'开发修正追加声明.json'
    if path.exists() and b.read_json(path)!=json.loads(json.dumps(data)):raise RuntimeError('revision declaration changed')
    b.write_json(path,data)


def build(env,seed):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,learning_rate=3e-5,gamma=b.GAMMA,
              n_steps=256,batch_size=256,n_epochs=5,gae_lambda=.98,target_kl=.01,
              policy_kwargs=dict(net_arch=[128,128],log_std_init=-2.5))
    with torch.no_grad():
        for layer in model.policy.mlp_extractor.policy_net.modules():
            if isinstance(layer,torch.nn.Linear) and layer.bias is not None:
                layer.bias.zero_();layer.bias.requires_grad_(False)
        model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_();model.policy.action_net.bias.requires_grad_(False)
    return model


def train(seed,stage):
    from stable_baselines3.common.logger import configure
    if (ROOT/'04_独立测试'/'freeze.json').exists():raise RuntimeError('test freeze exists')
    if stage=='development' and seed!=SEED:raise ValueError('seed')
    if stage=='formal':
        decision=b.read_json(ROOT/'03_追加训练'/'decision.json')
        if not decision['formal_eligible'] or seed not in b.FORMAL_SEEDS:raise ValueError('formal unavailable')
    declare()
    directory=ROOT/'03_追加训练'/stage/'PPO'/f'seed_{seed}'
    path=directory/'training_metadata.json'
    if path.exists():
        old=b.read_json(path)
        if old['status']=='complete' and old['source_sha256']==source_hash():return old
        raise RuntimeError('existing attempt retained')
    directory.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,directory/'revision_source.py');shutil.copy2(b.__file__,directory/'base_source.py')
    parameters=b.read_json(ROOT/'02_传统控制'/'PID_feedforward_preview.json')['parameters']
    env=b.make_env(seed,parameters,directory);env.interface=Interface(parameters)
    model=build(env,seed);model.set_logger(configure(str(directory/'optimizer_logs'),['csv']))
    metadata=dict(status='running',algorithm='PPO',seed=seed,stage=stage,source_sha256=source_hash(),base_source_sha256=b.source_hash(),
                  requested_steps=STEPS,actual_steps=0,base_parameters=parameters,checkpoints=[],best=None,test_responses_used=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),replay_buffer_saved=False)
    b.write_json(path,metadata);start=time.monotonic();val=b.scenes('validation')
    try:
        for step in range(0,STEPS+1,CHECKPOINT):
            if step:
                model.learn(CHECKPOINT,reset_num_timesteps=step==CHECKPOINT,log_interval=1)
                if model.num_timesteps!=step or env.total_steps!=step:raise RuntimeError('steps differ')
            checkpoint=directory/'checkpoints'/f'steps_{step:06d}';checkpoint.mkdir(parents=True,exist_ok=False)
            model.save(checkpoint/'model.zip')
            result=b.evaluate(val,Controller(model,parameters));stats=b.summary(result['rows'])
            b.write_csv(checkpoint/'validation.csv',result['rows'])
            item=dict(steps=step,model_path=b.relative(checkpoint/'model.zip'),model_sha256=b.sha256(checkpoint/'model.zip'),validation=stats,optimizer_updates=int(model._n_updates))
            metadata['checkpoints'].append(item)
            if step==0:metadata['initial']=item
            if metadata['best'] is None or (*b.rank(stats),-step)>(*b.rank(metadata['best']['validation']),-metadata['best']['steps']):
                shutil.copy2(checkpoint/'model.zip',directory/'best.zip');metadata['best']=dict(item,checkpoint_model_path=item['model_path'],model_path=b.relative(directory/'best.zip'))
            metadata['actual_steps']=int(model.num_timesteps);metadata['optimizer_updates']=int(model._n_updates)
            metadata['trained_checkpoint_improved_initial']=metadata['best']['steps']>0;metadata['elapsed_seconds']=time.monotonic()-start
            if source_hash()!=metadata['source_sha256']:raise RuntimeError('source changed')
            b.write_json(path,metadata)
            print(json.dumps(dict(event='revision_checkpoint',seed=seed,stage=stage,steps=step,validation=stats,elapsed_seconds=metadata['elapsed_seconds']),ensure_ascii=False),flush=True)
        model.save(directory/'final.zip');metadata['status']='complete'
    except BaseException as error:metadata['status']='incomplete';metadata['error']=repr(error);raise
    finally:
        metadata['source_sha256_after']=source_hash();metadata['actual_steps']=int(model.num_timesteps);metadata['elapsed_seconds']=time.monotonic()-start
        metadata['finished_utc']=datetime.now(timezone.utc).isoformat();b.write_json(path,metadata);env.close();model.logger.close()
    return metadata


def decide():
    path=ROOT/'03_追加训练'/'development'/'PPO'/f'seed_{SEED}'/'training_metadata.json';data=b.read_json(path)
    if data['status']!='complete' or data['source_sha256']!=source_hash():raise RuntimeError('development incomplete')
    best=data['best'];eligible=best['steps']>0 and best['validation']['physical_failures']==0
    result=dict(formal_eligible=eligible,selected=dict(algorithm='PPO',reward_name='dense_tracking',best=best),source_sha256=source_hash(),test_responses_used=False)
    b.write_json(ROOT/'03_追加训练'/'decision.json',result);return result


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['declare','train','decide']);parser.add_argument('--seed',type=int,default=SEED);parser.add_argument('--stage',default='development');a=parser.parse_args()
    if a.command=='declare':declare()
    elif a.command=='train':train(a.seed,a.stage)
    else:print(json.dumps(decide(),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
