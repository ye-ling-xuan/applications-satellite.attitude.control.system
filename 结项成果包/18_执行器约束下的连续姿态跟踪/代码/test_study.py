"""Independent physical, predictive-control and real learner checks."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
from scipy.integrate import solve_ivp
import study as s


def test_reference_derivatives_and_preview_shape():
    records=s.draw_scenes(np.random.default_rng(11),5,'checks');ref=s.Reference(records)
    t=np.linspace(0.,10.,5);e=1e-5
    angle,rate,acc=ref.at(t)
    before,br,_=ref.at(t-e);after,ar,_=ref.at(t+e)
    np.testing.assert_allclose((after-before)/(2*e),rate,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose((ar-br)/(2*e),acc,rtol=1e-7,atol=1e-9)
    assert ref.at(t[:,None]+s.PREVIEW)[0].shape==(5,6)


def test_ramp_physics_matches_independent_ode_and_cost_quad():
    from scipy.integrate import quad
    rng=np.random.default_rng(12);records=s.draw_scenes(rng,20,'checks');ref=s.Reference(records);p=s.Plant(20);p.reset(ref)
    for _ in range(25):
        oldtheta=p.theta.copy();oldomega=p.omega.copy();oldtau=p.torque.copy();command=rng.uniform(-s.LIMIT,s.LIMIT,20)
        step=p.step(command)
        for i in range(20):
            end=np.clip(command[i],oldtau[i]-s.SLEW*s.DT,oldtau[i]+s.SLEW*s.DT)
            sol=solve_ivp(lambda t,y:[y[1],(oldtau[i]+(end-oldtau[i])*t/s.DT)/s.INERTIA],(0,s.DT),[oldtheta[i],oldomega[i]],rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose([p.theta[i],p.omega[i]],sol.y[:,-1],atol=1e-10)
            cost=quad(lambda t:(oldtau[i]+(end-oldtau[i])*t/s.DT)**2,0,s.DT,epsabs=1e-12)[0]
            assert abs(cost-step['actual_cost'][i])<1e-11
        assert np.max(np.abs(p.torque))<=s.LIMIT+1e-12
        assert np.max(np.abs(p.torque-oldtau))<=s.SLEW*s.DT+1e-12


def test_mpc_prediction_matrices_match_physical_rollout_and_constraints():
    rng=np.random.default_rng(13);ref=s.Reference(s.draw_scenes(rng,3,'checks'));p=s.Plant(3);p.reset(ref)
    m=s.MPC();m.reset(3);start=np.column_stack([p.theta,p.omega]);old=p.torque.copy()
    targettorque=np.repeat(old[:,None],m.h,axis=1)
    for j in range(m.h):
        targettorque[:,j]=np.clip((old if j==0 else targettorque[:,j-1])+rng.uniform(-.004,.004,3),-s.LIMIT,s.LIMIT)
    predicted=start@m.F.T+old[:,None]*m.c+targettorque@m.G.T
    for k in range(m.h):
        p.step(targettorque[:,k])
        np.testing.assert_allclose(predicted[:,2*k],p.theta,atol=1e-11)
        np.testing.assert_allclose(predicted[:,2*k+1],p.omega,atol=1e-11)
    result=s.evaluate(ref.records,s.MPC())
    assert all(not row['failed'] for row in result['rows'])
    assert result['mpc_max_solver_residual'] is not None
    assert result['rows'][0]['max_slew']<=s.SLEW+1e-10


def test_zero_residual_and_training_deployment_features_identical():
    params=dict(kp=.4,kd=.9,ki=.03,feedforward=1.,preview_s=.2)
    records=s.draw_scenes(np.random.default_rng(14),8,'checks');ref=s.Reference(records);plant=s.Plant(8);plant.reset(ref)
    interface=s.PolicyInterface(params);interface.reset(8);pid=s.PID(params);pid.reset(8)
    for _ in range(50):
        features=interface.features(plant,ref)
        assert features.shape==(8,20) and features.dtype==np.float32
        command=interface.command(plant,ref,np.zeros((8,1)))
        np.testing.assert_array_equal(command,pid.act(plant,ref))
        np.testing.assert_array_equal(interface.pid.integral,pid.integral)
        plant.step(command)
    with tempfile.TemporaryDirectory() as tmp:
        env=s.make_env(18001,params,Path(tmp));features=env.reset();external=s.PolicyInterface(params);external.reset(8)
        try:
            for _ in range(20):
                np.testing.assert_array_equal(external.features(env.plant,env.ref),features)
                expected=external.command(env.plant,env.ref,np.full((8,1),.1,np.float32))
                features,_,done,_=env.step(np.full((8,1),.1,np.float32))
                assert not np.any(done)
                np.testing.assert_allclose(env.interface.previous_command,expected,atol=1e-9)
        finally:env.close()


@pytest.mark.parametrize('algorithm',['TD3','PPO'])
def test_real_learner_optimizer_and_save_load(algorithm):
    import stable_baselines3 as sb3
    from stable_baselines3.common.logger import configure
    params=dict(kp=.4,kd=.9,ki=0.,feedforward=1.,preview_s=.2)
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp);env=s.make_env(18001,params,directory)
        try:
            model=s.build_model(algorithm,env,18001,1024);model.learning_starts=16
            model.set_logger(configure(None,[]));features=env.reset();actions,_=model.predict(features,deterministic=True)
            np.testing.assert_array_equal(actions,0.)
            model.learn(1024)
            assert model.num_timesteps==env.total_steps==1024 and model._n_updates>0
            path=directory/'model.zip';model.save(path);restored=getattr(sb3,algorithm).load(path,device='cpu')
            actual,_=restored.predict(features,deterministic=True);expected,_=model.predict(features,deterministic=True)
            np.testing.assert_array_equal(actual,expected)
            optimizer=restored.actor.optimizer if algorithm=='TD3' else restored.policy.optimizer
            assert optimizer.state_dict()['state']
        finally:env.close()


def test_absorbing_failure_full_length_and_penalty():
    ref=s.Reference(s.draw_scenes(np.random.default_rng(19),1,'checks'));p=s.Plant(1);p.reset(ref)
    p.theta[:]=s.FAIL_ANGLE+.1
    step=p.step(np.array([0.]));assert step['failed'][0] and not step['done'][0]
    frozen=[p.theta.copy(),p.omega.copy(),p.torque.copy()]
    for _ in range(299):
        step=p.step(np.array([s.LIMIT]));np.testing.assert_array_equal([p.theta,p.omega,p.torque],frozen)
        assert s.normalized_interval_cost(np.array([0.]),np.array([0.]),step)[0]==201.
    assert step['done'][0] and p.count[0]==300
