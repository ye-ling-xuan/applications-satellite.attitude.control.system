from pathlib import Path
import tempfile
import numpy as np
import scipy.linalg
import scipy.integrate
import study as b
import revision as r

_runtime_warmup=float(np.trace(np.eye(100)@np.eye(100)))


def test_actor_odd_symmetry_before_and_after_real_training():
    from stable_baselines3.common.logger import configure
    import torch
    params=dict(kp=.2649456852206539,kd=.6429328040149732,ki=0.,feedforward=.5,preview_s=.6)
    with tempfile.TemporaryDirectory() as tmp:
        env=b.make_env(18202,params,Path(tmp));env.interface=r.Interface(params)
        try:
            model=r.build(env,18202);model.set_logger(configure(None,[]))
            obs=np.random.default_rng(21).normal(0.,.5,(40,20)).astype(np.float32);obs[:,-1]=0.
            a,_=model.predict(obs,deterministic=True);np.testing.assert_array_equal(a,0.)
            model.learn(2048)
            a,_=model.predict(obs,deterministic=True);other,_=model.predict(-obs,deterministic=True)
            np.testing.assert_allclose(other,-a,atol=1e-7,rtol=1e-6)
            for layer in model.policy.mlp_extractor.policy_net.modules():
                if isinstance(layer,torch.nn.Linear):
                    np.testing.assert_array_equal(layer.bias.detach().numpy(),0.)
                    assert not layer.bias.requires_grad
            assert model._n_updates>0
            state=env.reset();assert np.all(state[:,-1]==0.)
        finally:env.close()


def test_slsqp_mpc_is_feasible_and_physical_reference_stays_bounded():
    records=b.draw_scenes(np.random.default_rng(13),3,'checks');controller=r.ExactMPC();result=b.evaluate(records,controller)
    assert all(not row['failed'] for row in result['rows'])
    assert controller.max_solver_residual<1e-7
    assert result['rows'][0]['max_slew']<=b.SLEW+1e-10

