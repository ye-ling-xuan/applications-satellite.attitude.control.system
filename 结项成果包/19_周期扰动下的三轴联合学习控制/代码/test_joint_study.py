import unittest
import numpy as np
import joint_study as s

class JointTests(unittest.TestCase):
    def test_observation_and_action_are_joint_three_axis(self):
        p=s.Plant(s.random_scenes(np.random.default_rng(19000),3));c=s.Controller(dict(kp=2.,kd=2.))
        obs=p.measured();c.reset(3,obs);f=s.features(obs,c,np.zeros((3,3)),np.zeros(3))
        self.assertEqual(obs.shape,(3,15));self.assertEqual(f.shape,(3,22));self.assertEqual(c.act(obs).shape,(3,3))
        self.assertTrue(np.all(np.isfinite(f)))
    def test_zero_learned_residual_equals_full_three_axis_feedback(self):
        class Model:
            def predict(self,obs,deterministic=True):return np.zeros((len(obs),3)),None
        p=s.Plant(s.random_scenes(np.random.default_rng(19001),2));a=s.Controller(dict(kp=2.,kd=2.,ki=.1));b=s.LearnedController(dict(kp=2.,kd=2.,ki=.1),Model())
        a.reset(2,p.measured());b.reset(2,p.measured())
        for k in range(30):
            obs=p.measured();first=a.act(obs);second=b.act(obs,np.full(2,k*s.DT));np.testing.assert_array_equal(first,second);p.step(first)
    def test_gyroscopic_feedforward_has_correct_sign(self):
        c=s.Controller({});obs=np.zeros((1,15));omega=np.array([[.2,-.3,.4]]);obs[:,3:6]=omega;c.reset(1,obs)
        np.testing.assert_allclose(c.act(obs),np.cross(omega,s.INERTIA*omega),atol=1e-15)
    def test_real_three_action_TD3_smoke(self):
        _,TD3,_,_=s.imports();env=s.make_environment(dict(kp=2.,kd=2.),19100,n=2)
        model=TD3('MlpPolicy',env,buffer_size=128,learning_starts=8,batch_size=8,train_freq=(1,'step'),gradient_steps=1,policy_kwargs=dict(net_arch=[16,16]),seed=1,device='cpu')
        model.learn(total_timesteps=32);self.assertGreater(model._n_updates,0)
        action,_=model.predict(env.reset(),deterministic=True);self.assertEqual(action.shape,(2,3));env.close()

if __name__=='__main__':unittest.main()
