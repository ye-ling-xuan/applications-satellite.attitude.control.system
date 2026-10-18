"""Independent physical/control checks with synthetic scenes only."""
import math
import unittest
import numpy as np
import disturbance_study as s


def example():
    return dict(sample_id=0,angle_deg=8.,omega_deg_s=-.5,phase_rad=.7,frequency_rad_s=.5,
                amplitude_1_Nm=.035,amplitude_2_Nm=.01,attitude_coefficient_Nm=.02,rate_coefficient_Nm=.005)


class PhysicsTests(unittest.TestCase):
    def test_RK4_against_independent_small_step_ODE(self):
        r=example();plant=s.Plant([r]);command=.17
        x=np.array([np.radians(r['angle_deg']),np.radians(r['omega_deg_s']),0.])
        h=s.DT/1000
        def derivative(t,x):
            phase=r['phase_rad']+r['frequency_rad_s']*t
            disturbance=r['amplitude_1_Nm']*math.sin(phase)+r['amplitude_2_Nm']*math.sin(2*phase+.4)+r['attitude_coefficient_Nm']*math.sin(2*x[0])+r['rate_coefficient_Nm']*math.tanh(x[1]/.05)
            return np.array([x[1],(x[2]+disturbance)/s.INERTIA,(command-x[2])/s.LAG])
        for k in range(1000):
            t=k*h;a=derivative(t,x);b=derivative(t+h/2,x+h*a/2)
            c=derivative(t+h/2,x+h*b/2);d=derivative(t+h,x+h*c)
            x+=h*(a+2*b+2*c+d)/6
        result=plant.step(np.array([command]))
        np.testing.assert_allclose([plant.theta[0],plant.omega[0],plant.tau[0]],x,atol=5e-9,rtol=0.)
        nodes,weights=np.polynomial.legendre.leggauss(32)
        times=(nodes+1)*s.DT/2
        integral=float(weights@((command*(1-np.exp(-times/s.LAG)))**2)*s.DT/2)
        self.assertAlmostEqual(result['actual_cost'][0],integral,places=15)

    def test_saturation_and_failure_full_horizon(self):
        r=example();r['angle_deg']=89.9;r['omega_deg_s']=20.
        plant=s.Plant([r]);plant.step(np.array([100.]))
        self.assertTrue(plant.failed[0]);frozen=plant.theta.copy()
        for _ in range(s.STEPS-1):plant.step(np.array([100.]))
        np.testing.assert_array_equal(plant.theta,frozen)
        self.assertEqual(plant.count[0],s.STEPS)


class ControllerTests(unittest.TestCase):
    def test_zero_learned_residual_is_exact_same_PID(self):
        class Model:
            def predict(self,obs,deterministic=True):return np.zeros((len(obs),1)),None
        params=dict(kp=2.,kd=1.5,ki=.2)
        first=s.Controller(params);second=s.LearnedController(params,Model())
        plant=s.Plant([example()]);first.reset(1,plant.measured());second.reset(1,plant.measured())
        for k in range(100):
            obs=plant.measured();a=first.act(obs);b=second.act(obs,np.array([k*s.DT]))
            np.testing.assert_allclose(a,b,atol=0.,rtol=0.)
            plant.step(a)

    def test_observer_estimates_constant_unknown_external_torque(self):
        controller=s.Controller(dict(kp=0.,kd=0.,ki=0.,filter_time=.025,predict=0.),'observer')
        obs=np.array([[0.,0.,0.,1.,.5]]);controller.reset(1,obs)
        rate=0.;tau=0.;disturbance=.03
        for _ in range(120):
            u=controller.act(obs)[0]
            mean=u+(tau-u)*s.LAG*(1-np.exp(-s.DT/s.LAG))/s.DT
            rate+=(mean+disturbance)*s.DT/s.INERTIA
            tau=u+(tau-u)*np.exp(-s.DT/s.LAG)
            obs[0,1]=rate
        self.assertAlmostEqual(controller.estimate[0],disturbance,places=12)

    def test_harmonic_feedforward_includes_lag_phase_advance(self):
        coefficients=[.03,-.01,.005,.002];phase=.8;freq=.4
        obs=np.array([[0.,0.,np.sin(phase),np.cos(phase),freq]])
        controller=s.Controller({},'harmonic',coefficients);controller.reset(1,obs)
        u=controller.act(obs)[0]
        expected=-(coefficients[0]*np.sin(phase)+coefficients[1]*np.cos(phase)+coefficients[2]*np.sin(2*phase)+coefficients[3]*np.cos(2*phase))
        expected-=s.LAG*freq*(coefficients[0]*np.cos(phase)-coefficients[1]*np.sin(phase)+2*coefficients[2]*np.cos(2*phase)-2*coefficients[3]*np.sin(2*phase))
        self.assertAlmostEqual(u,expected,places=15)

    def test_real_continuous_TD3_smoke_and_terminal_reset(self):
        _,_,TD3,_,_=s._imports()
        env=s.make_environment(dict(kp=2.,kd=2.,ki=.1),17300,n=2)
        model=TD3('MlpPolicy',env,buffer_size=128,learning_starts=8,batch_size=8,
                  train_freq=(1,'step'),gradient_steps=1,policy_kwargs=dict(net_arch=[16,16]),seed=1,device='cpu')
        model.learn(total_timesteps=32)
        self.assertGreater(model._n_updates,0)
        obs=env.reset();self.assertEqual(obs.shape,(2,8));self.assertTrue(np.all(np.isfinite(obs)))
        for _ in range(s.STEPS):obs,reward,done,infos=env.step(np.zeros((2,1)))
        self.assertTrue(np.all(done));self.assertFalse(infos[0]['TimeLimit.truncated'])
        self.assertEqual(env.plant.count[0],0);env.close()


if __name__=='__main__':unittest.main()
