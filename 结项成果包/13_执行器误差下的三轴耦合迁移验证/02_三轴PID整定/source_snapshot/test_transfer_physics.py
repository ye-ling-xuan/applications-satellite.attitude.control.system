"""Independent predeclared representative physics and migration checks."""
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.spatial.transform import Rotation
import transfer_study as s


def independent_ode(t,y,u,parameter,kind):
    q=y[:4];w=y[4:7];a=y[7:10]
    torque=a if kind=='lag' and parameter>0. else u
    wx,wy,wz=w
    # Matrix quaternion kinematics, independently written from production.
    matrix=np.array([[0.,-wx,-wy,-wz],[wx,0.,wz,-wy],
                     [wy,-wz,0.,wx],[wz,wy,-wx,0.]])
    dq=.5*matrix@q
    dw=np.linalg.solve(np.diag(s.J),torque-np.cross(w,np.diag(s.J)@w))
    da=(u-a)/parameter if kind=='lag' and parameter>0. else np.zeros(3)
    return np.r_[dq,dw,da,np.dot(torque,torque)]


def independent_interval(q,w,a,u,p,kind):
    result=solve_ivp(independent_ode,(0.,s.DT),np.r_[q,w,a,0.],
                     args=(u,p,kind),rtol=3e-12,atol=2e-14,method='DOP853')
    assert result.success
    return result.y[:,-1]


class RigidBodyTests(unittest.TestCase):
    def test_rotation_error_matches_scipy_and_has_single_axis_sign(self):
        q=s.euler_to_quaternion(np.radians([[15.,0.,0.],[20.,-30.,40.]]))
        e=s.quaternion_error(q)
        np.testing.assert_allclose(e,Rotation.from_quat(q[:,[1,2,3,0]]).inv().as_rotvec(),atol=1e-14)
        self.assertAlmostEqual(e[0,0],-np.deg2rad(15.))
        np.testing.assert_allclose(s.quaternion_error(-q),e,atol=1e-14)

    def test_representative_interval_vs_independent_solve_ivp(self):
        cases=[('lag',.05,[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[.4,0.,0.]),
               ('lag',.30,[25.,-30.,18.],[.12,-.08,.07],[.2,-.25,.15],[-.5,.35,-.45]),
               ('lag',0.,[30.,20.,-25.],[.2,.1,-.15],[0.,0.,0.],[-.2,.4,.1]),
               ('delay',0.,[20.,-35.,22.],[.18,-.13,.1],[0.,0.,0.],[.5,-.3,.2])]
        for kind,p,euler,w,a,u in cases:
            with self.subTest(kind=kind,p=p):
                q=s.euler_to_quaternion(np.radians([euler]))[0]
                w,a,u=map(np.asarray,(w,a,u))
                q1,w1,a1,c=s.rigid_interval(q[None],w[None],a[None],u[None],np.array([p]),kind)
                independent=independent_interval(q,w,a,u,p,kind)
                # Numerical budget: much smaller than 1deg/.5deg/s acceptance.
                np.testing.assert_allclose(q1[0],independent[:4],atol=1e-7,rtol=0.)
                np.testing.assert_allclose(w1[0],independent[4:7],atol=1e-6,rtol=0.)
                if kind=='lag' and p>0.:
                    np.testing.assert_allclose(a1[0],independent[7:10],atol=2e-12)
                self.assertAlmostEqual(float(np.sum(c)),independent[10],places=11)

    def test_rk4_converges_with_more_substeps(self):
        q=s.euler_to_quaternion(np.radians([[25.,-30.,18.]]))
        w=np.array([[.12,-.08,.07]]);a=np.array([[.2,-.25,.15]]);u=np.array([[-.5,.35,-.45]])
        expected=independent_interval(q[0],w[0],a[0],u[0],.05,'lag')
        errors=[]
        for substeps in (1,2,4):
            q1,w1,_,_=s.rigid_interval(q,w,a,u,np.array([.05]),'lag',substeps=substeps)
            errors.append(np.max(np.abs(w1[0]-expected[4:7])))
        self.assertLess(errors[1],errors[0]/10)
        self.assertLess(errors[2],errors[1]/10)

    def test_lag_cost_reversals_vs_quadrature(self):
        for p in (.05,.30,.50):
            q=np.array([[1.,0.,0.,0.]])
            a=np.array([[.5,-.4,.25]]);u=np.array([[-.5,.4,-.25]])
            _,_,_,cost=s.rigid_interval(q,np.zeros((1,3)),a,u,np.array([p]),'lag')
            expected=[quad(lambda t:(u[0,k]+(a[0,k]-u[0,k])*np.exp(-t/p))**2,0.,s.DT,
                           epsabs=1e-13,epsrel=1e-13)[0] for k in range(3)]
            np.testing.assert_allclose(cost[0],expected,atol=2e-15)

    def test_free_rigid_body_has_coupling_and_conserves_energy(self):
        q=np.array([[1.,0.,0.,0.]]);w=np.array([[.4,.3,-.2]])
        initial=w.copy();energy=.5*np.sum(s.J*w*w)
        for _ in range(80):
            q,w,_,_=s.rigid_interval(q,w,np.zeros((1,3)),np.zeros((1,3)),np.array([0.]),'delay')
        self.assertGreater(np.max(np.abs(w-initial)),.01)
        self.assertAlmostEqual(.5*np.sum(s.J*w*w),energy,places=10)
        self.assertAlmostEqual(np.linalg.norm(q),1.,places=14)


class QueueAndControllersTests(unittest.TestCase):
    def _scene(self,p=0.):
        return dict(euler_deg=[0.,0.,0.],omega_deg_s=[0.,0.,0.],parameter_s=p)

    def test_delay_zero_one_and_six_steps(self):
        for m in (0,1,6):
            with self.subTest(m=m):
                plant=s.Plant('delay',[self._scene(m*s.DT)])
                commands=[]
                for k in range(10):
                    command=np.array([[.4 if k<4 else -.3,.1,-.2]])
                    commands.append(command[0].copy())
                    step=plant.step(command)
                    expected=commands[k-m] if k>=m else np.zeros(3)
                    np.testing.assert_allclose(step['tau_start'][0],expected,atol=1e-14)
                    self.assertAlmostEqual(np.sum(step['actual_cost_axis']),np.dot(expected,expected)*s.DT)

    def test_commands_are_clipped_before_physics_and_cost(self):
        plant=s.Plant('lag',[self._scene(0.)])
        step=plant.step(np.array([[3.,-2.,.1]]))
        np.testing.assert_allclose(step['command'][0],[.5,-.5,.1])
        np.testing.assert_allclose(step['actual_cost_axis'][0],np.array([.25,.25,.01])*s.DT)

    def test_failure_is_sticky_and_padding_cost_zero(self):
        scene=self._scene(.1);scene['omega_deg_s']=[301.,0.,0.]
        plant=s.Plant('lag',[scene])
        first=plant.step(np.array([[.4,.2,.1]]))
        self.assertTrue(plant.failed[0]);self.assertGreater(np.sum(first['actual_cost_axis']),0.)
        q,w=plant.q.copy(),plant.omega.copy()
        for _ in range(3):
            step=plant.step(np.array([[.5,.5,.5]]))
            np.testing.assert_array_equal(plant.q,q);np.testing.assert_array_equal(plant.omega,w)
            np.testing.assert_array_equal(step['command'],np.zeros((1,3)))
            np.testing.assert_array_equal(step['actual_cost_axis'],np.zeros((1,3)))

    def test_geometric_pid_batch_matches_individuals(self):
        cs=[dict(kp=.8,kd=1.2,ki=.01,inertia_scaled=True),dict(kp=1.1,kd=1.7,ki=0.,inertia_scaled=False)]
        batch=s.GeometricPID(cs[0],cs);batch.reset(6,np.zeros((6,6)))
        separate=[s.GeometricPID(c) for c in cs]
        for c in separate:c.reset(3,np.zeros((3,6)))
        rng=np.random.default_rng(1314)
        for _ in range(20):
            obs=rng.normal(0.,.1,(3,6))
            expected=np.concatenate([c.act(obs,0.) for c in separate])
            np.testing.assert_allclose(batch.act(np.tile(obs,(2,1)),0.),expected,atol=1e-14)

    def test_axis_transfer_same_sign_and_independent_integrals(self):
        p=dict(kp=1.,kd=1.,ki=.1)
        c=s.AxisTransfer('lag','pid_transfer',p)
        obs=np.array([[-.1,.2,-.3,.01,.02,.03]])
        c.reset(1,obs)
        command=c.act(obs,np.array([0.]))
        np.testing.assert_allclose(command[0],[-.1105,.181,-.3315])
        self.assertIsNot(c.controllers[0].integral,c.controllers[1].integral)

    def test_deployed_sac_uses_three_independent_histories_and_original_base(self):
        class DummyModel:
            def predict(self,features,deterministic=True):
                return np.zeros((len(features),1)),None
        base=dict(kp=.3,kd=.9,ki=.001)
        with mock.patch('stable_baselines3.SAC.load',return_value=DummyModel()):
            c=s.AxisTransfer('lag','residual',model_path='dummy.zip',base_parameters=base)
        obs=np.array([[-.1,.2,-.3,.01,.02,.03]])
        c.reset(1,obs);c.act(obs,np.array([0.]))
        for axis,controller in enumerate(c.controllers):
            self.assertEqual(controller.interface.pid.parameters,base)
            self.assertEqual(controller.interface.history.valid[0],1)
            np.testing.assert_array_equal(controller.interface.history.measurements[0,-1],obs[0,[axis,axis+3]])
        self.assertIsNot(c.controllers[0].interface.history,c.controllers[1].interface.history)


class OutputVerificationTests(unittest.TestCase):
    def test_end_to_end_outputs_and_recomputed_costs_for_every_region(self):
        with tempfile.TemporaryDirectory(prefix='three_axis_audit_',dir=s.STUDY) as task_dir:
            with mock.patch.object(s,'STUDY',Path(task_dir)):
                states=[dict(sample_id=0,euler_deg=[12.,-8.,5.],omega_deg_s=[.2,-.1,.3])]
                s.write_json(s.STUDY/'01_实验方案'/'test_initial_states.json',states)
                specs=[dict(id=kind+'_fixture',kind=kind,method='retuned',stage='traditional',
                            parameters=dict(kp=1.,kd=1.8,ki=0.,inertia_scaled=True)) for kind in s.KINDS]
                s.write_json(s.STUDY/'04_独立测试'/'freeze.json',dict(
                    controllers=specs,source_and_input_hashes=s.input_hashes()))
                # An empty result set cannot pass verification.
                with self.assertRaises(RuntimeError):s.verify_saved()
                totals=s.run_test()
                self.assertEqual(len(totals),6)
                verification=s.verify_saved()
                self.assertEqual(verification['verified_output_groups'],6)
                # Resuming validates persisted artifact hashes before reuse.
                self.assertEqual(s.run_test(),totals)
                metrics=s.STUDY/'04_独立测试'/'lag_fixture'/'ideal'/'metrics.csv'
                metrics.write_text(metrics.read_text(encoding='utf-8-sig')+'\n',encoding='utf-8-sig')
                with self.assertRaises(RuntimeError):s.run_test()


if __name__=='__main__':unittest.main()
