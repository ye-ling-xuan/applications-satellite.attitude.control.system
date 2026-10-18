"""Independent flexible-body ODE, conservation and objective checks."""
import unittest
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from scipy.integrate import solve_ivp

from flexible_study import (Plant,Controller,DT,JB,JF,LIMIT,STEPS,TOL,
                            independent_transition_matrices,reward,
                            episode_metrics,shaper_target)


class FlexiblePhysicsTests(unittest.TestCase):
    def test_independent_joint_ode_energy_and_angular_momentum(self):
        frequency = np.array([.4,.57,.8])
        damping = np.array([.01,.025,.04])
        plant = Plant(3)
        plant.reset([.1,-.3,.2],[.03,.01,-.02],frequency,damping)
        # Exercise nonzero initial flexure, as well as ordinary rest initial data.
        plant.state[:,2:] = np.array([[.003,-.01],[-.005,.02],[.001,-.03]])
        reference = np.column_stack((plant.state,np.zeros((3,2))))
        inv = 1/JB+1/JF
        k = (2*np.pi*frequency)**2/inv
        c = 2*damping*(2*np.pi*frequency)/inv

        def energy(state):
            return .5*JB*state[:,1]**2+.5*JF*(state[:,1]+state[:,3])**2+.5*k*state[:,2]**2

        initial_energy = energy(plant.state)
        initial_momentum = (JB+JF)*plant.state[:,1]+JF*plant.state[:,3]
        impulse = np.zeros(3)
        rng = np.random.default_rng(16001)
        for _ in range(30):
            requested = rng.uniform(-.7,.7,3)
            command = np.clip(requested,-LIMIT,LIMIT)
            result = plant.step(requested)
            self.assertFalse(np.any(result['failed']))
            for i,u in enumerate(command):
                def rhs(t,s):
                    del t
                    elastic = k[i]*s[2]+c[i]*s[3]
                    return [s[1],(u+elastic)/JB,s[3],-u/JB-elastic*inv,
                            u*s[1],c[i]*s[3]**2]
                solution = solve_ivp(rhs,(0.,DT),reference[i],method='DOP853',rtol=2e-12,atol=2e-14)
                self.assertTrue(solution.success)
                reference[i] = solution.y[:,-1]
            assert_allclose(plant.state,reference[:,:4],rtol=3e-11,atol=5e-13)
            assert_allclose(energy(plant.state)-initial_energy,
                            reference[:,4]-reference[:,5],rtol=5e-11,atol=3e-13)
            impulse += command*DT
            assert_allclose((JB+JF)*plant.state[:,1]+JF*plant.state[:,3]-initial_momentum,
                            impulse,rtol=0.,atol=5e-16)
            assert_allclose(result['cost'],command**2*DT,rtol=0.,atol=0.)

    def test_independent_state_matrix_exponential_for_full_maneuver(self):
        data = [dict(theta=.4,omega=-.02,frequency=f,damping=z)
                for f,z in [(.4,.01),(.6,.025),(.8,.04)]]
        f,g = independent_transition_matrices(data)
        plant = Plant(3)
        plant.reset(.4,-.02,[d['frequency'] for d in data],[d['damping'] for d in data])
        reference = plant.state.copy()
        for command in [np.array([.2,-.2,.1])]*12+[np.array([-.1,.2,-.2])]*15:
            reference = np.einsum('nij,nj->ni',f,reference)+g*command[:,None]
            result = plant.step(command)
            self.assertFalse(np.any(result['failed']))
            assert_allclose(plant.state,reference,rtol=0.,atol=2e-14)

    def test_failure_is_absorbing_but_horizon_is_fixed(self):
        plant = Plant(2)
        plant.reset(0.,0.,.4,.01)
        plant.state[0,2] = np.deg2rad(11.)
        result = plant.step(np.array([0.,.1]))
        self.assertTrue(result['new_failure'][0])
        self.assertFalse(np.any(result['done']))
        frozen = plant.state[0].copy()
        for _ in range(STEPS-1):
            result = plant.step(np.array([.5,0.]))
            self.assertFalse(result['new_failure'][0])
            self.assertEqual(result['command'][0],0.)
            assert_array_equal(plant.state[0],frozen)
        assert_array_equal(result['done'],True)

    def test_masked_reset_has_no_parameter_or_state_leakage(self):
        plant = Plant(3)
        plant.reset([.1,.2,.3],[.01,.02,.03],[.4,.5,.8],[.01,.02,.04])
        plant.step([.1,.2,.3])
        original = plant.state[1].copy()
        plant.reset([-.2,-.4],[0.,.01],[.7,.6],[.02,.03],np.array([True,False,True]))
        assert_array_equal(plant.state[1],original)
        assert_array_equal(plant.count,[0,1,0])
        assert_array_equal(plant.frequency,[.7,.5,.6])


class ObjectiveTests(unittest.TestCase):
    def test_failure_every_interval_is_more_costly_than_any_live_interval(self):
        plant = Plant(2)
        plant.reset(0.,0.,.6,.025)
        plant.state[:] = 1e5
        plant.count[:] = 200
        plant.failed[1] = True
        values = reward(plant,dict(command=np.array([LIMIT,LIMIT])))
        assert_allclose(values,[-40*DT,-50*DT],rtol=0.,atol=0.)
        self.assertLess(values[1],values[0])

    def test_whole_postdeadline_window_and_four_state_tolerances(self):
        states = np.zeros((6,STEPS+1,4))
        commands = np.zeros((6,STEPS))
        for i in range(4):
            states[i+1,160+10*i,i] = TOL[i]*1.001
        failed = np.array([False,False,False,False,False,True])
        assert_array_equal(episode_metrics(states,commands,failed)['success'],[True,False,False,False,False,False])

    def test_shaper_finishes_reference_transition_and_has_no_true_frequency(self):
        values = shaper_target(np.ones(3),np.array([0.,.9,2.]))
        self.assertTrue(0.<values[0]<1.)
        self.assertTrue(0.<values[1]<values[0])
        self.assertEqual(values[2],0.)

    def test_observation_contains_measured_flexure_but_not_modal_parameters(self):
        first,second = Plant(2),Plant(2)
        first.reset(.1,.01,[.4,.8],[.01,.04])
        second.reset(.1,.01,[.8,.4],[.04,.01])
        settings = dict(method='PID',kp=.3,kd=.9,ki=.001)
        a,b = Controller(settings,2),Controller(settings,2)
        a.reset(first);b.reset(second)
        self.assertEqual(a.observation(first).shape,(2,7))
        assert_array_equal(a.observation(first),b.observation(second))


if __name__=='__main__':
    unittest.main()
