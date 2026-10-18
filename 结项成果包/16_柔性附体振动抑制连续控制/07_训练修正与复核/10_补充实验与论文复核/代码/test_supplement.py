"""Consequential checks of noise/truth separation and old implementation compatibility."""
import unittest
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
import supplement as x

class SupplementTests(unittest.TestCase):
    def test_freeze_and_new_samples(self):
        x.assert_frozen()
        old=[tuple(r[k] for k in ('theta','omega','damping')) for split in ('development','validation','test') for r in x.read(x.OLD/f'01_实验方案/{split}.json')]
        new=[]
        for group in x.GROUPS:
            data=x.scenes(group)
            self.assertEqual(len(data),640)
            self.assertEqual(len(set((r['id'],r['frequency']) for r in data)),640)
            for r in data[:128]:
                values=tuple(r[k] for k in ('theta','omega','damping'))
                self.assertNotIn(values,old);self.assertNotIn(values,new);new.append(values)

    def test_zero_noise_matches_original_all_nine_controllers(self):
        data=[dict(theta=.1,omega=-.003,damping=.025,frequency=f,q=0.,qdot=0.) for f in (.4,.6,.8)]
        noise=np.zeros((3,x.s.STEPS+1,4))
        for item in x.registry():
            model=x.load(item);actual=x.rollout(data,item,noise,model)
            metrics,states,commands=x.s.simulate(data,item['settings'],model,True)
            assert_array_equal(actual['state'],states);assert_array_equal(actual['command'],commands)
            for k,value in metrics.items():assert_array_equal(actual[k],value)

    def test_nonzero_initial_vibration_is_preserved(self):
        data=[dict(theta=.01,omega=.001,damping=.02,frequency=.6,q=.0007,qdot=-.003)]
        item=dict(name='zero',settings=dict(method='PID',kp=0.,kd=0.,ki=0.))
        actual=x.rollout(data,item,np.zeros((1,x.s.STEPS+1,4)))
        assert_array_equal(actual['state'][:,0],x.initial_state(data))
        f,g=x.matrices(data);expected=(f@x.initial_state(data)[:,:,None])[:,:,0]
        assert_allclose(actual['state'][:,1],expected,rtol=0,atol=1e-14)

    def test_noise_affects_measurement_and_command_not_truth_acceptance(self):
        data=[dict(theta=0.,omega=0.,damping=.02,frequency=.6,q=0.,qdot=0.)]
        noise=np.ones((1,x.s.STEPS+1,4))*.01
        zero=dict(name='zero',settings=dict(method='PID',kp=0.,kd=0.,ki=0.))
        result=x.rollout(data,zero,noise)
        assert_array_equal(result['state'],0.);assert_array_equal(result['measurement'],noise)
        self.assertTrue(result['success'][0])
        self.assertFalse(np.all(np.abs(result['measurement'][:,160:])<=x.s.TOL))
        nonzero=dict(name='feedback',settings=dict(method='PID',kp=.6,kd=1.7,ki=0.))
        no_noise=x.rollout(data,nonzero,np.zeros_like(noise));with_noise=x.rollout(data,nonzero,noise)
        self.assertNotEqual(no_noise['command'][0,0],with_noise['command'][0,0])

    def test_shared_noise_matches_frozen_draw(self):
        expected=np.random.default_rng(x.NOISE_SEED).normal(0.,x.NOISE_STD,(640,x.s.STEPS+1,4))
        assert_array_equal(x.noise_for(x.GROUPS[2]),expected)
        self.assertTrue(np.all(x.noise_for(x.GROUPS[0])==0.))
        self.assertTrue(np.all(x.noise_for(x.GROUPS[1])==0.))

if __name__=='__main__':unittest.main()
