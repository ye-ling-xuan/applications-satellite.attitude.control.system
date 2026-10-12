"""Deployment equivalence, quaternion physics and shared acceptance checks."""
import numpy as np
import pytest
from multi_algorithm_benchmark.envs import (
    make_env, BatchSatelliteEnv, quaternion_error, euler_to_quaternion)
from multi_algorithm_benchmark.controllers import PIDController, ModelController
from multi_algorithm_benchmark.metrics import compute_metrics, evaluate_batch
from multi_algorithm_benchmark.scenarios import make_scenarios,validation_scenarios


def test_single_batch_matches_original_gym_physics_and_action_limits():
    scene = {"theta_deg":23.,"omega_deg_s":-2.,"noise_seed":1}
    gym = make_env("single")
    gym.reset(seed=1,options=scene)
    batch = BatchSatelliteEnv("single",[scene])
    for action in (.7,-.2,3.,0.,-.8):
        _,_,_,_,info = gym.step([action])
        batch.step([[action]])
        assert np.degrees(batch.theta[0,0]) == pytest.approx(info["theta_deg"],abs=1e-12)
        assert np.degrees(batch.omega[0,0]) == pytest.approx(info["omega_deg_s"],abs=1e-12)
        assert batch.torque[0,0] == pytest.approx(info["torque"])


def test_three_batch_and_gym_are_identical_and_do_not_end_on_success():
    scene = {"euler_deg":[22.,-17.,35.],"omega_deg_s":[1.,-2.,3.],
             "noise_theta_deg":.2,"noise_omega_deg":.1,"noise_seed":923,
             "disturbance_bias":[.01,-.01,.02]}
    gym = make_env("three")
    obs,_ = gym.reset(seed=1,options=scene)
    batch = BatchSatelliteEnv("three",[scene])
    np.testing.assert_array_equal(obs,batch.observation()[0])
    for _ in range(20):
        a = [-.2,.3,-.1]
        obs,_,term,trunc,_ = gym.step(a)
        bobs,_,_,_,_ = batch.step([a])
        np.testing.assert_array_equal(obs,bobs[0])
        np.testing.assert_allclose(gym.plant.q,batch.q,atol=1e-15)
        assert not term and not trunc
    gym.reset(seed=1,options={"quaternion":[1.,0,0,0],"omega_deg_s":[0,0,0]})
    _,_,term,trunc,_ = gym.step([0,0,0])
    assert not term and not trunc


def test_quaternion_shortest_path_zero_and_pi_are_representation_invariant():
    quaternions = np.array([[1.,0.,0.,0.],[-1.,0.,0.,0.],
                            [0.,1.,0.,0.],[0.,-1.,0.,0.]])
    errors = quaternion_error(quaternions)
    np.testing.assert_allclose(errors[:2],0,atol=1e-15)
    np.testing.assert_allclose(errors[2],errors[3],atol=1e-15)
    assert np.linalg.norm(errors[2]) == pytest.approx(np.pi)
    q = euler_to_quaternion(np.radians([179.,21.,-45.]))
    np.testing.assert_allclose(quaternion_error(q),quaternion_error(-q),atol=1e-14)
    assert np.linalg.norm(quaternion_error(q))<=np.pi+1e-12


def test_rk4_recovers_constant_axis_acceleration_and_quaternion_norm():
    env = BatchSatelliteEnv("three",[{"euler_deg":[0,0,0],"omega_deg_s":[0,0,0]}])
    # Rotation purely around X eliminates all cross coupling; analytic solution.
    for _ in range(50):
        env.step([[.2,0,0]])
    alpha = .4/1.2
    assert env.omega[0,0] == pytest.approx(alpha,abs=1e-13)
    np.testing.assert_allclose(env.error_vector[0],[-.5*alpha,0,0],atol=2e-11)
    assert np.linalg.norm(env.q[0]) == pytest.approx(1.,abs=1e-14)


def test_noise_is_identical_for_controllers_with_different_actions():
    scene = {"euler_deg":[20.,-10.,30.],"noise_theta_deg":.3,
             "noise_omega_deg":.6,"noise_seed":552}
    left = BatchSatelliteEnv("three",[scene]);right = BatchSatelliteEnv("three",[scene])
    for _ in range(20):
        left.step([[-.2,.3,.1]]);right.step([[.5,-.1,.2]])
        lnoise = left.observation().astype(float)-np.concatenate([left.error_vector,left.omega],axis=1)
        rnoise = right.observation().astype(float)-np.concatenate([right.error_vector,right.omega],axis=1)
        np.testing.assert_allclose(lnoise,rnoise,atol=6e-8)


def test_pid_rate_damping_antiwindup_and_alpha_one_equivalence():
    pid = PIDController("three","tuned",{"kp":4.,"ki":.5,"kd":3.})
    pid.reset(2)
    observations = [[1.,0,0,0,0,0],[-1.,0,0,0,0,0]]
    for _ in range(50):
        actions = pid.act(observations)
    np.testing.assert_array_equal(actions[:,0],[1.,-1.])
    np.testing.assert_array_equal(pid.integral,0)
    gains = {"kp":4.,"ki":.08,"kd":3.,"alpha":1.}
    tuned = PIDController("single","tuned",gains)
    improved = PIDController("single","improved",gains)
    for obs in ([.3,.1],[.1,.04],[.01,-.003]):
        np.testing.assert_allclose(tuned.act(obs),improved.act(obs),atol=1e-15)


def test_success_needs_complete_tail_angle_and_rate_bounds():
    time = np.arange(0,21,dtype=float)
    trace = {"time":time,"error_deg":np.r_[np.full(15,10.),np.zeros(6)],
             "rate_deg_s":np.zeros(21),"torque":np.zeros((21,3))}
    metrics = compute_metrics(trace)
    assert metrics["success"] and metrics["settling_time_s"]==15.
    trace["rate_deg_s"][-1] = .6
    assert not compute_metrics(trace)["success"]
    trace["rate_deg_s"][:] = 0
    trace["error_deg"][15] = 2.
    assert not compute_metrics(trace)["success"]
    short = {k:v[:19] for k,v in trace.items()}
    assert not compute_metrics(short)["success"]


def test_controller_adapter_retains_physical_torque_before_common_saturation():
    class Model:
        def predict(self,obs,deterministic=True):
            return np.full((len(obs),1),.4),None
    single = ModelController(Model(),"single",physical_action=True)
    np.testing.assert_allclose(single.act([[0,0]]),[[.8]])
    three = ModelController(Model(),"three",physical_action=True)
    np.testing.assert_allclose(three.act([[0,0]]),[[.2]])


def test_validation_test_states_are_separate_and_noise_scenes_are_paired():
    validation = validation_scenarios("three")
    test = make_scenarios("three","test",20)
    assert len(validation)==16 and len(test)==180
    assert not ({s["scenario_id"] for s in validation}&{s["scenario_id"] for s in test})
    vs = {tuple(s["euler_deg"])+tuple(s["omega_deg_s"]) for s in validation}
    ts = {tuple(s["euler_deg"])+tuple(s["omega_deg_s"]) for s in test}
    assert not vs&ts
    nominal = next(s for s in test if s["case"]=="nominal" and s["sample"]==4)
    combined = next(s for s in test if s["case"]=="combined" and s["sample"]==4)
    assert nominal["euler_deg"]==combined["euler_deg"]


def test_batch_trace_torque_cost_and_full_time_are_explicit():
    scene = {"theta_deg":0.,"omega_deg_s":0.,"noise_seed":77}
    result = evaluate_batch("single",[scene],lambda obs:np.full((len(obs),1),.01))
    trace = result[0]["trace"]
    assert len(trace["time"])==401 and trace["time"][-1]==20
    assert result[0]["metrics"]["control_cost"] == pytest.approx(20*.005**2)


def test_failure_padding_stops_actuator_cost_and_cannot_be_success():
    scene = {"theta_deg":169.99,"omega_deg_s":20.}
    result = evaluate_batch("single",[scene],lambda obs:np.ones((len(obs),1)))
    trace,metrics = result[0]["trace"],result[0]["metrics"]
    assert metrics["failed"] and not metrics["success"]
    assert metrics["failure_time_s"]==.05 and metrics["active_duration_s"]==.05
    assert metrics["control_cost"]==pytest.approx(.05*.5**2)
    assert np.all(trace["torque"][2:]==0)
    assert np.all(trace["error_deg"][2:]==trace["error_deg"][1])
