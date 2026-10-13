"""Independent, fixed, paired initial states for the inertia-only V3 study."""
import numpy as np
from multi_algorithm_benchmark.envs import axis_name
from .protocol import VALIDATION_SEED,TEST_SEED,VALIDATION_RHOS,TEST_RHOS,ANISOTROPIC_SCALES


def check_physical_inertia(inertia):
    """Principal rigid-body moments must be positive and obey triangle bounds."""
    values=np.asarray(inertia,float)
    if values.shape!=(3,) or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError("three positive finite principal moments required")
    if np.any(2*values>np.sum(values)+1e-12):
        raise ValueError("principal inertias violate triangle inequalities")
    return values


def initial_states(axis,split):
    axis=axis_name(axis)
    if split not in ("validation","test"):raise ValueError("unknown split")
    seed=VALIDATION_SEED if split=="validation" else TEST_SEED
    count=20 if split=="validation" else 30
    d=1 if axis=="single" else 3
    rng=np.random.default_rng(seed)
    # Draw two separate arrays in a fixed declared order; inertia assignment
    # never consumes this stream. No handpicked zero/representative states.
    angles=rng.uniform(-60.,60.,size=(count,d))
    rates=rng.uniform(-5.,5.,size=(count,d))
    states=[]
    for i in range(count):
        state={"axis":axis,"split":split,"sample_id":i,"initial_state_seed":seed}
        if axis=="single":state.update(theta_deg=float(angles[i,0]),omega_deg_s=float(rates[i,0]))
        else:state.update(euler_deg=angles[i].tolist(),omega_deg_s=rates[i].tolist())
        states.append(state)
    return states


def scene_from_state(state,scale,case_id,experiment="scalar",nominal_pid=False):
    axis=state["axis"]
    scalar=np.isscalar(scale)
    if axis=="three":
        check_physical_inertia(np.array([1.2,1.,.8])*np.asarray(scale))
    elif not scalar or not np.isfinite(scale) or scale<=0:
        raise ValueError("single-axis inertia scale must be positive scalar")
    split=state["split"]
    actual=1. if nominal_pid else scale
    id_suffix="_pid_nominal" if nominal_pid else ""
    return {**state,"scenario_id":f"{split}_{axis}_{case_id}_{state['sample_id']:03d}{id_suffix}",
            "case_id":case_id,"case":case_id,"family":"inertia_scalar" if scalar else "inertia_anisotropic",
            "experiment":experiment,"level":float(scale) if scalar else case_id,
            "rho":float(scale) if scalar else None,
            "domain":"nominal" if scalar and scale==1 else "in_domain" if scalar and .7<=scale<=1.3 else "out_of_domain" if scalar else "anisotropic_transfer",
            "inertia_scale":float(actual) if np.isscalar(actual) else list(actual),
            "planned_validation_rho":float(scale) if split=="validation" else None,
            "noise_seed":state["initial_state_seed"]+10000+state["sample_id"],
            "disturbance_bias":0.,"disturbance_amplitude":0.,"pulse_torque":0.,
            "noise_theta_deg":0.,"noise_omega_deg":0.}


def validation_scenarios(axis,nominal=False):
    states=initial_states(axis,"validation")
    return [scene_from_state(state,VALIDATION_RHOS[i//4],f"rho_{VALIDATION_RHOS[i//4]:.2f}",nominal_pid=nominal)
            for i,state in enumerate(states)]


def test_scenarios(axis):
    states=initial_states(axis,"test")
    scenes=[scene_from_state(state,rho,f"rho_{rho:.2f}") for rho in TEST_RHOS for state in states]
    if axis_name(axis)=="three":
        scenes.extend(scene_from_state(state,scale,f"anisotropic_{i+1}",experiment="anisotropic")
                      for i,scale in enumerate(ANISOTROPIC_SCALES) for state in states)
    return scenes


def make_scenarios(axis,split="test",samples=None):
    expected=30 if split=="test" else 20
    if samples is not None and samples!=expected:raise ValueError("Protocol forbids changing scene count")
    if split=="test":return test_scenarios(axis)
    if split=="validation":return validation_scenarios(axis)
    raise ValueError("unknown split")


make_validation_scenarios=validation_scenarios
make_test_scenarios=test_scenarios
test_scenarios.__test__=False
