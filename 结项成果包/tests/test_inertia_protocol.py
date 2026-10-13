"""Verify the predeclared V3 allocation, physics and equal PID search budgets."""
import json
import numpy as np
import pytest
from inertia_generalization.scenarios import (
    initial_states,validation_scenarios,test_scenarios,check_physical_inertia,make_scenarios)
from inertia_generalization.protocol import PROTOCOL,TEST_RHOS,ANISOTROPIC_SCALES,freeze_protocol,sha256
from inertia_generalization.pid import candidates,make_pid


@pytest.mark.parametrize("axis",["single","three"])
def test_new_split_states_are_independent_and_reproduce(axis):
    validation=initial_states(axis,"validation");test=initial_states(axis,"test")
    assert len(validation)==20 and len(test)==30
    assert validation==initial_states(axis,"validation") and test==initial_states(axis,"test")
    key="theta_deg" if axis=="single" else "euler_deg"
    vs={tuple(np.atleast_1d(s[key]))+tuple(np.atleast_1d(s["omega_deg_s"])) for s in validation}
    ts={tuple(np.atleast_1d(s[key]))+tuple(np.atleast_1d(s["omega_deg_s"])) for s in test}
    assert len(vs)==20 and len(ts)==30 and not vs&ts
    assert all(np.max(np.abs(s[key]))<=60 for s in validation+test)
    assert all(np.max(np.abs(s["omega_deg_s"]))<=5 for s in validation+test)


@pytest.mark.parametrize("axis",["single","three"])
def test_validation_is_twenty_fixed_states_four_per_rho_and_pid_paired(axis):
    robust=validation_scenarios(axis);nominal=validation_scenarios(axis,nominal=True)
    assert len(robust)==len(nominal)==20
    assert [s["rho"] for s in robust]==[rho for rho in (.7,.85,1.,1.15,1.3) for _ in range(4)]
    for r,n in zip(robust,nominal):
        assert n["inertia_scale"]==1 and r["inertia_scale"]==r["rho"]
        assert r["sample_id"]==n["sample_id"] and r["noise_seed"]==n["noise_seed"]
        key="theta_deg" if axis=="single" else "euler_deg"
        assert r[key]==n[key] and r["omega_deg_s"]==n["omega_deg_s"]
        assert r["split"]==n["split"]=="validation"


@pytest.mark.parametrize("axis,expected",[("single",330),("three",450)])
def test_test_counts_scalar_domain_labels_and_cross_rho_pairing(axis,expected):
    scenes=test_scenarios(axis)
    assert len(scenes)==expected and len({s["scenario_id"] for s in scenes})==expected
    scalar=[s for s in scenes if s["experiment"]=="scalar"]
    assert len(scalar)==330
    for rho in TEST_RHOS:
        group=[s for s in scalar if s["rho"]==rho]
        assert len(group)==30 and {s["sample_id"] for s in group}==set(range(30))
        assert all(s["domain"]==("nominal" if rho==1 else "in_domain" if .7<=rho<=1.3 else "out_of_domain") for s in group)
    for sample in range(30):
        key="theta_deg" if axis=="single" else "euler_deg"
        group=[s for s in scenes if s["sample_id"]==sample]
        assert all(s[key]==group[0][key] and s["omega_deg_s"]==group[0]["omega_deg_s"] for s in group)
    for s in scenes:
        assert all(s[k]==0 for k in ("disturbance_bias","disturbance_amplitude","pulse_torque","noise_theta_deg","noise_omega_deg"))


def test_anisotropic_moments_are_physical_and_separate_from_boundary():
    scenes=[s for s in test_scenarios("three") if s["experiment"]=="anisotropic"]
    assert len(scenes)==120 and len({s["case_id"] for s in scenes})==4
    for scale in ANISOTROPIC_SCALES:
        result=check_physical_inertia(np.array([1.2,1.,.8])*scale)
        assert np.all(2*result<=result.sum()+1e-12)
    assert all(s["rho"] is None and s["domain"]=="anisotropic_transfer" for s in scenes)
    for invalid in ([1.,1.,3.],[0.,1.,1.],[np.nan,1.,1.],[-1,1,1]):
        with pytest.raises(ValueError):check_physical_inertia(invalid)


@pytest.mark.parametrize("axis",["single","three"])
def test_pid_candidate_count_and_controller_structure_are_identical(axis):
    grid=list(candidates(axis));assert len(grid)==144
    assert len({tuple(g.values()) for g in grid})==144
    gains=grid[43]
    nominal=make_pid(axis,"nominal",gains);robust=make_pid(axis,"randomized",gains)
    dimension=1 if axis=="single" else 3
    for obs in (np.full(dimension*2,.1),np.full(dimension*2,-.03)):
        np.testing.assert_array_equal(nominal.act(obs),robust.act(obs))


def test_protocol_freeze_reuses_unchanged_bytes_and_rejects_mutation(tmp_path):
    original=freeze_protocol(tmp_path)
    assert original["counts"]["episodes"]==10920 and original["counts"]["pid_candidates"]==576
    assert original==freeze_protocol(tmp_path)
    for name,hash_value in original["frozen_sha256"].items():assert sha256(tmp_path/name)==hash_value
    path=tmp_path/"protocol.json"
    value=json.loads(path.read_text(encoding="utf-8"));value["duration"]=19
    path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(ValueError,match="Frozen protocol file changed"):freeze_protocol(tmp_path)


def test_protocol_rejects_posthoc_scene_count_changes():
    with pytest.raises(ValueError):make_scenarios("single","test",31)
    with pytest.raises(ValueError):make_scenarios("single","validation",19)
    assert PROTOCOL["planned_models"]==12 and PROTOCOL["test_used_for_selection"] is False
