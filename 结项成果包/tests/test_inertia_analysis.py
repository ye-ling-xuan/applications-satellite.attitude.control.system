"""Check boundary and pairing logic on synthetic outcomes before held-out evaluation."""
import pytest
from inertia_generalization.analysis import contiguous_interval, boundary_tables, paired_tables, aggregate
from inertia_generalization.protocol import TEST_RHOS
from inertia_generalization.scenarios import test_scenarios as scenarios


def synthetic_rows():
    rows=[]
    for axis in ("single","three"):
        for s in scenarios(axis):
            for controller in ("PID_N","PID_R","SAC_N","SAC_R","SAC_N_raw","SAC_R_raw"):
                for seed in ((0,) if controller.startswith("PID") else (11,22,33)):
                    # At nominal all SAC_R seeds have 27/30 successes. At rho=.9
                    # one seed fails 4/30 even though the pooled mean exceeds 90%.
                    success = s["sample_id"]<27
                    if controller=="SAC_R" and seed==33 and s["rho"]==.9:
                        success=s["sample_id"]<26
                    rows.append({**s,"axis":axis,"controller":controller,"model_seed":seed,
                                 "success":success,"failed":False,"settling_time_s":4. if success else None,
                                 "rms_error_deg":2.,"tail_mean_error_deg":.2,"tail_max_error_deg":.9,
                                 "tail_max_rate_deg_s":.4,"control_cost":1.,"trace_file":"synthetic.npz"})
    return rows


def test_interval_requires_nominal_and_never_jumps_over_failed_grid():
    passing=[True]*11
    assert contiguous_interval(TEST_RHOS,passing)["grid_points"]==11
    passing[5]=False
    assert contiguous_interval(TEST_RHOS,passing)["lower_rho"] is None
    passing[5]=True;passing[3]=False;passing[8]=False
    actual=contiguous_interval(TEST_RHOS,passing)
    assert (actual["lower_rho"],actual["upper_rho"],actual["grid_points"])==(.9,1.2,4)


def test_common_boundary_uses_worst_seed_and_exact_27_of_30():
    points,intervals=boundary_tables(synthetic_rows())
    p=next(r for r in points if r["axis"]=="three" and r["controller"]=="SAC_R" and r["model_seed"]=="common" and r["rho"]==.9)
    assert p["success_rate"]<.9 and not p["passed_90_percent"]
    interval=next(r for r in intervals if r["axis"]=="three" and r["controller"]=="SAC_R" and r["model_seed"]=="common")
    assert interval["lower_rho"]==1. and interval["upper_rho"]==1.5
    # Raising two seeds does not override the poor third seed.
    rows=synthetic_rows()
    for r in rows:
        if r["controller"]=="SAC_R" and r["model_seed"]!=33 and r["rho"]==.9:r["success"]=True
    points,_=boundary_tables(rows)
    p=next(r for r in points if r["axis"]=="three" and r["controller"]=="SAC_R" and r["model_seed"]=="common" and r["rho"]==.9)
    assert p["success_rate"]>.9 and not p["passed_90_percent"]


def test_pair_counts_duplicate_rejection_and_failure_preservation():
    rows=synthetic_rows()
    assert len(rows)==10920
    training,calibration,pid=paired_tables(rows)
    assert (len(training),len(calibration),len(pid))==(4680,4680,780)
    lost=[r for r in training if r["success_lost"]]
    assert len(lost)==2 and all(r["success_change"]==-1 for r in lost)
    assert sum(r["success_lost"] for r in calibration)==2
    assert all(r["success_change"]==0 for r in pid)
    with pytest.raises(ValueError):paired_tables(rows+[rows[0]])
    # Missing paired controllers must fail loudly, rather than silently drop data.
    with pytest.raises(KeyError):paired_tables([r for r in rows if r["controller"]!="SAC_R"])


def test_regions_and_success_only_settling_are_explicit():
    records=aggregate(synthetic_rows())
    bykey={(r["axis"],r["controller"],r["region"]):r for r in records}
    assert bykey["single","SAC_N","out_of_domain"]["n_episodes"]==360
    assert bykey["single","SAC_N","nominal"]["n_episodes"]==90
    assert bykey["three","PID_N","anisotropic"]["n_episodes"]==120
    assert bykey["three","SAC_N","all_scalar"]["settling_time_s_successful"]==4.
