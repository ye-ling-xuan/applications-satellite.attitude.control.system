"""Synthetic audit cases exercise boundaries and independent PID action replay."""
import csv
import json
import numpy as np
import pytest
from tools.verify_inertia_results import verify_boundaries,verify_controller_replay
from inertia_generalization.analysis import boundary_tables
from inertia_generalization.protocol import TEST_RHOS
from multi_algorithm_benchmark.controllers import PIDController
from multi_algorithm_benchmark.metrics import evaluate_batch


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def test_independent_boundary_audit_handles_gaps_nominal_failure_and_rejects_tamper(tmp_path):
    rows=[]
    for axis in ("single","three"):
        for label in ("PID_N","PID_R","SAC_N","SAC_R","SAC_N_raw","SAC_R_raw"):
            for seed in ((0,) if label.startswith("PID") else (11,22,33)):
                for rho in TEST_RHOS:
                    for sample in range(30):
                        success=not (label=="PID_N" and rho==.8) and not (label=="SAC_N" and seed==22 and rho==1. and sample<4)
                        rows.append({"axis":axis,"controller":label,"model_seed":seed,"rho":rho,
                                     "success":success,"experiment":"scalar"})
    points,intervals=boundary_tables(rows)
    write_csv(tmp_path/"episode_metrics.csv",rows);write_csv(tmp_path/"boundary_points.csv",points)
    write_csv(tmp_path/"empirical_boundaries.csv",intervals)
    report=verify_boundaries(tmp_path)
    assert report["contiguous_nominal_grid_recomputed"]
    pid=next(r for r in intervals if r["axis"]=="three" and r["controller"]=="PID_N" and r["model_seed"]=="common")
    assert pid["lower_rho"]==.9
    weak=next(r for r in intervals if r["axis"]=="three" and r["controller"]=="SAC_N" and r["model_seed"]=="common")
    assert weak["lower_rho"] is None and weak["grid_points"]==0
    weak["lower_rho"]=.5;write_csv(tmp_path/"empirical_boundaries.csv",intervals)
    with pytest.raises(AssertionError):verify_boundaries(tmp_path)


def test_independent_pid_formula_replays_exported_actions_and_rejects_tamper(tmp_path):
    gains={"kp":8.,"ki":.08,"kd":5.,"alpha":.5,"integral_limit":2.}
    scenes=[{"theta_deg":42.,"omega_deg_s":2.,"noise_seed":88},
            {"theta_deg":-31.,"omega_deg_s":-1.,"noise_seed":89,"inertia_scale":1.3}]
    results=evaluate_batch("single",scenes,PIDController("single","improved",gains))
    rows=[]
    for i,result in enumerate(results):
        trace=result["trace"];metrics=result["metrics"]
        metrics.update(max_error_deg=float(np.max(trace["error_deg"])),max_rate_deg_s=float(np.max(trace["rate_deg_s"])))
        path=tmp_path/f"scene_{i}.npz"
        np.savez_compressed(path,**trace,metadata_json=json.dumps({"metrics":metrics}))
        rows.append({"axis":"single","controller":"PID_N","model_seed":0,
                     "trace_file":path.name,"max_error_deg":metrics["max_error_deg"],"max_rate_deg_s":metrics["max_rate_deg_s"]})
    write_csv(tmp_path/"episode_metrics.csv",rows)
    manifest={"controllers":{"single":[{"controller":"PID_N","model_seed":0,"pid_gains":gains}]}}
    report=verify_controller_replay(tmp_path,manifest)
    assert report["largest_torque_difference"]<1e-12 and report["recomputed_intervals"]==800
    with np.load(tmp_path/"scene_0.npz",allow_pickle=False) as z:data={k:z[k] for k in z.files}
    data["torque"][10]*=-1
    np.savez_compressed(tmp_path/"scene_0.npz",**data)
    with pytest.raises(AssertionError,match="Controller replay"):verify_controller_replay(tmp_path,manifest)
