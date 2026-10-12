"""Independently verify multi-axis CSV/NPZ results and replay applied torques.

Usage: python tools/verify_multi_results.py [--results PATH]
Reads completed test outputs only for integrity verification, never selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from collections import defaultdict
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
PROJECT=PACKAGE.parent
sys.path.insert(0,str(PACKAGE))
DEFAULT=PACKAGE/"07_单轴三轴多算法对比"/"results"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def boolean(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if str(value).lower() in ("true","1"):return True
    if str(value).lower() in ("false","0"):return False
    raise AssertionError(f"Invalid boolean: {value}")


def equal(actual,expected,context):
    if actual is None or expected is None or actual=="" or expected=="":
        assert actual in (None,"") and expected in (None,""),context
    elif isinstance(actual,(bool,np.bool_)):
        assert actual==boolean(expected),context
    else:
        assert np.isfinite(float(actual)) and np.isfinite(float(expected)),context
        assert np.isclose(float(actual),float(expected),rtol=1e-10,atol=1e-8),f"{context}: {actual} != {expected}"


def independent_metrics(trace):
    t=trace["time"].astype(float)
    e=trace["error_deg"].astype(float)
    w=trace["rate_deg_s"].astype(float)
    failed=trace["failed"].astype(bool)
    assert t[0]==0 and np.all(np.diff(t)>0),"Time ordering"
    assert np.all(np.isfinite(e)) and np.all(e>=0),"Error values"
    assert np.all(np.isfinite(w)) and np.all(w>=0),"Rate values"
    assert np.all(~failed[:-1]|failed[1:]),"Failure must be sticky"
    tail=t>=15.-1e-9
    band=(e<=1.)&(w<=.5)&~failed
    complete=bool(t[-1]>=20.-1e-9)
    success=bool(complete and not failed.any() and tail.any() and band[tail].all())
    bad=np.flatnonzero(~band)
    start=int(bad[-1]+1) if bad.size else 0
    settling=float(t[start]) if success and start<len(t) and t[-1]-t[start]>=5.-1e-9 else None
    failure=np.flatnonzero(failed)
    failure_time=float(t[failure[0]]) if failure.size else None
    torque=np.asarray(trace["torque"],float).reshape(len(t),-1)
    cost=float(np.sum(np.sum(torque[1:]**2,axis=1)*np.diff(t)))
    return {"success":success,"settling_time_s":settling,
            "rms_error_deg":float(np.sqrt(np.trapz(e*e,t)/(t[-1]-t[0]))),
            "tail_mean_error_deg":float(np.mean(e[tail])),"tail_max_error_deg":float(np.max(e[tail])),
            "tail_max_rate_deg_s":float(np.max(w[tail])),"control_cost":cost,
            "final_error_deg":float(e[-1]),"final_rate_deg_s":float(w[-1]),
            "failed":bool(failed.any()),"complete":complete,"failure_time_s":failure_time,
            "active_duration_s":failure_time if failure_time is not None else float(t[-1]-t[0])}


def summarize_independently(rows):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in ("axis","controller","family","case_id","level"))].append(r)
    summaries={}
    fields=("rms_error_deg","tail_mean_error_deg","tail_max_error_deg","tail_max_rate_deg_s", "control_cost","final_error_deg","final_rate_deg_s")
    for key,group in groups.items():
        seeds=defaultdict(list)
        for r in group:seeds[r["model_seed"]].append(r)
        rates=[np.mean([boolean(r["success"]) for r in g]) for g in seeds.values()]
        tails=[np.mean([float(r["tail_mean_error_deg"]) for r in g]) for g in seeds.values()]
        settled=[float(r["settling_time_s"]) for r in group if r["settling_time_s"]!=""]
        summary={"n_episodes":len(group),"n_model_seeds":len(seeds),
                 "success_rate":float(np.mean([boolean(r["success"]) for r in group])),
                 "success_seed_std":float(np.std(rates,ddof=1)) if len(rates)>1 else None,
                 "tail_error_seed_std":float(np.std(tails,ddof=1)) if len(tails)>1 else None,
                 "settling_time_s_mean_successful":float(np.mean(settled)) if settled else None,
                 "settled_episodes":len(settled),"failed_rate":float(np.mean([boolean(r["failed"]) for r in group]))}
        for field in fields:summary[field+"_mean"]=float(np.mean([float(r[field]) for r in group]))
        summaries[key]=summary
    return summaries


def verify_snapshot(snapshot_path,new=False):
    data=json.loads(snapshot_path.read_text(encoding="utf-8"))
    records=data["files"] if new else [{"source_relative":key,**value} for key,value in data.items()]
    bad=[]
    for r in records:
        path=PROJECT/r["source_relative"]
        if not path.is_file() or digest(path)!=r["sha256"]:bad.append(r["source_relative"])
    assert not bad,f"Original files changed: {bad}"
    return {"checked_file_count":len(records),"changed_or_missing":bad,"preserved":True}


def verify_copies(manifest_path):
    records=json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    for r in records:
        assert digest(PROJECT/r["source_relative"])==r["sha256"],"Copied source changed"
        assert digest(PROJECT/r["copy_relative"])==r["sha256"],"Archive copy changed"
    return {"checked_file_count":len(records),"copies_identical":True}


def verify_dataset(results):
    from multi_algorithm_benchmark.envs import BatchSatelliteEnv
    rows=csv_rows(results/"episode_metrics.csv")
    manifest=json.loads((results/"evaluation_manifest.json").read_text(encoding="utf-8"))
    assert len(rows)==manifest["episode_count"],"Manifest episode count"
    assert rows,"No episodes"
    groups=defaultdict(list)
    pair_initial={}
    traces_seen=set()
    for r in rows:
        path=(results/r["trace_file"]).resolve()
        assert path.is_relative_to(results.resolve()),"Trace path must remain in result folder"
        assert path not in traces_seen,"Duplicate trace"
        traces_seen.add(path)
        with np.load(path,allow_pickle=False) as z:
            trace={k:z[k] for k in z.files if k!="metadata_json"}
            meta=json.loads(str(z["metadata_json"]))
        assert meta["axis"]==r["axis"] and meta["scenario"]["scenario_id"]==r["scenario_id"]
        assert meta["controller"]["controller"]==r["controller"]
        assert str(meta["controller"]["model_seed"])==r["model_seed"]
        assert int(r["scenario_seed"])==meta["scenario"]["noise_seed"]
        recomputed=independent_metrics(trace)
        for k,v in recomputed.items():
            equal(v,r[k],f"{r['trace_file']} CSV {k}")
            equal(v,meta["metrics"][k],f"{r['trace_file']} metadata {k}")
        np.testing.assert_allclose(trace["error_deg"],np.degrees(np.linalg.norm(trace["error_vector_rad"],axis=1)),atol=1e-9)
        np.testing.assert_allclose(trace["rate_deg_s"],np.degrees(np.linalg.norm(trace["omega_rad_s"],axis=1)),atol=1e-9)
        np.testing.assert_array_equal(trace["time"],trace["t"])
        np.testing.assert_array_equal(trace["signed_error_vector"],trace["error_vector_rad"])
        limit=.5 if r["axis"]=="single" else 2.
        assert np.max(np.abs(trace["torque"]))<=limit+1e-12,"Torque bounds"
        if r["axis"]=="three":
            q=trace["quaternion"]
            np.testing.assert_allclose(np.linalg.norm(q,axis=1),1.,atol=1e-12)
            angle=np.degrees(2*np.arctan2(np.linalg.norm(q[:,1:],axis=1),np.abs(q[:,0])))
            np.testing.assert_allclose(angle,trace["error_deg"],atol=1e-9)
            initial=np.r_[q[0],trace["omega_rad_s"][0]]
        else:
            wrapped=(trace["theta_rad"][:,0]+np.pi)%(2*np.pi)-np.pi
            np.testing.assert_allclose(np.abs(np.degrees(wrapped)),trace["error_deg"],atol=1e-9)
            initial=np.r_[trace["theta_rad"][0],trace["omega_rad_s"][0]]
        pair_key=(r["axis"],r["scenario_id"])
        if pair_key in pair_initial:np.testing.assert_array_equal(initial,pair_initial[pair_key])
        else:pair_initial[pair_key]=initial
        groups[(r["axis"],r["controller"],r["model_seed"])].append((r,meta["scenario"],path))
    expected=summarize_independently(rows)
    summary=csv_rows(results/"summary.csv")
    assert len(summary)==len(expected),"Summary group count"
    for row in summary:
        key=tuple(row[k] for k in ("axis","controller","family","case_id","level"))
        for k,v in expected[key].items():equal(v,row[k],f"Summary {key} {k}")
    # Replaying exported physical torques checks every interval and every state.
    max_state_error=0.;max_sensor_error=0.;replay_steps=0
    scenes_by_axis={axis:json.loads((results/axis/"test_scenarios.json").read_text(encoding="utf-8")) for axis in {r["axis"] for r in rows}}
    for (axis,controller,seed),group in groups.items():
        scene_ids={s["scenario_id"] for s in scenes_by_axis[axis]}
        assert {scene["scenario_id"] for _,scene,_ in group}==scene_ids,"Every controller must cover every scene"
        stored=[]
        for _,_,path in group:
            with np.load(path,allow_pickle=False) as z:stored.append({k:z[k] for k in z.files if k!="metadata_json"})
        env=BatchSatelliteEnv(axis,[s for _,s,_ in group])
        torque=np.stack([t["torque"] for t in stored],axis=1)
        obs=np.stack([t["observation"] for t in stored],axis=1)
        rates=np.stack([t["omega_rad_s"] for t in stored],axis=1)
        attitudes=np.stack([t["quaternion" if axis=="three" else "theta_rad"] for t in stored],axis=1)
        failed=np.stack([t["failed"] for t in stored],axis=1)
        disturbances=np.stack([t["disturbance"] for t in stored],axis=1)
        for k in range(env.steps+1):
            state=env.q if axis=="three" else env.theta
            state_error=max(float(np.max(np.abs(state-attitudes[k]))),float(np.max(np.abs(env.omega-rates[k]))))
            sensor_error=float(np.max(np.abs(env.observation().astype(float)-obs[k].astype(float))))
            assert state_error<1e-8,f"State replay {axis}/{controller}/{seed} step {k}: {state_error}"
            assert sensor_error<2e-7,f"Sensor replay {axis}/{controller}/{seed} step {k}: {sensor_error}"
            np.testing.assert_array_equal(env.failed,failed[k])
            np.testing.assert_allclose(env.disturbance,disturbances[k],atol=1e-12)
            max_state_error=max(max_state_error,state_error);max_sensor_error=max(max_sensor_error,sensor_error)
            if k<env.steps:
                env.step(torque[k+1]/env.max_torque)
                replay_steps+=env.n
        print(f"Replayed {axis} {controller} seed={seed}: {len(group)} scenes",flush=True)
    return {"episode_count":len(rows),"scene_count":len(pair_initial),"summary_group_count":len(summary),
            "controller_seed_group_count":len(groups),"replayed_interaction_count":replay_steps,
            "max_state_replay_difference":max_state_error,"max_sensor_replay_difference":max_sensor_error,
            "all_metrics_recomputed":True,"paired_initial_states_identical":True,
            "paired_sensor_noise_replayed":True,"dynamics_replayed":True}


def verify_pairwise(results):
    rows=csv_rows(results/"episode_metrics.csv")
    grouped=defaultdict(list)
    for row in rows:grouped[(row["axis"],row["scenario_id"],row["controller"])].append(row)
    means={key:{f:float(np.mean([boolean(r["success"]) if f=="success_rate" else float(r[f]) for r in g]))
                for f in ("rms_error_deg","tail_mean_error_deg","control_cost","success_rate")} for key,g in grouped.items()}
    pairwise=csv_rows(results/"pairwise_results.csv")
    seen=set()
    for r in pairwise:
        key=(r["axis"],r["scenario_id"],r["controller_a"],r["controller_b"])
        assert key not in seen;seen.add(key)
        a=means[key[:2]+(key[2],)];b=means[key[:2]+(key[3],)]
        for f in a:
            equal(a[f],r[f+"_a"],"Pairwise A "+f)
            equal(b[f],r[f+"_b"],"Pairwise B "+f)
            equal(b[f]-a[f],r[f+"_b_minus_a"],"Pairwise difference "+f)
    per_scene=defaultdict(set)
    for axis,scene,controller in means:per_scene[(axis,scene)].add(controller)
    expected=sum(len(cs)*(len(cs)-1)//2 for cs in per_scene.values())
    assert len(pairwise)==expected,"Every pair of controller families must be compared"
    improvements=csv_rows(results/"paired_improvements.csv")
    for r in improvements:
        key=(r["axis"],r["scenario_id"])
        a=means[key+(r["baseline"],)];b=means[key+(r["improved"],)]
        for f in a:
            equal(a[f],r[f+"_baseline"],"Improvement baseline "+f)
            equal(b[f],r[f+"_improved"],"Improvement final "+f)
            equal(b[f]-a[f],r[f+"_change"],"Improvement difference "+f)
    return {"pairwise_rows":len(pairwise),"improvement_rows":len(improvements),"all_differences_recomputed":True}


def verify_calibration_pairs(results):
    rows=csv_rows(results/"episode_metrics.csv")
    keyed={(r["axis"],r["controller"],r["model_seed"],r["scenario_id"]):r for r in rows}
    assert len(keyed)==len(rows),"Duplicate formal controller-seed-scene key"
    pairs=csv_rows(results/"policy_calibration_pairs.csv")
    expected=sum(r["controller"].endswith("_calibrated") for r in rows)
    assert len(pairs)==expected,"Every calibrated episode must have one raw partner"
    fields=("rms_error_deg","tail_mean_error_deg","tail_max_error_deg","tail_max_rate_deg_s",
            "control_cost","final_error_deg","final_rate_deg_s")
    seen=set()
    for p in pairs:
        key=(p["axis"],p["algorithm"],p["model_seed"],p["scenario_id"])
        assert key not in seen;seen.add(key)
        raw=keyed[key]
        calibrated=keyed[(key[0],key[1]+"_calibrated",key[2],key[3])]
        sr,sc=boolean(raw["success"]),boolean(calibrated["success"])
        equal(sr,p["success_raw"],"Calibration raw success")
        equal(sc,p["success_calibrated"],"Calibration processed success")
        equal(int(sc)-int(sr),p["success_change"],"Calibration success difference")
        equal(sc and not sr,p["success_recovered"],"Calibration recovered")
        equal(sr and not sc,p["success_lost"],"Calibration lost")
        equal(raw["settling_time_s"],p["settling_time_s_raw"],"Calibration raw settling")
        equal(calibrated["settling_time_s"],p["settling_time_s_calibrated"],"Calibration processed settling")
        assert raw["trace_file"]==p["trace_file_raw"] and calibrated["trace_file"]==p["trace_file_calibrated"]
        for f in fields:
            a,b=float(raw[f]),float(calibrated[f])
            equal(a,p[f+"_raw"],"Calibration raw "+f)
            equal(b,p[f+"_calibrated"],"Calibration processed "+f)
            equal(b-a,p[f+"_change"],"Calibration difference "+f)
    return {"paired_episode_count":len(pairs),"same_seed_and_scene":True,"all_differences_recomputed":True}


def verify_completed_models(manifest):
    unique={}
    for records in manifest["controllers"].values():
        for controller in records:
            if "model_path" in controller:
                p=Path(controller["model_path"])
                assert digest(p)==controller["model_sha256"],"Evaluated model changed"
                unique[p]=controller
    output=[]
    budgets=defaultdict(set)
    source={p.name:digest(p) for p in (PACKAGE/"multi_algorithm_benchmark").glob("*.py")}
    for path,descriptor in unique.items():
        metadata_path=path.parent/"training_metadata.json"
        final_path=path.parent/"final_model.zip"
        assert metadata_path.exists() and final_path.exists(),"Incomplete training output"
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["fresh_training"] and not metadata["test_used_for_selection"],"Training protocol"
        assert digest(path)==metadata["model_sha256"],"Metadata best model hash"
        with zipfile.ZipFile(path) as z:best=json.loads(z.read("data"))
        with zipfile.ZipFile(final_path) as z:final=json.loads(z.read("data"))
        assert int(best["num_timesteps"])==metadata["best_checkpoint_steps"],"Checkpoint step count"
        assert int(final["num_timesteps"])==metadata["actual_steps"],"Final step count"
        assert metadata["actual_steps"]>=metadata["requested_steps"],"Incomplete interaction budget"
        budgets[metadata["axis"]].add(metadata["requested_steps"])
        if metadata["algorithm"] in ("SAC","TD3"):
            assert (path.parent/"replay_buffer.pkl").is_file(),"Off-policy replay buffer missing"
        output.append({"axis":metadata["axis"],"algorithm":metadata["algorithm"],"seed":metadata["seed"],
                       "best_sha256":digest(path),"final_sha256":digest(final_path),
                       "metadata_sha256":digest(metadata_path),"actual_steps":metadata["actual_steps"],
                       "best_checkpoint_steps":metadata["best_checkpoint_steps"],
                       "new_source_files_since_training":sorted(set(source)-set(metadata["source_sha256"])),
                       "source_hash_differences":{k:{"recorded":v,"current":source.get(k)}
                                                  for k,v in metadata["source_sha256"].items() if source.get(k)!=v}})
    assert all(len(values)==1 for values in budgets.values()),"Unequal budget inside an axis"
    expected={(axis,algorithm,int(seed)) for axis in manifest["scene_counts"]
              for algorithm in ("PPO","SAC","TD3") for seed in manifest["training_seeds"]}
    actual={(r["axis"],r["algorithm"],int(r["seed"])) for r in output}
    assert actual==expected,"All requested algorithms, axes and seeds must be complete"
    return {"unique_complete_model_count":len(unique),"all_saved_step_counts_and_hashes_match":True,
            "requested_budgets":{axis:list(values) for axis,values in budgets.items()},
            "source_differences_are_recorded_not_rejected":True,"models":output}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=DEFAULT)
    args=parser.parse_args()
    results=args.results.resolve()
    report={"status":"running","purpose":"Integrity checking only; test results never used for parameter selection"}
    try:
        manifest=json.loads((results/"evaluation_manifest.json").read_text(encoding="utf-8"))
        for axis,hash_value in manifest["test_scenario_sha256"].items():assert digest(results/axis/"test_scenarios.json")==hash_value,"Scenario hash changed"
        report["completed_models"]=verify_completed_models(manifest)
        report["formal"]=verify_dataset(results)
        report["pairwise"]=verify_pairwise(results)
        report["calibration_pairs"]=verify_calibration_pairs(results)
        if (results/"historical"/"evaluation_manifest.json").exists():report["historical"]=verify_dataset(results/"historical")
        archive=PACKAGE/"01_遴选原始参考"/"05_中期前任务收尾新增材料"
        report["original_first_delivery"]=verify_snapshot(PACKAGE/"06_结项说明与报告"/"original_snapshot.json")
        report["original_new_delivery"]=verify_snapshot(archive/"source_snapshot.json",new=True)
        report["archive_first_delivery"]=verify_copies(PACKAGE/"01_遴选原始参考"/"copy_manifest.json")
        report["archive_new_delivery"]=verify_copies(archive/"copy_manifest.json")
        report["status"]="passed"
    except Exception as exc:
        report["status"]="failed";report["error_type"]=type(exc).__name__;report["error"]=str(exc)
        raise
    finally:
        out=PACKAGE/"07_单轴三轴多算法对比"/"核验记录.json"
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        summary={"status":report["status"],"verification_record":str(out),
                 "formal_episodes":report.get("formal",{}).get("episode_count"),
                 "historical_episodes":report.get("historical",{}).get("episode_count"),
                 "complete_models":report.get("completed_models",{}).get("unique_complete_model_count")}
        if "error" in report:summary["error"]=report["error"]
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
