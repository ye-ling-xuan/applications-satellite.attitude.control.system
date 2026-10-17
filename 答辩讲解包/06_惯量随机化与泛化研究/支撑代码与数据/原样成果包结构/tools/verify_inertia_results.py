"""Independently audit the frozen inertia study without selecting on test data.

python tools/verify_inertia_results.py --pretest   # validation and preservation only
python tools/verify_inertia_results.py             # completed final results + replay
"""
from pathlib import Path
import sys
import json
import csv
import argparse
import zipfile
from collections import defaultdict
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_multi_results import verify_dataset,verify_snapshot,verify_copies,equal,boolean,csv_rows,digest
from inertia_generalization.protocol import CONFIG_DIR,MODELS_DIR,RESULTS_DIR,ANALYSIS_DIR,TEST_RHOS,TRAINING_SEEDS,write_json


def verify_freeze():
    freeze=json.loads((CONFIG_DIR/"protocol_freeze.json").read_text(encoding="utf-8"))
    for relative,hash_value in freeze["frozen_sha256"].items():
        assert digest(CONFIG_DIR/relative)==hash_value,"Frozen input changed: "+relative
    for name,hash_value in freeze["protocol_source_sha256"].items():
        assert digest(PACKAGE/"inertia_generalization"/name)==hash_value,"Frozen protocol implementation changed: "+name
    protocol=json.loads((CONFIG_DIR/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["planned_models"]==12 and protocol["pid_total_candidates"]==576
    counts={}
    for axis,expected in (("single",330),("three",450)):
        validation=json.loads((CONFIG_DIR/f"validation_{axis}.json").read_text(encoding="utf-8"))
        nominal=json.loads((CONFIG_DIR/f"validation_nominal_pid_{axis}.json").read_text(encoding="utf-8"))
        test=json.loads((CONFIG_DIR/f"test_{axis}.json").read_text(encoding="utf-8"))
        assert len(validation)==len(nominal)==20 and len(test)==expected
        assert len({s["scenario_id"] for s in test})==expected
        assert [s["inertia_scale"] for s in validation]==[rho for rho in (.7,.85,1.,1.15,1.3) for _ in range(4)]
        for r,n in zip(validation,nominal):
            assert n["inertia_scale"]==1.
            key="theta_deg" if axis=="single" else "euler_deg"
            assert r[key]==n[key] and r["omega_deg_s"]==n["omega_deg_s"]
        teststates={(tuple(np.atleast_1d(s["theta_deg"] if axis=="single" else s["euler_deg"])),tuple(np.atleast_1d(s["omega_deg_s"]))) for s in test}
        valstates={(tuple(np.atleast_1d(s["theta_deg"] if axis=="single" else s["euler_deg"])),tuple(np.atleast_1d(s["omega_deg_s"]))) for s in validation}
        assert not teststates&valstates and len(teststates)==30 and len(valstates)==20
        def state_keys(scenes):
            return {(tuple(np.atleast_1d(s["theta_deg"] if axis=="single" else s["euler_deg"])),tuple(np.atleast_1d(s["omega_deg_s"]))) for s in scenes}
        from multi_algorithm_benchmark.scenarios import validation_scenarios as v2_validation
        old_scenes=json.loads((PACKAGE/"07_单轴三轴多算法对比"/"results"/axis/"test_scenarios.json").read_text(encoding="utf-8"))+v2_validation(axis)
        if axis=="single":
            from satellite_benchmark.evaluation import validation_scenarios as v1_validation
            old_scenes+=json.loads((PACKAGE/"04_实验数据与图表"/"test_scenarios.json").read_text(encoding="utf-8"))+v1_validation()
        assert not (teststates|valstates)&state_keys(old_scenes),"V3 initial state overlaps V1/V2"
        for scene in test+validation+nominal:
            assert all(scene[k]==0. for k in ("disturbance_bias","disturbance_amplitude","pulse_torque","noise_theta_deg","noise_omega_deg"))
            if axis=="three":
                moments=np.array([1.2,1.,.8])*np.asarray(scene["inertia_scale"])
                assert np.isfinite(moments).all() and np.all(moments>0) and np.all(2*moments<=moments.sum()+1e-12)
        counts[axis]={"validation":len(validation),"test":len(test)}
    return {"frozen_input_count":len(freeze["frozen_sha256"]),"all_hashes_match":True,
            "validation_and_test_initial_states_disjoint":True,"v1_v2_initial_states_disjoint":True,"counts":counts}


def verify_pid(recompute=True):
    from multi_algorithm_benchmark.metrics import evaluate_batch,validation_score
    from multi_algorithm_benchmark.controllers import PIDController
    configs=json.loads((MODELS_DIR/"pid_configs.json").read_text(encoding="utf-8"))
    summaries=[]
    count=0
    for axis in ("single","three"):
        rows=csv_rows(MODELS_DIR/f"pid_search_{axis}.csv")
        assert len(rows)==288
        for mode in ("nominal","randomized"):
            group=[r for r in rows if r["mode"]==mode]
            assert len(group)==144 and len({r["candidate_index"] for r in group})==144
            scenes=json.loads((CONFIG_DIR/f"validation{'_nominal_pid' if mode=='nominal' else ''}_{axis}.json").read_text(encoding="utf-8"))
            best=None;failures=0;max_delta=0.
            for row in group:
                gains={k:float(row[k]) for k in ("kp","ki","kd","alpha","integral_limit")}
                stats={k:float(row[k]) for k in ("score","success_rate","rms_error_deg","control_cost")}
                if recompute:
                    results=evaluate_batch(axis,scenes,PIDController(axis,"improved",gains),keep_traces=False)
                    _,actual=validation_score(results)
                    for key in stats:
                        equal(actual[key],stats[key],f"PID validation {axis} {mode} {row['candidate_index']} {key}")
                        max_delta=max(max_delta,abs(actual[key]-stats[key]))
                    failed=sum(r["metrics"]["failed"] for r in results)
                    assert failed==int(row["failed_count"])
                    failures+=failed
                rank=(stats["score"],-stats["rms_error_deg"],-stats["control_cost"],-int(row["candidate_index"]))
                if best is None or rank>best[0]:best=(rank,gains)
            assert best[1]==configs[axis][mode],"PID selection differs from fixed tie-break"
            record=json.loads((MODELS_DIR/f"pid_selected_{axis}_{mode}.json").read_text(encoding="utf-8"))
            assert record["gains"]==best[1] and not record["test_used_for_selection"]
            summaries.append({"axis":axis,"mode":mode,"candidates":144,"selected_gains":best[1],
                              "max_validation_recompute_difference":max_delta,"failure_count":failures})
            count+=len(group)
    return {"candidate_count":count,"all_candidates_recomputed":recompute,"selection_matches":True,"groups":summaries}


def verify_models(manifest=None,pretest=False):
    import torch
    from stable_baselines3 import SAC
    from multi_algorithm_benchmark.controllers import ModelController
    from multi_algorithm_benchmark.improvements import CalibratedModelController
    from multi_algorithm_benchmark.metrics import evaluate_batch,validation_score
    torch.set_num_threads(1)
    reports=[]
    for axis in ("single","three"):
        scenes=json.loads((CONFIG_DIR/f"validation_{axis}.json").read_text(encoding="utf-8"))
        for mode in ("nominal","randomized"):
            for seed in TRAINING_SEEDS:
                folder=MODELS_DIR/axis/mode/f"seed_{seed}"
                metadata=json.loads((folder/"training_metadata.json").read_text(encoding="utf-8"))
                checkpoint=json.loads((folder/"best_checkpoint.json").read_text(encoding="utf-8"))
                assert metadata["status"]=="complete" and metadata["fresh_training"] and not metadata["development"]
                assert metadata["requested_steps"]==metadata["actual_steps"]==1000000
                assert metadata["n_envs"]==8 and metadata["validation_interval"]==100000
                assert metadata["selection_policy"]=="calibrated" and not metadata["test_used_for_selection"]
                assert metadata["initial_state_stream_independent_of_inertia"]
                assert metadata["runtime_source_unchanged"]
                assert json.loads((folder/"validation_scenarios.json").read_text(encoding="utf-8"))==scenes
                assert metadata["validation_scenarios_sha256"]==digest(folder/"validation_scenarios.json")
                assert metadata["selection_score"]=="100*success_rate-rms_error_deg-0.2*control_cost"
                assert metadata["raw_ablation_uses_calibrated_selected_checkpoint"]
                hyper=metadata["algorithm_hyperparameters"]
                for key,value in {"buffer_size":200000,"learning_starts":20000,"batch_size":256,
                                  "tau":.005,"train_freq":[4,"step"],"gradient_steps":2,"ent_coef":"auto_0.01"}.items():
                    assert hyper[key]==value,"SAC hyperparameter drift: "+key
                assert hyper["target_entropy"]==(-1. if axis=="single" else -3.)
                common=metadata["common_hyperparameters"]
                assert common["learning_rate"]==.0003 and common["gamma"]==.995
                # SB3 SAC injects its default use_sde=False into the provided
                # policy_kwargs dictionary during construction.
                assert common["policy_kwargs"] in ({"net_arch":[64,64]}, {"net_arch":[64,64],"use_sde":False})
                assert common["seed"]==seed
                for name,record in metadata["artifacts"].items():
                    path=folder/name
                    assert path.stat().st_size==record["bytes"] and digest(path)==record["sha256"],"Saved artifact changed"
                assert digest(folder/"best_model.zip")==metadata["model_sha256"]==checkpoint["model_sha256"]
                with zipfile.ZipFile(folder/"best_model.zip") as z:best=json.loads(z.read("data"))
                with zipfile.ZipFile(folder/"final_model.zip") as z:final=json.loads(z.read("data"))
                assert best["num_timesteps"]==metadata["best_checkpoint_steps"]==checkpoint["timesteps"]
                assert final["num_timesteps"]==1000000
                raw_rows=csv_rows(folder/"validation_history_raw.csv")
                cal_rows=csv_rows(folder/"validation_history_calibrated.csv")
                assert [int(r["timesteps"]) for r in raw_rows]==list(range(100000,1000001,100000))
                assert [int(r["timesteps"]) for r in cal_rows]==list(range(100000,1000001,100000))
                chosen=max(cal_rows,key=lambda r:(float(r["score"]),-float(r["rms_error_deg"]),-float(r["control_cost"]),-int(r["timesteps"])))
                assert int(chosen["timesteps"])==metadata["best_checkpoint_steps"]
                model=SAC.load(folder/"best_model.zip",device="cpu")
                corrected=CalibratedModelController(model,axis)
                np.testing.assert_array_equal(corrected.zero_action,np.asarray(metadata["best_zero_action_normalized"]))
                np.testing.assert_array_equal(corrected.zero_action,np.asarray(checkpoint["zero_action_normalized"]))
                validation={}
                for variant,controller in (("raw",ModelController(model,axis)),("calibrated",corrected)):
                    _,stats=validation_score(evaluate_batch(axis,scenes,controller,keep_traces=False))
                    for key,value in stats.items():equal(value,checkpoint[variant][key],f"Model validation {axis} {mode} {seed} {variant} {key}")
                    validation[variant]=stats
                env=metadata["env_config"]
                assert env["axis"]==axis and env["mode"]==mode and env["duration"]==20.
                assert env["dt"]==(.05 if axis=="single" else .02)
                assert env["max_torque"]==(.5 if axis=="single" else 2.)
                assert env["inertia"]==([1.] if axis=="single" else [1.2,1.,.8])
                assert env["initial_angle_range_deg"]==[-60.,60.] and env["initial_rate_range_deg_s"]==[-5.,5.]
                assert env["reward_angle_weight"]==6. and env["reward_omega_weight"]==.6 and env["reward_torque_weight"]==.03
                assert env["divergence_penalty"]==1000.
                assert env["inertia_scale_distribution"]==("fixed 1" if mode=="nominal" else "Uniform[0.7,1.3] per episode")
                assert all(env[key]==0. for key in ("disturbance_bias","disturbance_amplitude","pulse_torque","noise_theta_deg","noise_omega_deg"))
                source_differences={k:{"recorded":v,"current":digest(PACKAGE/k) if (PACKAGE/k).is_file() else None}
                                    for k,v in metadata["source_sha256"].items() if not (PACKAGE/k).is_file() or digest(PACKAGE/k)!=v}
                reports.append({"axis":axis,"mode":mode,"seed":seed,"steps":1000000,
                                "checkpoint_steps":metadata["best_checkpoint_steps"],"zero_action":corrected.zero_action.tolist(),
                                "validation_recomputed":validation,"model_sha256":metadata["model_sha256"],
                                "evidence_sha256":{name:digest(folder/name) for name in
                                    ("training_metadata.json","best_checkpoint.json","validation_history_raw.csv",
                                     "validation_history_calibrated.csv","training_manifest.json","validation_scenarios.json",
                                     "session_status.json",*metadata["artifacts"])},
                                "source_differences":source_differences})
                print(f"Verified V3 model {axis} {mode} seed={seed} including validation-only selection",flush=True)
    return {"model_count":len(reports),"exact_total_interactions":len(reports)*1000000,
            "all_best_final_artifact_hashes_match":True,"common_validation_recomputed":True,"models":reports}


def verify_training_initials():
    checked=0;matched=0;output=[]
    for axis in ("single","three"):
        for seed in TRAINING_SEEDS:
            by_mode={}
            for mode in ("nominal","randomized"):
                folder=MODELS_DIR/axis/mode/f"seed_{seed}"
                paths=sorted(folder.glob("training_episode_initial_states*.csv"))
                assert paths,"No training initial-state records"
                rows=[row for path in paths for row in csv_rows(path)]
                keyed={}
                for row in rows:
                    rank,index=int(row["rank"]),int(row["episode_index"])
                    assert 0<=rank<8 and row["mode"]==mode and int(row["stream_seed"])==seed+rank
                    key=(rank,index);assert key not in keyed
                    angle=np.atleast_1d(float(row["theta_deg"]) if axis=="single" else json.loads(row["euler_deg"]))
                    rate=np.atleast_1d(float(row["omega_deg_s"]) if axis=="single" else json.loads(row["omega_deg_s"]))
                    rho=float(row["inertia_scale"])
                    assert np.max(np.abs(angle))<=60 and np.max(np.abs(rate))<=5
                    assert rho==1. if mode=="nominal" else .7<=rho<=1.3
                    keyed[key]=(angle,rate,row["noise_seed"],rho);checked+=1
                assert {key[0] for key in keyed}==set(range(8))
                for rank in range(8):
                    aseq,iseq,nseq=np.random.SeedSequence(seed+rank).spawn(3)
                    arng=np.random.default_rng(aseq);irng=np.random.default_rng(iseq);nrng=np.random.default_rng(nseq)
                    recorded={index:value for (r,index),value in keyed.items() if r==rank}
                    assert sorted(recorded)==list(range(max(recorded)+1)),"Training reset log has missing episode indices"
                    dimensions=1 if axis=="single" else 3
                    for index in range(max(recorded)+1):
                        angle=arng.uniform(-60.,60.,dimensions);rate=arng.uniform(-5.,5.,dimensions)
                        rho=float(irng.uniform(.7,1.3));noise=int(nrng.integers(0,2**31))
                        value=recorded[index]
                        np.testing.assert_array_equal(angle,value[0]);np.testing.assert_array_equal(rate,value[1]);assert str(noise)==value[2]
                        assert value[3]==(1. if mode=="nominal" else rho)
                by_mode[mode]=keyed
            common=set(by_mode["nominal"])&set(by_mode["randomized"])
            for key in common:
                n,r=by_mode["nominal"][key],by_mode["randomized"][key]
                np.testing.assert_array_equal(n[0],r[0]);np.testing.assert_array_equal(n[1],r[1]);assert n[2]==r[2]
                matched+=1
            output.append({"axis":axis,"seed":seed,"matched_rank_episode_count":len(common),
                           "nominal_episode_count":len(by_mode["nominal"]),"randomized_episode_count":len(by_mode["randomized"])})
    return {"checked_episode_records":checked,"matched_initial_state_pairs":matched,
            "independent_inertia_stream_confirmed":True,"every_record_regenerated_from_declared_seed":True,"runs":output}


def verify_evaluation_sources(results,manifest):
    """Report a post-test narrative revision; execution/statistics changes block."""
    narrative_revisions=[];unchanged=[]
    for name,recorded in manifest["evaluation_source_sha256"].items():
        path=PACKAGE/"inertia_generalization"/name
        current=digest(path)
        if current==recorded:unchanged.append(name)
        else:
            assert name=="reporting.py","Post-test execution/statistics source changed: "+name
            narrative_revisions.append({"path":"inertia_generalization/"+name,"evaluation_sha256":recorded,
                "current_sha256":current,"classification":"post-test reporting revision",
                "interpretation":"Narrative/figure generation may be revised after observing frozen results; execution and statistics sources remain bound to the original evaluation manifest."})
    runtime_records=0
    for axis in ("single","three"):
        for mode in ("nominal","randomized"):
            for seed in TRAINING_SEEDS:
                metadata=json.loads((MODELS_DIR/axis/mode/f"seed_{seed}"/"training_metadata.json").read_text(encoding="utf-8"))
                assert metadata["source_sha256"]==metadata["source_sha256_after"],"Training execution source changed during run"
                for relative,recorded in metadata["source_sha256"].items():
                    assert digest(PACKAGE/relative)==recorded,"Training/controller/dynamics source changed: "+relative
                    runtime_records+=1
    return {"evaluation_source_count":len(manifest["evaluation_source_sha256"]),"unchanged_sources":unchanged,
            "post_test_reporting_revisions":narrative_revisions,"evaluation_and_statistics_sources_unchanged":True,
            "training_runtime_hash_records_checked":runtime_records,"training_controller_dynamics_sources_unchanged":True,
            "original_evaluation_manifest_preserved":True}


def verify_deployment(results,manifest):
    """Bind the final controller inventory to the independently audited evidence."""
    deployment=json.loads((results/"deployment_freeze.json").read_text(encoding="utf-8"))
    pretest_path=ANALYSIS_DIR/"测试前独立核验.json"
    pretest=json.loads(pretest_path.read_text(encoding="utf-8"))
    assert pretest["status"]=="passed" and pretest["mode"]=="pretest"
    assert deployment["pretest_audit_sha256"]==digest(pretest_path),"Pre-test evidence changed after deployment freeze"
    assert deployment["test_used_for_selection"] is False
    for name,path in (("protocol_freeze_sha256",CONFIG_DIR/"protocol_freeze.json"),
                      ("pid_configs_sha256",MODELS_DIR/"pid_configs.json")):
        assert deployment[name]==pretest[name]==manifest[name]==digest(path)
    expected={(axis,mode,seed) for axis in ("single","three")
              for mode in ("nominal","randomized") for seed in TRAINING_SEEDS}
    audited={(r["axis"],r["mode"],int(r["seed"])):r for r in pretest["models"]["models"]}
    deployed={(r["axis"],r["mode"],int(r["seed"])):r for r in deployment["models"]}
    assert len(audited)==len(pretest["models"]["models"])==12
    assert len(deployed)==len(deployment["models"])==12 and set(audited)==set(deployed)==expected
    evidence_count=0
    for key in sorted(expected):
        axis,mode,seed=key;folder=MODELS_DIR/axis/mode/f"seed_{seed}"
        a,d=audited[key],deployed[key]
        metadata=json.loads((folder/"training_metadata.json").read_text(encoding="utf-8"))
        required={"training_metadata.json","best_checkpoint.json","validation_history_raw.csv",
                  "validation_history_calibrated.csv","training_manifest.json","validation_scenarios.json",
                  "session_status.json",*metadata["artifacts"]}
        assert set(a["evidence_sha256"])==required,"Incomplete pre-test evidence binding"
        for name,value in a["evidence_sha256"].items():
            assert digest(folder/name)==value,"Pre-test model evidence changed: "+str(folder/name)
            evidence_count+=1
        assert d["model_sha256"]==a["model_sha256"]==digest(folder/"best_model.zip")
        assert d["checkpoint_steps"]==a["checkpoint_steps"]==metadata["best_checkpoint_steps"]
        np.testing.assert_array_equal(d["zero_action_normalized"],a["zero_action"])
    pids=json.loads((MODELS_DIR/"pid_configs.json").read_text(encoding="utf-8"))
    frozen_scenes={};descriptors={}
    for axis in ("single","three"):
        records=manifest["controllers"][axis]
        inventory={(r["controller"],int(r["model_seed"])):r for r in records}
        expected_inventory={(label,0) for label in ("PID_N","PID_R")}|{
            (label,seed) for label in ("SAC_N","SAC_R","SAC_N_raw","SAC_R_raw") for seed in TRAINING_SEEDS}
        assert len(inventory)==len(records)==14 and set(inventory)==expected_inventory
        exported=json.loads((results/axis/"controllers.json").read_text(encoding="utf-8"))
        assert exported==records,"Per-axis controller export differs from manifest"
        scenes=json.loads((CONFIG_DIR/f"test_{axis}.json").read_text(encoding="utf-8"))
        assert json.loads((results/axis/"test_scenarios.json").read_text(encoding="utf-8"))==scenes
        frozen_scenes[axis]={s["scenario_id"]:s for s in scenes}
        for (label,seed),record in inventory.items():
            descriptors[axis,label,str(seed)]=record
            mode="nominal" if label.startswith(("PID_N","SAC_N")) else "randomized"
            assert record["mode"]==mode
            if label.startswith("PID"):
                assert record["source"]=="validation_selected_pid" and record["pid_gains"]==pids[axis][mode]
                continue
            d=deployed[axis,mode,seed]
            expected_path=(MODELS_DIR/axis/mode/f"seed_{seed}"/"best_model.zip").resolve()
            assert Path(record["model_path"]).resolve()==expected_path
            assert record["model_sha256"]==d["model_sha256"]
            assert record["checkpoint_steps"]==d["checkpoint_steps"]
            assert record["actual_training_steps"]==record["requested_training_steps"]==1000000
            assert record["selection"]=="calibrated common in-domain validation"
            raw=label.endswith("_raw")
            assert record["source"]==("same_checkpoint_raw_ablation" if raw else "fresh_inertia_only_training_calibrated")
            if not raw:np.testing.assert_array_equal(record["zero_action_normalized"],d["zero_action_normalized"])
            else:
                calibrated=inventory[label[:-4],seed]
                assert all(record[k]==calibrated[k] for k in ("model_path","model_sha256","checkpoint_steps","selection"))
    episode_rows=csv_rows(results/"episode_metrics.csv")
    covered=set()
    for row in episode_rows:
        axis=row["axis"];identity=(axis,row["controller"],row["model_seed"])
        scene=frozen_scenes[axis][row["scenario_id"]]
        path=(results/row["trace_file"]).resolve()
        assert path.is_relative_to(results.resolve())
        with np.load(path,allow_pickle=False) as z:metadata=json.loads(str(z["metadata_json"]))
        assert metadata["axis"]==axis and metadata["scenario"]==scene,"Episode scene differs from frozen test protocol"
        assert metadata["controller"]==descriptors[identity],"Episode controller differs from deployment manifest"
        for field in ("case_id","family","level","sample_id","experiment","domain"):
            assert str(scene[field])==row[field],"Episode classification differs from frozen test scene: "+field
        equal(scene["rho"],row["rho"],"Episode frozen rho")
        equal(scene["noise_seed"],row["scenario_seed"],"Episode frozen noise seed")
        key=identity+(scene["scenario_id"],)
        assert key not in covered,"Duplicate frozen controller-seed-scene episode"
        covered.add(key)
    expected_episodes={identity+(sid,) for identity in descriptors for sid in frozen_scenes[identity[0]]}
    assert covered==expected_episodes and len(covered)==10920
    session=json.loads((results/"evaluation_session.json").read_text(encoding="utf-8"))
    assert session["status"]=="complete" and session["episode_count"]==10920
    assert manifest["split"]=="test" and manifest["controls_per_scene"]==14
    assert manifest["scene_counts"]=={"single":330,"three":450}
    return {"pretest_evidence_files_rehashed":evidence_count,"pretest_report_bound_to_deployment":True,
            "exact_controller_inventory":True,"same_checkpoint_raw_calibrated_pairs":True,
            "frozen_pid_gains_match":True,"completed_session_preserved":True,
            "every_npz_scene_and_controller_matches_frozen_deployment":True,"bound_episode_count":len(covered)}


def verify_controller_replay(results,manifest):
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    rows=csv_rows(results/"episode_metrics.csv");groups=defaultdict(list)
    for row in rows:groups[(row["axis"],row["controller"],row["model_seed"])].append(row)
    descriptors={(axis,r["controller"],str(r["model_seed"])):r for axis,records in manifest["controllers"].items() for r in records}
    greatest=0.;intervals=0
    for key,group in groups.items():
        axis,label,seed=key;desc=descriptors[key];d=1 if axis=="single" else 3
        data=[]
        for row in group:
            with np.load(results/row["trace_file"],allow_pickle=False) as z:
                data.append({k:z[k] for k in ("observation","torque","failed")})
                meta=json.loads(str(z["metadata_json"]))
                for name,value in (("max_error_deg",float(np.max(z["error_deg"]))),
                                   ("max_rate_deg_s",float(np.max(z["rate_deg_s"])))):
                    equal(value,row[name],"Whole-trajectory maximum CSV "+name)
                    equal(value,meta["metrics"][name],"Whole-trajectory maximum metadata "+name)
        observations=np.stack([r["observation"] for r in data],axis=1)
        torques=np.stack([r["torque"] for r in data],axis=1)
        failures=np.stack([r["failed"] for r in data],axis=1)
        limit=.5 if axis=="single" else 2.;dt=.05 if axis=="single" else .02
        if label.startswith("SAC"):
            model=SAC.load(desc["model_path"],device="cpu")
            zero,_=model.predict(np.zeros((1,d*2),np.float32),deterministic=True)
            zero=np.asarray(zero,float)[0]
            if not label.endswith("_raw"):
                np.testing.assert_array_equal(zero,np.asarray(desc["zero_action_normalized"]))
        else:
            gains=desc["pid_gains"];integral=np.zeros((len(group),d));filtered=np.zeros_like(integral)
        for k in range(len(observations)-1):
            obs=observations[k]
            if label.startswith("SAC"):
                actions,_=model.predict(obs,deterministic=True)
                actions=np.asarray(actions,float)
                if not label.endswith("_raw"):actions=np.clip(actions-zero,-1,1)
                commanded=np.clip(actions,-1,1)*limit
            else:
                error=obs[:,:d].astype(float);rate=obs[:,d:].astype(float)
                correction=-error if axis=="single" else error
                candidate=np.clip(integral+correction*dt,-gains["integral_limit"],gains["integral_limit"])
                filtered=rate.copy() if k==0 else filtered+gains["alpha"]*(rate-filtered)
                base=gains["kp"]*correction-gains["kd"]*filtered
                trial=base+gains["ki"]*candidate;delta=gains["ki"]*(candidate-integral)
                allowed=(np.abs(trial)<=limit)|((trial>limit)&(delta<0))|((trial<-limit)&(delta>0))
                integral=np.where(allowed,candidate,integral)
                commanded=np.clip(base+gains["ki"]*integral,-limit,limit)
            commanded=np.where(failures[k,:,None],0.,commanded)
            diff=float(np.max(np.abs(commanded-torques[k+1])))
            assert diff<2e-7,f"Controller replay {key} interval {k}: {diff}"
            greatest=max(greatest,diff);intervals+=len(group)
        print(f"Recomputed controller actions {axis} {label} seed={seed}: {len(group)} scenes",flush=True)
    return {"controller_seed_groups":len(groups),"recomputed_intervals":intervals,
            "largest_torque_difference":greatest,"independent_pid_equations":True,
            "same_checkpoint_raw_and_calibrated":True}


def verify_pairs(results):
    rows=csv_rows(results/"episode_metrics.csv")
    keyed={(r["axis"],r["controller"],r["model_seed"],r["scenario_id"]):r for r in rows}
    assert len(keyed)==len(rows)
    configs=(("training_mode_pairs.csv","randomized_minus_nominal",4680),
             ("calibration_pairs.csv","calibrated_minus_raw",4680),
             ("pid_pairs.csv","robust_minus_nominal_pid",780))
    fields=("rms_error_deg","tail_mean_error_deg","control_cost","tail_max_error_deg","tail_max_rate_deg_s")
    report={}
    for name,label,expected in configs:
        pairs=csv_rows(results/name);assert len(pairs)==expected
        seen=set()
        for p in pairs:
            assert p["comparison"]==label
            key=(p["axis"],p["controller_a"],p["model_seed"],p["scenario_id"])
            pair_key=key+(p["controller_b"],)
            assert pair_key not in seen;seen.add(pair_key)
            a=keyed[key];b=keyed[(key[0],p["controller_b"],key[2],key[3])]
            if name.startswith("training"):
                assert a["controller"] in ("SAC_N","SAC_N_raw") and b["controller"]==a["controller"].replace("SAC_N","SAC_R")
            elif name.startswith("calibration"):
                assert a["controller"].endswith("_raw") and b["controller"]==a["controller"][:-4]
            else:assert a["controller"]=="PID_N" and b["controller"]=="PID_R"
            sa,sb=boolean(a["success"]),boolean(b["success"])
            equal(sa,p["success_a"],name+" success_a");equal(sb,p["success_b"],name+" success_b")
            equal(int(sb)-int(sa),p["success_change"],name+" success_change")
            equal(sb and not sa,p["success_recovered"],name+" recovered")
            equal(sa and not sb,p["success_lost"],name+" lost")
            equal(a["settling_time_s"],p["settling_time_s_a"],name+" settle A")
            equal(b["settling_time_s"],p["settling_time_s_b"],name+" settle B")
            assert a["trace_file"]==p["trace_file_a"] and b["trace_file"]==p["trace_file_b"]
            for f in fields:
                av,bv=float(a[f]),float(b[f])
                equal(av,p[f+"_a"],name+" A "+f);equal(bv,p[f+"_b"],name+" B "+f)
                equal(bv-av,p[f+"_change"],name+" difference "+f)
        report[name]={"rows":len(pairs),"all_same_seed_same_scene_differences_match":True}
    return report


def verify_boundaries(results):
    rows=csv_rows(results/"episode_metrics.csv")
    groups=defaultdict(list)
    for r in rows:
        if r["experiment"]=="scalar":groups[(r["axis"],r["controller"],r["model_seed"],float(r["rho"]))].append(r)
    points=csv_rows(results/"boundary_points.csv")
    rates={};expected_points={};controllers=defaultdict(set)
    for (axis,controller,seed,rho),group in groups.items():
        assert len(group)==30
        success=sum(boolean(r["success"]) for r in group)
        rates[(axis,controller,seed,rho)]=success/30
        expected_points[(axis,controller,seed,rho)]={"successes":success,"n_episodes":30,"success_rate":success/30,"passed_90_percent":success>=27}
        controllers[(axis,controller)].add(seed)
    for (axis,controller),seeds in controllers.items():
        for rho in TEST_RHOS:
            values=[rates[axis,controller,seed,rho] for seed in seeds]
            expected_points[(axis,controller,"common",rho)]={"successes":None,"n_episodes":30*len(seeds),"success_rate":np.mean(values),
                  "worst_seed_success_rate":min(values),"success_seed_std":np.std(values,ddof=1) if len(values)>1 else None,"passed_90_percent":min(values)>=.9}
    assert len(points)==len(expected_points)
    for p in points:
        key=(p["axis"],p["controller"],p["model_seed"],float(p["rho"]))
        for field,value in expected_points[key].items():equal(value,p[field],"Boundary point "+field)
    intervals=csv_rows(results/"empirical_boundaries.csv")
    assert len(intervals)==sum(len(seeds)+1 for seeds in controllers.values())
    for r in intervals:
        key=(r["axis"],r["controller"]);seed=r["model_seed"]
        passed=[(min(rates[key[0],key[1],s,rho] for s in controllers[key])>=.9 if seed=="common" else rates[key[0],key[1],seed,rho]>=.9) for rho in TEST_RHOS]
        center=list(TEST_RHOS).index(1.)
        if not passed[center]:expected={"lower_rho":None,"upper_rho":None,"grid_points":0,"nominal_passed":False,"reaches_test_lower_limit":False,"reaches_test_upper_limit":False}
        else:
            left=right=center
            for i in range(center-1,-1,-1):
                if not passed[i]:break
                left=i
            for i in range(center+1,len(TEST_RHOS)):
                if not passed[i]:break
                right=i
            expected={"lower_rho":TEST_RHOS[left],"upper_rho":TEST_RHOS[right],"grid_points":right-left+1,"nominal_passed":True,"reaches_test_lower_limit":left==0,"reaches_test_upper_limit":right==len(TEST_RHOS)-1}
        equal(.9,r["threshold"],"Boundary threshold")
        for field,value in expected.items():equal(value,r[field],"Boundary interval "+field)
    return {"point_rows":len(points),"interval_rows":len(intervals),"conservative_all_seed_rule":True,"contiguous_nominal_grid_recomputed":True,"anisotropic_excluded":True}


def verify_aggregate(results):
    rows=csv_rows(results/"episode_metrics.csv");groups=defaultdict(list)
    fields=("rms_error_deg","tail_mean_error_deg","control_cost","tail_max_error_deg","tail_max_rate_deg_s",
            "max_error_deg","max_rate_deg_s")
    for r in rows:
        region="anisotropic" if r["experiment"]=="anisotropic" else "nominal" if float(r["rho"])==1. else "in_domain" if .7<=float(r["rho"])<=1.3 else "out_of_domain"
        for label in (region,"all_scalar" if r["experiment"]=="scalar" else "all_anisotropic"):groups[(r["axis"],r["controller"],label)].append(r)
    actual=json.loads((results/"aggregate_results.json").read_text(encoding="utf-8"))
    assert len(actual)==len(groups)
    for r in actual:
        group=groups[(r["axis"],r["controller"],r["region"])];seeds=defaultdict(list)
        for row in group:seeds[row["model_seed"]].append(row)
        values=[np.mean([boolean(row["success"]) for row in g]) for g in seeds.values()]
        settled=[float(row["settling_time_s"]) for row in group if row["settling_time_s"]!=""]
        expected={"n_episodes":len(group),"success_rate":np.mean(values),"worst_seed_success_rate":min(values),
                  "success_seed_std":np.std(values,ddof=1) if len(values)>1 else None,
                  "settling_time_s_successful":np.mean(settled) if settled else None,
                  "failed_rate":np.mean([boolean(row["failed"]) for row in group])}
        expected.update({f:np.mean([float(row[f]) for row in group]) for f in fields})
        for field,value in expected.items():equal(value,r[field],"Aggregate "+field)
    return {"groups":len(actual),"all_statistics_recomputed":True}


def verify_prior_assets():
    snapshot=json.loads((CONFIG_DIR/"prior_assets_snapshot.json").read_text(encoding="utf-8"))
    changed=[]
    for record in snapshot["files"]:
        path=PACKAGE/record["relative"]
        if not path.is_file() or digest(path)!=record["sha256"]:changed.append(record["relative"])
    assert not changed,"V1/V2 assets changed: "+str(changed)
    archive=PACKAGE/"01_遴选原始参考"/"05_中期前任务收尾新增材料"
    original_new=json.loads((archive/"source_snapshot.json").read_text(encoding="utf-8"))
    content_changed=[];administrative=[]
    for record in original_new["files"]:
        relative=record["source_relative"];path=PACKAGE.parent/relative
        current=digest(path) if path.is_file() else None
        if current!=record["sha256"]:
            item={"path":relative,"snapshot_sha256":record["sha256"],"current_sha256":current,
                  "snapshot_bytes":record["size"],"current_bytes":path.stat().st_size if path.is_file() else None}
            if relative.endswith("/.git/index"):administrative.append(item)
            else:content_changed.append(item)
    assert not content_changed,"New original project contents changed: "+str(content_changed)
    return {"prior_v1_v2_assets":{"file_count":len(snapshot["files"]),"preserved":True},
            "original_v1":verify_snapshot(PACKAGE/"06_结项说明与报告"/"original_snapshot.json"),
            "original_v2":{"checked_file_count":original_new["file_count"],"all_file_bytes_preserved":not administrative,
                           "project_content_preserved":True,"changed_project_content":content_changed,
                           "administrative_metadata_drift":administrative,
                           "interpretation":"Only explicitly listed Git index metadata drift is nonblocking; original snapshot is preserved and no cause is inferred."},
            "copies_v1":verify_copies(PACKAGE/"01_遴选原始参考"/"copy_manifest.json"),
            "copies_v2":verify_copies(archive/"copy_manifest.json")}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--pretest",action="store_true")
    p.add_argument("--inputs-only",action="store_true",help="frozen inputs, all PID validation candidates and preservation; models may still be running")
    p.add_argument("--results",type=Path,default=RESULTS_DIR);args=p.parse_args()
    report={"status":"running","test_used_for_selection":False,"mode":"inputs_only" if args.inputs_only else "pretest" if args.pretest else "final_integrity",
            "protocol_freeze_sha256":digest(CONFIG_DIR/"protocol_freeze.json"),
            "pid_configs_sha256":digest(MODELS_DIR/"pid_configs.json")}
    try:
        report["protocol"]=verify_freeze();report["pid"]=verify_pid()
        report["preservation"]=verify_prior_assets()
        if not args.inputs_only:
            report["models"]=verify_models(pretest=args.pretest)
            report["training_initials"]=verify_training_initials()
        if not args.pretest and not args.inputs_only:
            manifest=json.loads((args.results/"evaluation_manifest.json").read_text(encoding="utf-8"))
            assert manifest["episode_count"]==10920 and manifest["training_model_count"]==12 and not manifest["test_used_for_selection"]
            assert manifest["protocol_freeze_sha256"]==digest(CONFIG_DIR/"protocol_freeze.json")
            assert manifest["deployment_freeze_sha256"]==digest(args.results/"deployment_freeze.json")
            assert manifest["pid_configs_sha256"]==digest(MODELS_DIR/"pid_configs.json")
            for axis,hash_value in manifest["scenario_sha256"].items():assert digest(CONFIG_DIR/f"test_{axis}.json")==hash_value
            report["evaluation_source_evidence"]=verify_evaluation_sources(args.results,manifest)
            report["deployment_evidence"]=verify_deployment(args.results,manifest)
            report["episodes_and_dynamics"]=verify_dataset(args.results)
            report["controller_replay"]=verify_controller_replay(args.results,manifest)
            report["pairs"]=verify_pairs(args.results);report["boundaries"]=verify_boundaries(args.results)
            report["aggregate"]=verify_aggregate(args.results)
        report["status"]="passed"
    except Exception as exc:
        report["status"]="failed";report["error_type"]=type(exc).__name__;report["error"]=str(exc)
        raise
    finally:
        path=ANALYSIS_DIR/("协议与PID独立核验.json" if args.inputs_only else "测试前独立核验.json" if args.pretest else "独立核验记录.json")
        write_json(path,report)
        print(json.dumps({"status":report["status"],"record":str(path),"error":report.get("error")},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
