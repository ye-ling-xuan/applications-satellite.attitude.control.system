"""Matched held-out tests after all V3 checkpoints and PID gains are frozen."""
import json
from pathlib import Path
import numpy as np
from multi_algorithm_benchmark.controllers import ModelController, PIDController
from multi_algorithm_benchmark.improvements import CalibratedModelController
from multi_algorithm_benchmark.evaluation import summarize, write_csv, write_json
from multi_algorithm_benchmark.metrics import evaluate_batch
from .protocol import MODELS_DIR, RESULTS_DIR, CONFIG_DIR, ANALYSIS_DIR, TRAINING_SEEDS, freeze_protocol, sha256
from .analysis import boundary_tables, paired_tables, aggregate


def _export_results(output_dir,axis,scenes,descriptors):
    """V3 export retains V2 metrics and adds full-horizon maxima from saved traces."""
    output_dir=Path(output_dir); trace_dir=output_dir/"raw_traces"
    trace_dir.mkdir(parents=True,exist_ok=True)
    rows,records=[],[]
    for descriptor in descriptors:
        controller=descriptor.pop("object");records.append(dict(descriptor))
        results=evaluate_batch(axis,scenes,controller)
        for result in results:
            scene,metrics,trace=result["scenario"],result["metrics"],result["trace"]
            metrics.update(max_rate_deg_s=float(np.max(trace["rate_deg_s"])),max_error_deg=float(np.max(trace["error_deg"])))
            relative=Path("raw_traces")/f"{descriptor['controller']}_seed_{descriptor['model_seed']}_{scene['scenario_id']}.npz"
            metadata={"axis":axis,"controller":descriptor,"scenario":scene,"metrics":metrics}
            np.savez_compressed(output_dir/relative,**trace,metadata_json=json.dumps(metadata,ensure_ascii=False))
            rows.append({"axis":axis,"scenario_id":scene["scenario_id"],"case_id":scene["case_id"],
                         "family":scene["family"],"level":scene["level"],"sample_id":scene["sample_id"],
                         "scenario_seed":scene["noise_seed"],"controller":descriptor["controller"],
                         "model_seed":descriptor["model_seed"],**metrics,"trace_file":relative.as_posix()})
        print(f"Evaluated {axis} {descriptor['controller']} seed={descriptor['model_seed']}: {len(results)} full-horizon scenes",flush=True)
    write_json(output_dir/"test_scenarios.json",scenes);write_json(output_dir/"controllers.json",records)
    write_csv(output_dir/"episode_metrics.csv",rows);write_csv(output_dir/"summary.csv",summarize(rows))
    return rows,records


def _evaluate_models(models_dir=MODELS_DIR, output_dir=RESULTS_DIR):
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    freeze = freeze_protocol()
    models_dir, output_dir = Path(models_dir), Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    if (output_dir/"evaluation_manifest.json").exists():
        raise FileExistsError("Completed final results are preserved; choose a new output directory")
    pid_path=models_dir/"pid_configs.json"
    pids=json.loads(pid_path.read_text(encoding="utf-8"))
    pretest_path=ANALYSIS_DIR/"测试前独立核验.json"
    pretest=json.loads(pretest_path.read_text(encoding="utf-8"))
    if pretest.get("status")!="passed":raise ValueError("Independent pre-test audit must pass before held-out evaluation")
    if pretest.get("protocol_freeze_sha256")!=sha256(CONFIG_DIR/"protocol_freeze.json") or pretest.get("pid_configs_sha256")!=sha256(pid_path):
        raise ValueError("Audited protocol or PID artifact changed")
    audited={(r["axis"],r["mode"],int(r["seed"])):r for r in pretest["models"]["models"]}
    loaded = {}
    for axis in ("single","three"):
        for mode in ("nominal","randomized"):
            for seed in TRAINING_SEEDS:
                folder=models_dir/axis/mode/f"seed_{seed}"
                meta=json.loads((folder/"training_metadata.json").read_text(encoding="utf-8"))
                path=folder/"best_model.zip"
                if meta.get("status")!="complete" or meta.get("development") or not meta.get("eligible_for_final_research"):
                    raise ValueError("Incomplete or development training is not formal research evidence")
                if (meta.get("axis"),meta.get("mode"),meta.get("seed"))!=(axis,mode,seed):
                    raise ValueError("Training identity does not match model directory")
                if not meta.get("runtime_source_unchanged") or meta.get("source_sha256")!=meta.get("source_sha256_after"):
                    raise ValueError("Training runtime changed during formal execution")
                if meta.get("selection_policy")!="calibrated" or meta.get("test_used_for_selection"):
                    raise ValueError("Unexpected selection policy or use of held-out responses")
                if meta.get("validation_checkpoint_steps")!=list(range(100000,1000001,100000)):
                    raise ValueError("Formal validation cadence changed")
                if meta["requested_steps"]!=1000000 or meta["actual_steps"]!=1000000:
                    raise ValueError("All formal SAC runs must reach the predeclared million-step budget")
                if meta["model_sha256"]!=sha256(path): raise ValueError("Model hash changed")
                audit_record=audited.get((axis,mode,seed),{})
                if audit_record.get("model_sha256")!=sha256(path):raise ValueError("Model differs from independently audited artifact")
                if not audit_record.get("evidence_sha256"):raise ValueError("Pre-test audit must bind checkpoint selection evidence")
                for relative,expected in audit_record["evidence_sha256"].items():
                    if sha256(folder/relative)!=expected:raise ValueError("Audited training evidence changed: "+relative)
                model=SAC.load(path,device="cpu")
                if model.num_timesteps!=meta["best_checkpoint_steps"]:raise ValueError("Best checkpoint step mismatch")
                bias=CalibratedModelController(model,axis).zero_action
                if not np.allclose(bias,meta["best_zero_action_normalized"],rtol=0,atol=1e-7):raise ValueError("Calibration bias mismatch")
                loaded[axis,mode,seed]=(model,meta,path)
    # This records deployment artifacts before evaluating a single held-out scene.
    deployment={"test_used_for_selection":False,"protocol_freeze_sha256":sha256(CONFIG_DIR/"protocol_freeze.json"),
                "pid_configs_sha256":sha256(pid_path),"pretest_audit_sha256":sha256(pretest_path),"models":[]}
    for (axis,mode,seed),(model,meta,path) in loaded.items():
        deployment["models"].append({"axis":axis,"mode":mode,"seed":seed,"model_sha256":sha256(path),
                                     "checkpoint_steps":int(model.num_timesteps),
                                     "zero_action_normalized":CalibratedModelController(model,axis).zero_action.tolist()})
    write_json(output_dir/"deployment_freeze.json",deployment)
    all_rows,controller_records,counts=[],{},{}
    for axis in ("single","three"):
        scene_path=CONFIG_DIR/f"test_{axis}.json"
        scenes=json.loads(scene_path.read_text(encoding="utf-8"))
        descriptors=[]
        for mode,label in (("nominal","PID_N"),("randomized","PID_R")):
            gains=pids[axis][mode]
            if "gains" in gains: gains=gains["gains"]
            descriptors.append({"controller":label,"model_seed":0,"source":"validation_selected_pid", "mode":mode,
                                "pid_gains":gains,"object":PIDController(axis,"improved",gains=gains)})
        for mode,label in (("nominal","SAC_N"),("randomized","SAC_R")):
            for seed in TRAINING_SEEDS:
                model,meta,path=loaded[axis,mode,seed]
                corrected=CalibratedModelController(model,axis)
                common={"model_seed":seed,"mode":mode,"model_path":str(path),"model_sha256":sha256(path),
                        "actual_training_steps":meta["actual_steps"],"requested_training_steps":meta["requested_steps"],
                        "checkpoint_steps":int(model.num_timesteps),"selection":"calibrated common in-domain validation"}
                descriptors.append({**common,"controller":label,"source":"fresh_inertia_only_training_calibrated",
                                    "zero_action_normalized":corrected.zero_action.tolist(),"object":corrected})
                descriptors.append({**common,"controller":label+"_raw","source":"same_checkpoint_raw_ablation",
                                    "object":ModelController(model,axis)})
        rows,records=_export_results(output_dir/axis,axis,scenes,descriptors)
        scene_by_id={s["scenario_id"]:s for s in scenes}
        for row in rows:
            scene=scene_by_id[row["scenario_id"]]
            row.update(experiment=scene["experiment"],rho=scene["rho"],domain=scene["domain"])
            row["trace_file"]=axis+"/"+row["trace_file"]
        all_rows.extend(rows); controller_records[axis]=records; counts[axis]=len(scenes)
    if len(all_rows)!=10920: raise ValueError("Final scene/controller count differs from predeclared protocol")
    write_csv(output_dir/"episode_metrics.csv",all_rows)
    write_csv(output_dir/"summary.csv",summarize(all_rows))
    training,calibration,pid=paired_tables(all_rows)
    for name,rows in (("training_mode_pairs.csv",training),("calibration_pairs.csv",calibration),("pid_pairs.csv",pid)):
        write_csv(output_dir/name,rows)
    points,intervals=boundary_tables(all_rows)
    write_csv(output_dir/"boundary_points.csv",points);write_csv(output_dir/"empirical_boundaries.csv",intervals)
    write_json(output_dir/"aggregate_results.json",aggregate(all_rows))
    manifest={"split":"test","episode_count":len(all_rows),"scene_counts":counts,"training_model_count":len(loaded),
              "controls_per_scene":14,"training_mode_pair_count":len(training),"calibration_pair_count":len(calibration),
              "pid_pair_count":len(pid),"controllers":controller_records,"test_used_for_selection":False,
              "protocol_freeze_sha256":sha256(CONFIG_DIR/"protocol_freeze.json"),"deployment_freeze_sha256":sha256(output_dir/"deployment_freeze.json"),
              "pid_configs_sha256":sha256(pid_path),"scenario_sha256":{axis:sha256(CONFIG_DIR/f"test_{axis}.json") for axis in ("single","three")},
              "evaluation_source_sha256":{p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")},
              "raw_ablation_policy":"same checkpoint chosen by calibrated validation, no independent raw selection",
              "boundary_policy":"contiguous measured grid through rho=1; all three seeds must individually pass 27/30 for common interval"}
    write_json(output_dir/"evaluation_manifest.json",manifest)
    return {k:manifest[k] for k in ("episode_count","scene_counts","training_model_count","training_mode_pair_count","calibration_pair_count","pid_pair_count")}


def evaluate_models(models_dir=MODELS_DIR,output_dir=RESULTS_DIR):
    """A failed test run remains reviewable; an explicit new folder is needed to retry."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("Final test output must be empty; completed and interrupted runs are preserved")
    marker=output/"evaluation_session.json"
    with marker.open("x",encoding="utf-8") as stream:json.dump({"status":"running"},stream)
    status={"status":"running","outputs_preserved_on_failure":True}
    try:
        result=_evaluate_models(models_dir,output)
        status.update(status="complete",**result)
        return result
    except BaseException as exc:
        status.update(status="incomplete",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:write_json(marker,status)
