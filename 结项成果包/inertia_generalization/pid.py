"""Equal-budget nominal and robust PID selection, restricted to validation."""
from pathlib import Path
import csv
import json
import numpy as np
from multi_algorithm_benchmark.controllers import PIDController
from multi_algorithm_benchmark.metrics import evaluate_batch,validation_score
from multi_algorithm_benchmark.envs import axis_name
from .protocol import PID_GRID,MODELS_DIR,PROTOCOL,sha256,write_json
from .scenarios import validation_scenarios


def candidates(axis):
    axis=axis_name(axis)
    for kp in PID_GRID["kp"]:
        for ki in PID_GRID["ki"]:
            for kd in PID_GRID["kd_"+axis]:
                for alpha in PID_GRID["alpha"]:
                    yield {"kp":kp,"ki":ki,"kd":kd,"alpha":alpha,
                           "integral_limit":PROTOCOL["pid_integral_limit"][axis]}


def make_pid(axis,mode,gains):
    if mode not in ("nominal","randomized"):raise ValueError("mode must be nominal or randomized")
    return PIDController(axis,kind="improved",gains=gains)


def tune_pid(axis,output_dir=MODELS_DIR):
    axis=axis_name(axis);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    selections={};all_rows=[]
    for mode in ("nominal","randomized"):
        scenes=validation_scenarios(axis,nominal=mode=="nominal")
        write_json(output/f"pid_validation_{axis}_{mode}.json",scenes)
        best=None
        for index,gains in enumerate(candidates(axis)):
            results=evaluate_batch(axis,scenes,make_pid(axis,mode,gains),keep_traces=False)
            score,stats=validation_score(results)
            row={"axis":axis,"mode":mode,"candidate_index":index,**gains,**stats,
                 "failed_count":sum(r["metrics"]["failed"] for r in results),"validation_scene_count":len(scenes)}
            all_rows.append(row)
            rank=(score,-stats["rms_error_deg"],-stats["control_cost"],-index)
            if best is None or rank>best[0]:best=(rank,gains,stats,index)
        selections[mode]=best[1]
        write_json(output/f"pid_selected_{axis}_{mode}.json",{"axis":axis,"mode":mode,"gains":best[1],
                   "validation_stats":best[2],"candidate_index":best[3],"candidate_count":144,
                   "selection_split":"validation","test_used_for_selection":False,
                   "validation_scenes_sha256":sha256(output/f"pid_validation_{axis}_{mode}.json")})
        print(f"PID {axis} {mode}: {best[1]} validation={best[2]}",flush=True)
    with (output/f"pid_search_{axis}.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    path=output/"pid_configs.json"
    configurations=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    configurations[axis]=selections;write_json(path,configurations)
    return selections
