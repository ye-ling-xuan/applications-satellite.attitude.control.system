"""Independent validation-only audit. Never loads the final test scenarios."""
from pathlib import Path
import sys
import csv
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def main():
    import torch
    from stable_baselines3 import PPO,SAC,TD3
    from multi_algorithm_benchmark.controllers import PIDController,ModelController
    from multi_algorithm_benchmark.metrics import evaluate_batch,validation_score
    from multi_algorithm_benchmark.scenarios import validation_scenarios
    torch.set_num_threads(1)
    output=Path(__file__).resolve().parent
    models=output.parent/"models"
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"multi_algorithm_benchmark").glob("*.py")}
    configs=json.loads((models/"pid_configs.json").read_text(encoding="utf-8"))
    pid={"selection_split":"validation","test_read":False,"source_sha256":source,
         "source_change":"Only failed-row padding was changed: after a failure actuator torque and disturbance are zero, state remains frozen. First failure interval still incurs torque cost. Dynamics and every no-failure trajectory are identical.",
         "axes":{}}
    for axis in ("single","three"):
        scenes=validation_scenarios(axis)
        rows=list(csv.DictReader((models/f"pid_search_{axis}.csv").open(encoding="utf-8-sig")))
        family_best={}
        failures=0
        max_difference=0.
        for row in rows:
            gains={k:float(row[k]) for k in ("kp","ki","kd","alpha","integral_limit")}
            result=evaluate_batch(axis,scenes,PIDController(axis,row["kind"],gains),keep_traces=False)
            score,stats=validation_score(result)
            failed=sum(r["metrics"]["failed"] for r in result)
            failures+=failed
            max_difference=max(max_difference,abs(score-float(row["score"])))
            rank=(score,-stats["rms_error_deg"],-stats["control_cost"])
            if row["kind"] not in family_best or rank>family_best[row["kind"]][0]:
                family_best[row["kind"]]=(rank,gains)
        selected={}
        for kind,gains in configs[axis].items():
            result=evaluate_batch(axis,scenes,PIDController(axis,kind,gains),keep_traces=False)
            _,stats=validation_score(result)
            selected[kind]={"gains":gains,"stats":stats,
                            "failure_count":sum(r["metrics"]["failed"] for r in result)}
        same=all(family_best[kind][1]==configs[axis][kind] for kind in ("tuned","improved"))
        pid["axes"][axis]={"candidate_count":len(rows),"validation_scene_count":len(scenes),
                           "failure_count_across_candidate_scenes":failures,
                           "max_score_difference_after_padding_fix":max_difference,
                           "chosen_gains_unchanged":same,"selected":selected}
        print(axis,"PID audited",len(rows),"candidates; same selection",same,flush=True)
    write_json(output/"PID验证选择复核.json",pid)
    audits=[]
    for algorithm,loader in (("PPO",PPO),("SAC",SAC),("TD3",TD3)):
        for seed in (11,22,33):
            folder=models/"single"/algorithm/f"seed_{seed}"
            model=loader.load(folder/"best_model.zip",device="cpu")
            checkpoint=json.loads((folder/"best_checkpoint.json").read_text(encoding="utf-8"))
            metadata=json.loads((folder/"training_metadata.json").read_text(encoding="utf-8"))
            scenes=json.loads((folder/"validation_scenarios.json").read_text(encoding="utf-8"))
            assert scenes==validation_scenarios("single"),"Saved validation scenes changed"
            result=evaluate_batch("single",scenes,ModelController(model,"single"),keep_traces=False)
            score,stats=validation_score(result)
            deltas={k:float(stats[k]-checkpoint[k]) for k in stats}
            audits.append({"algorithm":algorithm,"seed":seed,"selected_timesteps":checkpoint["timesteps"],
                           "validation_scene_count":len(scenes),"failure_count":sum(r["metrics"]["failed"] for r in result),
                           "recomputed":stats,"stored":checkpoint,"differences":deltas,
                           "matches_within_1e_8":max(abs(v) for v in deltas.values())<1e-8,
                           "source_hash_differences":{k:{"recorded":v,"current":source.get(k)}
                                                      for k,v in metadata["source_sha256"].items() if source.get(k)!=v}})
            print(algorithm,seed,"best validation",stats,"differences",deltas,flush=True)
    write_json(output/"单轴九模型验证复核.json",{"selection_split":"validation","test_read":False,
                                               "source_sha256":source,"models":audits,
                                               "all_match":all(r["matches_within_1e_8"] for r in audits)})


if __name__=="__main__":main()
