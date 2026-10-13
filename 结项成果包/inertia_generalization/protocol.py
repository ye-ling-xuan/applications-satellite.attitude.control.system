"""Predeclared V3 protocol. No values are selected using held-out responses."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
STUDY_ROOT=ROOT/"08_惯量域随机化与泛化边界研究"
CONFIG_DIR=STUDY_ROOT/"01_实验协议与配置"
MODELS_DIR=STUDY_ROOT/"02_模型与验证日志"
RESULTS_DIR=STUDY_ROOT/"03_测试数据与配对结果"
ANALYSIS_DIR=STUDY_ROOT/"05_研究分析与核验"
PACKAGE=ROOT
PROTOCOL_DIR=CONFIG_DIR
REPORT_DIR=ANALYSIS_DIR
TRAINING_SEEDS=(11,22,33)
VALIDATION_SEED=2026101202
TEST_SEED=2026101203
VALIDATION_RHOS=(.7,.85,1.,1.15,1.3)
TEST_RHOS=(.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4,1.5)
ANISOTROPIC_SCALES=((.6,1.,1.),(1.4,1.,1.),(1.,.6,1.2),(1.,1.4,.8))
PID_GRID={"kp":(2.,4.,8.,12.),"ki":(0.,.08,.2,.5),
          "kd_single":(1.5,3.,5.),"kd_three":(2.,4.,6.),"alpha":(.2,.5,1.)}

PROTOCOL={
    "schema_version":1,"study":"inertia_only_domain_randomization",
    "axes":["single","three"],"modes":["nominal","randomized"],
    "training_seeds":list(TRAINING_SEEDS),"validation_seed":VALIDATION_SEED,"test_seed":TEST_SEED,
    "initial_angles_deg":[-60.,60.],"initial_rates_deg_s":[-5.,5.],
    "nominal_inertia":{"single":[1.],"three":[1.2,1.,.8]},
    "dt":{"single":.05,"three":.02},"max_torque":{"single":.5,"three":2.},
    "duration":20.,"hold_seconds":5.,"angle_tolerance_deg":1.,"rate_tolerance_deg_s":.5,
    "reward":{"angle":6.,"rate":.6,"torque":.03,"first_divergence_penalty":1000.},
    "training_inertia":{"nominal":[1.,1.],"randomized":[.7,1.3]},
    "inertia_is_constant_within_episode":True,"inertia_observed_by_controller":False,
    "external_disturbance":0.,"pulse":0.,"sensor_noise":0.,
    "validation_rhos":list(VALIDATION_RHOS),"validation_states_per_axis":20,
    "validation_scene_allocation":"4 independently sampled states per rho, ordered by rho",
    "test_rhos":list(TEST_RHOS),"test_states_per_axis":30,
    "anisotropic_scales":[list(x) for x in ANISOTROPIC_SCALES],
    "test_scenes":{"single":330,"three":450},"controls_per_scene":14,"planned_episodes":10920,
    "planned_models":12,"requested_steps_per_model":1000000,"n_envs":8,
    "validation_interval":100000,
    "sac_hyperparameters":{"net_arch":[64,64],"learning_rate":.0003,"gamma":.995,
                           "buffer_size":200000,"learning_starts":20000,"batch_size":256,
                           "tau":.005,"train_freq_steps":4,"gradient_steps":2,
                           "ent_coef":"auto_0.01"},
    "selection_score":"100*success_rate-rms_error_deg-0.2*control_cost",
    "checkpoint_tie_break":["higher_score","lower_rms","lower_control_cost","earlier_checkpoint"],
    "checkpoint_selection_policy":"common domain-internal 20-scene validation using calibrated deployment",
    "calibration_rule":"clip(predict(measured_observation)-predict(zero_observation),-1,1)",
    "raw_ablation":"same checkpoint selected using calibrated validation, no separate raw selection",
    "pid_grid":{k:list(v) for k,v in PID_GRID.items()},"pid_candidates_per_axis_and_mode":144,
    "pid_total_candidates":576,"pid_structure":"rate damping, conditional anti-windup, first-order rate filter",
    "pid_integral_limit":{"single":2.,"three":5.},
    "boundary_success_threshold":.9,"boundary_min_successes":27,
    "boundary":"contiguous measured scalar grid through rho=1; common SAC interval requires every seed >=90%",
    "failure_padding":"freeze after physical failure, zero subsequent torque, never count success",
    "test_used_for_selection":False,
}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def freeze_protocol(output_dir=CONFIG_DIR):
    """Generate fixed split lists and capture their bytes before formal tests."""
    from .scenarios import validation_scenarios,test_scenarios,initial_states
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/"protocol_freeze.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative,hash_value in manifest["frozen_sha256"].items():
            if sha256(output/relative)!=hash_value:raise ValueError("Frozen protocol file changed: "+relative)
        return manifest
    write_json(output/"protocol.json",PROTOCOL)
    files=[output/"protocol.json"]
    for axis in ("single","three"):
        entries={f"validation_{axis}.json":validation_scenarios(axis),
                 f"validation_nominal_pid_{axis}.json":validation_scenarios(axis,nominal=True),
                 f"test_{axis}.json":test_scenarios(axis),
                 f"validation_initial_states_{axis}.json":initial_states(axis,"validation"),
                 f"test_initial_states_{axis}.json":initial_states(axis,"test")}
        for name,value in entries.items():
            path=output/name;write_json(path,value);files.append(path)
        for mode in ("nominal","randomized"):
            path=output/f"training_{axis}_{mode}.json"
            write_json(path,{"axis":axis,"mode":mode,"dt":PROTOCOL["dt"][axis],"duration":20.,
                             "max_torque":PROTOCOL["max_torque"][axis],
                             "inertia_range":PROTOCOL["training_inertia"][mode],
                             "initial_state_rng_independent_of_inertia_rng":True,
                             "disturbance_bias":0.,"disturbance_amplitude":0.,"pulse_torque":0.,
                             "noise_theta_deg":0.,"noise_omega_deg":0.,
                             "reward_angle_weight":6.,"reward_omega_weight":.6,"reward_torque_weight":.03,
                             "divergence_penalty":1000.,"requested_steps":1000000,
                             "training_seeds":list(TRAINING_SEEDS)})
            files.append(path)
    plan=STUDY_ROOT/"研究计划.md"
    manifest={"schema_version":1,"frozen_before_test":True,"test_controller_responses_read":False,
              "protocol_source":"08_惯量域随机化与泛化边界研究/研究计划.md",
              "plan_sha256":sha256(plan),"frozen_sha256":{p.name:sha256(p) for p in files},
              "protocol_source_sha256":{p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("scenarios.py"),Path(__file__).with_name("pid.py"))},
              "counts":{"validation_states_per_axis":20,"test_states_per_axis":30,
                        "test_scenes_single":330,"test_scenes_three":450,"episodes":10920,
                        "models":12,"pid_candidates":576}}
    write_json(manifest_path,manifest)
    return manifest


if __name__=="__main__":print(json.dumps(freeze_protocol(),ensure_ascii=False,indent=2))
