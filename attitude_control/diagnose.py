"""Reproduce archived PPO and inspect v1 policies without changing or retraining them."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from .config import Config
from .core import Scenario
from .controllers import PID
from .evaluate import load_policy, summarize, write_csv
from .experiments import create_output, provenance, save_json
from .metrics import metrics
from .simulation import rollout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/diagnosis_v1")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    out = create_output(args.output)
    root = Path(__file__).resolve().parents[1]
    legacy = root / "yxy的学习笔记" / "gpt改良版"
    if not (legacy / "sat_env.py").is_file():
        legacy = root / "大一立项AI在卫星姿态调整中的应用" / "yxy的学习笔记" / "gpt改良版"
    spec = importlib.util.spec_from_file_location("archived_attitude_env", legacy / "sat_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archived = PPO.load(legacy / "ppo_satellite_improved.zip", device="cpu")
    cap = 2 * math.tanh(1)
    cfg = replace(Config(), max_torque=cap)
    rows = []
    for initial in [-60, -30, -10, 10, 30, 60]:
        env = module.SatelliteAttitudeEnv()
        env.reset()
        env.sat.set_state(initial, 0)
        obs = env._get_obs()
        angles, rates, torques = [initial], [0.0], []
        reward_sum = 0.0
        failed = False
        for _ in range(cfg.steps):
            action, _ = archived.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            reward_sum += reward
            angles.append(env.sat.get_angle_deg())
            rates.append(env.sat.get_omega_deg())
            torques.append(float(2 * np.tanh(action[0])))
            if terminated and env.current_step < cfg.steps:
                failed = True
                break
        t = np.arange(len(angles)) * cfg.dt
        row = metrics(t, angles, rates, torques, cfg, failed)
        u = np.asarray(torques)
        cost = cfg.dt * np.sum(
            (np.asarray(angles[1:]) / cfg.angle_scale_deg) ** 2
            + cfg.rate_weight * (np.asarray(rates[1:]) / cfg.rate_scale_deg_s) ** 2
            + cfg.torque_weight * (u / cfg.max_torque) ** 2
            + cfg.slew_weight * (np.diff(np.r_[0.0, u]) / cfg.max_torque) ** 2)
        rows.append({"controller": "archived_PPO_original_dynamics", "group": "nominal",
                     "scenario": str(initial), "cost_integral": float(cost),
                     "episode_return": float(reward_sum), **row})
        np.savez_compressed(out / f"archived_{initial}.npz", time_s=t, angle_deg=angles,
                            rate_deg_s=rates, torque_nm=torques)
        # Same archived policy, exact new dynamics, and an identically limited PD.
        scenario = Scenario(angle_deg=initial)
        def command(ep, obs):
            legacy_obs = np.array([ep.plant.theta / math.radians(40),
                                   ep.plant.omega / math.radians(50)], np.float32)
            action = archived.predict(legacy_obs, deterministic=True)[0]
            return float(2 * np.tanh(action[0]))
        result, _ = rollout(cfg, scenario, command)
        rows.append({"controller": "archived_PPO_exact_dynamics", "group": "nominal",
                     "scenario": str(initial), **result})
        controller = PID(6, 3)
        result, _ = rollout(cfg, scenario, lambda ep, obs: controller.command(ep))
        rows.append({"controller": "PD_same_torque_cap", "group": "nominal",
                     "scenario": str(initial), **result})
    tuning = json.loads((root / "artifacts/tuning_v1/tuning.json").read_text(encoding="utf-8"))
    new_cfg = Config(**tuning["pd_config"])
    probes = []
    for mode, directory in [("pure", "pure_seed7_v1"), ("residual", "residual_seed7_v1")]:
        model, _ = load_policy(root / "artifacts" / directory, new_cfg, mode)
        for angle in [-10, -1, 0, 1, 10]:
            obs = np.array([-math.radians(angle) / math.pi, 0, 0], np.float32)
            a = float(model.predict(obs, deterministic=True)[0][0])
            torque = 2 * a if mode == "pure" else -new_cfg.kp * math.radians(angle) + 0.5 * a
            probes.append({"mode": mode, "angle_deg": angle, "action": a,
                           "command_at_zero_rate_nm": torque})
    write_csv(out / "episodes.csv", rows)
    summaries = summarize(rows)
    save_json(out / "summary.json", summaries)
    save_json(out / "policy_probes.json", probes)
    save_json(out / "manifest.json", {"archived_effective_torque_cap_nm": cap,
        "archived_model_sha256": hashlib.sha256((legacy / "ppo_satellite_improved.zip").read_bytes()).hexdigest(),
        "archived_env_sha256": hashlib.sha256((legacy / "sat_env.py").read_bytes()).hexdigest(),
        "note": "Reproduction/diagnostic cases, not an untouched final test set. Cost uses the same normalized quadratic definition; episode_return follows each original environment and is not comparable across reward definitions.",
        "provenance": provenance()})
    for row in summaries:
        if row["group"] == "all":
            print(json.dumps(row))
    print("Policy action probes:", json.dumps(probes))


if __name__ == "__main__":
    main()
