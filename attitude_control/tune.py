"""Tune classical controllers on a validation set, never on the held-out suite."""
import argparse
from dataclasses import replace
from itertools import product
import numpy as np
from .config import Config
from .controllers import PID
from .core import Scenario
from .experiments import create_output, save_json, provenance, scenario_dict
from .simulation import rollout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/tuning_v1")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)
    out = create_output(args.output)
    base = Config()
    rng = np.random.default_rng(args.seed)
    scenarios = [Scenario(name=f"validation_{i}", angle_deg=float(rng.uniform(-60, 60)),
                          rate_deg_s=float(rng.uniform(-10, 10)),
                          inertia=[0.7, 1.0, 1.3][i % 3],
                          bias_torque=[-0.03, 0.0, 0.03][i % 3], seed=args.seed + i)
                 for i in range(12)]
    trials = []
    best = {}
    for name, integrals in [("pd", [0.0]), ("pid", [0.1, 0.3, 0.6])]:
        winner = None
        for kp, kd, ki in product([1.0, 3.0, 6.0], [1.0, 3.0, 5.0], integrals):
            cfg = replace(base, kp=kp, kd=kd, ki=ki)
            costs = []
            for scenario in scenarios:
                controller = PID(kp, kd, ki)
                result, _ = rollout(cfg, scenario, lambda ep, obs: controller.command(ep))
                costs.append(result["cost_integral"] + (100.0 if result["failed"] else 0.0))
            trial = {"controller": name, "kp": kp, "kd": kd, "ki": ki,
                     "validation_cost": float(np.mean(costs))}
            trials.append(trial)
            if winner is None or trial["validation_cost"] < winner["validation_cost"]:
                winner = trial
        best[name] = winner
        print(f"{name.upper()}: {winner}")
    pd_cfg = replace(base, **{k: best["pd"][k] for k in ["kp", "kd", "ki"]})
    pid_cfg = replace(base, **{k: best["pid"][k] for k in ["kp", "kd", "ki"]})
    save_json(out / "tuning.json", {"seed": args.seed, "selection": "minimum mean validation cost",
              "pd_config": pd_cfg.to_dict(), "pid_config": pid_cfg.to_dict(),
              "validation_scenarios": list(map(scenario_dict, scenarios)),
              "trials": trials, "provenance": provenance()})


if __name__ == "__main__":
    main()
