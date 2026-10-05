"""Checkpoint selection on fixed validation cases, independent of final test cases."""
import csv
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .controllers import action_command
from .core import Scenario
from .simulation import rollout
from .tasks import PolicyAdapter
from .experiments import save_json, scenario_dict


class ValidationCallback(BaseCallback):
    def __init__(self, config, mode, task, output, interval=20000):
        super().__init__()
        self.config, self.mode, self.task = config, mode, task
        self.output = Path(output)
        self.interval = interval
        self.next_check = interval
        self.best_key = None
        self.best_step = None
        self.rows = []
        self.scenarios = [Scenario(name=f"validation_{i}", angle_deg=config.target_deg + angle, rate_deg_s=rate,
                          inertia=config.inertia, seed=901 + i)
                          for i, (angle, rate) in enumerate([
                              (-35, 0), (-15, 0), (15, 0), (35, 0),
                              (-25, -5), (-25, 5), (25, -5), (25, 5)])]
        save_json(self.output / "validation_scenarios.json", list(map(scenario_dict, self.scenarios)))

    def evaluate(self):
        policy = PolicyAdapter(self.model, self.task)
        results = []
        for scenario in self.scenarios:
            result, _ = rollout(self.config, scenario, lambda ep, obs: action_command(
                policy.predict(obs, deterministic=True)[0], ep, self.mode))
            results.append(result)
        success = float(np.mean([r["success"] for r in results]))
        failure = float(np.mean([r["failed"] for r in results]))
        error = float(np.mean([r["tail_mean_abs_error_deg"] for r in results]))
        cost = float(np.mean([r["cost_integral"] for r in results]))
        # Declare selection order in advance: task success, failure, accuracy, then cost.
        key = (1 - success, failure, error, cost)
        row = dict(timesteps=self.num_timesteps, success_rate=success, failure_rate=failure,
                   tail_mean_abs_error_deg=error, cost_integral=cost)
        self.rows.append(row)
        with (self.output / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        if self.best_key is None or key < self.best_key:
            self.best_key, self.best_step = key, self.num_timesteps
            self.model.save(self.output / "model")
        print(f"validation steps={self.num_timesteps} success={success:.0%} tail_error={error:.3f} deg", flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next_check:
            self.evaluate()
            self.next_check = self.num_timesteps + self.interval
        return True

    def _on_training_end(self):
        self.evaluate()
