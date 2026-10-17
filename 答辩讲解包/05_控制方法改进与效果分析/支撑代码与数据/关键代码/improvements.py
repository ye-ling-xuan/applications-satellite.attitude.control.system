"""Explicit policy post-processing, tested separately from original algorithms.

At the known zero-error/zero-rate reference, nominal corrective torque should
be zero. Removing a learned constant action offset enforces this fixed point.
This is neither extra RL training nor integral disturbance rejection. It can
reduce nominal bias while harming performance under persistent disturbance;
both effects are measured on validation and held-out scenes.
"""
from pathlib import Path
import json

import numpy as np

from .controllers import ModelController


class CalibratedModelController(ModelController):
    def __init__(self, model, axis):
        super().__init__(model, axis)
        dimensions = 2 if self.axis == "single" else 6
        self.zero_action = super().act(np.zeros((1, dimensions), np.float32))[0].copy()

    def act(self, observations):
        raw = super().act(observations)
        return np.clip(raw - self.zero_action, -1., 1.)


def validate_calibration(models_dir):
    """Freeze an offset-only intervention using validation; retain every result."""
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    from .metrics import evaluate_batch, validation_score
    from .scenarios import validation_scenarios
    from .training import write_csv, write_json

    torch.set_num_threads(1)
    models_dir = Path(models_dir)
    classes = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    rows = []
    for axis in ("single", "three"):
        scenes = validation_scenarios(axis)
        for algorithm, cls in classes.items():
            for seed in (11, 22, 33):
                path = models_dir / axis / algorithm / f"seed_{seed}" / "best_model.zip"
                model = cls.load(path, device="cpu")
                raw = ModelController(model, axis)
                corrected = CalibratedModelController(model, axis)
                raw_score, raw_stats = validation_score(evaluate_batch(axis, scenes, raw))
                calibrated_score, calibrated_stats = validation_score(evaluate_batch(axis, scenes, corrected))
                row = {"axis": axis, "algorithm": algorithm, "seed": seed,
                       "zero_action_normalized": json.dumps(corrected.zero_action.tolist()),
                       "validation_scene_count": len(scenes), "raw_score": raw_score,
                       "calibrated_score": calibrated_score,
                       **{f"raw_{k}": v for k, v in raw_stats.items() if k != "score"},
                       **{f"calibrated_{k}": v for k, v in calibrated_stats.items() if k != "score"}}
                rows.append(row)
                print(f"Calibration validation {axis} {algorithm} seed={seed}: "
                      f"{raw_stats['success_rate']:.3f} -> {calibrated_stats['success_rate']:.3f}", flush=True)
    write_csv(models_dir / "policy_calibration_validation.csv", rows)
    record = {"intervention": "clip(a(obs)-a(zero_error_and_zero_rate), -1, 1)",
              "split": "validation", "model_weights_changed": False,
              "uses_true_state_or_inertia": False, "has_integral_state": False,
              "test_used_for_design_or_selection": False,
              "selection": "No favorable-only selection: all raw and all calibrated policies enter final test",
              "runs": rows}
    write_json(models_dir / "policy_calibration_manifest.json", record)
    return record
