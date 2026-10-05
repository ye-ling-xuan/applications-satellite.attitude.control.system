import numpy as np
import pytest
pytest.importorskip("stable_baselines3")
from attitude_control.config import Config
from attitude_control.benchmark import held_out_suite, across_seed_summary


def test_final_suite_is_reproducible_and_nominal_group_is_separate():
    cfg = Config()
    first = held_out_suite(cfg, 9040, 6)
    second = held_out_suite(cfg, 9040, 6)
    assert first == second
    assert len(first) == 42
    for group, scenario in first[:6]:
        assert group == "in_distribution"
        assert scenario.rate_deg_s == 0 and scenario.inertia == 1 and scenario.bias_torque == 0
        assert 5 <= abs(scenario.angle_deg) <= 40
    assert first != held_out_suite(cfg, 9041, 6)


def test_seed_sd_is_computed_across_models():
    rows = []
    for name, success in [("PPO_seed7", 1.0), ("PPO_seed19", 0.0), ("PD", 1.0), ("PID", 1.0)]:
        rows.append({"group": "all", "controller": name, "success_rate": success,
            "mean_tail_mean_abs_error_deg": 0.1, "mean_torque_effort_nm2_s": 1.0,
            "mean_cost_integral": 1.0})
    result = across_seed_summary(rows, ["PPO_seed7", "PPO_seed19"])[0]
    assert result["ppo_success_rate_mean_across_seeds"] == 0.5
    assert result["ppo_success_rate_sample_sd_across_seeds"] == np.sqrt(0.5)
    assert result["training_seeds"] == 2
