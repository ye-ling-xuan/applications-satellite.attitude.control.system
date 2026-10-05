from dataclasses import replace
import math
import numpy as np
import pytest
from attitude_control.config import Config
from attitude_control.core import Episode, Scenario, observation
from attitude_control.tasks import Task, PolicyAdapter


def test_precision_reward_requires_low_rate_too():
    task = Task("precision_nominal")
    stopped = Episode(Config(), Scenario(angle_deg=0))
    moving = Episode(Config(), Scenario(angle_deg=0, rate_deg_s=1))
    assert task.reward(stopped, 0, 0) == 2
    assert task.reward(moving, 0, 0) < 0


def test_new_scale_and_saved_policy_adapter_match_training_observation():
    task = Task("precision_nominal")
    ep = Episode(Config(), Scenario(angle_deg=30, rate_deg_s=5))
    class Model:
        def predict(self, obs, deterministic):
            return obs, None
    adapted, _ = PolicyAdapter(Model(), task).predict(observation(ep.plant, 0))
    np.testing.assert_array_equal(adapted, task.obs(ep))
    assert adapted[0] == pytest.approx(-0.75)


def test_nominal_profile_has_fixed_inertia_zero_rate_and_no_disturbance():
    task = Task("precision_nominal")
    for i in range(10):
        scenario = task.sample(Config(), np.random.default_rng(i), True)
        assert scenario.inertia == 1 and scenario.rate_deg_s == 0 and scenario.bias_torque == 0
        assert abs(scenario.angle_deg) <= 40


def test_v1_observation_and_reward_are_unchanged():
    task = Task()
    ep = Episode(Config(), Scenario())
    np.testing.assert_array_equal(task.obs(ep), observation(ep.plant, 0))
    assert task.reward(ep, 1, -3.5) == -3.5


def test_task_saved_version_is_checked():
    assert Task.from_dict(Task("precision_robust").to_dict()).profile == "precision_robust"
    with pytest.raises(ValueError):
        Task.from_dict({"profile": "precision_nominal", "version": "unknown"})


def test_reset_does_not_sample_outside_failure_boundary():
    task = Task()
    c = replace(Config(), failure_angle_deg=10)
    for i in range(10):
        assert abs(task.sample(c, np.random.default_rng(i), False).angle_deg) <= 10


def test_torque_ablation_changes_only_quadratic_torque_term():
    ep = Episode(Config(), Scenario(angle_deg=10, rate_deg_s=5))
    strong = Task("precision_nominal", 0.5)
    weak = Task("precision_nominal", 0.05)
    assert weak.reward(ep, 2, 0) - strong.reward(ep, 2, 0) == pytest.approx(1.8)
    np.testing.assert_array_equal(strong.obs(ep), weak.obs(ep))
    assert strong.sample(Config(), np.random.default_rng(7), True) == weak.sample(Config(), np.random.default_rng(7), True)


def test_v1_task_checkpoint_recovers_original_torque_penalty():
    task = Task.from_dict({"version": "explicit-ppo-task-v1", "profile": "precision_nominal"})
    assert task.torque_penalty == 0.5


@pytest.mark.parametrize("penalty", [float("nan"), float("inf"), -1])
def test_invalid_torque_penalty_is_rejected(penalty):
    with pytest.raises(ValueError):
        Task("precision_nominal", penalty)
