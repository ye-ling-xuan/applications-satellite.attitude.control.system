from dataclasses import replace
import numpy as np
import pytest
from attitude_control.config import Config
from attitude_control.metrics import metrics


def samples(angle, rate=None, torque=None):
    config = Config(dt=1, duration=len(angle) - 1)
    return metrics(np.arange(len(angle)), angle, rate or [0] * len(angle),
                   torque or [0] * (len(angle) - 1), config)


def test_overshoot_excludes_initial_error_and_handles_both_directions():
    assert samples([30, 10, -2, 0, 0])["overshoot_deg"] == 2
    assert samples([-30, -10, 2, 0, 0])["overshoot_deg"] == 2


def test_first_crossing_is_not_settling():
    result = samples([30, 0, 10, 0, 0])
    assert result["settling_time_s"] == 3


def test_high_rate_and_inadequate_hold_are_not_success():
    assert not samples([30, 0, 0], [0, 10, 10])["success"]
    assert not samples([30, 10, 0])["success"]


def test_effort_includes_sample_interval():
    cfg = Config(dt=0.5, duration=1)
    result = metrics([0, 0.5, 1], [0, 0, 0], [0, 0, 0], [2, 2], cfg)
    assert result["torque_effort_nm2_s"] == 4


def test_truncated_failure_cannot_be_success():
    cfg = Config(dt=1, duration=10)
    assert not metrics([0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 0], cfg)["success"]
