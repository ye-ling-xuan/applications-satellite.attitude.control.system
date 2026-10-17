"""Independent development checks; never load held-out controller responses.

Run with the package's tested Python and pytest.  Synthetic counterexamples
check the acceptance rule, and the original plant is the dynamics oracle.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent.parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(CODE))

from deadline_study.evaluation import Plant, evaluate, metric_key, metrics_from_arrays
from deadline_study.protocol import ANGLE_TOL, DEADLINES, DT, DURATION, LIMIT, RATE_TOL, scenarios
from multi_algorithm_benchmark.envs import BatchSatelliteEnv


def synthetic(time=None, error=None, rate=None, torque=None, failed=None):
    """A trace fixture, independent of the rollout implementation."""
    t = np.array([0., 3.95, 4., 5.98, 6., 7.98, 8., 19.98, 20.]) if time is None else np.asarray(time, float)
    n = len(t)
    e = np.zeros(n) if error is None else np.asarray(error, float)
    w = np.zeros(n) if rate is None else np.asarray(rate, float)
    tau = np.zeros((n, 1)) if torque is None else np.asarray(torque, float)
    f = np.zeros(n, bool) if failed is None else np.asarray(failed, bool)
    return metrics_from_arrays(t, e, w, tau, f)[0]


def successes(row):
    return tuple(row[metric_key(t)] for t in DEADLINES)


def test_published_task_constants():
    assert DURATION == 20.
    assert DEADLINES == (4., 6., 8.)
    assert ANGLE_TOL == 1.
    assert RATE_TOL == .5
    assert DT == {"single": .05, "three": .02}
    assert LIMIT == {"single": .5, "three": 2.}


def test_thresholds_are_inclusive_and_early_exceedance_is_allowed():
    error = np.full(9, 1.)
    error[:2] = [60., 1.001]
    row = synthetic(error=error, rate=np.full(9, .5))
    assert successes(row) == (True, True, True)
    assert row["settling_time_s"] == 4.


@pytest.mark.parametrize("field,index,expected", [
    ("error", 2, (False, True, True)),  # exactly t=4 must count
    ("rate", 4, (False, False, True)),  # exactly t=6 must count
    ("error", 6, (False, False, False)),  # exactly t=8 must count
    ("rate", 7, (False, False, False)),  # transient just before the end
    ("error", 8, (False, False, False)),  # final sample must count
])
def test_one_out_of_band_sample_rejects_every_affected_deadline(field, index, expected):
    values = np.zeros(9)
    values[index] = 1.00001 if field == "error" else .50001
    row = synthetic(**{field: values})
    assert successes(row) == expected
    if index == 7:
        assert row["final_rate_deg_s"] == 0.
        assert row["max_rate_after_8s"] > .5


def test_early_rate_violation_does_not_disqualify_later_deadline():
    rate = np.zeros(9)
    rate[3] = .6  # t=5.98, before the 6-second deadline
    row = synthetic(rate=rate)
    assert successes(row) == (False, True, True)
    assert row["settling_time_s"] == 6.


def test_incomplete_and_physically_failed_are_distinct():
    incomplete = synthetic(time=[0., 4., 6., 8., 19.98])
    assert incomplete["complete"] is False
    assert incomplete["failed"] is False
    assert incomplete["settling_time_s"] is None
    assert successes(incomplete) == (False, False, False)
    failed = np.zeros(9, bool)
    failed[1] = True  # sticky physical failure is not excused by later padding
    physical = synthetic(failed=failed)
    assert physical["complete"] is True
    assert physical["failed"] is True
    assert physical["settling_time_s"] is None
    assert successes(physical) == (False, False, False)


def test_zoh_effort_uses_applied_interval_torque_and_full_vector_norm():
    # The first sample has no preceding interval: its absurd value is ignored.
    # Applied scalar intervals: 1^2*4 + 2^2*2 + 3^2*2 + 4^2*12 = 222.
    time = np.array([0., 4., 6., 8., 20.])
    scalar = np.array([999., 1., 2., 3., 4.])
    row = synthetic(time=time, torque=scalar[:, None])
    assert row["control_cost"] == pytest.approx(222., abs=1e-12)
    vector = scalar[:, None, None] * np.array([[[1., -2., .5]]])
    rows = metrics_from_arrays(time, np.zeros((5, 1)), np.zeros((5, 1)), vector, np.zeros((5, 1), bool))
    assert rows[0]["control_cost"] == pytest.approx(222.*5.25, abs=1e-12)


def test_two_scenes_are_scored_independently():
    time = np.array([0., 4., 6., 8., 20.])
    error = np.zeros((5, 2))
    error[-1, 1] = 2.
    torque = np.zeros((5, 2, 1))
    torque[1:, 0, 0] = .25
    torque[1:, 1, 0] = .5
    rows = metrics_from_arrays(time, error, np.zeros_like(error), torque, np.zeros_like(error, bool))
    assert successes(rows[0]) == (True, True, True)
    assert successes(rows[1]) == (False, False, False)
    assert rows[0]["control_cost"] == pytest.approx(1.25)
    assert rows[1]["control_cost"] == pytest.approx(5.)


def test_three_axis_angular_speed_is_euclidean_norm():
    plant = Plant("three", [{"euler_deg": [0., 0., 0.], "omega_deg_s": [.4, .4, 0.]}])
    assert plant.rate_deg_s[0] == pytest.approx(np.sqrt(.4**2+.4**2), abs=1e-12)
    assert plant.rate_deg_s[0] > RATE_TOL  # every component is below the limit
    angle_plant = Plant("three", [{"euler_deg": [.8, .8, 0.], "omega_deg_s": [0., 0., 0.]}])
    assert angle_plant.true_error_deg[0] > ANGLE_TOL


def _comparison_scenes(axis):
    if axis == "single":
        return [{"theta_deg": 38., "omega_deg_s": -3.},
                {"theta_deg": -51., "omega_deg_s": 4.},
                {"theta_deg": 169.95, "omega_deg_s": 5.}]
    return [{"euler_deg": [38., -27., 19.], "omega_deg_s": [-3., 2., 1.]},
            {"quaternion": [2., -.1, .2, -.4], "omega_deg_s": [4., -1., 2.]},
            {"euler_deg": [0., 0., 0.], "omega_deg_s": [299.9, 0., 0.]}]


@pytest.mark.parametrize("axis", ["single", "three"])
def test_memory_optimization_preserves_original_plant_all_samples(axis):
    scenes = _comparison_scenes(axis)
    config = {"duration": DURATION, "dt": DT[axis], "max_torque": LIMIT[axis]}
    original = BatchSatelliteEnv(axis, scenes, config)
    optimized = Plant(axis, scenes, config)
    # The view carries zero noise without an O(horizon*batch) allocation.
    assert not optimized._noise.flags.owndata
    assert optimized._noise.base.nbytes <= 2*optimized.dimensions*8
    np.testing.assert_array_equal(optimized.observation(), original.observation())
    rng = np.random.default_rng(11808)
    for k in range(original.steps):
        actions = rng.uniform(-2.5, 2.5, (original.n, original.dimensions))
        actions[2] = 3.  # exercises saturation followed by physical failure
        left = optimized.step(actions)
        right = original.step(actions)
        for index in range(4):
            np.testing.assert_allclose(left[index], right[index], rtol=0., atol=1e-14)
        for key in right[4]:
            np.testing.assert_allclose(left[4][key], right[4][key], rtol=0., atol=1e-14)
        np.testing.assert_allclose(optimized.omega, original.omega, rtol=0., atol=1e-14)
        np.testing.assert_allclose(optimized.theta if axis == "single" else optimized.q,
                                   original.theta if axis == "single" else original.q, rtol=0., atol=1e-14)
        assert np.max(np.abs(optimized.torque)) <= LIMIT[axis]
        if k == 0:
            assert optimized.failed[2]
            assert np.max(np.abs(optimized.torque[2])) == LIMIT[axis]
        else:
            np.testing.assert_array_equal(optimized.torque[2], 0.)
    assert optimized.time == DURATION
    assert optimized.failed[2]
    with pytest.raises(RuntimeError, match="finished"):
        optimized.step(np.zeros((optimized.n, optimized.dimensions)))


@pytest.mark.parametrize("key", ["disturbance_bias", "disturbance_amplitude", "pulse_torque", "noise_theta_deg", "noise_omega_deg"])
def test_zero_noise_plant_refuses_nonzero_nuisance_inputs(key):
    with pytest.raises(ValueError, match="zero nuisance"):
        Plant("single", [{"theta_deg": 0., key: .01}])


class InitialPulse:
    def __init__(self, command):
        self.command = np.asarray(command, float)

    def act(self, observations, time):
        value = self.command if time == 0. else np.zeros_like(self.command)
        return np.broadcast_to(value, (len(observations), len(value)))


@pytest.mark.parametrize("axis,scene,command,expected", [
    ("single", {"theta_deg": 0., "omega_deg_s": 0.}, [2.], .5**2*.05),
    ("three", {"euler_deg": [0., 0., 0.], "omega_deg_s": [0., 0., 0.]}, [3., -3., .5], (2.**2+(-2.)**2+1.**2)*.02),
])
def test_rollout_charges_saturated_physical_torque_once(axis, scene, command, expected):
    result = evaluate(axis, [scene], InitialPulse(command), keep_traces=True)
    row, trace = result["metrics"][0], result["traces"]
    assert row["complete"]
    assert not row["failed"]
    assert row["control_cost"] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(trace["torque"][0], 0.)
    np.testing.assert_allclose(trace["torque"][1, 0], np.clip(command, -1., 1.)*LIMIT[axis], atol=0., rtol=0.)
    np.testing.assert_array_equal(trace["torque"][2:], 0.)
    if axis == "three":
        np.testing.assert_allclose(trace["rate_deg_s"], np.degrees(np.linalg.norm(trace["omega_rad_s"], axis=2)), atol=1e-12)


@pytest.mark.parametrize("axis", ["single", "three"])
def test_development_and_validation_scenarios_are_reproducible_and_disjoint(axis):
    development = scenarios(axis, "development")
    validation = scenarios(axis, "validation")
    assert development == scenarios(axis, "development")
    assert validation == scenarios(axis, "validation")
    assert not ({r["scenario_id"] for r in development} & {r["scenario_id"] for r in validation})
    def state(scene):
        angle = scene["theta_deg"] if axis == "single" else scene["euler_deg"]
        return tuple(np.r_[angle, scene["omega_deg_s"]])
    assert not ({state(r) for r in development} & {state(r) for r in validation})
    for row in development + validation:
        assert row["inertia_scale"] == 1.
        assert row["disturbance_bias"] == row["noise_theta_deg"] == row["noise_omega_deg"] == 0.


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q", "-p", "no:cacheprovider"]))
