"""V3 development checks; these never read held-out controller responses."""
import csv
import json

import numpy as np
import pytest

from inertia_generalization.envs import BatchSatelliteEnv, EpisodeSampler, InertiaSatelliteEnv
from inertia_generalization.training import checkpoint_rank, train_model, sha256
from inertia_generalization.vector_training import FastThreeVecEnv
from multi_algorithm_benchmark.envs import BatchSatelliteEnv as PreviousBatchEnv


@pytest.mark.parametrize("axis", ["single", "three"])
def test_initial_state_stream_matched_across_modes_including_overridden_resets(axis):
    nominal = EpisodeSampler(axis, "nominal", 110)
    randomized = EpisodeSampler(axis, "randomized", 110)
    random_rhos = []
    for index in range(100):
        options = {"inertia_scale": 1.2} if index == 5 else None
        left, _ = nominal.sample(options)
        right, _ = randomized.sample(options)
        key = "theta_deg" if axis == "single" else "euler_deg"
        np.testing.assert_array_equal(left[key], right[key])
        np.testing.assert_array_equal(left["omega_deg_s"], right["omega_deg_s"])
        assert left["noise_seed"] == right["noise_seed"]
        if options is None:
            assert left["inertia_scale"] == 1
            random_rhos.append(right["inertia_scale"])
        for key in ("disturbance_bias", "disturbance_amplitude", "pulse_torque", "noise_theta_deg", "noise_omega_deg"):
            assert left[key] == right[key] == 0
    assert .7 <= min(random_rhos) < .8
    assert 1.2 < max(random_rhos) <= 1.3


@pytest.mark.parametrize("axis", ["single", "three"])
def test_new_batch_preserves_old_physics_observations_and_rewards(axis):
    scenes = [EpisodeSampler(axis, "randomized", 5 + i).sample()[0] for i in range(4)]
    old, new = PreviousBatchEnv(axis, scenes), BatchSatelliteEnv(axis, scenes)
    dimensions = 1 if axis == "single" else 3
    np.testing.assert_array_equal(old.observation(), new.observation())
    rng = np.random.default_rng(4)
    for _ in range(45):
        action = rng.uniform(-1.1, 1.1, (4, dimensions))
        left, right = old.step(action), new.step(action)
        for i in range(4):
            np.testing.assert_array_equal(left[i], right[i])
        np.testing.assert_array_equal(old.omega, new.omega)
        np.testing.assert_array_equal(old.q if dimensions == 3 else old.theta,
                                      new.q if dimensions == 3 else new.theta)


def test_physical_triangle_and_nuisance_rejection():
    with pytest.raises(ValueError, match="triangle"):
        BatchSatelliteEnv("three", [{"inertia_scale": [10, 1, 1]}])
    with pytest.raises(ValueError, match="disturbance_bias"):
        BatchSatelliteEnv("single", [{"disturbance_bias": .001}])
    for scale in ([.6, 1, 1], [1.4, 1, 1], [1, .6, 1.2], [1, 1.4, .8]):
        BatchSatelliteEnv("three", [{"inertia_scale": scale}])


@pytest.mark.parametrize("mode", ["nominal", "randomized"])
def test_fast_vs_gym_multiple_auto_resets_and_actual_scene_logs(mode, tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv
    config = {"duration": .095}
    fast = FastThreeVecEnv(3, 101, tmp_path, mode, config)
    dummy = DummyVecEnv([lambda: InertiaSatelliteEnv("three", mode, config) for _ in range(3)])
    dummy.seed(101)
    np.testing.assert_array_equal(fast.reset(), dummy.reset())
    rng = np.random.default_rng(88)
    terminal_count = 0
    for _ in range(30):
        action = rng.uniform(-1.2, 1.2, (3, 3))
        result, reference = fast.step(action), dummy.step(action)
        for i in range(3):
            np.testing.assert_array_equal(result[i], reference[i])
        for rank in np.flatnonzero(result[2]):
            terminal_count += 1
            np.testing.assert_array_equal(result[3][rank]["terminal_observation"], reference[3][rank]["terminal_observation"])
            assert result[3][rank]["episode"]["l"] == 5
    assert terminal_count == 18
    fast.close(); dummy.close()
    with (tmp_path/"training_episode_initial_states.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21  # Initial three rows + eighteen actual auto-resets.
    assert all(.7 <= float(json.loads(row["inertia_scale"])) <= 1.3 for row in rows)


def test_fast_row_failure_does_not_shift_other_rows_and_records_monitor(tmp_path):
    fast = FastThreeVecEnv(2, 4, tmp_path, "randomized", {"duration": .055})
    fast.set_options([{"euler_deg": [10, 0, 0], "omega_deg_s": [301, 0, 0]},
                      {"euler_deg": [0, 0, 0], "omega_deg_s": [0, 0, 0]}])
    fast.reset()
    _, rewards, dones, infos = fast.step(np.zeros((2, 3)))
    assert dones.tolist() == [True, False]
    assert infos[0]["failed"] and not infos[0]["TimeLimit.truncated"]
    assert infos[0]["episode"]["l"] == 1
    assert infos[0]["episode"]["r"] == pytest.approx(float(rewards[0]), rel=1e-6)
    assert fast.step_count.tolist() == [0, 1]
    fast.close()


def test_selection_uses_calibrated_score_then_rms_cost_and_earlier_checkpoint():
    high_success = {"score": 75., "rms_error_deg": 20., "control_cost": 3.}
    low_success = {"score": 70., "rms_error_deg": 1., "control_cost": 1.}
    assert checkpoint_rank(high_success, 200000) > checkpoint_rank(low_success, 100000)
    equal_score_low_rms = {**high_success, "rms_error_deg": 19.}
    assert checkpoint_rank(equal_score_low_rms, 300000) > checkpoint_rank(high_success, 100000)
    equal_rms_low_cost = {**equal_score_low_rms, "control_cost": 2.}
    assert checkpoint_rank(equal_rms_low_cost, 400000) > checkpoint_rank(equal_score_low_rms, 100000)
    assert checkpoint_rank(equal_rms_low_cost, 100000) > checkpoint_rank(equal_rms_low_cost, 200000)


def test_formal_budget_guard_and_run_directory_preservation(tmp_path):
    with pytest.raises(ValueError, match="Formal"):
        train_model(tmp_path, "single", "nominal", 11, steps=32)
    run = tmp_path/"single"/"nominal"/"seed_11"
    run.mkdir(parents=True)
    marker = run/"original.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train_model(tmp_path, "single", "nominal", 11)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_developer_checkpoint_artifacts_bias_and_latest_hash_are_consistent(tmp_path):
    from stable_baselines3 import SAC
    metadata = train_model(tmp_path, "single", "nominal", 11, steps=32, n_envs=2,
                           validation_interval=32, development=True)
    run = tmp_path/"single"/"nominal"/"seed_11"
    assert metadata["actual_steps"] == 32
    assert metadata["checkpoint_count"] == 1
    assert not metadata["eligible_for_final_research"]
    assert metadata["runtime_source_unchanged"]
    for filename, record in metadata["artifacts"].items():
        assert sha256(run/filename) == record["sha256"]
        assert record["timesteps"] == 32
    latest = json.loads((run/"latest_checkpoint.json").read_text(encoding="utf-8"))
    assert latest["model_sha256"] == sha256(run/"latest_model.zip")
    best = json.loads((run/"best_checkpoint.json").read_text(encoding="utf-8"))
    model = SAC.load(run/"best_model.zip", device="cpu")
    actual_zero, _ = model.predict(np.zeros((1, 2), dtype=np.float32), deterministic=True)
    np.testing.assert_array_equal(actual_zero[0], best["zero_action_normalized"])
    assert json.loads((run/"session_status.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_developer_interruption_records_incomplete_and_recovery_state(tmp_path, monkeypatch):
    from stable_baselines3 import SAC
    def interrupted(self, *args, **kwargs):
        raise RuntimeError("intentional developer interruption")
    monkeypatch.setattr(SAC, "learn", interrupted)
    with pytest.raises(RuntimeError, match="intentional"):
        train_model(tmp_path, "single", "randomized", 22, steps=32, n_envs=2,
                    validation_interval=32, development=True)
    run = tmp_path/"single"/"randomized"/"seed_22"
    status = json.loads((run/"session_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert status["error_type"] == "RuntimeError"
    assert status["actual_steps"] == 0
    assert status["interrupted_model_sha256"] == sha256(run/"interrupted_model.zip")
    assert status["interrupted_replay_sha256"] == sha256(run/"interrupted_replay_buffer.pkl")
