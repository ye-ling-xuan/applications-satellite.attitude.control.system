import hashlib
import numpy as np
import pytest
pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
import torch
from attitude_control.config import Config
from attitude_control.env import AttitudeEnv
from attitude_control.evaluate import load_policy
from attitude_control.experiments import contract, save_json
from attitude_control.tasks import Task


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    config, task = Config(), Task("precision_nominal")
    env = AttitudeEnv(config, mode="pure", task=task)
    model = PPO("MlpPolicy", env, seed=7, n_steps=16, batch_size=16)
    model.save(tmp_path / "model")
    metadata = {"mode": "pure", "contract": contract(config), "task": task.to_dict(),
                "status": "complete", "seed": 7, "actual_timesteps": 0,
                "model_sha256": hashlib.sha256((tmp_path / "model.zip").read_bytes()).hexdigest()}
    save_json(tmp_path / "metadata.json", metadata)
    yield tmp_path, config, task, model
    env.close()


def test_checkpoint_adapter_preserves_network_predictions(checkpoint):
    directory, config, task, model = checkpoint
    policy, _ = load_policy(directory, config, "pure")
    canonical = np.array([-30 / 180, 0.2, 0.1], np.float32)
    actual = policy.predict(canonical)[0]
    transformed = canonical.copy()
    transformed[0] *= 180 / task.observation_angle_scale_deg
    expected = model.predict(transformed, deterministic=True)[0]
    np.testing.assert_array_equal(actual, expected)


def test_corrupted_checkpoint_is_rejected_before_loading(checkpoint):
    directory, config, _, _ = checkpoint
    with (directory / "model.zip").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_policy(directory, config, "pure")
