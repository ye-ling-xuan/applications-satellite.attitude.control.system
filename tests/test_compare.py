from copy import deepcopy
import json
import pytest
pytest.importorskip("stable_baselines3")
from attitude_control.compare import check_paired_manifests, load_recorded_metadata


def test_unpaired_test_conditions_are_rejected():
    manifest = {"status": "complete", "test_seed": 1, "contract": {}, "pid_config": {},
                "scenarios": [{"angle_deg": 30}], "models": {"PPO_seed7": {"training_seed": 7}}}
    check_paired_manifests(manifest, deepcopy(manifest))
    changed = deepcopy(manifest)
    changed["scenarios"][0]["angle_deg"] = 40
    with pytest.raises(ValueError, match="scenarios"):
        check_paired_manifests(manifest, changed)


def test_training_seed_mismatch_is_rejected():
    manifest = {"status": "complete", "test_seed": 1, "contract": {}, "pid_config": {},
                "scenarios": [], "models": {"PPO_seed7": {"training_seed": 7}}}
    changed = deepcopy(manifest)
    changed["models"]["PPO_seed7"]["training_seed"] = 19
    with pytest.raises(ValueError, match="seed sets"):
        check_paired_manifests(manifest, changed)


def test_moved_project_resolves_recorded_windows_model_path(tmp_path):
    model = tmp_path / "artifacts" / "seed7"
    evaluation = model.parent / "evaluation"
    model.mkdir(parents=True)
    evaluation.mkdir()
    metadata = {"status": "complete", "model_sha256": "expected"}
    (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    record = {"directory": r"Z:\old-computer\artifacts\seed7", "sha256": "expected"}
    assert load_recorded_metadata(record, evaluation) == metadata


def test_relocated_checkpoint_metadata_mismatch_is_rejected(tmp_path):
    model = tmp_path / "artifacts" / "seed7"
    model.mkdir(parents=True)
    (model / "metadata.json").write_text(
        json.dumps({"status": "complete", "model_sha256": "different"}), encoding="utf-8")
    record = {"directory": r"Z:\old-computer\artifacts\seed7", "sha256": "expected"}
    with pytest.raises(ValueError, match="checkpoint"):
        load_recorded_metadata(record, model.parent / "evaluation")
