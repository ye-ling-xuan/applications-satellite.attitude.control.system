"""Regression checks for continuation integrity and accidental-overwrite guards."""
import csv
import hashlib
import json

import pytest

from satellite_benchmark import training


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_existing_or_incomplete_run_is_not_replaced(tmp_path):
    run = tmp_path / "nominal" / "seed_11"
    run.mkdir(parents=True)
    marker = run / "incomplete.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        training.train_model(tmp_path, "nominal", 11, total_timesteps=1)
    with pytest.raises(FileNotFoundError, match="original manifest"):
        training.train_model(tmp_path, "nominal", 11, total_timesteps=1, resume=True)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert list(run.iterdir()) == [marker]


def test_resume_preserves_best_history_manifest_and_monitors(tmp_path, monkeypatch):
    # Use real PPO updates/save/load; deterministic validation scores let this
    # regression exercise a worse continuation without relying on training luck.
    monkeypatch.setattr(training, "evaluate_batch", lambda *args, **kwargs: [])
    score = [90.0]
    monkeypatch.setattr(training, "validation_score", lambda _: (score[0], {
        "success_rate": 1.0, "rms_error_deg": 1.0,
        "control_cost": 0.1, "score": score[0]}))
    first = training.train_model(tmp_path, "nominal", 11, total_timesteps=1,
                                 n_envs=1, validation_interval=1024)
    run = tmp_path / "nominal" / "seed_11"
    preserved = {name: digest(run / name) for name in (
        "training_manifest.json", "validation_scenarios.json", "best_model.zip",
        "best_checkpoint.json", "monitor_0.monitor.csv")}
    old_rows = read_rows(run / "validation_history.csv")
    old_steps = first["actual_timesteps"]
    # Simulate an interruption: latest_model is the only completed checkpoint.
    (run / "final_model.zip").unlink()
    score[0] = 10.0
    resumed = training.train_model(tmp_path, "nominal", 11, total_timesteps=1,
                                   n_envs=1, validation_interval=1024, resume=True)
    assert resumed["initial_timesteps"] == old_steps
    assert resumed["actual_timesteps"] > old_steps
    assert resumed["best_validation_score"] == first["best_validation_score"]
    assert resumed["best_checkpoint_timesteps"] == first["best_checkpoint_timesteps"]
    for name, old_hash in preserved.items():
        assert digest(run / name) == old_hash, name
    rows = read_rows(run / "validation_history.csv")
    assert rows[:len(old_rows)] == old_rows
    assert all(int(row["timesteps"]) > old_steps for row in rows[len(old_rows):])
    segment = run / "resume_0001"
    assert (segment / "monitor_0.monitor.csv").exists()
    assert json.loads((segment / "training_manifest.json").read_text(encoding="utf-8"))["resume_from"] == "latest_model.zip"
    assert json.loads((segment / "session_status.json").read_text(encoding="utf-8"))["status"] == "completed"
    # A changed configuration must fail before creating a new resume segment or
    # replacing any artifact, even when the existing run is otherwise valid.
    snapshot = {p.relative_to(run): digest(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="changed n_envs"):
        training.train_model(tmp_path, "nominal", 11, total_timesteps=1,
                             n_envs=2, validation_interval=1024, resume=True)
    assert {p.relative_to(run): digest(p) for p in run.rglob("*") if p.is_file()} == snapshot
