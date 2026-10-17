"""Synthetic delivery checks. Never read actual held-out response artifacts."""
from pathlib import Path
import copy
import csv
import json
import sys
import numpy as np
import pytest

CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(CODE.parent.parent)]
from deadline_study import delivery
from deadline_study.protocol import DEADLINES, sha256, write_json


@pytest.fixture(autouse=True)
def synthetic_initial_state_definition(monkeypatch):
    """Supply synthetic membership, never generate or inspect real test states."""
    monkeypatch.setattr(delivery, "scenarios", lambda axis, split: [
        {"sample_id": i, "scenario_id": f"synthetic_{axis}_{i}"} for i in range(8)])


def synthetic_results(n=8):
    controllers, rows = [], []
    for axis in ("single", "three"):
        for deadline in DEADLINES:
            controllers.append({"id": f"{axis}_pid_{int(deadline)}s", "axis": axis,
                                "kind": "pid", "group": "pid", "seed": -1,
                                "optimized_deadlines": [deadline]})
        for group, seeds in (("legacy_sac", (11, 22, 33)), ("fresh_sac", (41, 42, 43))):
            for seed in seeds:
                controllers.append({"id": f"{axis}_{group}_{seed}", "axis": axis,
                                    "kind": "legacy" if group == "legacy_sac" else "fresh",
                                    "group": group, "seed": seed, "optimized_deadlines": list(DEADLINES)})
    for c in controllers:
        for i in range(n):
            # Paired differences are exactly -0.5 for all states and all models.
            cost = 1.+i*.1-(0. if c["group"] == "pid" else .5)
            rows.append({"controller_id": c["id"], "axis": c["axis"], "group": c["group"],
                         "seed": c["seed"], "sample_id": i, "scenario_id": f"synthetic_{c['axis']}_{i}",
                         "control_cost": cost, "settling_time_s": 3., "rms_error_deg": 2.,
                         "failed": False, "complete": True,
                         **{f"success_{int(d)}s": True for d in DEADLINES}})
    return {"controllers": controllers, "test_scenes_per_axis": n}, rows


def test_pairing_bootstrap_and_direction_are_correct_for_constant_differences():
    record, rows = synthetic_results()
    by_id, summaries = delivery.summarize(record, list(reversed(rows)))
    assert len(summaries) == len(record["controllers"])*3
    values = delivery.comparisons(record, by_id)
    assert len(values) == 12
    for row in values:
        assert row["all_controllers_passed_all_scenes"]
        assert row["all_seed_cost_differences_negative"]
        assert row["supports_lower_effort_under_identical_requirements"]
        np.testing.assert_allclose(row["mean_paired_cost_difference_each_seed"], -.5, atol=1e-15)
        np.testing.assert_allclose(row["paired_cost_difference_95pct_bootstrap_CI"], [-.5, -.5], atol=1e-15)
        assert row["n_initial_states"] == 8  # three models do not make 24 initial states


@pytest.mark.parametrize("target", ["pid", "one_SAC_seed"])
def test_one_failed_hold_blocks_lower_effort_claim(target):
    record, rows = synthetic_results()
    selected = "single_pid_8s" if target == "pid" else "single_fresh_sac_43"
    next(r for r in rows if r["controller_id"] == selected)["success_8s"] = False
    by_id, _ = delivery.summarize(record, rows)
    comparisons = delivery.comparisons(record, by_id)
    value = next(r for r in comparisons if r["axis"] == "single" and r["deadline_s"] == 8.
                 and r["method"] == "fresh_sac")
    assert not value["all_controllers_passed_all_scenes"]
    assert not value["supports_lower_effort_under_identical_requirements"]


def test_one_training_seed_with_higher_cost_blocks_claim_despite_lower_group_mean():
    record, rows = synthetic_results()
    for row in rows:
        if row["controller_id"] == "single_fresh_sac_43":
            row["control_cost"] += .7  # +0.2 vs PID; other two seeds are -0.5
    by_id, _ = delivery.summarize(record, rows)
    value = next(r for r in delivery.comparisons(record, by_id) if r["axis"] == "single"
                 and r["deadline_s"] == 8. and r["method"] == "fresh_sac")
    assert value["relative_cost_change_pct"] < 0
    assert not value["all_seed_cost_differences_negative"]
    assert not value["supports_lower_effort_under_identical_requirements"]


def test_initial_state_uncertainty_can_block_claim_with_three_negative_seed_means():
    record, rows = synthetic_results(n=8)
    for row in rows:
        if row["group"] == "fresh_sac":
            pid = 1.+row["sample_id"]*.1
            row["control_cost"] = pid+(-.8 if row["sample_id"] < 4 else .7)
    by_id, _ = delivery.summarize(record, rows)
    value = next(r for r in delivery.comparisons(record, by_id) if r["axis"] == "single"
                 and r["deadline_s"] == 8. and r["method"] == "fresh_sac")
    assert value["all_seed_cost_differences_negative"]
    assert value["paired_cost_difference_95pct_bootstrap_CI"][1] > 0
    assert not value["supports_lower_effort_under_identical_requirements"]


@pytest.mark.parametrize("damage", ["missing_all", "duplicate_all", "wrong_pairing"])
def test_incomplete_duplicate_or_misaligned_initial_state_rows_are_rejected(damage):
    record, rows = synthetic_results()
    if damage == "missing_all":
        rows = [r for r in rows if r["sample_id"] != 7]
    elif damage == "duplicate_all":
        rows.extend(copy.deepcopy([r for r in rows if r["sample_id"] == 0]))
    else:
        row = next(r for r in rows if r["controller_id"] == "single_fresh_sac_43")
        row["scenario_id"] = "synthetic_wrong_initial_state"
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        by_id, _ = delivery.summarize(record, rows)
        delivery.comparisons(record, by_id)


def test_parse_rows_preserves_false_boolean_null_and_integer_types(tmp_path, monkeypatch):
    study = tmp_path/"synthetic_study"
    path = study/"04_独立测试"/"episode_metrics.csv"
    row = {"controller_id": "synthetic", "axis": "single", "group": "pid", "seed": -1,
           "sample_id": 2, "scenario_id": "synthetic_2", "control_cost": .125,
           "settling_time_s": None, "failed": False, "complete": True,
           "success_4s": False, "success_6s": False, "success_8s": False}
    delivery.csv_write(path, [row])
    monkeypatch.setattr(delivery, "STUDY", study)
    actual = delivery.parse_rows()[0]
    assert actual == row
    assert actual["failed"] is False and actual["complete"] is True
    assert actual["settling_time_s"] is None
    assert type(actual["sample_id"]) is int


def make_freeze_fixture(tmp_path, monkeypatch):
    package = tmp_path/"synthetic_package"
    study = package/"synthetic_study"
    monkeypatch.setattr(delivery, "STUDY", study)
    monkeypatch.setattr(delivery, "PACKAGE", package)
    source = study/"代码"/"synthetic_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic fixture only\n", encoding="utf-8")
    hashes = {str(source.relative_to(package)).replace("\\", "/"): sha256(source)}
    for name in ("envs.py", "vector_training.py"):
        path = package/"multi_algorithm_benchmark"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic fixture only\n", encoding="utf-8")
    write_json(study/"01_实验方案"/"任务定义.json", {"synthetic": True})
    for axis in ("single", "three"):
        write_json(study/"01_实验方案"/f"test_{axis}.json", [])
    pid = {}
    for axis in ("single", "three"):
        value = {"parameters": {"kp": [1.], "ki": [0.], "kd": [2.]}, "validation": {"synthetic": True}}
        pid[axis] = {"deadlines": {str(int(d)): value for d in DEADLINES},
                     "auxiliary_validation_time_margin_0_25s": {str(int(d)): value for d in DEADLINES}}
    write_json(study/"02_PID整定"/"selected_pid.json", pid)
    write_json(study/"03_SAC训练"/"正式训练配置冻结.json", {"config": "B", "source_sha256": hashes})
    legacy = []
    for axis in ("single", "three"):
        for seed in (11, 22, 33):
            path = package/"synthetic_legacy"/f"{axis}_{seed}.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic model {axis} {seed}".encode())
            for deadline in DEADLINES:
                legacy.append({"axis": axis, "seed": seed, "action_scale": 1., "deadline_s": deadline,
                               "model_path": str(path.relative_to(package)).replace("\\", "/"),
                               "model_sha256": sha256(path)})
        for seed in delivery.SEEDS:
            path = study/"03_SAC训练"/"formal"/axis/f"seed_{seed}"/"best_model.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic fresh {axis} {seed}".encode())
            best = {"model_path": str(path.relative_to(study)).replace("\\", "/"), "model_sha256": sha256(path),
                    "calibrated": True, "timesteps": 100000}
            write_json(path.with_name("training_metadata.json"), {"axis": axis, "seed": seed, "config": "B",
                       "status": "complete", "actual_steps": 1000000, "runtime_source_unchanged": True,
                       "source_sha256": hashes, "source_sha256_after": hashes, "best": best})
    write_json(study/"03_SAC训练"/"旧模型验证与动作缩放"/"selected_legacy.json", {"models": legacy})
    return package, study, source


def test_freeze_schema_and_legacy_deduplication(tmp_path, monkeypatch):
    _, _, _ = make_freeze_fixture(tmp_path, monkeypatch)
    frozen = delivery.freeze()
    controllers = frozen["controllers"]
    assert len(controllers) == 30  # 6 PID +6 margin +6 legacy instances +12 fresh/ablation
    legacy = [c for c in controllers if c["kind"] == "legacy"]
    assert len(legacy) == 6
    assert all(c["optimized_deadlines"] == list(DEADLINES) for c in legacy)
    assert len({c["id"] for c in controllers}) == len(controllers)
    assert frozen["test_controller_responses_read"] is False
    delivery.check_frozen(frozen)


@pytest.mark.parametrize("damage", ["training_source_changed", "legacy_model_changed", "dedup_hash_conflict"])
def test_pretest_freeze_rejects_changed_source_or_model(tmp_path, monkeypatch, damage):
    package, study, source = make_freeze_fixture(tmp_path, monkeypatch)
    if damage == "training_source_changed":
        source.write_text("# changed after synthetic training\n", encoding="utf-8")
    else:
        path = study/"03_SAC训练"/"旧模型验证与动作缩放"/"selected_legacy.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if damage == "legacy_model_changed":
            (package/value["models"][0]["model_path"]).write_bytes(b"changed model")
        else:
            value["models"][1]["model_sha256"] = "0"*64
            write_json(path, value)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        delivery.freeze()


def test_root_pid_cli_forwards_options_without_running_or_overwriting(monkeypatch):
    from deadline_study import baselines, cli
    called = []
    monkeypatch.setattr(cli, "freeze_design", lambda: None)
    monkeypatch.setattr(baselines, "main", lambda argv=None: called.append(argv))
    monkeypatch.setattr(sys, "argv", ["synthetic_cli", "pid", "--axis", "single", "--batch-size", "8"])
    cli.main()
    assert called == [["--axis", "single", "--batch-size", "8"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q", "-p", "no:cacheprovider"]))
