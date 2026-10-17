"""Independently audit saved episode metrics and both summary layers.

Uses only the Python standard library. Does not import study code, load models,
run controllers, read trajectory arrays, or change frozen research artifacts.
Run: python -X utf8 -B verify_summary_semantics.py
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
PACKAGE = STUDY.parent
TEST = STUDY / "04_独立测试"
AUDIT = STUDY / "05_独立核验"
REGIONS = ("in_domain", "ideal", "out_of_domain")
REL_TOL = ABS_TOL = 1e-12
errors: list[str] = []


def require(condition, message):
    if not condition:
        errors.append(str(message))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"duplicate CSV headers: {path}")
        return list(reader)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite_float(value, context):
    result = float(value)
    require(math.isfinite(result), f"nonfinite float: {context}")
    return result


def boolean(value, context):
    require(value in ("True", "False"), f"invalid boolean: {context}: {value!r}")
    return value == "True"


def same_number(expected, actual, context):
    if expected is None:
        require(actual is None or actual == "", f"expected null: {context}: {actual!r}")
    else:
        actual_value = finite_float(actual, context)
        require(math.isclose(expected, actual_value, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                f"numeric mismatch: {context}: expected {expected!r}, found {actual_value!r}")


def fingerprint(paths):
    return {str(path.relative_to(PACKAGE)).replace("\\", "/"): sha256(path)
            for path in sorted(set(paths))}


def verify():
    freeze_path = TEST / "freeze.json"
    frozen = read_json(freeze_path)
    declaration = read_json(STUDY / "01_实验方案" / "执行配置.json")
    states = read_json(STUDY / "01_实验方案" / "test_initial_states.json")
    state_ids = [int(state["sample_id"]) for state in states]
    require(len(states) == 128, "expected 128 independent test initial states")
    require(len(set(state_ids)) == 128, "test initial sample IDs are not unique")
    require(declaration["counts"]["test"] == 128, "declared test count differs from 128")
    states_by_id = {int(state["sample_id"]): state for state in states}

    controllers = frozen["controllers"]
    identifiers = [spec["id"] for spec in controllers]
    require(len(controllers) == 14, "expected exactly 14 frozen controllers")
    require(len(set(identifiers)) == len(identifiers), "duplicate frozen controller IDs")
    require({spec["kind"] for spec in controllers} == {"lag", "delay"}, "unexpected kinds")
    expected = {(spec["id"], region) for spec in controllers for region in REGIONS}
    require(len(expected) == 42, "expected exactly 42 controller-region groups")
    specs = {spec["id"]: spec for spec in controllers}

    expected_metrics = {TEST / controller / region / "metrics.csv" for controller, region in expected}
    expected_summaries = {path.with_name("summary.json") for path in expected_metrics}
    expected_completions = {path.with_name("complete.json") for path in expected_metrics}
    require(set(TEST.glob("*/*/metrics.csv")) == expected_metrics, "metrics group coverage mismatch")
    require(set(TEST.glob("*/*/summary.json")) == expected_summaries, "summary group coverage mismatch")
    require(set(TEST.glob("*/*/complete.json")) == expected_completions, "completion group coverage mismatch")

    single = PACKAGE / "12_执行器滞后与纯延迟对比研究"
    single_freeze = single / "04_独立测试" / "freeze.json"
    tracked = [freeze_path, TEST / "all_summaries.csv", single_freeze]
    tracked += [PACKAGE / relative for relative in frozen["source_and_input_hashes"]]
    tracked += [Path(spec["model_path"]) for spec in controllers if spec.get("model_path")]
    tracked += [Path(spec["metadata_path"]) for spec in controllers if spec.get("metadata_path")]
    tracked += list(expected_metrics | expected_summaries | expected_completions)
    before = fingerprint(tracked)
    for relative, expected_hash in frozen["source_and_input_hashes"].items():
        require(sha256(PACKAGE / relative) == expected_hash, f"frozen source/input mismatch: {relative}")
    require(sha256(single_freeze) == frozen["single_axis_freeze_sha256"], "single-axis freeze hash mismatch")
    for spec in controllers:
        if spec.get("model_path"):
            require(sha256(Path(spec["model_path"])) == spec["model_sha256"],
                    f"frozen model mismatch: {spec['id']}")
    require(sha256(STUDY / "01_实验方案" / "test_initial_states.json") == frozen["test_initial_sha256"],
            "test initial-state hash mismatch")

    total_rows = 0
    expected_total = 0
    global_episode_keys = set()
    paired_scenario_keys = set()
    groups = []
    freeze_hash = sha256(freeze_path)
    grid = [float(value) for value in declaration["in_domain_s"]]
    require(len(grid) == 6 and len(set(grid)) == 6, "expected six in-domain parameters")

    for controller, region in sorted(expected):
        spec = specs[controller]
        kind = spec["kind"]
        parameters = grid if region == "in_domain" else [0.] if region == "ideal" else [
            float(value) for value in declaration["out_of_domain_s"][kind]]
        require(len(parameters) == (6 if region == "in_domain" else 1 if region == "ideal"
                                  else 2 if kind == "lag" else 3),
                f"parameter count mismatch: {kind}/{region}")
        expected_pairs = {(parameter, sample) for parameter in parameters for sample in state_ids}
        expected_count = 128 * len(parameters)
        expected_total += expected_count
        directory = TEST / controller / region
        rows = read_csv(directory / "metrics.csv")
        require(len(rows) == expected_count, f"episode count mismatch: {controller}/{region}")
        summary = read_json(directory / "summary.json")
        completion = read_json(directory / "complete.json")
        require(completion["complete"] is True, f"incomplete group: {controller}/{region}")
        require(completion["freeze_sha256"] == freeze_hash, f"completion freeze mismatch: {controller}/{region}")
        require(completion["controller"] == controller and completion["region"] == region,
                f"completion identity mismatch: {controller}/{region}")
        require(set(completion["artifact_sha256"]) == {"metrics.csv", "summary.json", "traces.npz"},
                f"completion artifact key coverage mismatch: {controller}/{region}")
        for name in ("metrics.csv", "summary.json"):
            require(sha256(directory / name) == completion["artifact_sha256"][name],
                    f"completion artifact mismatch: {controller}/{region}/{name}")

        pairs = []
        scenario_ids = []
        successes = []
        failures = []
        costs = []
        command_costs = []
        settling = []
        for index, row in enumerate(rows):
            context = f"{controller}/{region}/{index}"
            require(row["kind"] == kind and row["split"] == "test" and row["region"] == region,
                    f"row identity mismatch: {context}")
            parameter = finite_float(row["parameter_s"], context)
            sample_id = int(row["sample_id"])
            pair = (parameter, sample_id)
            pairs.append(pair)
            scenario_ids.append(row["scenario_id"])
            require(row["scenario_id"] == f"{kind}_{region}_test_{parameter:.2f}_{sample_id:03d}",
                    f"scenario ID mismatch: {context}")
            episode_key = (controller, kind, region, parameter, sample_id)
            require(episode_key not in global_episode_keys, f"duplicate global episode: {episode_key}")
            global_episode_keys.add(episode_key)
            paired_scenario_keys.add((kind, region, parameter, sample_id))
            state = states_by_id.get(sample_id)
            require(state is not None, f"unknown initial sample ID: {context}")
            if state is not None:
                for name in ("euler_deg", "omega_deg_s"):
                    values = ast.literal_eval(row[name])
                    require(len(values) == 3, f"initial vector length mismatch: {context}/{name}")
                    for axis, (expected_value, actual_value) in enumerate(zip(state[name], values)):
                        same_number(expected_value, actual_value, f"{context}/{name}/{axis}")
            success = boolean(row["success"], context + "/success")
            failed = boolean(row["failed"], context + "/failed")
            require(not (success and failed), f"failed episode marked successful: {context}")
            successes.append(success)
            failures.append(failed)
            costs.append(finite_float(row["actual_cost"], context + "/actual_cost"))
            command_costs.append(finite_float(row["command_cost"], context + "/command_cost"))
            require(costs[-1] >= 0 and command_costs[-1] >= 0, f"negative cost: {context}")
            if success:
                settling.append(finite_float(row["settling_time_s"], context + "/settling_time_s"))
            else:
                require(row["settling_time_s"] == "", f"unsuccessful settling time is not null: {context}")

        require(len(pairs) == len(set(pairs)), f"duplicate scenario pair: {controller}/{region}")
        require(set(pairs) == expected_pairs, f"parameter-sample Cartesian coverage mismatch: {controller}/{region}")
        require(len(scenario_ids) == len(set(scenario_ids)), f"duplicate scenario ID: {controller}/{region}")
        values = dict(count=len(rows), passes=sum(successes), physical_failures=sum(failures),
                      success_rate=math.fsum(successes) / len(rows),
                      mean_actual_cost=math.fsum(costs) / len(rows),
                      mean_command_cost=math.fsum(command_costs) / len(rows),
                      max_settling_time_s=max(settling) if settling else None)
        for key in ("count", "passes", "physical_failures"):
            require(summary[key] == values[key], f"summary integer mismatch: {controller}/{region}/{key}")
        for key in ("success_rate", "mean_actual_cost", "mean_command_cost", "max_settling_time_s"):
            same_number(values[key], summary[key], f"summary/{controller}/{region}/{key}")
        identity = dict(controller=controller, kind=kind, method=spec["method"], stage=spec["stage"],
                        region=region, diagnostic_only=spec.get("diagnostic_only", False))
        for key, value in identity.items():
            require(summary[key] == value, f"summary identity mismatch: {controller}/{region}/{key}")
        groups.append(dict(**identity, **values, expected_count=expected_count,
                           unique_parameter_sample_pairs=len(set(pairs)),
                           metrics_and_summary_completion_hashes_match=True))
        total_rows += len(rows)

    global_rows = read_csv(TEST / "all_summaries.csv")
    global_keys = [(row["controller"], row["region"]) for row in global_rows]
    require(len(global_rows) == 42 and len(set(global_keys)) == 42, "global summary count/duplicates mismatch")
    require(set(global_keys) == expected, "global summary controller-region coverage mismatch")
    by_key = {(row["controller"], row["region"]): row for row in global_rows}
    for group in groups:
        key = (group["controller"], group["region"])
        row = by_key.get(key)
        require(row is not None, f"missing global group: {key}")
        if row is None:
            continue
        for field in ("controller", "kind", "method", "stage", "region"):
            require(row[field] == group[field], f"global identity mismatch: {key}/{field}")
        require(boolean(row["diagnostic_only"], str(key)) == group["diagnostic_only"],
                f"global diagnostic flag mismatch: {key}")
        for field in ("count", "passes", "physical_failures"):
            require(int(row[field]) == group[field], f"global integer mismatch: {key}/{field}")
        for field in ("success_rate", "mean_actual_cost", "mean_command_cost", "max_settling_time_s"):
            same_number(group[field], row[field], f"global/{key}/{field}")
    require(total_rows == expected_total == 17024, "expected exactly 17,024 episode rows")
    require(len(global_episode_keys) == total_rows, "global episode rows are not unique")
    require(len(paired_scenario_keys) == 2432, "expected exactly 2,432 shared kind-region scenarios")
    require(sum(int(row["count"]) for row in global_rows) == total_rows, "global count sum mismatch")
    after = fingerprint(tracked)
    require(before == after, "input files changed during audit")

    return dict(
        passed=not errors, errors=errors, created_utc=datetime.now(timezone.utc).isoformat(),
        audit="independent standard-library CSV -> per-group JSON -> global CSV semantic recomputation",
        numeric_method="math.fsum means; math.isclose rel_tol=1e-12 and abs_tol=1e-12",
        frozen_controller_count=len(controllers), expected_output_groups=42,
        verified_output_groups=len(groups), global_summary_rows=len(global_rows),
        expected_episode_records=expected_total, verified_episode_records=total_rows,
        total_test_episodes=total_rows,
        globally_unique_controller_scenario_records=len(global_episode_keys),
        paired_kind_region_scenarios=len(paired_scenario_keys), independent_test_initial_states=len(states),
        all_summary_fields_recomputed=["count", "passes", "physical_failures", "success_rate",
                                       "mean_actual_cost", "mean_command_cost", "max_settling_time_s"],
        frozen_source_inputs_models_and_all_audit_inputs_unchanged=before == after,
        trace_arrays_read=False, model_loading_or_predictions=False,
        study_code_imported=False, research_files_modified=False,
        scope="saved metric semantics and frozen input integrity; trajectory physics are checked by the separate verify_saved audit",
        script_sha256=sha256(Path(__file__).resolve()), input_sha256=after,
        groups=groups, in_domain_results=[group for group in groups if group["region"] == "in_domain"])


if __name__ == "__main__":
    result = verify()
    output = AUDIT / "summary_semantic_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "errors", "frozen_controller_count",
        "verified_output_groups", "verified_episode_records", "paired_kind_region_scenarios",
        "frozen_source_inputs_models_and_all_audit_inputs_unchanged")}, ensure_ascii=False))
    for row in result["in_domain_results"]:
        print(json.dumps(row, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
