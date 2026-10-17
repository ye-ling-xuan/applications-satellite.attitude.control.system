"""Recalculate paired improvements; no training, no inference, no file writes."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / "结果数据" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    pid_groups = defaultdict(list)
    for row in load("paired_improvements.csv"):
        pid_groups[row["axis"], row["baseline"]].append(row)
    pid = []
    for (axis, baseline), rows in sorted(pid_groups.items()):
        assert len(rows) == 180
        pid.append({"axis": axis, "baseline": baseline, "paired_scenarios": len(rows),
                    "success_change_percentage_points": 100 * statistics.mean(float(r["success_rate_change"]) for r in rows),
                    "tail_mean_error_deg_baseline": statistics.mean(float(r["tail_mean_error_deg_baseline"]) for r in rows),
                    "tail_mean_error_deg_improved": statistics.mean(float(r["tail_mean_error_deg_improved"]) for r in rows),
                    "control_cost_change": statistics.mean(float(r["control_cost_change"]) for r in rows)})
    rl_groups = defaultdict(list)
    pairs = load("policy_calibration_pairs.csv")
    assert len(pairs) == 3240
    keys = [(r["axis"], r["algorithm"], r["model_seed"], r["scenario_id"]) for r in pairs]
    assert len(set(keys)) == len(keys)
    for row in pairs:
        rl_groups[row["axis"], row["algorithm"]].append(row)
    rl = []
    for (axis, algorithm), rows in sorted(rl_groups.items()):
        assert len(rows) == 540
        rl.append({"axis": axis, "algorithm": algorithm, "pairs": len(rows),
                   "raw_success_percent": 100 * statistics.mean(r["success_raw"] == "True" for r in rows),
                   "calibrated_success_percent": 100 * statistics.mean(r["success_calibrated"] == "True" for r in rows),
                   "success_recovered": sum(r["success_recovered"] == "True" for r in rows),
                   "success_lost": sum(r["success_lost"] == "True" for r in rows),
                   "raw_tail_error_deg": statistics.mean(float(r["tail_mean_error_deg_raw"]) for r in rows),
                   "calibrated_tail_error_deg": statistics.mean(float(r["tail_mean_error_deg_calibrated"]) for r in rows)})
    print(json.dumps({"verification": "passed", "pid": pid, "calibration": rl}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
