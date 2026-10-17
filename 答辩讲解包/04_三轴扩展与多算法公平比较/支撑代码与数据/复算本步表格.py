"""Recalculate V2 tables from copied CSV. Standard library only; no training."""
from pathlib import Path
import csv
import json
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parent


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    rows = load(ROOT / "结果数据/episode_metrics.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[row["axis"], row["controller"]].append(row)
    output = []
    for (axis, controller), values in sorted(groups.items()):
        seeds = defaultdict(list)
        for value in values:
            seeds[value["model_seed"]].append(value["success"] == "True")
        seed_rates = [statistics.mean(v) for v in seeds.values()]
        success = sum(v["success"] == "True" for v in values)
        output.append({"axis": axis, "controller": controller, "episodes": len(values),
                       "successes": success, "success_rate_percent": 100 * success / len(values),
                       "seed_sample_sd_percentage_points": 100 * statistics.stdev(seed_rates) if len(seed_rates) > 1 else None,
                       "rms_error_deg": statistics.mean(float(v["rms_error_deg"]) for v in values),
                       "tail_mean_error_deg": statistics.mean(float(v["tail_mean_error_deg"]) for v in values),
                       "control_cost": statistics.mean(float(v["control_cost"]) for v in values)})
    assert len(rows) == 7560
    assert len(output) == 18
    lookup = {(r["axis"], r["controller"]): r for r in output}
    for axis in ("single", "three"):
        for controller in ("baseline_PID", "tuned_PID", "improved_PID", "PPO", "SAC", "TD3"):
            assert lookup[axis, controller]["episodes"] == (180 if "PID" in controller else 540)
    print(json.dumps({"verification": "passed", "total_episodes": len(rows), "tables": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
