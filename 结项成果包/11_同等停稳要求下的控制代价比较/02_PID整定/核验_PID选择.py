"""Recompute frozen PID validation selections from per-episode records only."""
from pathlib import Path
import csv
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent/"代码"), str(ROOT.parent.parent)]
from deadline_study.protocol import write_json


def main():
    selected = json.loads((ROOT/"selected_pid.json").read_text(encoding="utf-8"))
    checks = []
    for axis, result in selected.items():
        cached = {}
        for group in ("deadlines", "auxiliary_validation_time_margin_0_25s"):
            for deadline, item in result[group].items():
                if item is None:
                    continue
                summary = item["validation"]
                stage = summary["stage"]
                if stage not in cached:
                    with (ROOT/axis/f"{stage}_validation_episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
                        cached[stage] = list(csv.DictReader(stream))
                rows = [r for r in cached[stage] if r["candidate_id"] == summary["candidate_id"]]
                mean = sum(float(r["control_cost"]) for r in rows)/len(rows)
                passed = sum(r[f"success_{deadline}s"] == "True" for r in rows)
                settling = max(float(r["settling_time_s"]) for r in rows)
                assert len(rows) == 96 and passed == summary[f"success_{deadline}s"] == 96
                assert math.isclose(mean, summary["mean_control_cost"], rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(settling, summary["max_settling_time_s"], abs_tol=1e-12)
                if group != "deadlines":
                    assert settling <= float(deadline)-.25+1e-9
                checks.append({"axis": axis, "group": group, "deadline_s": int(deadline),
                               "episode_count": len(rows), "passing": passed,
                               "mean_control_cost_recomputed": mean,
                               "max_settling_time_recomputed": settling, "passed": True})
    value = {"test_responses_read": False, "all_passed": True, "checks": checks}
    write_json(ROOT/"selected_validation_recomputation.json", value)
    print(json.dumps({"checks": len(checks), "all_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
