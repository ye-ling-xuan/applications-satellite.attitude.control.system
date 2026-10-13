"""Predeclared paired statistics and empirical grid boundaries for V3."""
from collections import defaultdict
import numpy as np
from .protocol import TEST_RHOS, TRAINING_SEEDS

MAIN_CONTROLLERS = ("PID_N", "PID_R", "SAC_N", "SAC_R")
ALL_CONTROLLERS = (*MAIN_CONTROLLERS, "SAC_N_raw", "SAC_R_raw")
FIELDS = ("rms_error_deg", "tail_mean_error_deg", "control_cost", "tail_max_error_deg", "tail_max_rate_deg_s")


def contiguous_interval(levels, passed):
    """Only measured connected points through nominal; no interpolation."""
    levels = list(levels)
    center = levels.index(1.)
    if not passed[center]:
        return {"lower_rho": None, "upper_rho": None, "grid_points": 0, "nominal_passed": False,
                "reaches_test_lower_limit": False, "reaches_test_upper_limit": False}
    left = right = center
    while left > 0 and passed[left-1]: left -= 1
    while right+1 < len(levels) and passed[right+1]: right += 1
    return {"lower_rho": levels[left], "upper_rho": levels[right], "grid_points": right-left+1,
            "nominal_passed": True, "reaches_test_lower_limit": left == 0,
            "reaches_test_upper_limit": right == len(levels)-1}


def boundary_tables(rows):
    points, intervals = [], []
    for axis in ("single", "three"):
        for controller in ALL_CONTROLLERS:
            group = [r for r in rows if r["axis"] == axis and r["controller"] == controller and r["experiment"] == "scalar"]
            seeds = sorted({int(r["model_seed"]) for r in group})
            rates = {}
            for rho in TEST_RHOS:
                values = []
                for seed in seeds:
                    sample = [r for r in group if r["rho"] == rho and int(r["model_seed"]) == seed]
                    successes = sum(bool(r["success"]) for r in sample)
                    assert len(sample) == 30
                    rate = successes / len(sample)
                    rates[seed, rho] = rate
                    values.append(rate)
                    points.append({"axis": axis, "controller": controller, "model_seed": seed, "rho": rho,
                                   "successes": successes, "n_episodes": len(sample), "success_rate": rate,
                                   "passed_90_percent": successes >= 27})
                points.append({"axis": axis, "controller": controller, "model_seed": "common", "rho": rho,
                               "successes": None, "n_episodes": 30*len(seeds),
                               "success_rate": float(np.mean(values)), "worst_seed_success_rate": min(values),
                               "success_seed_std": float(np.std(values, ddof=1)) if len(values)>1 else None,
                               "passed_90_percent": min(values) >= .9})
            for seed in (*seeds, "common"):
                passed = [(min(rates[s,rho] for s in seeds) >= .9 if seed == "common" else rates[seed,rho] >= .9) for rho in TEST_RHOS]
                intervals.append({"axis": axis, "controller": controller, "model_seed": seed,
                                  "threshold": .9, "definition": "contiguous measured grid through nominal",
                                  **contiguous_interval(TEST_RHOS, passed)})
    return points, intervals


def paired_tables(rows):
    keyed = {(r["axis"], r["controller"], int(r["model_seed"]), r["scenario_id"]): r for r in rows}
    if len(keyed) != len(rows): raise ValueError("Duplicate result key")
    training_pairs, calibration_pairs, pid_pairs = [], [], []

    def compare(left, right, comparison):
        pair = {k:left[k] for k in ("axis", "scenario_id", "case_id", "experiment", "domain", "rho", "sample_id", "model_seed")}
        pair.update(comparison=comparison, controller_a=left["controller"], controller_b=right["controller"],
                    success_a=left["success"], success_b=right["success"],
                    success_change=int(right["success"])-int(left["success"]),
                    success_recovered=bool(right["success"] and not left["success"]),
                    success_lost=bool(left["success"] and not right["success"]),
                    settling_time_s_a=left["settling_time_s"], settling_time_s_b=right["settling_time_s"],
                    trace_file_a=left["trace_file"], trace_file_b=right["trace_file"])
        for field in FIELDS:
            pair[field+"_a"] = left[field]
            pair[field+"_b"] = right[field]
            pair[field+"_change"] = right[field]-left[field]
        return pair

    for (axis, controller, seed, scene), row in keyed.items():
        if controller in ("SAC_N", "SAC_N_raw"):
            other = controller.replace("SAC_N", "SAC_R")
            training_pairs.append(compare(row, keyed[axis,other,seed,scene], "randomized_minus_nominal"))
        if controller.endswith("_raw"):
            calibration_pairs.append(compare(row, keyed[axis,controller[:-4],seed,scene], "calibrated_minus_raw"))
        if controller == "PID_N":
            pid_pairs.append(compare(row, keyed[axis,"PID_R",seed,scene], "robust_minus_nominal_pid"))
    return training_pairs, calibration_pairs, pid_pairs


def aggregate(rows):
    records = []
    groups = defaultdict(list)
    for row in rows:
        region = "anisotropic" if row["experiment"] == "anisotropic" else "nominal" if row["rho"] == 1. else "in_domain" if .7 <= row["rho"] <= 1.3 else "out_of_domain"
        for label in (region, "all_scalar" if row["experiment"] == "scalar" else "all_anisotropic"):
            groups[row["axis"], row["controller"], label].append(row)
    for (axis, controller, region), group in groups.items():
        seed_groups = defaultdict(list)
        for row in group: seed_groups[int(row["model_seed"])].append(row)
        rates = [float(np.mean([bool(r["success"]) for r in values])) for values in seed_groups.values()]
        settled = [r["settling_time_s"] for r in group if r["settling_time_s"] is not None]
        records.append({"axis": axis, "controller": controller, "region": region,
                        "n_episodes": len(group), "success_rate": float(np.mean(rates)),
                        "worst_seed_success_rate": min(rates),
                        "success_seed_std": float(np.std(rates,ddof=1)) if len(rates)>1 else None,
                        "settling_time_s_successful": float(np.mean(settled)) if settled else None,
                        "failed_rate": float(np.mean([r["failed"] for r in group])),
                        **{field:float(np.mean([r[field] for r in group])) for field in ("max_rate_deg_s","max_error_deg") if field in group[0]},
                        **{field:float(np.mean([r[field] for r in group])) for field in FIELDS}})
    return records
