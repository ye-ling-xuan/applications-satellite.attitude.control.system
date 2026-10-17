"""PID tuning for the declared deadline/effort study.

Only development and validation scenes are used by this module. All torque
gains are physical gains (observations are radians and rad/s). A candidate can
use common gains, diagonal-inertia-scaled gains, or independent axis gains.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from .protocol import STUDY, DEADLINES, DT, LIMIT, scenarios, write_json
from .evaluation import evaluate, metric_key

OUTPUT = STUDY / "02_PID整定"
SEARCH_SEED = 2026101744


class PIDController:
    """Rate-feedback PID with conditional integration and actuator clipping."""
    def __init__(self, axis, parameters):
        self.axis = axis
        self.dimensions = 1 if axis == "single" else 3
        self.parameters = dict(parameters)
        self.candidates = [dict(parameters)]
        self.reset(1)

    def reset(self, n):
        d = self.dimensions
        if n % len(self.candidates):
            raise ValueError("candidate-major scene count is not divisible")
        repeat = n // len(self.candidates)
        self.gains = {}
        for key in ("kp", "ki", "kd"):
            values = np.array([np.broadcast_to(c.get(key, 0.), (d,)) for c in self.candidates])
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("gains must be finite and nonnegative")
            self.gains[key] = np.repeat(values, repeat, axis=0)
        self.integral_limit = np.repeat(np.array([np.broadcast_to(c.get("integral_limit", 2.), (d,))
                                                  for c in self.candidates]), repeat, axis=0)
        self.integral = np.zeros((n, d))

    def act(self, observation, time=0.):
        obs = np.asarray(observation, float).reshape(-1, 2*self.dimensions)
        if obs.shape[0] != self.integral.shape[0]:
            self.reset(obs.shape[0])
        error, rate = np.split(obs, 2, axis=1)
        correction = -error if self.axis == "single" else error
        kp, ki, kd = (self.gains[k] for k in ("kp", "ki", "kd"))
        base = kp*correction-kd*rate
        proposed_integral = np.clip(self.integral+correction*DT[self.axis],
                                    -self.integral_limit, self.integral_limit)
        proposed = base+ki*proposed_integral
        delta = ki*(proposed_integral-self.integral)
        limit = LIMIT[self.axis]
        accept = ((np.abs(proposed) <= limit) | ((proposed > limit) & (delta < 0))
                  | ((proposed < -limit) & (delta > 0)))
        self.integral = np.where(accept, proposed_integral, self.integral)
        return np.clip((base+ki*self.integral)/limit, -1., 1.)


class CandidateBatchPID(PIDController):
    def __init__(self, axis, candidates):
        self.axis = axis
        self.dimensions = 1 if axis == "single" else 3
        self.candidates = [dict(c) for c in candidates]
        self.parameters = None


def _candidate(axis, kp, kd, ki=0., family="common_PD", stage="broad"):
    d = 1 if axis == "single" else 3
    return {"kp": np.broadcast_to(kp, (d,)).tolist(),
            "ki": np.broadcast_to(ki, (d,)).tolist(),
            "kd": np.broadcast_to(kd, (d,)).tolist(),
            "integral_limit": 2., "family": family, "stage": stage}


def _unique(candidates):
    result, seen = [], set()
    for candidate in candidates:
        key = tuple(round(v, 11) for k in ("kp", "ki", "kd") for v in candidate[k])
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def broad_candidates(axis):
    rng = np.random.default_rng(np.random.SeedSequence([SEARCH_SEED, 1 if axis == "single" else 3]))
    inertia = np.array([1.]) if axis == "single" else np.array([1.2, 1., .8])
    rows = []
    # Grid expressed through familiar second-order frequency/damping parameters;
    # these are only gain-generation coordinates, not a linearized simulator.
    for wn in np.geomspace(.35, 5., 25):
        for damping in np.linspace(.45, 1.8, 19):
            rows.append(_candidate(axis, wn**2, 2*damping*wn))
            if axis == "three":
                rows.append(_candidate(axis, inertia*wn**2, inertia*2*damping*wn,
                                       family="inertia_scaled_PD"))
    # Integral variants are included, even though the plant has no static bias.
    for _ in range(180):
        wn = np.exp(rng.uniform(np.log(.45), np.log(4.)))
        damping = rng.uniform(.55, 1.5)
        scale = inertia if axis == "three" and rng.random() < .5 else np.ones(len(inertia))
        ki = scale*np.exp(rng.uniform(np.log(.001), np.log(.5)))
        rows.append(_candidate(axis, scale*wn**2, scale*2*damping*wn, ki,
                               family="common_or_scaled_PID"))
    if axis == "three":
        for _ in range(420):
            wn = np.exp(rng.uniform(np.log(.45), np.log(3.5)))
            damping = rng.uniform(.55, 1.5)
            kp = inertia*wn**2*np.exp(rng.normal(0., .3, 3))
            kd = inertia*2*damping*wn*np.exp(rng.normal(0., .22, 3))
            ki = np.zeros(3) if rng.random() < .8 else np.exp(rng.uniform(np.log(.001), np.log(.2), 3))
            rows.append(_candidate(axis, kp, kd, ki, family="independent_axis_PD_PID"))
    # Existing commonly used gains are explicitly represented for reference.
    rows.append(_candidate(axis, 4. if axis == "single" else 3., 3. if axis == "single" else 4.))
    return _unique(rows)


def summarize(candidate, rows, axis, split, stage, index):
    result = {"candidate_id": f"{axis}_{stage}_{index:05d}", "axis": axis,
              "split": split, "stage": stage, "family": candidate["family"],
              "count": len(rows), "mean_control_cost": float(np.mean([r["control_cost"] for r in rows])),
              "max_settling_time_s": max((r["settling_time_s"] if r["settling_time_s"] is not None else 1e9 for r in rows)),
              "mean_rms_error_deg": float(np.mean([r["rms_error_deg"] for r in rows])),
              "failed": sum(r["failed"] for r in rows)}
    for deadline in DEADLINES:
        key = metric_key(deadline)
        result[key] = sum(r[key] for r in rows)
    return result


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidates(axis, candidates, split, stage, batch_size=48, save_episodes=False):
    """Batch candidate-major copies of exactly the same non-test scenes."""
    if split not in ("development", "validation"):
        raise ValueError("PID tuning cannot evaluate held-out test responses")
    scenes = scenarios(axis, split)
    summaries, episodes = [], []
    start = time.perf_counter()
    for offset in range(0, len(candidates), batch_size):
        chunk = candidates[offset:offset+batch_size]
        result = evaluate(axis, scenes*len(chunk), CandidateBatchPID(axis, chunk))
        for local, c in enumerate(chunk):
            rows = result["metrics"][local*len(scenes):(local+1)*len(scenes)]
            summary = summarize(c, rows, axis, split, stage, offset+local)
            summaries.append(summary)
            if save_episodes:
                for scene, row in zip(scenes, rows):
                    episodes.append({"candidate_id": summary["candidate_id"],
                                     "scenario_id": scene["scenario_id"], **row})
        print(f"PID {axis} {stage} {split}: {min(offset+len(chunk), len(candidates))}/{len(candidates)}, "
              f"{time.perf_counter()-start:.1f}s", flush=True)
    directory = OUTPUT/axis
    _write_csv(directory/f"{stage}_{split}_ranking.csv", summaries)
    if save_episodes:
        _write_csv(directory/f"{stage}_{split}_episodes.csv", episodes)
    write_json(directory/f"{stage}_{split}_candidates.json", candidates)
    return summaries


def _rank(summary, deadline):
    return (-summary[metric_key(deadline)], summary["mean_control_cost"])


def margin_selections(candidates, summaries):
    """Prespecified auxiliary: all validation scenes settle 0.25s early."""
    output = {}
    for deadline in DEADLINES:
        eligible = [i for i, row in enumerate(summaries)
                    if row[metric_key(deadline)] == row["count"]
                    and row["max_settling_time_s"] <= deadline-.25+1e-9]
        if eligible:
            i = min(eligible, key=lambda i: summaries[i]["mean_control_cost"])
            output[str(int(deadline))] = {
                "parameters": candidates[i], "validation": summaries[i],
                "selection_rule": "auxiliary: all validation pass and max settling <=deadline-0.25s, then min mean cost",
                "reserved_validation_time_margin_s": .25,
            }
        else:
            output[str(int(deadline))] = None
    return output


def add_existing_margin_selections(axis):
    """Read already-completed validation search; no extra controller responses."""
    directory = OUTPUT/axis
    summaries, candidates = [], []
    for stage in ("shortlist", "refine1", "refine2"):
        cp = directory/f"{stage}_validation_candidates.json"
        rp = directory/f"{stage}_validation_ranking.csv"
        if not cp.exists() or not rp.exists():
            continue
        cs = json.loads(cp.read_text(encoding="utf-8"))
        with rp.open(encoding="utf-8-sig", newline="") as stream:
            rs = list(csv.DictReader(stream))
        if len(rs) != len(cs):
            raise ValueError("candidate/ranking count differs")
        for row in rs:
            for key in ("count", "failed", "success_4s", "success_6s", "success_8s"):
                row[key] = int(row[key])
            for key in ("mean_control_cost", "max_settling_time_s", "mean_rms_error_deg"):
                row[key] = float(row[key])
        candidates.extend(cs)
        summaries.extend(rs)
    path = directory/"selected_pid.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["auxiliary_validation_time_margin_0_25s"] = margin_selections(candidates, summaries)
    write_json(path, result)
    return result


def shortlist(candidates, summaries, per_deadline=60):
    indexes = set()
    for deadline in DEADLINES:
        indexes.update(sorted(range(len(summaries)), key=lambda i: _rank(summaries[i], deadline))[:per_deadline])
    # Preserve the best from every family as well as the global leaders.
    for family in sorted({c["family"] for c in candidates}):
        indexes_in_family = [i for i, c in enumerate(candidates) if c["family"] == family]
        for deadline in DEADLINES:
            indexes.update(sorted(indexes_in_family, key=lambda i: _rank(summaries[i], deadline))[:8])
    return [candidates[i] for i in sorted(indexes)]


def refine_candidates(axis, candidates, summaries, round_number):
    rng = np.random.default_rng(np.random.SeedSequence([SEARCH_SEED, len(candidates), round_number,
                                                      1 if axis == "single" else 3]))
    rows = list(candidates)
    for deadline in DEADLINES:
        leaders = sorted(range(len(candidates)), key=lambda i: _rank(summaries[i], deadline))[:6]
        for i in leaders:
            parent = candidates[i]
            # Reduce search scale each round, retaining exact prior candidates.
            for j in range(20):
                sigma = (.16 if round_number == 1 else .07)
                if axis == "single" or j < 10:
                    kp = np.array(parent["kp"])*np.exp(rng.normal(0., sigma))
                    kd = np.array(parent["kd"])*np.exp(rng.normal(0., sigma))
                else:
                    kp = np.array(parent["kp"])*np.exp(rng.normal(0., sigma, 3))
                    kd = np.array(parent["kd"])*np.exp(rng.normal(0., sigma, 3))
                ki = np.array(parent["ki"])
                if j % 5 == 0:
                    ki = np.zeros_like(ki)
                elif np.any(ki):
                    ki *= np.exp(rng.normal(0., sigma))
                rows.append(_candidate(axis, kp, kd, ki, family=parent["family"],
                                       stage=f"refine{round_number}"))
    return _unique(rows)


def equivalence_check():
    """Check candidate batching and the established tuned-PID implementation."""
    from multi_algorithm_benchmark.controllers import PIDController as EstablishedPID
    results = []
    class Adapter:
        def __init__(self, controller):
            self.controller = controller
        def reset(self, n):
            self.controller.reset(n)
        def act(self, obs, time):
            return self.controller.act(obs)
    for axis in ("single", "three"):
        scenes = scenarios(axis, "development")[:3]
        candidates = [_candidate(axis, 1.2, 1.8, .02), _candidate(axis, .7, 1.4)]
        batched = evaluate(axis, scenes*2, CandidateBatchPID(axis, candidates), True)
        max_batch = 0.
        for i, candidate in enumerate(candidates):
            scalar = evaluate(axis, scenes, PIDController(axis, candidate), True)
            for key in ("error_deg", "rate_deg_s", "torque"):
                max_batch = max(max_batch, float(np.max(np.abs(
                    batched["traces"][key][:, i*len(scenes):(i+1)*len(scenes)]-scalar["traces"][key]))))
        candidate = candidates[0]
        established = EstablishedPID(axis, "tuned", {"kp": 1.2, "ki": .02, "kd": 1.8, "integral_limit": 2.})
        old = evaluate(axis, scenes, Adapter(established), True)
        new = evaluate(axis, scenes, PIDController(axis, candidate), True)
        max_old = max(float(np.max(np.abs(old["traces"][key]-new["traces"][key])))
                      for key in ("error_deg", "rate_deg_s", "torque"))
        if max_batch > 1e-12 or max_old > 1e-12:
            raise AssertionError((axis, max_batch, max_old))
        results.append({"axis": axis, "candidate_batch_max_abs_error": max_batch,
                        "established_tuned_PID_max_abs_error": max_old,
                        "scenes": len(scenes), "passed": True})
    write_json(OUTPUT/"implementation_equivalence.json", results)
    return results


def run_axis(axis, batch_size=48):
    start = time.perf_counter()
    candidates = broad_candidates(axis)
    development = evaluate_candidates(axis, candidates, "development", "broad", batch_size)
    candidates = shortlist(candidates, development)
    validation = evaluate_candidates(axis, candidates, "validation", "shortlist", max(8, batch_size//3), True)
    counts = {"broad_development_candidates": len(development), "shortlist_validation_candidates": len(candidates)}
    for round_number in (1, 2):
        # Refinement is entirely on declared validation data; no test responses.
        candidates = refine_candidates(axis, candidates, validation, round_number)
        validation = evaluate_candidates(axis, candidates, "validation", f"refine{round_number}",
                                         max(8, batch_size//3), True)
        counts[f"refine{round_number}_validation_candidates"] = len(candidates)
    selections = {}
    for deadline in DEADLINES:
        i = min(range(len(candidates)), key=lambda i: _rank(validation[i], deadline))
        selections[str(int(deadline))] = {"parameters": candidates[i], "validation": validation[i],
                                         "selection_rule": "max passing count, then min mean full-horizon torque-squared integral"}
    result = {"axis": axis, "search_seed": SEARCH_SEED,
              "test_responses_used": False, "gain_unit": "physical torque / rad and rad/s",
              "counts": counts, "wall_time_s": time.perf_counter()-start,
              "deadlines": selections,
              "auxiliary_validation_time_margin_0_25s": margin_selections(candidates, validation)}
    write_json(OUTPUT/axis/"selected_pid.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("single", "three", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    equivalence_check()
    if args.check_only:
        return
    write_json(OUTPUT/"search_design.json", {
        "random_seed": SEARCH_SEED, "development_scenes": 32, "validation_scenes": 96,
        "test_responses_used": False, "batch_size_development": args.batch_size,
        "broad_grid": {"natural_frequency_rad_s": [.35, 5., 25], "damping_ratio": [.45, 1.8, 19]},
        "families": ["common_PD", "inertia_scaled_PD (three-axis)", "common_or_scaled_PID",
                     "independent_axis_PD_PID (three-axis)"],
        "random_integral_candidates_per_axis": 180, "random_independent_three_axis_candidates": 420,
        "integral_limit_rad_s": 2., "conditional_anti_windup": True,
        "selection_rule": "maximize passing count then minimize full-horizon mean control effort",
        "shortlist": "top60 per deadline plus top8 per deadline and family on development",
        "refinement": "2 rounds: top6 per deadline each get20 perturbations; log-gain sigma0.16 then0.07; retain past candidates",
        "validation_reuse": "all refinement and selection uses same validation set; final reliability must be checked on independent test set",
        "prespecified_auxiliary": "Among already evaluated validation candidates, choose minimum mean cost with all scenes passing and max settling<=deadline-0.25s; no additional search; main selection unchanged",
        "no_guarantee_of_global_PID_optimum": True,
    })
    for axis in (("single", "three") if args.axis == "both" else (args.axis,)):
        run_axis(axis, args.batch_size)
    completed = {}
    for axis in ("single", "three"):
        path = OUTPUT/axis/"selected_pid.json"
        if path.exists():
            completed[axis] = json.loads(path.read_text(encoding="utf-8"))
    write_json(OUTPUT/"selected_pid.json", completed)


if __name__ == "__main__":
    main()
