"""Full-horizon acceptance and failure-aware, paired experiment summaries.

``actual_cost`` always means the physical torque-square integral.  It is never
replaced by a training penalty.  A separate ``penalized_selection_cost`` charges
the maximum possible remaining torque-square integral after a physical failure,
so a stopped physical simulation cannot win model selection by stopping early.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .protocol import DT, DURATION, DEADLINE, LIMIT, INERTIA, ANGLE_TOL, RATE_TOL


def summary(rows):
    if not rows:
        raise ValueError('no evaluation records')
    costs = np.array([r['actual_cost'] for r in rows])
    if not np.all(np.isfinite(costs)) or np.any(costs < -1e-12):
        raise ValueError('nonfinite/negative control costs')
    passed = [r for r in rows if r['success']]
    selection = np.array([r.get('penalized_selection_cost', r['actual_cost']) for r in rows])
    if not np.all(np.isfinite(selection)) or np.any(selection < costs-1e-12):
        raise ValueError('invalid failure-penalized selection costs')
    return dict(count=len(rows), passes=len(passed), success_rate=len(passed)/len(rows),
                physical_failures=sum(bool(r['failed']) for r in rows),
                mean_actual_cost=float(np.mean(costs)),
                mean_penalized_selection_cost=float(np.mean(selection)),
                cost_comparison_eligible=len(passed) == len(rows),
                mean_command_cost=float(np.mean([r['command_cost'] for r in rows])),
                mean_rms_error_deg=float(np.mean([r['rms_error_deg'] for r in rows])),
                max_settling_time_s=max((r['settling_time_s'] for r in passed), default=None),
                mean_settling_time_success_only_s=float(np.mean([r['settling_time_s'] for r in passed])) if passed else None)


def ranking(stats):
    """Larger is better; unsafe or early-failing policies cannot gain priority."""
    return (-stats.get('physical_failures', 0), stats['passes'],
            -stats.get('mean_penalized_selection_cost', stats['mean_actual_cost']))


def evaluate(kind, scenes, controller, keep_traces=False):
    from .physics import BatchPlant
    if not scenes:
        raise ValueError('no scenarios')
    n = len(scenes)
    plant = BatchPlant(kind, n)
    plant.reset(np.radians([s['angle_deg'] for s in scenes]),
                np.radians([s['omega_deg_s'] for s in scenes]),
                np.array([s['parameter_s'] for s in scenes]))
    obs = plant.observation()
    controller.reset(n, obs)
    steps = int(round(DURATION/DT))
    errors = np.abs(np.degrees(plant.theta))
    rates = np.abs(np.degrees(plant.omega))
    outside = (errors > ANGLE_TOL) | (rates > RATE_TOL)
    last_bad = np.where(outside, 0, -1)
    successful = np.ones(n, dtype=bool)
    tail_pass = np.ones(n, dtype=bool)
    max_error, max_rate = errors.copy(), rates.copy()
    integral_error = np.zeros(n)
    deadline_violation = np.zeros(n)
    actual_cost, command_cost = np.zeros(n), np.zeros(n)
    failure_time = np.full(n, np.nan)
    traces = None
    if keep_traces:
        traces = {'time': np.arange(steps+1, dtype=float)*DT,
                  'theta': np.empty((steps+1, n)), 'omega': np.empty((steps+1, n)),
                  'failed': np.empty((steps+1, n), dtype=bool)}
        for key in ('sent_command', 'command', 'tau_start', 'tau_end', 'actual_cost_interval', 'command_cost_interval'):
            traces[key] = np.empty((steps, n))
        traces['theta'][0], traces['omega'][0], traces['failed'][0] = plant.theta, plant.omega, plant.failed
    for k in range(steps):
        command = np.asarray(controller.act(obs, plant.elapsed), dtype=float).reshape(-1)
        if command.shape != (n,) or not np.all(np.isfinite(command)):
            raise ValueError('invalid controller command')
        if np.any(np.abs(command) > LIMIT+1e-9):
            raise ValueError('controller did not apply common command limit')
        step = plant.step(command)
        next_errors = np.abs(np.degrees(plant.theta))
        next_rates = np.abs(np.degrees(plant.omega))
        current_time = (k+1)*DT
        outside = (next_errors > ANGLE_TOL) | (next_rates > RATE_TOL) | plant.failed
        last_bad = np.where(outside, k+1, last_bad)
        if current_time >= DEADLINE-1e-10:
            successful &= ~outside
            deadline_violation += np.maximum(next_errors / ANGLE_TOL - 1., 0.) * DT
            deadline_violation += np.maximum(next_rates / RATE_TOL - 1., 0.) * DT
        if current_time >= 20.-1e-10:
            tail_pass &= ~outside
        failure_time = np.where(step['new_failure'], current_time, failure_time)
        actual_cost += step['actual_cost']
        command_cost += step['command_cost']
        integral_error += .5*(errors**2+next_errors**2)*DT
        max_error = np.maximum(max_error, next_errors)
        max_rate = np.maximum(max_rate, next_rates)
        if traces is not None:
            traces['theta'][k+1], traces['omega'][k+1], traces['failed'][k+1] = plant.theta, plant.omega, plant.failed
            # Keep the controller's issued command distinct from the plant's
            # safety shutdown after an absorbing physical failure.
            traces['sent_command'][k] = command
            for key in ('command', 'tau_start', 'tau_end'):
                traces[key][k] = step[key]
            traces['actual_cost_interval'][k] = step['actual_cost']
            traces['command_cost_interval'][k] = step['command_cost']
        errors, rates = next_errors, next_rates
        obs = plant.observation()
    rows = []
    for i, scene in enumerate(scenes):
        failure_time_s = None if np.isnan(failure_time[i]) else float(failure_time[i])
        penalty = 0. if failure_time_s is None else max(0., DURATION-failure_time_s)*LIMIT**2
        sustained_settling = float(max(0, int(last_bad[i])+1)*DT)
        if int(last_bad[i]) >= steps or plant.failed[i]:
            sustained_settling = None
        rows.append(dict(scene, success=bool(successful[i] and not plant.failed[i]),
                         failed=bool(plant.failed[i]), complete=bool(plant.elapsed[i] >= DURATION-1e-10),
                         failure_time_s=failure_time_s,
                         actual_cost=float(actual_cost[i]), command_cost=float(command_cost[i]),
                         remaining_failure_cost_penalty=float(penalty),
                         penalized_selection_cost=float(actual_cost[i]+penalty),
                         deadline_violation=float(deadline_violation[i]),
                         rms_error_deg=float(np.sqrt(integral_error[i]/DURATION)),
                         settling_time_s=float(max(0, int(last_bad[i])+1)*DT) if successful[i] and not plant.failed[i] else None,
                         sustained_settling_time_s=sustained_settling,
                         max_error_deg=float(max_error[i]), max_rate_deg_s=float(max_rate[i]),
                         final_error_deg=float(errors[i]), final_rate_deg_s=float(rates[i]),
                         extension_pass=bool(tail_pass[i] and not plant.failed[i])))
    return {'rows': rows, 'traces': traces}


def paired_cluster_bootstrap(reference, candidate, metric='actual_cost', *,
                             confidence=.95, resamples=20000, seed=150771,
                             require_all_success=True):
    """Pair every scene, then resample independent initial-state clusters.

    Parameter-grid repeats share ``sample_id`` and must not count as extra
    independent initial states.  Positive differences and reductions favour
    the candidate.  Successful-only subsets are deliberately never selected:
    when reliability is insufficient, raw cost superiority is ineligible.
    """
    if not 0. < confidence < 1. or int(resamples) != resamples or resamples < 100:
        raise ValueError('invalid confidence or resample count')
    if not reference or len(reference) != len(candidate):
        raise ValueError('paired nonempty records are required')
    def keyed(rows):
        result = {}
        for row in rows:
            key = row.get('scenario_id')
            if key is None or key in result or 'sample_id' not in row:
                raise ValueError('unique scenario_id and sample_id are required')
            result[key] = row
        return result
    first, second = keyed(reference), keyed(candidate)
    if first.keys() != second.keys():
        raise ValueError('candidate and reference do not cover identical scenes')
    clusters = {}
    all_success = True
    for key, before in first.items():
        after = second[key]
        if before['sample_id'] != after['sample_id']:
            raise ValueError('paired initial-state identifiers differ')
        for field in ('kind', 'split', 'region', 'parameter_s', 'angle_deg', 'omega_deg_s'):
            if before.get(field) != after.get(field):
                raise ValueError('paired physical scenarios differ: '+field)
        a, b = float(before[metric]), float(after[metric])
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError('paired metric must be finite')
        clusters.setdefault(before['sample_id'], []).append((a, b))
        all_success &= bool(before['success']) and bool(after['success'])
    sizes = {len(items) for items in clusters.values()}
    if len(sizes) != 1:
        raise ValueError('each initial-state cluster must have equal parameter coverage')
    group = np.array([np.mean(items, axis=0) for items in clusters.values()])
    means = group.mean(axis=0)
    eligible = all_success or not require_all_success
    result = dict(metric=metric, initial_state_clusters=len(group), paired_scenes=len(first),
                  parameter_repeats_per_cluster=sizes.pop(), all_success=bool(all_success),
                  cost_comparison_eligible=bool(eligible), confidence=float(confidence),
                  resamples=int(resamples), bootstrap_seed=int(seed),
                  reference_mean=float(means[0]), candidate_mean=float(means[1]),
                  mean_reference_minus_candidate=float(means[0]-means[1]),
                  relative_reduction_fraction=float((means[0]-means[1])/means[0]) if means[0] != 0. else None,
                  difference_interval=None, relative_reduction_interval=None)
    if not eligible:
        result['ineligible_reason'] = 'At least one paired controller failed the common full-horizon acceptance.'
        return result
    rng = np.random.default_rng(seed)
    boot = np.empty((int(resamples), 2))
    for offset in range(0, int(resamples), 1000):
        count = min(1000, int(resamples)-offset)
        indexes = rng.integers(0, len(group), size=(count, len(group)))
        boot[offset:offset+count] = group[indexes].mean(axis=1)
    quantiles = ((1.-confidence)/2., 1.-(1.-confidence)/2.)
    result['difference_interval'] = np.quantile(boot[:, 0]-boot[:, 1], quantiles).tolist()
    if np.all(boot[:, 0] != 0.):
        reductions = (boot[:, 0]-boot[:, 1])/boot[:, 0]
        result['relative_reduction_interval'] = np.quantile(reductions, quantiles).tolist()
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('no rows to write')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
