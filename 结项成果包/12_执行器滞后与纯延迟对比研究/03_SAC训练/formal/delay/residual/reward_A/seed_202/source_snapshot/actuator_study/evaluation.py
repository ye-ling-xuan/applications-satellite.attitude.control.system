"""Full, uniformly sampled acceptance with interval-exact actual torque costs."""
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
    return dict(count=len(rows), passes=len(passed), success_rate=len(passed)/len(rows),
                physical_failures=sum(bool(r['failed']) for r in rows),
                mean_actual_cost=float(np.mean(costs)),
                mean_command_cost=float(np.mean([r['command_cost'] for r in rows])),
                mean_rms_error_deg=float(np.mean([r['rms_error_deg'] for r in rows])),
                max_settling_time_s=max((r['settling_time_s'] for r in passed), default=None),
                mean_settling_time_success_only_s=float(np.mean([r['settling_time_s'] for r in passed])) if passed else None)


def ranking(stats):
    return (stats['passes'], -stats['mean_actual_cost'])


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
    actual_cost, command_cost = np.zeros(n), np.zeros(n)
    failure_time = np.full(n, np.nan)
    traces = None
    if keep_traces:
        traces = {'time': np.arange(steps+1, dtype=float)*DT,
                  'theta': np.empty((steps+1, n)), 'omega': np.empty((steps+1, n)),
                  'failed': np.empty((steps+1, n), dtype=bool)}
        for key in ('command', 'tau_start', 'tau_end', 'actual_cost_interval', 'command_cost_interval'):
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
            for key in ('command', 'tau_start', 'tau_end'):
                traces[key][k] = step[key]
            traces['actual_cost_interval'][k] = step['actual_cost']
            traces['command_cost_interval'][k] = step['command_cost']
        errors, rates = next_errors, next_rates
        obs = plant.observation()
    rows = []
    for i, scene in enumerate(scenes):
        rows.append(dict(scene, success=bool(successful[i] and not plant.failed[i]),
                         failed=bool(plant.failed[i]), complete=bool(plant.elapsed[i] >= DURATION-1e-10),
                         failure_time_s=None if np.isnan(failure_time[i]) else float(failure_time[i]),
                         actual_cost=float(actual_cost[i]), command_cost=float(command_cost[i]),
                         rms_error_deg=float(np.sqrt(integral_error[i]/DURATION)),
                         settling_time_s=float(max(0, int(last_bad[i])+1)*DT) if successful[i] and not plant.failed[i] else None,
                         max_error_deg=float(max_error[i]), max_rate_deg_s=float(max_rate[i]),
                         final_error_deg=float(errors[i]), final_rate_deg_s=float(rates[i]),
                         extension_pass=bool(tail_pass[i] and not plant.failed[i])))
    return {'rows': rows, 'traces': traces}


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
