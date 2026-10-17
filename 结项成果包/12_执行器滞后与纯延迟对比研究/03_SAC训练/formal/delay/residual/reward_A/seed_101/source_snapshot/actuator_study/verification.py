"""Independent frozen-artifact, trajectory, acceptance and action audit.

Production physics/evaluation/controller/history code is never imported. Lag
states use a matrix exponential and torque costs use 16-point Gauss integration;
delay uses absolute past-command indices. Only frozen SAC network prediction is
shared, with features reconstructed here directly from recorded measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import protocol as p


TOLERANCES = dict(state=2e-10, torque=2e-11, interval_cost=2e-12,
                  traditional_action=2e-10, sac_action=2e-6, metric=2e-9)
METRICS = ('success', 'failed', 'complete', 'failure_time_s', 'actual_cost',
           'command_cost', 'rms_error_deg', 'settling_time_s', 'max_error_deg',
           'max_rate_deg_s', 'final_error_deg', 'final_rate_deg_s', 'extension_pass')

# Fixed Gaussian rule avoids initializing a separate BLAS/OpenMP runtime merely
# to diagonalize the Legendre companion matrix in a process also using PyTorch.
_POSITIVE_GAUSS_NODES = (.095012509837637440185, .281603550779258913230,
                         .458016777657227386342, .617876244402643748447,
                         .755404408355003033895, .865631202387831743880,
                         .944575023073232576078, .989400934991649932596)
_POSITIVE_GAUSS_WEIGHTS = (.189450610455068496285, .182603415044923588867,
                           .169156519395002538189, .149595988816576732082,
                           .124628971255533872052, .095158511682492784810,
                           .062253523938647892863, .027152459411754094852)


def _expm4(matrix):
    """Scaling, convergent Taylor series and squaring for a real 4x4 matrix.

    Plain scalar arithmetic makes this independent of production closed-form
    propagation and avoids loading a second BLAS runtime beside a frozen SAC.
    The scaled infinity norm is at most 1/2; terms are retained below double
    precision before undoing the scaling by repeated squaring.
    """
    matrix = np.asarray(matrix, float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('matrix exponential requires a finite 4x4 matrix')
    norm = max(sum(abs(float(value)) for value in row) for row in matrix)
    scaling = max(0, int(math.ceil(math.log2(norm/.5)))) if norm else 0
    scaled = [[float(value)/(2.**scaling) for value in row] for row in matrix]
    result = [[float(i == j) for j in range(4)] for i in range(4)]
    term = [row[:] for row in result]

    def multiply(left, right):
        return [[math.fsum(left[i][k]*right[k][j] for k in range(4))
                 for j in range(4)] for i in range(4)]

    for order in range(1, 50):
        product = multiply(term, scaled)
        term = [[value/order for value in row] for row in product]
        result = [[result[i][j]+term[i][j] for j in range(4)] for i in range(4)]
        if max(abs(value) for row in term for value in row) < 1e-18:
            break
    for _ in range(scaling):
        result = multiply(result, result)
    return np.asarray(result)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _path(study, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('artifact paths must be relative to the study')
    resolved = (study/relative).resolve()
    try:
        resolved.relative_to(study.resolve())
    except ValueError as error:
        raise ValueError('artifact path escapes the study') from error
    return resolved


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _equal(actual, expected, tolerance, label, errors, maxima, category):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        errors.append(f'{label}: shape {actual.shape} != {expected.shape}')
        return
    if actual.dtype == bool or expected.dtype == bool:
        if not np.array_equal(actual, expected):
            errors.append(f'{label}: boolean mismatch in {np.count_nonzero(actual != expected)} values')
        return
    difference = float(np.max(np.abs(actual-expected), initial=0.))
    maxima[category] = max(maxima.get(category, 0.), difference)
    if not np.all(np.isfinite(actual)) or difference > tolerance:
        errors.append(f'{label}: maximum absolute error {difference:.6g} > {tolerance:.6g}')


def _physical_reconstruction(kind, scenes, commands):
    """Reconstruct globally from initial conditions and saved sent commands."""
    steps, n = commands.shape
    parameters = np.array([scene['parameter_s'] for scene in scenes], float)
    if np.any(~np.isfinite(parameters)) or np.any(parameters < 0.):
        raise ValueError('invalid physical parameter')
    theta, omega = np.empty((steps+1, n)), np.empty((steps+1, n))
    failed = np.zeros((steps+1, n), dtype=bool)
    theta[0] = np.deg2rad([scene['angle_deg'] for scene in scenes])
    omega[0] = np.deg2rad([scene['omega_deg_s'] for scene in scenes])
    tau = np.zeros(n)
    start, end, cost = (np.zeros((steps, n)) for _ in range(3))
    command_cost = np.zeros((steps, n))
    lag_groups = []
    if kind == 'lag':
        nodes = np.array(tuple(-value for value in _POSITIVE_GAUSS_NODES[::-1])+_POSITIVE_GAUSS_NODES)
        weights = np.array(_POSITIVE_GAUSS_WEIGHTS[::-1]+_POSITIVE_GAUSS_WEIGHTS)
        for parameter in np.unique(parameters[parameters > 0.]):
            rows = np.flatnonzero(parameters == parameter)
            generator = np.zeros((4, 4))
            generator[0, 1], generator[1, 2] = 1., 1./p.INERTIA
            generator[2, 2], generator[2, 3] = -1./parameter, 1./parameter
            transition = _expm4(generator*p.DT)
            attenuation = np.exp(-(nodes+1.)*p.DT/(2.*parameter))
            lag_groups.append((rows, transition, attenuation, weights*p.DT/2.))
    elif kind == 'delay':
        ratios = parameters/p.DT
        if not np.allclose(ratios, np.rint(ratios), rtol=0., atol=1e-9):
            raise ValueError('delay is not an integer number of sample intervals')
        delays = np.rint(ratios).astype(int)
    else:
        raise ValueError('unknown plant kind')
    for k in range(steps):
        active = ~failed[k]
        u = np.where(active, commands[k], 0.)
        command_cost[k] = u*u*p.DT
        next_theta, next_omega = theta[k].copy(), omega[k].copy()
        if kind == 'lag':
            ideal = active & (parameters == 0.)
            start[k, ideal] = end[k, ideal] = u[ideal]
            cost[k, ideal] = u[ideal]**2*p.DT
            next_theta[ideal] += omega[k, ideal]*p.DT+u[ideal]*p.DT**2/(2.*p.INERTIA)
            next_omega[ideal] += u[ideal]*p.DT/p.INERTIA
            for group, transition, attenuation, weights in lag_groups:
                rows = group[active[group]]
                if not len(rows):
                    continue
                initial = np.column_stack((theta[k, rows], omega[k, rows], tau[rows], u[rows]))
                propagated = sum(initial[:, column, None]*transition[None, :, column]
                                 for column in range(4))
                next_theta[rows], next_omega[rows] = propagated[:, 0], propagated[:, 1]
                start[k, rows], end[k, rows] = tau[rows], propagated[:, 2]
                quadrature_torque = (tau[rows, None]*attenuation[None, :]
                                     + u[rows, None]*(1.-attenuation[None, :]))
                cost[k, rows] = np.sum(quadrature_torque**2*weights[None, :], axis=1)
        else:
            rows = np.flatnonzero(active)
            past = k-delays[rows]
            available = past >= 0
            applied = np.zeros(len(rows))
            applied[available] = commands[past[available], rows[available]]
            start[k, rows] = end[k, rows] = applied
            cost[k, rows] = applied**2*p.DT
            next_theta[rows] += omega[k, rows]*p.DT+applied*p.DT**2/(2.*p.INERTIA)
            next_omega[rows] += applied*p.DT/p.INERTIA
        theta[k+1], omega[k+1] = next_theta, next_omega
        new_failure = active & ((np.abs(next_theta) > np.deg2rad(p.FAIL_ANGLE))
                                | (np.abs(next_omega) > np.deg2rad(p.FAIL_RATE)))
        failed[k+1] = failed[k] | new_failure
        tau = end[k].copy()
    return dict(theta=theta, omega=omega, failed=failed, tau_start=start,
                tau_end=end, actual_cost_interval=cost,
                command_cost_interval=command_cost)


def recompute_rows(scenes, traces, physical):
    """Independent acceptance and measurement statistics on the complete grid."""
    times = traces['time']
    error = np.abs(np.rad2deg(traces['theta']))
    rate = np.abs(np.rad2deg(traces['omega']))
    failed = physical['failed']
    outside = (error > p.ANGLE_TOL) | (rate > p.RATE_TOL) | failed
    success = np.all(~outside[times >= p.DEADLINE-1e-10], axis=0) & ~failed[-1]
    extension = np.all(~outside[times >= 20.-1e-10], axis=0) & ~failed[-1]
    last_bad = np.max(np.where(outside, np.arange(len(times))[:, None], -1), axis=0)
    rms = np.sqrt(np.trapz(error**2, x=times, axis=0)/p.DURATION)
    rows = []
    for i, scene in enumerate(scenes):
        failures = np.flatnonzero(failed[:, i])
        rows.append(dict(scene, success=bool(success[i]), failed=bool(failed[-1, i]),
                         complete=bool(abs(times[-1]-p.DURATION) < 1e-12),
                         failure_time_s=float(times[failures[0]]) if len(failures) else None,
                         actual_cost=float(np.sum(physical['actual_cost_interval'][:, i])),
                         command_cost=float(np.sum(physical['command_cost_interval'][:, i])),
                         rms_error_deg=float(rms[i]),
                         settling_time_s=float(max(0, int(last_bad[i])+1)*p.DT) if success[i] else None,
                         max_error_deg=float(np.max(error[:, i])),
                         max_rate_deg_s=float(np.max(rate[:, i])),
                         final_error_deg=float(error[-1, i]), final_rate_deg_s=float(rate[-1, i]),
                         extension_pass=bool(extension[i])))
    return rows


class _PID:
    """Independent scalar-parameter PID algebra; no production controller call."""
    def __init__(self, parameters, n):
        self.kp, self.ki, self.kd = (float(parameters.get(key, 0.)) for key in ('kp', 'ki', 'kd'))
        self.bound = float(parameters.get('integral_limit', 2.))
        if not all(np.isfinite(value) and value >= 0. for value in
                   (self.kp, self.ki, self.kd, self.bound)):
            raise ValueError('invalid PID parameters')
        self.integral = np.zeros(n)

    def propose(self, error, rate):
        candidate = np.maximum(-self.bound, np.minimum(self.bound, self.integral+p.DT*error))
        return self.kp*error-self.kd*rate+self.ki*candidate, candidate

    def commit(self, candidate, raw_total, clipped):
        deepens = (raw_total-clipped)*self.ki*(candidate-self.integral) > 1e-15
        self.integral = np.where(deepens, self.integral, candidate)


def _policy_features(traces, k, method, raw=None, integral=None):
    """Build chronological windows by absolute indexing, without a history object."""
    n = traces['theta'].shape[1]
    result = np.zeros((n, 94 if method == 'direct' else 96), dtype=np.float32)
    indices = np.maximum(np.arange(k-p.HISTORY, k+1), 0)
    measurements = np.stack((-traces['theta'][indices].T/(np.pi/3.),
                              traces['omega'][indices].T/np.deg2rad(30.)), axis=2)
    result[:, :62] = measurements.reshape(n, 62)
    valid = min(k, p.HISTORY)
    if valid:
        result[:, 92-valid:92] = traces['command'][k-valid:k].T/p.LIMIT
    result[:, 92] = (k*p.DT)/p.DURATION
    result[:, 93] = valid/p.HISTORY
    if method == 'residual':
        result[:, 94], result[:, 95] = raw/p.LIMIT, integral/2.
    return result


def _audit_actions(traces, controller, model, errors, maxima):
    family, kind = controller['family'], controller['kind']
    steps, n = traces['command'].shape
    if family not in ('pid', 'model_compensated', 'direct', 'residual'):
        raise ValueError('unknown controller family')
    pid = None
    if family in ('pid', 'model_compensated', 'residual'):
        pid = _PID(controller['base_parameters'] if family == 'residual' else controller['parameters'], n)
    if family in ('direct', 'residual') and model is None:
        raise ValueError('frozen SAC model is required for policy action auditing')
    parameter = controller.get('parameters', {})
    torque_estimate = np.zeros(n)
    previous = np.zeros(n)
    if family == 'model_compensated':
        nominal = float(parameter.get('nominal_parameter_s', parameter.get(
            'nominal_lag_s' if kind == 'lag' else 'nominal_delay_s', .175 if kind == 'lag' else .15)))
        if not np.isfinite(nominal) or nominal < 0. or (kind == 'lag' and nominal == 0.):
            raise ValueError('invalid nominal compensation parameter')
        if kind == 'lag':
            decay = np.exp(-p.DT/nominal)
            impulse = nominal*(-np.expm1(-p.DT/nominal))
            ka = float(parameter.get('actuator_feedback', 0.))
            gain = float(parameter.get('observer_gain', 1.))
        else:
            nominal_steps = int(round(nominal/p.DT))
            if abs(nominal_steps*p.DT-nominal) > 1e-10:
                raise ValueError('nominal delay is not on the sample grid')
            attempted = np.zeros((steps, n))
    for k in range(steps):
        error, rate = -traces['theta'][k], traces['omega'][k]
        feedback_error, feedback_rate = error, rate
        if family == 'model_compensated' and kind == 'delay':
            predicted_angle = -error+rate*nominal_steps*p.DT
            predicted_rate = rate.copy()
            # Future pending intervals are addressed by their issue time.
            for offset in range(nominal_steps):
                issue_index = k-nominal_steps+offset
                if issue_index >= 0:
                    pending = attempted[issue_index]
                    predicted_angle = predicted_angle+pending*(nominal_steps-offset-.5)*p.DT**2/p.INERTIA
                    predicted_rate = predicted_rate+pending*p.DT/p.INERTIA
            feedback_error, feedback_rate = -predicted_angle, predicted_rate
        if pid is not None:
            raw, candidate = pid.propose(feedback_error, feedback_rate)
        if family in ('direct', 'residual'):
            features = _policy_features(traces, k, family,
                                        raw if pid is not None else None,
                                        pid.integral if pid is not None else None)
            action, _ = model.predict(features, deterministic=True)
            action = np.asarray(action, float).reshape(-1)
            if action.shape != (n,) or not np.all(np.isfinite(action)):
                raise ValueError('invalid frozen-policy action')
            raw_total = (p.LIMIT*np.clip(action, -1., 1.) if family == 'direct'
                         else raw+p.ALPHA*p.LIMIT*np.clip(action, -1., 1.))
        elif family == 'model_compensated' and kind == 'lag':
            if k:
                measured_impulse = p.INERTIA*(rate-traces['omega'][k-1])
                inferred_start = previous+(measured_impulse-previous*p.DT)/impulse
                inferred_end = previous+(inferred_start-previous)*decay
                predicted_end = previous+(torque_estimate-previous)*decay
                torque_estimate = (1.-gain)*predicted_end+gain*inferred_end
            raw_total = raw-ka*torque_estimate
        else:
            raw_total = raw
        command = np.clip(raw_total, -p.LIMIT, p.LIMIT)
        if pid is not None:
            pid.commit(candidate, raw_total, command)
        expected = np.where(~traces['failed'][k], command, 0.)
        tolerance = TOLERANCES['sac_action'] if family in ('direct', 'residual') else TOLERANCES['traditional_action']
        _equal(traces['command'][k], expected, tolerance, f'command at step {k}',
               errors, maxima, 'action')
        if family == 'model_compensated' and kind == 'delay':
            attempted[k] = command
        previous = command
        if len(errors) > 30:
            errors.append('further action mismatches omitted')
            break


def verify_shard(kind, traces, scenes, rows, controller, model=None):
    """Audit one homogeneous shard in memory; useful for independent unit checks."""
    errors, maxima = [], {}
    steps, n = int(round(p.DURATION/p.DT)), len(scenes)
    required = {'time': (steps+1,), 'theta': (steps+1, n), 'omega': (steps+1, n),
                'failed': (steps+1, n), 'command': (steps, n), 'tau_start': (steps, n),
                'tau_end': (steps, n), 'actual_cost_interval': (steps, n),
                'command_cost_interval': (steps, n)}
    if not n or len(rows) != n:
        return dict(passed=False, errors=['empty shard or scene/row count mismatch'], max_absolute_errors={})
    for key, shape in required.items():
        value = np.asarray(traces.get(key, []))
        if value.shape != shape or not np.all(np.isfinite(value)):
            errors.append(f'{key}: invalid shape or nonfinite data')
        if key == 'failed' and value.dtype != bool:
            errors.append('failed must have boolean dtype')
    if errors:
        return dict(passed=False, errors=errors, max_absolute_errors=maxima)
    _equal(traces['time'], np.arange(steps+1)*p.DT, 1e-12, 'uniform full time grid',
           errors, maxima, 'time')
    if np.any(np.abs(traces['command']) > p.LIMIT+1e-12):
        errors.append('physical command limit exceeded')
    if controller['kind'] != kind:
        errors.append('controller and shard plant kinds differ')
    try:
        physical = _physical_reconstruction(kind, scenes, traces['command'])
        for key in ('theta', 'omega', 'failed', 'tau_start', 'tau_end',
                    'actual_cost_interval', 'command_cost_interval'):
            category = ('state' if key in ('theta', 'omega', 'failed') else
                        'torque' if key.startswith('tau_') else 'interval_cost')
            _equal(traces[key], physical[key], TOLERANCES[category], key,
                   errors, maxima, category)
        if np.any(traces['command'][physical['failed'][:-1]] != 0.):
            errors.append('failed-row issued commands were not padded with zero')
        independent_rows = recompute_rows(scenes, traces, physical)
        for i, (scene, stored, independent) in enumerate(zip(scenes, rows, independent_rows)):
            for key, value in scene.items():
                if stored.get(key) != value:
                    errors.append(f'row {i} scene metadata differs: {key}')
            for key in METRICS:
                actual, expected = stored.get(key, '__missing__'), independent[key]
                if expected is None or isinstance(expected, bool):
                    if actual != expected or (isinstance(expected, bool) and not isinstance(actual, bool)):
                        errors.append(f'row {i} metric differs: {key}')
                elif not isinstance(actual, (int, float)) or not np.isfinite(actual):
                    errors.append(f'row {i} invalid metric: {key}')
                else:
                    _equal(np.asarray(actual), np.asarray(expected), TOLERANCES['metric'],
                           f'row {i} {key}', errors, maxima, 'metric')
        _audit_actions(traces, controller, model, errors, maxima)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append('shard audit error: '+str(error))
    return dict(passed=not errors, episodes=n, intervals=steps*n,
                errors=errors[:50], max_absolute_errors=maxima)


def _check_hashes(study, artifacts, label, errors):
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append(label+' has no artifact hashes')
        return 0
    checked = 0
    for relative, expected in artifacts.items():
        try:
            path = _path(study, relative)
            if _hash(path) != expected:
                errors.append(label+' hash mismatch: '+relative)
            checked += 1
        except (OSError, ValueError) as error:
            errors.append(label+' artifact unavailable: '+relative+': '+str(error))
    return checked


def _coverage(study, freeze, manifest, errors):
    controllers = freeze['controllers']
    identifiers = [controller['id'] for controller in controllers]
    if len(set(identifiers)) != len(identifiers):
        errors.append('duplicate controller IDs in freeze')
    registry = {controller['id']: controller for controller in controllers}
    initial_files = [relative for relative in freeze['artifacts']
                     if Path(relative).name == 'test_initial_states.json']
    if len(initial_files) != 1:
        errors.append('exactly one frozen test_initial_states.json is required')
        return registry, {}
    initials = _read(_path(study, initial_files[0]))
    rng = np.random.default_rng(p.SPLIT_SEEDS['test'])
    angles, rates = rng.uniform(-60., 60., p.COUNTS['test']), rng.uniform(-5., 5., p.COUNTS['test'])
    expected = [dict(sample_id=i, angle_deg=float(angle), omega_deg_s=float(rate))
                for i, (angle, rate) in enumerate(zip(angles, rates))]
    if initials != expected:
        errors.append('frozen test initials differ from the predeclared independent seed')
    keys = set()
    for entry in manifest['entries']:
        key = (entry['controller_id'], entry['kind'], entry['region'], float(entry['parameter_s']))
        if key in keys:
            errors.append('duplicate test shard: '+str(key))
        keys.add(key)
        if entry['controller_id'] not in registry:
            errors.append('test shard uses an unfrozen controller')
    expected_keys = set()
    for controller in controllers:
        regions = controller.get('regions', ['ideal', 'in_domain', 'out_of_domain'])
        if controller['stage'] != 'development_diagnostic' and set(regions) != {'ideal', 'in_domain', 'out_of_domain'}:
            errors.append('non-diagnostic controller omits required test regions')
        for region in regions:
            values = ((0.,) if region == 'ideal' else p.IN_DOMAIN if region == 'in_domain'
                      else p.OUT_DOMAIN[controller['kind']] if region == 'out_of_domain' else ())
            if not values:
                errors.append('unknown test region: '+str(region))
            for value in values:
                expected_keys.add((controller['id'], controller['kind'], region, float(value)))
    if keys != expected_keys:
        errors.append(f'test coverage differs: {len(expected_keys-keys)} missing, {len(keys-expected_keys)} unexpected shards')
    return registry, {row['sample_id']: row for row in expected}


def verify(freeze_path=None):
    """Verify all frozen inputs and complete test outputs, and save a report.

    ``freeze.artifacts`` hashes inputs fixed before testing. Its sibling
    ``test_manifest.json`` has ``entries``, output ``artifacts``, ``complete``
    and ``freeze_sha256``. Each entry's JSON is ``{scenes: [...], rows: [...]}``.
    Every artifact/model/entry path is relative to the study root.
    """
    freeze_path = Path(freeze_path or p.STUDY/'04_独立测试'/'freeze.json').resolve()
    study = freeze_path.parent.parent
    output = study/'05_独立核验'/'核验结果.json'
    started = time.perf_counter()
    report = dict(passed=False, audit_version=1, verified_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_input_artifacts_checked=0, test_output_artifacts_checked=0,
                  shards_verified=0, episodes_verified=0, intervals_verified=0,
                  max_absolute_errors={}, errors=[], shards=[], tolerances=TOLERANCES,
                  methodology=dict(lag_state='independent scalar scaling/Taylor/squaring matrix exponential',
                                   actual_cost='independent 16-point Gauss-Legendre torque integration',
                                   delay='absolute past-command indexing and ideal interval dynamics',
                                   metrics='independent complete-grid success, settling, failure, RMS and costs',
                                   actions='independent PID/compensation; reconstructed history and frozen deterministic SAC'))
    errors = report['errors']
    try:
        freeze = _read(freeze_path)
        report['freeze_sha256'] = _hash(freeze_path)
        manifest_path = (_path(study, freeze['test_manifest_path']) if 'test_manifest_path' in freeze
                         else freeze_path.with_name('test_manifest.json'))
        manifest = _read(manifest_path)
        report['test_manifest_sha256'] = _hash(manifest_path)
        if manifest.get('complete') is not True:
            errors.append('test manifest is not marked complete')
        if manifest.get('freeze_sha256') != report['freeze_sha256']:
            errors.append('test manifest is not bound to this frozen input manifest')
        report['frozen_input_artifacts_checked'] = _check_hashes(study, freeze.get('artifacts'), 'freeze', errors)
        report['test_output_artifacts_checked'] = _check_hashes(study, manifest.get('artifacts'), 'test', errors)
        registry, initials = _coverage(study, freeze, manifest, errors)
        for controller in freeze['controllers']:
            if controller['family'] in ('direct', 'residual') and controller.get('model_path') not in freeze['artifacts']:
                errors.append('SAC model lacks a frozen input hash: '+controller['id'])
        for entry in manifest['entries']:
            for key in ('trace_path', 'data_path'):
                if entry[key] not in manifest['artifacts']:
                    errors.append('test shard lacks an output hash: '+entry[key])
        # Never load a policy archive or claim a trajectory audit under a broken seal.
        if not errors:
            models = {}
            for entry in manifest['entries']:
                controller = registry[entry['controller_id']]
                if controller['family'] in ('direct', 'residual') and controller['id'] not in models:
                    from stable_baselines3 import SAC
                    import torch
                    torch.set_num_threads(1)
                    models[controller['id']] = SAC.load(_path(study, controller['model_path']), device='cpu')
                data = _read(_path(study, entry['data_path']))
                shard_errors = []
                scene_ids = [scene['sample_id'] for scene in data['scenes']]
                if len(scene_ids) != p.COUNTS['test'] or set(scene_ids) != set(initials):
                    shard_errors.append('shard does not contain all independent test sample IDs exactly once')
                for scene in data['scenes']:
                    reference = initials.get(scene['sample_id'])
                    if (reference is None or any(scene.get(key) != value for key, value in reference.items())
                            or scene.get('split') != 'test' or scene.get('kind') != entry['kind']
                            or scene.get('region') != entry['region']
                            or scene.get('parameter_s') != entry['parameter_s']):
                        shard_errors.append('scene identity/initial condition differs from frozen test inputs')
                        break
                with np.load(_path(study, entry['trace_path']), allow_pickle=False) as archive:
                    traces = {key: archive[key] for key in archive.files}
                audit = verify_shard(entry['kind'], traces, data['scenes'], data['rows'],
                                     controller, models.get(controller['id']))
                audit['errors'] = shard_errors+audit['errors']
                audit['passed'] = not audit['errors']
                audit.update(controller_id=controller['id'], stage=controller['stage'],
                             kind=entry['kind'], region=entry['region'], parameter_s=entry['parameter_s'],
                             trace_path=entry['trace_path'], data_path=entry['data_path'])
                report['shards'].append(audit)
                report['shards_verified'] += 1
                report['episodes_verified'] += audit.get('episodes', 0)
                report['intervals_verified'] += audit.get('intervals', 0)
                for category, value in audit['max_absolute_errors'].items():
                    report['max_absolute_errors'][category] = max(report['max_absolute_errors'].get(category, 0.), value)
                if not audit['passed']:
                    errors.append(f"failed shard {controller['id']}/{entry['region']}/{entry['parameter_s']}: "+'; '.join(audit['errors'][:3]))
                print(f"independent audit {report['shards_verified']}/{len(manifest['entries'])}: "
                      f"{controller['id']} {entry['region']} {entry['parameter_s']} passed={audit['passed']}", flush=True)
    except (OSError, KeyError, TypeError, ValueError, ImportError, RuntimeError) as error:
        errors.append('verification could not complete: '+str(error))
    report['passed'] = not errors and report['shards_verified'] > 0
    report['wall_time_s'] = time.perf_counter()-started
    _write(output, report)
    return report


def verify_all(freeze_path=None):
    return verify(freeze_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-path', type=Path)
    args = parser.parse_args(argv)
    result = verify(args.freeze_path)
    print(json.dumps({key: value for key, value in result.items() if key != 'shards'}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
