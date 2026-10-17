"""Independent action, feedback, continuous collision and dynamics replay.

No production physics, feature, planner, PID or summary helper is imported.
"""
from __future__ import annotations
import csv
import numpy as np
from . import protocol as p


def verify():
    frozen = p.read_json(p.STUDY/'04_独立测试'/'freeze.json')
    if frozen['source_sha256'] != p.source_hashes():
        raise RuntimeError('frozen source differs')
    for relative, digest in frozen['artifact_sha256'].items():
        if p.sha256(p.STUDY/relative) != digest:
            raise RuntimeError('frozen artifact differs')
    manifest = p.read_json(p.STUDY/'04_独立测试'/'responses'/'response_manifest.json')
    if manifest['freeze_sha256'] != p.sha256(p.STUDY/'04_独立测试'/'freeze.json'):
        raise RuntimeError('response manifest refers to a different held-out freeze')
    for relative, digest in manifest['files'].items():
        if p.sha256(p.STUDY/relative) != digest:
            raise RuntimeError('response checksum differs')
    model_info = {item['name']: item for item in frozen['models']}
    response_root = p.STUDY/'04_独立测试'/'responses'
    expected = {response_root/method/split/'traces.npz'
                for method in ['direct_PID', 'analytic_route_PID', *model_info]
                for split in ('test', 'detour', 'edge')}
    actual = set(response_root.glob('*/*/traces.npz'))
    if actual != expected:
        raise RuntimeError('held-out response shards are missing or unexpected')
    results = []
    for path in sorted(actual):
        method, region = path.parents[1].name, path.parent.name
        with np.load(path) as archive:
            trace = {name: archive[name] for name in archive.files}
        initial, target = trace['population'][:, 0], trace['population'][:, 1]
        n = len(initial)
        scenarios = p.scenes(region)
        expected_population = np.array([[row['theta_rad'], row['target_rad']] for row in scenarios])
        np.testing.assert_array_equal(trace['population'], expected_population)
        ticks = int(round(p.DURATION/p.DT))
        np.testing.assert_allclose(trace['time'], np.arange(ticks+1)*p.DT, atol=1e-14, rtol=0.)
        for name in ('theta', 'omega', 'collision'):
            if trace[name].shape != (ticks+1, n):
                raise RuntimeError('incorrect full state trajectory dimensions')
        for name in ('command', 'interval_cost'):
            if trace[name].shape != (ticks, n):
                raise RuntimeError('incorrect full interval trajectory dimensions')
        np.testing.assert_array_equal(trace['theta'][0], initial)
        np.testing.assert_array_equal(trace['omega'][0], np.zeros(n))
        np.testing.assert_array_equal(trace['collision'][0], np.zeros(n, bool))
        if method == 'direct_PID':
            actions = np.full(n, -1)
        elif method == 'analytic_route_PID':
            actions = (target-initial > 0.).astype(int)
        else:
            from stable_baselines3 import DQN
            import torch
            network = DQN.load(p.STUDY/model_info[method]['model_path'], device='cpu').q_net
            independent_features = np.stack([np.sin(initial), np.cos(initial), np.sin(target), np.cos(target)], axis=1).astype(np.float32)
            with torch.no_grad():
                actions = network(torch.from_numpy(independent_features)).argmax(dim=1).numpy()
        np.testing.assert_array_equal(actions, trace['action'])
        theta, omega, integral = initial.copy(), np.zeros(n), np.zeros(n)
        failed, accepted_task = np.zeros(n, bool), np.ones(n, bool)
        cost, first_collision = np.zeros(n), np.full(n, -1, int)
        physical_intervals = np.zeros(n, int)
        last_bad = np.where(abs(np.arctan2(np.sin(target-initial), np.cos(target-initial))) > p.ANGLE_TOL, 0, -1)
        lifted = np.where(actions == 1, target+2*np.pi*np.ceil((initial-target)/(2*np.pi)),
                          target+2*np.pi*np.floor((initial-target)/(2*np.pi)))
        parameter = frozen['parameters']
        worst = 0.
        for k in range(len(trace['command'])):
            physical_intervals += ~failed
            error = np.where(actions == -1, np.arctan2(np.sin(target-theta), np.cos(target-theta)), lifted-theta)
            candidate = np.minimum(2., np.maximum(-2., integral+p.DT*error))
            raw = parameter['kp']*error-parameter['kd']*omega+parameter['ki']*candidate
            command = np.maximum(-p.LIMIT, np.minimum(p.LIMIT, raw))
            command[failed] = 0.
            accept = (raw-command)*(candidate-integral)*parameter['ki'] <= 1e-15
            integral = np.where(accept & ~failed, candidate, integral)
            np.testing.assert_allclose(command, trace['command'][k], atol=2e-10, rtol=0.)
            end = theta+p.DT*omega+p.DT**2*command/(2*p.INERTIA)
            turning = np.divide(-omega*p.INERTIA, command, out=np.zeros(n), where=command != 0.)
            turning = np.clip(turning, 0., p.DT)
            extremum = theta+omega*turning+command*turning**2/(2*p.INERTIA)
            low = np.minimum(np.minimum(theta, end), extremum)
            high = np.maximum(np.maximum(theta, end), extremum)
            center = np.pi+2*np.pi*np.rint(((low+high)/2-np.pi)/(2*np.pi))
            crossing = np.zeros(n, bool)
            for offset in (-2*np.pi, 0., 2*np.pi):
                crossing |= (low <= center+offset+np.deg2rad(20.)+1e-12) & (high >= center+offset-np.deg2rad(20.)-1e-12)
            crossing &= ~failed
            first_collision[crossing] = k
            theta = np.where(failed, theta, end)
            omega = np.where(failed, omega, omega+p.DT*command/p.INERTIA)
            failed |= crossing
            cost += p.DT*command*command
            worst = max(worst, float(np.max(abs(theta-trace['theta'][k+1]))), float(np.max(abs(omega-trace['omega'][k+1]))))
            np.testing.assert_allclose(theta, trace['theta'][k+1], atol=2e-10, rtol=0.)
            np.testing.assert_allclose(omega, trace['omega'][k+1], atol=2e-10, rtol=0.)
            np.testing.assert_array_equal(failed, trace['collision'][k+1])
            np.testing.assert_allclose(p.DT*command**2, trace['interval_cost'][k], atol=2e-11, rtol=0.)
            outside = failed | (abs(np.arctan2(np.sin(target-theta), np.cos(target-theta))) > p.ANGLE_TOL) | (abs(omega) > p.RATE_TOL)
            last_bad = np.where(outside, k+1, last_bad)
            if (k+1)*p.DT >= p.DEADLINE-1e-10:
                accepted_task &= ~outside
        with (path.parent/'episodes.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if [row['scenario_id'] for row in rows] != [row['scenario_id'] for row in scenarios]:
            raise RuntimeError('episode identities/order differ from frozen scenario split')
        np.testing.assert_array_equal(actions, [int(row['action']) for row in rows])
        np.testing.assert_array_equal(accepted_task, [row['success'] == 'True' for row in rows])
        np.testing.assert_array_equal(failed, [row['collision'] == 'True' for row in rows])
        np.testing.assert_array_equal(first_collision, [int(row['collision_interval']) for row in rows])
        np.testing.assert_allclose(cost, [float(row['torque_cost']) for row in rows], atol=2e-10, rtol=0.)
        np.testing.assert_array_equal(physical_intervals, [int(row['physical_intervals']) for row in rows])
        np.testing.assert_allclose(theta, [float(row['final_theta_rad']) for row in rows], atol=2e-10, rtol=0.)
        np.testing.assert_allclose(np.rad2deg(abs(np.arctan2(np.sin(target-theta), np.cos(target-theta)))),
                                   [float(row['final_error_deg']) for row in rows], atol=2e-9, rtol=0.)
        np.testing.assert_allclose(np.rad2deg(abs(omega)), [float(row['final_rate_deg_s']) for row in rows], atol=2e-9, rtol=0.)
        for i, row in enumerate(rows):
            if accepted_task[i]:
                np.testing.assert_allclose(float(row['settling_time_s']), max(0, last_bad[i]+1)*p.DT, atol=2e-10, rtol=0.)
            elif row['settling_time_s'] != '':
                raise RuntimeError('failed scenario has a reported settling time')
        results.append(dict(method=method, region=region, count=n, max_state_difference=worst, passed=True,
                            successes=int(accepted_task.sum()), collisions=int(failed.sum()),
                            mean_torque_cost=float(cost.mean()),
                            mean_success_cost=float(cost[accepted_task].mean()) if np.any(accepted_task) else None))
    if manifest['total_scenario_responses'] != sum(item['count'] for item in results):
        raise RuntimeError('response manifest total count differs')
    with (response_root/'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        aggregate = list(csv.DictReader(stream))
    if {(row['method'], row['region']) for row in aggregate} != {(row['method'], row['region']) for row in results} or len(aggregate) != len(results):
        raise RuntimeError('summary has missing or duplicate method/region records')
    for expected_row in results:
        row = next(item for item in aggregate if item['method'] == expected_row['method'] and item['region'] == expected_row['region'])
        for key in ('count', 'successes', 'collisions'):
            if int(row[key]) != expected_row[key]:
                raise RuntimeError('aggregate response counts differ')
        for key, value in [('success_rate', expected_row['successes']/expected_row['count']),
                           ('safety_rate', 1-expected_row['collisions']/expected_row['count']),
                           ('mean_torque_cost', expected_row['mean_torque_cost'])]:
            np.testing.assert_allclose(float(row[key]), value, atol=2e-10, rtol=0.)
        if expected_row['mean_success_cost'] is None:
            if row['mean_success_cost'] != '':
                raise RuntimeError('aggregate successful cost is present without success')
        else:
            np.testing.assert_allclose(float(row['mean_success_cost']), expected_row['mean_success_cost'], atol=2e-10, rtol=0.)
    result = dict(passed=bool(results) and all(item['passed'] for item in results), checks=results,
                  helpers_imported_from_production=False,
                  independently_recomputed=['DQN q-network argmax', 'analytic direction', 'PID antiwindup', 'exact held-torque propagation', 'continuous periodic-arc intersection', 'costs', 'full acceptance window'])
    p.write_json(p.STUDY/'05_独立核验'/'verification.json', result)
    return result
