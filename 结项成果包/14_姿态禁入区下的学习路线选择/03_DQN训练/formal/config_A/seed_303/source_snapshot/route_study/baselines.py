"""Broad independently declared PID search, shared by every route selector."""
from __future__ import annotations
import csv
import time
import numpy as np
from . import protocol as p
from .physics import simulate, analytic_actions, summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _population(scenes):
    return np.array([[row['theta_rad'], row['target_rad']] for row in scenes])


def rank(stats):
    return (stats['successes'], -stats['collisions'], -stats['mean_torque_cost'])


def tune():
    if (p.STUDY/'04_独立测试'/'freeze.json').exists():
        raise RuntimeError('held-out study already frozen')
    directory = p.STUDY/'02_传统对照'
    if (directory/'selected_pid.json').exists():
        raise FileExistsError('baseline selection is retained and cannot be overwritten')
    start = time.monotonic()
    source = p.source_hashes()
    # 625 PD choices plus 125 integral choices: 750 physically evaluated
    # candidates, followed by complete validation of the best 40.
    candidates = [dict(kp=float(kp), kd=float(kd), ki=0.)
                  for kp in np.geomspace(.08, 8., 25) for kd in np.geomspace(.3, 12., 25)]
    candidates += [dict(kp=float(kp), kd=float(kd), ki=float(ki))
                   for kp in np.geomspace(.2, 2., 5) for kd in np.geomspace(.8, 5., 5)
                   for ki in (.0001, .001, .005, .02, .05)]
    manifest = dict(candidate_count=len(candidates), selection='successes first, collisions second, cost third',
                    information='all selectors use the same finally selected physical PID',
                    source_sha256=source, test_responses_used=False,
                    shared_pid_selection_population='analytic-safe-route development then validation')
    p.write_json(directory/'search_manifest.json', manifest)
    population = _population(p.scenes('development'))
    development = []
    for first in range(0, len(candidates), 32):
        group = candidates[first:first+32]
        tiled = np.tile(population, (len(group), 1))
        parameters = {key: np.repeat([item[key] for item in group], len(population)) for key in ('kp', 'kd', 'ki')}
        rows = simulate(tiled, analytic_actions(tiled), parameters)['rows']
        for j, item in enumerate(group):
            stats = summary(rows[j*len(population):(j+1)*len(population)])
            development.append(dict(candidate_id=first+j, **item, **stats))
    write_csv(directory/'all_development_candidates.csv', development)
    shortlist = sorted(range(len(candidates)), key=lambda i: rank(development[i]), reverse=True)[:40]
    validation = []
    scene_list = p.scenes('validation')
    population = _population(scene_list)
    for index in shortlist:
        stats = summary(simulate(scene_list, analytic_actions(population), candidates[index])['rows'])
        validation.append(dict(candidate_id=index, **candidates[index], **stats))
    write_csv(directory/'all_validation_candidates.csv', validation)
    selected = max(validation, key=rank)
    parameters = candidates[selected['candidate_id']]
    selected_validation = simulate(scene_list, analytic_actions(population), parameters)['rows']
    write_csv(directory/'analytic_validation.csv', selected_validation)
    direct_validation = simulate(scene_list, np.full(len(population), -1), parameters)['rows']
    write_csv(directory/'direct_validation.csv', direct_validation)
    final_source = p.source_hashes()
    if source != final_source:
        raise RuntimeError('source changed during classical search')
    result = dict(parameters=parameters, reported_feedback='PD' if parameters['ki'] == 0. else 'PID',
                  validation=summary(selected_validation), direct_validation=summary(direct_validation),
                  candidate_count=len(candidates), validation_candidate_count=len(shortlist),
                  analytic_feasibility_passed=all(row['success'] for row in selected_validation),
                  source_sha256=source, source_sha256_after=final_source,
                  elapsed_seconds=time.monotonic()-start, test_responses_used=False)
    p.write_json(directory/'selected_pid.json', result)
    print(result, flush=True)
    return result


def selected():
    return p.read_json(p.STUDY/'02_传统对照'/'selected_pid.json')
