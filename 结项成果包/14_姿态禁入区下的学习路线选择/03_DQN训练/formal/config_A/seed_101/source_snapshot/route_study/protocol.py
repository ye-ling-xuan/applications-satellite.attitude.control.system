"""Fixed geometry, independent scenario splits, and immutable declarations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
DT, DURATION, DEADLINE = .05, 22., 12.
INERTIA, LIMIT = 1., 2.
CENTER, HALF_WIDTH, GUARD = np.pi, np.deg2rad(20.), np.deg2rad(15.)
BOUND = np.pi-HALF_WIDTH-GUARD
ANGLE_TOL, RATE_TOL = np.deg2rad(1.), np.deg2rad(.5)
MIN_SEPARATION = np.deg2rad(15.)
SPLIT_SEEDS = dict(development=2026101701401, validation=2026101701402,
                   test=2026101701403, training=2026101701404,
                   detour=2026101701405, edge=2026101701406)
COUNTS = dict(development=64, validation=128, test=512, detour=128, edge=128, training=20000)
FORMAL_SEEDS = (101, 202, 303)
DEVELOPMENT_SEEDS = dict(A=1401, B=1402)
DEVELOPMENT_DECISIONS, FORMAL_DECISIONS, CHECK_EVERY = 10000, 20000, 2000
N_ENVS = 16


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {path.relative_to(STUDY).as_posix(): sha256(path)
            for path in sorted((STUDY/'代码').rglob('*.py')) if '__pycache__' not in path.parts}


def sample(rng, n, region='natural'):
    """Rejection sampling specifies populations, never policy action labels."""
    theta, target = [], []
    while len(theta) < n:
        a, b = rng.uniform(-BOUND, BOUND, size=(2, max(64, n-len(theta))))
        keep = np.abs(a-b) >= MIN_SEPARATION
        if region == 'detour':
            keep &= np.abs(a-b) > np.pi
        elif region == 'edge':
            # At least one endpoint lies within 5 deg of the declared guarded
            # sampling boundary; this is not the actual forbidden boundary.
            keep &= np.maximum(np.abs(a), np.abs(b)) >= BOUND-np.deg2rad(5.)
        elif region != 'natural':
            raise ValueError(region)
        theta.extend(a[keep].tolist())
        target.extend(b[keep].tolist())
    return np.column_stack((theta[:n], target[:n]))


def scenes(split):
    return read_json(STUDY/'01_方案与场景'/f'{split}.json')


def configuration():
    return dict(task='one terminal contextual route decision, followed by fixed PID; not continuous learned feedback',
                dt_s=DT, duration_s=DURATION, deadline_s=DEADLINE, inertia=INERTIA,
                limit_Nm=LIMIT, forbidden_center_deg=180., forbidden_half_width_deg=20.,
                sampling_boundary_guard_deg=15., minimum_endpoint_separation_deg=15.,
                initial_omega_rad_s=0., angle_tolerance_deg=1., rate_tolerance_deg_s=.5,
                collision='closed forbidden arc, checked over exact quadratic hold interval including interior extrema',
                failed_interval='integrate complete first contact interval; later physical states frozen and cost zero',
                action_0='clockwise/decreasing unwrapped angle', action_1='counterclockwise/increasing unwrapped angle',
                direct_pid='shortest wrapped current goal error, without route planning',
                analytic='unique safe direction in the connected allowed arc',
                counts=COUNTS, split_seeds=SPLIT_SEEDS,
                training_population='20000 independent natural initial/target pairs, disjoint generator from all assessment sets',
                training_database='both actions precomputed by full physical feedback rollout; online actions chosen by epsilon-greedy DQN, no oracle labels or corrections',
                development_decisions=DEVELOPMENT_DECISIONS, formal_decisions=FORMAL_DECISIONS,
                maximum_route_decisions=80000, formal_seeds=list(FORMAL_SEEDS),
                development_seeds=DEVELOPMENT_SEEDS, check_every=CHECK_EVERY,
                development_gate=dict(validation_collision_count=0, validation_success_fraction=.95),
                primary='paired task success and forbidden-arc avoidance versus direct target PID on natural test distribution',
                primary_inference=dict(metric='paired task success difference, separately for each of three formal seeds',
                                       bootstrap_resamples=20000, bootstrap_seed=2026101701499,
                                       family_alpha=.05, simultaneous_intervals=3,
                                       conclusion='all three Bonferroni-adjusted interval lower bounds >0'),
                reference='analytic safe route + exactly the same fixed PID; no assumed learned superiority',
                cost_comparison='secondary; do not interpret low collision-terminated costs as a benefit',
                regions=dict(natural='uniform independent allowed endpoints, conditioned on >=15deg separation',
                             detour='same population conditioned on absolute canonical endpoint difference >180deg',
                             edge='same population conditioned on either endpoint within 5deg of guarded sampling boundary'))


def design():
    settings = configuration()
    destination = STUDY/'01_方案与场景'/'执行配置.json'
    if destination.exists() and read_json(destination) != settings:
        raise RuntimeError('route study execution declaration changed')
    write_json(destination, settings)
    # Training population is intentionally generated only by prepare_database.
    for split in ('development', 'validation', 'test', 'detour', 'edge'):
        population = sample(np.random.default_rng(SPLIT_SEEDS[split]), COUNTS[split], split if split in ('detour', 'edge') else 'natural')
        records = [dict(scenario_id=f'{split}_{i:05d}', split=split,
                        theta_rad=float(row[0]), target_rad=float(row[1]), omega_rad_s=0.)
                   for i, row in enumerate(population)]
        path = destination.parent/f'{split}.json'
        if path.exists() and read_json(path) != records:
            raise RuntimeError('scenario split changed')
        write_json(path, records)
    return settings
