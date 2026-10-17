"""Predeclared conditions and independent scenario splits."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
PACKAGE = STUDY.parent
DT = .05
DURATION = 22.
DEADLINE = 8.
LIMIT = .5
INERTIA = 1.
HISTORY = 30
ALPHA = .5
ANGLE_TOL = 1.
RATE_TOL = .5
FAIL_ANGLE = 170.
FAIL_RATE = 300.
KINDS = ('lag', 'delay')
METHODS = ('direct', 'residual')
SPLIT_SEEDS = {'development': 202610170101, 'validation': 202610170102, 'test': 202610170103}
COUNTS = {'development': 32, 'validation': 96, 'test': 256}
IN_DOMAIN = tuple(round(.05*k, 2) for k in range(1, 7))
OUT_DOMAIN = {'lag': (.40, .50), 'delay': (.35, .40, .50)}
FORMAL_SEEDS = (101, 202, 303)
DEVELOPMENT_SEEDS = {'A': 9101, 'B': 9102}
DEVELOPMENT_STEPS = 150000
FORMAL_STEPS = 500000
VALIDATE_EVERY = 50000
N_ENVS = 8


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05*(attempt+1))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def initial_states(split):
    return read_json(STUDY/'01_实验方案'/f'{split}_initial_states.json')


def scenes(kind, split, region='in_domain'):
    if kind not in KINDS:
        raise ValueError(kind)
    parameters = IN_DOMAIN if region == 'in_domain' else OUT_DOMAIN[kind] if region == 'out_of_domain' else (0.,) if region == 'ideal' else None
    if parameters is None:
        raise ValueError(region)
    return [dict(kind=kind, split=split, region=region, parameter_s=p,
                 scenario_id=f'{kind}_{split}_{p:.2f}_{state["sample_id"]:03d}', **state)
            for p in parameters for state in initial_states(split)]


def training_parameters(kind, rng, n):
    return rng.uniform(.05, .30, size=n) if kind == 'lag' else rng.integers(1, 7, size=n)*DT


def source_hashes():
    return {p.relative_to(STUDY).as_posix(): sha256(p)
            for p in sorted((STUDY/'代码').rglob('*.py')) if '__pycache__' not in p.parts}


def design():
    directory = STUDY/'01_实验方案'
    for split, seed in SPLIT_SEEDS.items():
        path = directory/f'{split}_initial_states.json'
        rng = np.random.default_rng(seed)
        angles = rng.uniform(-60., 60., COUNTS[split])
        rates = rng.uniform(-5., 5., COUNTS[split])
        data = [dict(sample_id=i, angle_deg=float(a), omega_deg_s=float(w)) for i, (a, w) in enumerate(zip(angles, rates))]
        if path.exists():
            if read_json(path) != data:
                raise RuntimeError('scenario split changed: '+str(path))
        else:
            write_json(path, data)
    settings = dict(dt_s=DT, duration_s=DURATION, deadline_s=DEADLINE, inertia_kg_m2=INERTIA,
                    command_limit_Nm=LIMIT, angle_tolerance_deg=ANGLE_TOL, rate_tolerance_deg_s=RATE_TOL,
                    physical_failure_angle_deg=FAIL_ANGLE, physical_failure_rate_deg_s=FAIL_RATE,
                    history_intervals=HISTORY, residual_alpha=ALPHA, in_domain_grid_s=IN_DOMAIN,
                    out_of_domain_grid_s=OUT_DOMAIN, lag_training='uniform continuous [0.05,0.30] s',
                    delay_training='uniform discrete {1,...,6}*0.05 s', parameter_constant_within_episode=True,
                    hidden_deployment_quantities=['true parameter', 'actual torque', 'physical input queue'],
                    initial_actual_torque_Nm=0., prehistory_commands_Nm=0.,
                    formal_seeds=FORMAL_SEEDS, development_seeds=DEVELOPMENT_SEEDS,
                    development_steps_per_run=DEVELOPMENT_STEPS, formal_steps_per_run=FORMAL_STEPS,
                    validation_every_steps=VALIDATE_EVERY, development_gate_success_rate=.95,
                    development_gate_physical_failures=0, primary_comparisons=4, correction_intervals=12,
                    family_alpha=.05, emphasized_cost_reduction_fraction=.05,
                    independent_test_initial_states=256, continuous_time_stability_claim=False,
                    training_budget_steps_upper_bound=7200000)
    path = directory/'执行配置.json'
    # JSON roundtrip makes tuple/list comparison explicit and reproducible.
    settings = json.loads(json.dumps(settings))
    if path.exists():
        if read_json(path) != settings:
            raise RuntimeError('predeclared execution settings changed')
    else:
        write_json(path, settings)
    return settings
