"""Fresh scenario splits and declared budgets for corrected continuous control."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
PACKAGE = STUDY.parent
DT = .05
DURATION = 22.
DEADLINE = 8.
LIMIT = .5
INERTIA = 1.
HISTORY = 12
ALPHA = .5
ANGLE_TOL = 1.
RATE_TOL = .5
FAIL_ANGLE = 170.
FAIL_RATE = 300.
KINDS = ('lag', 'delay')
METHODS = ('direct', 'residual')
ALGORITHMS = ('SAC', 'TD3', 'PPO')
N_ENVS = 8
GAMMA = .999
SANITY_STEPS = 98304
DEVELOPMENT_STEPS = 196608
FORMAL_STEPS = 393216
VALIDATE_EVERY = 24576
FORMAL_SEEDS = (111, 222, 333)
DEVELOPMENT_SEEDS = {'A': 15101, 'B': 15202}
SANITY_SEEDS = {'SAC': 15011, 'TD3': 15012, 'PPO': 15013}
COUNTS = {'development': 48, 'validation': 64, 'test': 128}
SPLIT_SEEDS = {'development': 202610181501, 'validation': 202610181502, 'test': 202610181503}
IN_DOMAIN = tuple(round(.05*k, 2) for k in range(1, 7))
OUT_DOMAIN = {'lag': (.40, .50), 'delay': (.35, .40, .50)}
TRAINING = {
    'algorithms': ALGORITHMS,
    'rewards': ('A', 'B'),
    'sanity_steps': SANITY_STEPS,
    'development_steps': DEVELOPMENT_STEPS,
    'formal_steps': FORMAL_STEPS,
    'validate_every': VALIDATE_EVERY,
    'n_envs': N_ENVS,
    'formal_seeds': FORMAL_SEEDS,
    'sanity_kind': 'ideal',
    'primary_mode': 'residual',
    'development_gate_success_rate': .95,
    'development_gate_physical_failures': 0,
    'reward_B_condition': 'all three algorithms complete A; no A model has zero physical failures, >=95% validation success AND lower mean actual cost than selected PID',
    'formal_selection': 'eligible means >=95% passes, zero physical failures, selected checkpoint steps > 0; most passes then lowest penalized selection cost then earliest checkpoint then fixed algorithm order as exact tie break; step-zero PID fallback is not a learned success',
    'primary_cost_requirement': 'PID and every formal seed pass every in-domain test; all corrected paired cost confidence intervals below zero',
    'primary_success_requirement': 'every seed paired success improvement interval above zero; no additional physical failures',
    'family_alpha': .05,
    'intervals_upper_bound': 12,
    'emphasized_cost_reduction': .05,
    'held_out_tests_used_to_choose': False,
    'observed_variables': ['measured error', 'measured angular velocity', 'past sent commands', 'own PID state', 'elapsed time'],
    'hidden_variables': ['actual torque', 'true lag/delay parameter', 'simulator command queue'],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_hashes():
    return {path.relative_to(STUDY).as_posix(): sha256(path)
            for path in sorted((STUDY/'代码'/'continuous_study').glob('*.py'))}


def initial_states(split):
    return read_json(STUDY/'01_实验方案'/f'{split}_initial_states.json')


def scenes(kind, split, region='in_domain'):
    if kind == 'ideal':
        # Sanity validation only: distinct seed and narrower known-feasible starts.
        if split != 'validation':
            raise ValueError('ideal scenarios are development validation, not held-out research tests')
        rng = np.random.default_rng(202610181504)
        return [dict(kind='ideal', split='validation', region='ideal', parameter_s=0., sample_id=i,
                     scenario_id=f'ideal_validation_{i:03d}', angle_deg=float(a), omega_deg_s=float(w))
                for i, (a, w) in enumerate(zip(rng.uniform(-20., 20., 64), rng.uniform(-2., 2., 64)))]
    if kind not in KINDS:
        raise ValueError(kind)
    parameters = (IN_DOMAIN if region == 'in_domain' else OUT_DOMAIN[kind] if region == 'out_of_domain'
                  else (0.,) if region == 'ideal' else None)
    if parameters is None:
        raise ValueError(region)
    return [dict(kind=kind, split=split, region=region, parameter_s=value,
                 scenario_id=f'{kind}_{split}_{value:.2f}_{state["sample_id"]:03d}', **state)
            for value in parameters for state in initial_states(split)]


def training_parameters(kind, rng, n):
    if kind == 'ideal':
        return np.zeros(n)
    if kind == 'lag':
        return rng.uniform(.05, .30, size=n)
    if kind == 'delay':
        return rng.integers(1, 7, size=n)*DT
    raise ValueError(kind)


def training_initial(rng, n, sanity=False):
    return (rng.uniform(-20. if sanity else -60., 20. if sanity else 60., size=n),
            rng.uniform(-2. if sanity else -5., 2. if sanity else 5., size=n))


def design():
    for split, seed in SPLIT_SEEDS.items():
        rng = np.random.default_rng(seed)
        data = [dict(sample_id=i, angle_deg=float(a), omega_deg_s=float(w)) for i, (a, w) in
                enumerate(zip(rng.uniform(-60., 60., COUNTS[split]), rng.uniform(-5., 5., COUNTS[split])))]
        path = STUDY/'01_实验方案'/f'{split}_initial_states.json'
        if path.exists() and read_json(path) != data:
            raise RuntimeError('scenario split changed')
        if not path.exists():
            write_json(path, data)
    config = json.loads(json.dumps(dict(dt=DT, duration=DURATION, deadline=DEADLINE,
        torque_limit=LIMIT, inertia=INERTIA, angle_tolerance=ANGLE_TOL, rate_tolerance=RATE_TOL,
        history_intervals=HISTORY, in_domain=IN_DOMAIN, out_of_domain=OUT_DOMAIN,
        counts=COUNTS, split_seeds=SPLIT_SEEDS, training=TRAINING,
        physical_failure='absorbing frozen state; clock and reward continue until the complete 440-step horizon',
        deployment_weights_frozen=True, online_weight_updates=False,
        primary_comparisons='one selected algorithm per actuator; 3 fresh formal seeds; 12 intervals reserved for success/cost across two kinds',
        previous_studies_read_only=True)))
    path = STUDY/'01_实验方案'/'执行配置.json'
    if path.exists() and read_json(path) != config:
        raise RuntimeError('execution configuration changed')
    if not path.exists():
        write_json(path, config)
    return config
