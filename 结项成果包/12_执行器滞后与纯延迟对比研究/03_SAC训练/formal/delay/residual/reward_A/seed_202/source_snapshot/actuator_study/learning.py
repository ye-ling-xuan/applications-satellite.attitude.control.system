"""Predeclared SAC experiments, with one shared training/deployment interface.

Only measured [-theta, omega], sent total commands and the controller's own
state enter policy features. Simulator parameters/actual torque are used only
for environment sampling, reward and audit logs. No deployment calibration is
applied. All arrays consumed by the policy/replay buffer are float32.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from . import protocol as p


# These are the ONLY two development rewards, declared before any experiment.
# State terms are endpoint sampled, while the torque term uses the interval's
# exact physical integral. Excess ratios are max(abs(state)/tolerance - 1, 0).
# r = -DT * [we*min((theta/5deg)^2,100)
#           +ww*min((omega/5deg/s)^2,100)
#           +wu*integral(tau_actual^2)/(DT*LIMIT^2)
#           +1[t_end>=8]*wv*(1[outside] + min(excess_e^2,25)
#                                             + min(excess_w^2,25))].
# A newly failing transition instead receives -1000. True finite termination
# applies at failure and at 22 s: neither terminal is time-limit bootstrapped.
REWARDS = {
    'A': dict(error_weight=1., rate_weight=.5, torque_weight=.10, violation_weight=8.),
    'B': dict(error_weight=1., rate_weight=.5, torque_weight=.02, violation_weight=12.),
}
ERROR_SCALE = np.pi/3
RATE_SCALE = np.deg2rad(30.)
INTEGRAL_SCALE = 2.
REWARD_STATE_SCALE = np.deg2rad(5.)
SAC_SETTINGS = dict(
    policy='MlpPolicy', net_arch=[128, 128], learning_rate=3e-4,
    batch_size=256, gamma=.995, tau=.005, buffer_size=300000,
    ent_coef='auto_0.01', target_entropy=-1., learning_starts=10000,
    train_freq_vector_steps=1, gradient_steps=1, n_envs=p.N_ENVS,
    feature_dtype='float32', device='cpu', torch_threads=1,
)


def configuration():
    """JSON-compatible frozen declaration, independent of run results."""
    return dict(
        rewards=REWARDS, sac=SAC_SETTINGS,
        reward_formula=__doc_reward(),
        feature_order=['31 chronological measured error/rate pairs',
                       '30 chronological past sent clipped total commands',
                       'elapsed/22', 'valid_intervals/30',
                       'residual only: proposed raw PID / 0.5',
                       'residual only: committed PID integral / 2'],
        feature_scales=dict(error_rad=float(ERROR_SCALE), rate_rad_s=float(RATE_SCALE),
                            command_Nm=p.LIMIT, integral_rad_s=INTEGRAL_SCALE),
        feature_dimensions=dict(direct=94, residual=96),
        history_intervals=p.HISTORY, prehistory_measurements='repeat initial measurement',
        prehistory_commands=0., prehistory_valid=0,
        action_direct='0.5*clip(action,-1,1)',
        action_residual='clip(raw_PID+0.5*0.5*clip(action,-1,1),-0.5,0.5)',
        anti_windup='PID commit uses the final clipped total command',
        training_random_stream='SeedSequence([seed, kind_id, method_id, rank]); lag=0,delay=1; direct=0,residual=1',
        terminal='physical failure or t=22; TimeLimit.truncated=False; complete terminal_observation',
        selection='maximize validation passes, minimize mean actual cost, earliest checkpoint',
        development_steps=p.DEVELOPMENT_STEPS, formal_steps=p.FORMAL_STEPS,
        validation_every=p.VALIDATE_EVERY, validation_count=576,
        development_seeds=dict(p.DEVELOPMENT_SEEDS), formal_seeds=list(p.FORMAL_SEEDS),
        development_gate=dict(success_rate=.95, physical_failures=0),
        hidden_from_features=['true parameter', 'actual torque', 'simulator delay queue'],
        test_response_calibration=False,
        replay_buffer_persisted=False,
    )


def __doc_reward():
    return dict(state_scale_deg=5., rate_scale_deg_s=5., state_square_cap=100.,
                excess_square_cap=25., physical_failure_reward=-1000.,
                interval_reward='-dt*(we*min((theta/5deg)^2,100)+ww*min((omega/5deg/s)^2,100)+wu*actual_cost/(dt*limit^2)+1[t_end>=8]*wv*(outside+min(excess_angle^2,25)+min(excess_rate^2,25)))',
                excess='max(abs(state)/acceptance_tolerance-1,0)',
                deadline_sample_included=True)


def predeclare():
    """Persist both rewards/settings; an existing declaration cannot change."""
    settings = json.loads(json.dumps(configuration()))
    path = p.STUDY/'03_SAC训练'/'predeclared_learning.json'
    if path.exists():
        if p.read_json(path) != settings:
            raise RuntimeError('predeclared SAC configuration changed')
    else:
        p.write_json(path, settings)
    return settings


def reward_values(observation, elapsed, actual_cost, new_failure, reward_name):
    """Bounded ordinary task reward; hidden actual torque never enters features."""
    weights = REWARDS[reward_name]
    obs = np.asarray(observation, dtype=float)
    angle = np.abs(obs[:, 0])
    rate = np.abs(obs[:, 1])
    angle_ratio = angle/np.deg2rad(p.ANGLE_TOL)
    rate_ratio = rate/np.deg2rad(p.RATE_TOL)
    outside = (angle_ratio > 1.) | (rate_ratio > 1.)
    violation = (outside.astype(float)
                 + np.minimum(np.maximum(angle_ratio-1., 0.)**2, 25.)
                 + np.minimum(np.maximum(rate_ratio-1., 0.)**2, 25.))
    cost = (weights['error_weight']*np.minimum((angle/REWARD_STATE_SCALE)**2, 100.)
            + weights['rate_weight']*np.minimum((rate/REWARD_STATE_SCALE)**2, 100.)
            + weights['torque_weight']*np.asarray(actual_cost)/(p.DT*p.LIMIT**2)
            + (np.asarray(elapsed) >= p.DEADLINE-1e-10)*weights['violation_weight']*violation)
    reward = -p.DT*cost
    reward[np.asarray(new_failure, dtype=bool)] = -1000.
    return reward.astype(np.float32)


class FeatureHistory:
    """Chronological measured history, aligned with past sent total commands.

    ``observe`` is idempotent until ``record_command``. The next observation
    shifts exactly one measurement; recording a command shifts exactly one
    command. Thus repeated policy introspection cannot lengthen the history.
    """

    def __init__(self, method):
        if method not in p.METHODS:
            raise ValueError(method)
        self.method = method
        self.n = 0

    @property
    def dimension(self):
        return 94 if self.method == 'direct' else 96

    def reset(self, n, initial_observation):
        obs = np.asarray(initial_observation, dtype=float)
        if obs.shape != (n, 2) or not np.all(np.isfinite(obs)):
            raise ValueError('initial observation must be finite (n,2)')
        self.n = n
        self.measurements = np.repeat(obs[:, None, :], p.HISTORY+1, axis=1)
        self.commands = np.zeros((n, p.HISTORY))
        self.valid = np.zeros(n, dtype=np.int64)
        self.pending = np.zeros(n, dtype=bool)

    def reset_rows(self, mask, observation):
        mask = np.asarray(mask, dtype=bool)
        obs = np.asarray(observation, dtype=float)
        if mask.shape != (self.n,) or obs.shape != (self.n, 2):
            raise ValueError('invalid partial reset shapes')
        self.measurements[mask] = obs[mask, None, :]
        self.commands[mask] = 0.
        self.valid[mask] = 0
        self.pending[mask] = False

    def observe(self, observation):
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (self.n, 2) or not np.all(np.isfinite(obs)):
            raise ValueError('observation must be finite (n,2)')
        rows = np.flatnonzero(self.pending)
        if rows.size:
            self.measurements[rows, :-1] = self.measurements[rows, 1:]
            self.measurements[rows, -1] = obs[rows]
            self.pending[rows] = False
        # Also validate that callers did not change a sampled observation
        # without a corresponding sent command/interval.
        if not np.array_equal(self.measurements[:, -1], obs):
            raise ValueError('measurement changed without an intervening command')

    def record_command(self, command):
        command = np.asarray(command, dtype=float).reshape(-1)
        if command.shape != (self.n,) or not np.all(np.isfinite(command)):
            raise ValueError('command must be finite length n')
        if np.any(np.abs(command) > p.LIMIT+1e-12) or np.any(self.pending):
            raise ValueError('commands must be clipped and followed by a measurement')
        self.commands[:, :-1] = self.commands[:, 1:]
        self.commands[:, -1] = command
        self.valid = np.minimum(self.valid+1, p.HISTORY)
        self.pending[:] = True

    def features(self, elapsed, raw_pid=None, integral=None):
        elapsed = np.broadcast_to(np.asarray(elapsed, dtype=float), (self.n,))
        if not np.all(np.isfinite(elapsed)) or np.any(elapsed < 0.):
            raise ValueError('elapsed must be finite and nonnegative')
        result = np.empty((self.n, self.dimension), dtype=np.float32)
        scaled = self.measurements/np.array([ERROR_SCALE, RATE_SCALE])
        result[:, :62] = scaled.reshape(self.n, 62)
        result[:, 62:92] = self.commands/p.LIMIT
        result[:, 92] = elapsed/p.DURATION
        result[:, 93] = self.valid/p.HISTORY
        if self.method == 'residual':
            if raw_pid is None or integral is None:
                raise ValueError('residual features require own PID output and integral')
            result[:, 94] = np.asarray(raw_pid)/p.LIMIT
            result[:, 95] = np.asarray(integral)/INTEGRAL_SCALE
        return result


class _PolicyInterface:
    """The identical feature/action/anti-windup path for train and deployment."""

    def __init__(self, method, base_parameters=None):
        self.method = method
        self.base_parameters = base_parameters
        self.history = FeatureHistory(method)
        self.pid = None
        if method == 'residual':
            if base_parameters is None:
                raise ValueError('residual SAC requires fixed base PID parameters')
            from .baselines import PIDController
            self.pid = PIDController(base_parameters)

    def reset(self, n, initial_observation):
        self.history.reset(n, initial_observation)
        if self.pid is not None:
            self.pid.reset(n, initial_observation)

    def reset_rows(self, mask, observation):
        self.history.reset_rows(mask, observation)
        if self.pid is not None:
            self.pid.integral[mask] = 0.

    def features(self, observation, elapsed):
        self.history.observe(observation)
        raw = self.pid.propose(observation) if self.pid is not None else None
        integral = self.pid.integral if self.pid is not None else None
        return self.history.features(elapsed, raw, integral)

    def command(self, observation, actions):
        action = np.asarray(actions, dtype=float).reshape(-1)
        if action.shape != (self.history.n,) or not np.all(np.isfinite(action)):
            raise ValueError('actions must be finite length n')
        action = np.clip(action, -1., 1.)
        self.history.observe(observation)
        if self.pid is None:
            command = p.LIMIT*action
        else:
            raw_total = self.pid.propose(observation)+p.ALPHA*p.LIMIT*action
            command = np.clip(raw_total, -p.LIMIT, p.LIMIT)
            # Saturation belongs to the final total command. Comparing the
            # base PID alone with a residual-shifted command would falsely
            # suppress integration even when no total-command clipping occurs.
            self.pid.commit(observation, command, raw=raw_total)
        self.history.record_command(command)
        return command


class SACController:
    """Deterministic deployed SAC, without hidden state or output calibration."""

    def __init__(self, model, method, base_parameters=None):
        self.model = model
        self.interface = _PolicyInterface(method, base_parameters)
        self.method = method

    def reset(self, n, initial_observation):
        self.interface.reset(n, initial_observation)

    def features(self, observation, elapsed):
        return self.interface.features(observation, elapsed)

    def act(self, observation, elapsed):
        features = self.features(observation, elapsed)
        actions, _ = self.model.predict(features, deterministic=True)
        return self.interface.command(observation, actions)


# VecEnv/SB3 imports are intentionally delayed until this class is requested,
# while the module and predeclaration stay usable without a training runtime.
def _make_vec_env_class():
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import VecEnv
    from .physics import BatchPlant

    class _SACVecEnv(VecEnv):
        render_mode = None

        def __init__(self, kind, method, reward_name, seed, base_parameters=None,
                     n_envs=p.N_ENVS, log_directory=None):
            if kind not in p.KINDS or method not in p.METHODS or reward_name not in REWARDS:
                raise ValueError('unknown study environment configuration')
            if n_envs != p.N_ENVS:
                raise ValueError('predeclared SAC uses exactly eight vector environments')
            self.kind, self.method, self.reward_name = kind, method, reward_name
            self.run_seed = int(seed)
            self.interface = _PolicyInterface(method, base_parameters)
            self.plant = BatchPlant(kind, n_envs)
            self.rngs = [np.random.default_rng(np.random.SeedSequence(
                [self.run_seed, p.KINDS.index(kind), p.METHODS.index(method), rank]))
                for rank in range(n_envs)]
            self.episode_ids = np.zeros(n_envs, dtype=np.int64)
            self.episode_returns = np.zeros(n_envs)
            self.episode_actual_cost = np.zeros(n_envs)
            self.episode_command_cost = np.zeros(n_envs)
            self.total_steps = 0
            self.actions = None
            self.closed = False
            self._initial_log = self._episode_log = None
            if log_directory is not None:
                directory = Path(log_directory)
                directory.mkdir(parents=True, exist_ok=True)
                self._initial_log = (directory/'training_episode_initials.jsonl').open('x', encoding='utf-8')
                self._episode_log = (directory/'training_episodes.jsonl').open('x', encoding='utf-8')
            super().__init__(n_envs, Box(-np.inf, np.inf, (self.interface.history.dimension,), dtype=np.float32),
                             Box(-1., 1., (1,), dtype=np.float32))

        def seed(self, seed=None):
            # SB3 seeds the VecEnv during construction. The declared streams
            # already include kind, method and rank and are never shared.
            if seed is not None and int(seed) != self.run_seed:
                raise ValueError('training VecEnv seed is fixed at construction')
            return [int(np.random.SeedSequence([self.run_seed, p.KINDS.index(self.kind),
                         p.METHODS.index(self.method), rank]).generate_state(1)[0])
                    for rank in range(self.num_envs)]

        def _reset_rows(self, mask, first=False):
            theta, omega, parameter = [], [], []
            for rank in np.flatnonzero(mask):
                rng = self.rngs[rank]
                angle = float(rng.uniform(-60., 60.))
                rate = float(rng.uniform(-5., 5.))
                value = float(p.training_parameters(self.kind, rng, 1)[0])
                theta.append(np.deg2rad(angle))
                omega.append(np.deg2rad(rate))
                parameter.append(value)
                self.reset_infos[rank] = dict(rank=int(rank), episode=int(self.episode_ids[rank]),
                                             seed=self.run_seed, kind=self.kind, method=self.method,
                                             initial_angle_deg=angle, initial_omega_deg_s=rate,
                                             parameter_s=value, total_training_steps=self.total_steps)
                if self._initial_log is not None:
                    self._initial_log.write(json.dumps(self.reset_infos[rank], allow_nan=False)+'\n')
                self.episode_ids[rank] += 1
            self.plant.reset(np.array(theta), np.array(omega), np.array(parameter), mask=mask)
            obs = self.plant.observation()
            if first:
                self.interface.reset(self.num_envs, obs)
            else:
                self.interface.reset_rows(mask, obs)
            self.episode_returns[mask] = 0.
            self.episode_actual_cost[mask] = 0.
            self.episode_command_cost[mask] = 0.
            if self._initial_log is not None:
                self._initial_log.flush()

        def reset(self):
            self._reset_rows(np.ones(self.num_envs, dtype=bool), first=True)
            self._reset_seeds()
            self._reset_options()
            return self.interface.features(self.plant.observation(), self.plant.elapsed)

        def step_async(self, actions):
            if self.actions is not None:
                raise RuntimeError('step_async called twice')
            self.actions = np.asarray(actions).copy()

        def step_wait(self):
            if self.actions is None:
                raise RuntimeError('step_wait requires step_async')
            command = self.interface.command(self.plant.observation(), self.actions)
            self.actions = None
            step = self.plant.step(command)
            self.total_steps += self.num_envs
            measured = self.plant.observation()
            reward = reward_values(measured, self.plant.elapsed, step['actual_cost'],
                                   step['new_failure'], self.reward_name)
            self.episode_returns += reward
            self.episode_actual_cost += step['actual_cost']
            self.episode_command_cost += step['command_cost']
            features = self.interface.features(measured, self.plant.elapsed)
            dones = step['done']
            infos = []
            for rank in range(self.num_envs):
                info = {'TimeLimit.truncated': False}
                if dones[rank]:
                    episode = dict(r=float(self.episode_returns[rank]), l=int(self.plant.step_count[rank]),
                                   actual_cost=float(self.episode_actual_cost[rank]),
                                   command_cost=float(self.episode_command_cost[rank]),
                                   failed=bool(self.plant.failed[rank]))
                    info.update(terminal_observation=features[rank].copy(), episode=episode,
                                termination_reason='physical_failure' if self.plant.failed[rank] else 'finite_horizon_22s')
                    if self._episode_log is not None:
                        self._episode_log.write(json.dumps(dict(self.reset_infos[rank], **episode), allow_nan=False)+'\n')
                infos.append(info)
            if np.any(dones):
                self._reset_rows(dones)
                features = self.interface.features(self.plant.observation(), self.plant.elapsed)
                if self._episode_log is not None:
                    self._episode_log.flush()
            return features, reward, dones, infos

        def close(self):
            if not self.closed:
                for stream in (self._initial_log, self._episode_log):
                    if stream is not None:
                        stream.close()
                self.closed = True

        def get_attr(self, attr_name, indices=None):
            selected = self._get_indices(indices)
            value = getattr(self, attr_name)
            return [value for _ in selected]

        def set_attr(self, attr_name, value, indices=None):
            raise AttributeError('study environment settings are fixed')

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            raise AttributeError('no per-environment public mutation methods')

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return _SACVecEnv


_VEC_ENV_CLASS = None


def SACVecEnv(*args, **kwargs):
    """Construct the eight-environment SB3 vector interface lazily."""
    global _VEC_ENV_CLASS
    if _VEC_ENV_CLASS is None:
        _VEC_ENV_CLASS = _make_vec_env_class()
    return _VEC_ENV_CLASS(*args, **kwargs)


def _relative(path):
    return Path(path).relative_to(p.STUDY).as_posix()


def _rank(candidate):
    stats = candidate['validation']
    return (stats['passes'], -stats['mean_actual_cost'], -candidate['steps'])


def _snapshot(directory):
    for source in sorted(Path(__file__).resolve().parents[1].rglob('*.py')):
        if '__pycache__' in source.parts:
            continue
        destination = directory/source.relative_to(Path(__file__).resolve().parents[1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_directory(kind, method, reward_name, seed, stage):
    return p.STUDY/'03_SAC训练'/stage/kind/method/f'reward_{reward_name}'/f'seed_{seed}'


def _load_base(kind):
    return p.read_json(p.STUDY/'02_传统控制'/f'selected_{kind}.json')['pid']['parameters']


def train_run(kind, method, reward_name, seed, steps, stage, base_parameters=None):
    """Train exactly the declared aggregate step count and retain every attempt.

    Research execution is orchestrated by the root CLI after physics/baseline
    checks and predeclaration. No implicit resume, overwrite, or extra seed is
    permitted. Checkpoints are evaluated after their optimizer update at each
    exact 50,000-step boundary; the same validation data select best.zip.
    """
    if (p.STUDY/'04_独立测试'/'freeze.json').exists():
        raise RuntimeError('held-out response freeze exists; further research training is prohibited')
    if kind not in p.KINDS or method not in p.METHODS or reward_name not in REWARDS:
        raise ValueError('unknown training configuration')
    if stage not in ('development', 'formal'):
        raise ValueError(stage)
    if steps != (p.DEVELOPMENT_STEPS if stage == 'development' else p.FORMAL_STEPS):
        raise ValueError('research step budget must equal the predeclared stage budget')
    if steps % p.N_ENVS or p.VALIDATE_EVERY % p.N_ENVS:
        raise ValueError('step counts must be exact multiples of eight')
    if (stage == 'development' and seed != p.DEVELOPMENT_SEEDS[reward_name]) or (stage == 'formal' and seed not in p.FORMAL_SEEDS):
        raise ValueError('seed is outside the predeclared stage seeds')
    declaration = p.STUDY/'03_SAC训练'/'predeclared_learning.json'
    if not declaration.exists() or p.read_json(declaration) != json.loads(json.dumps(configuration())):
        raise RuntimeError('save and freeze predeclared SAC settings before research training')
    selected_base = _load_base(kind)
    if base_parameters is None:
        base_parameters = selected_base
    if base_parameters != selected_base:
        raise ValueError('base PID must be the independently selected PID')
    if stage == 'formal':
        decision = p.read_json(p.STUDY/'03_SAC训练'/'development_decisions'/f'{kind}_{method}.json')
        if not decision['passed_gate'] or decision['reward_name'] != reward_name or decision['base_parameters'] != base_parameters:
            raise RuntimeError('formal training configuration has not passed its development gate')

    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure
    import stable_baselines3
    import gymnasium
    import torch
    from .evaluation import evaluate, summary, write_csv

    directory = _run_directory(kind, method, reward_name, seed, stage)
    directory.mkdir(parents=True, exist_ok=False)
    source_before = p.source_hashes()
    _snapshot(directory/'source_snapshot')
    metadata = dict(status='running', kind=kind, method=method, reward_name=reward_name,
                    seed=int(seed), stage=stage, requested_steps=int(steps), actual_steps=0,
                    best=None, checkpoints=[], base_parameters=base_parameters,
                    source_sha256=source_before, source_sha256_after=None,
                    configuration=configuration(), run_path=_relative(directory),
                    declaration_sha256=p.sha256(declaration),
                    training_elapsed_seconds=0., validation_elapsed_seconds=0.,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    versions=dict(numpy=np.__version__, torch=torch.__version__,
                                  stable_baselines3=stable_baselines3.__version__, gymnasium=gymnasium.__version__),
                    replay_buffer_saved=False, test_responses_used=False)
    metadata_path = directory/'training_metadata.json'
    p.write_json(metadata_path, metadata)
    env = model = optimizer_log = None
    start = time.monotonic()
    try:
        torch.set_num_threads(SAC_SETTINGS['torch_threads'])
        torch.use_deterministic_algorithms(True)
        env = SACVecEnv(kind, method, reward_name, seed, base_parameters, log_directory=directory)
        optimizer_log = (directory/'optimizer_steps.jsonl').open('x', encoding='utf-8')

        class LoggedSAC(SAC):
            def train(self, gradient_steps, batch_size=64):
                super().train(gradient_steps, batch_size)
                metrics = {key: float(value) for key, value in self.logger.name_to_value.items()
                           if key.startswith('train/') and isinstance(value, (float, int, np.number))}
                metrics.update(timesteps=int(self.num_timesteps), optimizer_updates=int(self._n_updates))
                optimizer_log.write(json.dumps(metrics, allow_nan=False)+'\n')

        model = LoggedSAC(
            'MlpPolicy', env, seed=int(seed), device='cpu', verbose=0,
            policy_kwargs=dict(net_arch=SAC_SETTINGS['net_arch']),
            learning_rate=SAC_SETTINGS['learning_rate'], batch_size=SAC_SETTINGS['batch_size'],
            gamma=SAC_SETTINGS['gamma'], tau=SAC_SETTINGS['tau'],
            buffer_size=min(SAC_SETTINGS['buffer_size'], steps),
            ent_coef=SAC_SETTINGS['ent_coef'], target_entropy=SAC_SETTINGS['target_entropy'],
            learning_starts=SAC_SETTINGS['learning_starts'], train_freq=(1, 'step'),
            gradient_steps=SAC_SETTINGS['gradient_steps'],
        )
        model.set_logger(configure(str(directory/'sb3_logs'), ['csv']))
        validation_scenes = p.scenes(kind, 'validation')
        if len(validation_scenes) != 576:
            raise RuntimeError('validation set must contain all 576 predefined scenes')
        for checkpoint_steps in range(p.VALIDATE_EVERY, steps+1, p.VALIDATE_EVERY):
            train_started = time.monotonic()
            model.learn(p.VALIDATE_EVERY, reset_num_timesteps=(checkpoint_steps == p.VALIDATE_EVERY), log_interval=1)
            metadata['training_elapsed_seconds'] += time.monotonic()-train_started
            if model.num_timesteps != checkpoint_steps or env.total_steps != checkpoint_steps:
                raise RuntimeError('aggregate environment-step count differs from the declared budget')
            expected_updates = max(0, (checkpoint_steps-SAC_SETTINGS['learning_starts'])//p.N_ENVS)
            if model._n_updates != expected_updates:
                raise RuntimeError('optimizer/environment update ratio changed')
            optimizer_log.flush()
            checkpoint_dir = directory/'checkpoints'/f'steps_{checkpoint_steps:06d}'
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            model_path = checkpoint_dir/'model.zip'
            model.save(model_path)
            validation_started = time.monotonic()
            validation = evaluate(kind, validation_scenes, SACController(model, method, base_parameters))
            metadata['validation_elapsed_seconds'] += time.monotonic()-validation_started
            stats = summary(validation['rows'])
            write_csv(checkpoint_dir/'validation.csv', validation['rows'])
            checkpoint = dict(model_path=_relative(model_path), model_sha256=p.sha256(model_path),
                              steps=int(model.num_timesteps), validation=stats,
                              optimizer_updates=int(model._n_updates))
            p.write_json(checkpoint_dir/'checkpoint_metadata.json', dict(checkpoint, source_sha256=source_before,
                                                                        base_parameters=base_parameters,
                                                                        reward_name=reward_name, seed=int(seed)))
            metadata['checkpoints'].append(checkpoint)
            if metadata['best'] is None or _rank(checkpoint) > _rank(metadata['best']):
                shutil.copy2(model_path, directory/'best.zip')
                metadata['best'] = dict(checkpoint)
                metadata['best']['checkpoint_model_path'] = checkpoint['model_path']
                metadata['best']['model_path'] = _relative(directory/'best.zip')
            metadata['actual_steps'] = int(model.num_timesteps)
            metadata['optimizer_updates'] = int(model._n_updates)
            metadata['training_steps_per_second'] = model.num_timesteps/metadata['training_elapsed_seconds']
            metadata['elapsed_seconds'] = time.monotonic()-start
            metadata['source_sha256_after'] = p.source_hashes()
            if metadata['source_sha256_after'] != source_before:
                raise RuntimeError('study source changed during training')
            p.write_json(metadata_path, metadata)
            print(json.dumps(dict(event='SAC checkpoint', kind=kind, method=method, stage=stage,
                                  reward=reward_name, seed=seed, steps=checkpoint_steps,
                                  validation=stats, elapsed_seconds=metadata['elapsed_seconds']), ensure_ascii=False), flush=True)
        final_path = directory/'final.zip'
        model.save(final_path)
        metadata['final'] = dict(model_path=_relative(final_path), model_sha256=p.sha256(final_path), steps=int(model.num_timesteps))
        if p.source_hashes() != source_before:
            raise RuntimeError('study source changed before run completion')
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'incomplete'
        metadata['error'] = repr(error)
        metadata['traceback'] = traceback.format_exc()
        if model is not None:
            metadata['actual_steps'] = int(model.num_timesteps)
            model.save(directory/'incomplete_model.zip')
        raise
    finally:
        metadata['source_sha256_after'] = p.source_hashes()
        metadata['elapsed_seconds'] = time.monotonic()-start
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        p.write_json(metadata_path, metadata)
        if optimizer_log is not None:
            optimizer_log.close()
        if model is not None:
            model.logger.close()
        if env is not None:
            env.close()
    return metadata


def decide_development(kind, method):
    """Select only complete declared attempts, retaining all attempted outcomes."""
    attempts = []
    candidates = []
    for reward_name, seed in p.DEVELOPMENT_SEEDS.items():
        path = _run_directory(kind, method, reward_name, seed, 'development')/'training_metadata.json'
        if path.exists():
            run = p.read_json(path)
            attempts.append(dict(metadata_path=_relative(path), status=run['status'],
                                 reward_name=reward_name, actual_steps=run['actual_steps'], best=run['best']))
            if run['status'] == 'complete' and run['actual_steps'] == p.DEVELOPMENT_STEPS and run['best'] is not None:
                candidates.append(run)
    if not attempts:
        raise RuntimeError('no recorded development attempts')
    selected = max(candidates, key=lambda run: _rank(run['best'])) if candidates else None
    stats = selected['best']['validation'] if selected else None
    passed = bool(stats and stats['passes']/stats['count'] >= .95 and stats['physical_failures'] == 0)
    decision = dict(kind=kind, method=method, passed_gate=passed,
                    reward_name=selected['reward_name'] if selected else None,
                    base_parameters=selected['base_parameters'] if selected else _load_base(kind),
                    best=selected['best'] if selected else None, validation=stats,
                    attempts=attempts, test_responses_used=False,
                    reason='development gate passed' if passed else 'development did not succeed within this budget; no formal training',
                    source_sha256=p.source_hashes())
    path = p.STUDY/'03_SAC训练'/'development_decisions'/f'{kind}_{method}.json'
    if path.exists() and p.read_json(path) != decision:
        raise RuntimeError('development decision already frozen; attempts are never overwritten')
    p.write_json(path, decision)
    return decision
