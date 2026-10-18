"""Corrected, shared continuous-control interface for SAC, TD3 and PPO.

Policy inputs contain measured error/rate histories and previously issued
commands only.  Physical failures are absorbing states, never early episode
ends.  Every training and deployment episode has the complete 440 samples.
The same observation and action code is used by all three algorithms.
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

ERROR_SCALE = np.deg2rad(60.)
RATE_SCALE = np.deg2rad(30.)
INTEGRAL_SCALE = 2.
REWARDS = {
    'A': dict(torque_weight=1., error_weight=.01, rate_weight=.002, violation_weight=5.),
    'B': dict(torque_weight=1., error_weight=.03, rate_weight=.006, violation_weight=10.),
}
COMMON_SETTINGS = dict(net_arch=[128, 128], learning_rate=3e-4, gamma=p.GAMMA,
                       n_envs=p.N_ENVS, torch_threads=1, device='cpu',
                       zero_actor_mean=True, feature_dtype='float32')
ALGORITHM_SETTINGS = {
    'SAC': dict(batch_size=256, tau=.005, buffer_size=200000,
                learning_starts=4096, train_freq_vector_steps=1, gradient_steps=1,
                ent_coef='auto_0.0005', target_entropy=-1., initial_log_std=-3.),
    'TD3': dict(batch_size=256, tau=.005, buffer_size=200000,
                learning_starts=4096, train_freq_vector_steps=1, gradient_steps=1,
                exploration_std=.10, policy_delay=2,
                target_policy_noise=.10, target_noise_clip=.20),
    'PPO': dict(n_steps=128, batch_size=256, n_epochs=10,
                gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5,
                max_grad_norm=.5, initial_log_std=-2.3,
                normalize_advantage=True, target_kl=None),
}


def configuration():
    return dict(
        algorithms=ALGORITHM_SETTINGS, common=COMMON_SETTINGS, rewards=REWARDS,
        protocol_training=p.TRAINING,
        feature_dimensions=dict(direct=3*p.HISTORY+4, residual=3*p.HISTORY+6),
        feature_order=['13 chronological measured error/rate pairs',
                       '12 chronological clipped total sent commands',
                       'elapsed/22', 'valid_intervals/12',
                       'residual only: proposed raw PID/0.5',
                       'residual only: committed PID integral/2'],
        feature_scales=dict(error_rad=float(ERROR_SCALE), rate_rad_s=float(RATE_SCALE),
                            command_Nm=p.LIMIT, integral_rad_s=INTEGRAL_SCALE),
        prehistory='repeat initial measurement; zero commands; zero valid count',
        residual_action='clip(raw_PID+0.25*clip(action,-1,1),-0.5,0.5)',
        direct_action='0.5*clip(action,-1,1)',
        anti_windup='conditional integration uses final total command and final total raw torque',
        test_response_calibration=False, replay_buffer_saved=False,
        reward=dict(
            cost='wu*actual_tau_square_integral/(dt*limit^2)+we*min((error/60deg)^2,100)+ww*min((rate/30deg_s)^2,100)+1[t_end>=8]*wv*(outside+min(max(angle_ratio-1,0)^2,25)+min(max(rate_ratio-1,0)^2,25))',
            interval='-dt*cost+gamma*potential_next-potential_current',
            potential='-4*min((error/60deg)^2,100)-0.1*min((rate/30deg_s)^2,100)',
            final_potential=0., failure='each remaining interval costs the maximum legal ordinary rate cost plus one',
            failure_rate_costs={name: maximum_rate_cost(name)+1. for name in REWARDS},
            ordinary_rate_cost_bounds={name: maximum_rate_cost(name) for name in REWARDS},
            terminal='440 samples only; no time-limit bootstrap; failures remain absorbing until sample 440',
            shaping_property='discounted shaping sums to minus the common initial potential; terminal potential is exactly zero'),
        model_selection='fewest physical failures, most passes, lowest penalized_selection_cost, earliest checkpoint',
        initialization='actor deterministic mean exactly zero; initial checkpoint retained and evaluated; zero residual is exactly the selected PID',
        ideal_sanity='direct continuous action, lag physical plant with T=0, initial angle +/-20deg and rate +/-2deg/s',
        hidden_from_features=['actual torque', 'true actuator parameter', 'simulator delay queue'],
        optimizer_step_count_definition='actual calls to each Torch optimizer.step; SB3 _n_updates is retained separately',
    )


def maximum_rate_cost(reward_name):
    w = REWARDS[reward_name]
    return w['torque_weight'] + 100*w['error_weight'] + 100*w['rate_weight'] + 51*w['violation_weight']


def potential(observation):
    obs = np.asarray(observation, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 2 or not np.all(np.isfinite(obs)):
        raise ValueError('potential observation must be finite (n,2)')
    return -4.*np.minimum((obs[:, 0]/ERROR_SCALE)**2, 100.)-.1*np.minimum((obs[:, 1]/RATE_SCALE)**2, 100.)


def reward_values(observation, elapsed, actual_cost, failed, reward_name='A',
                  previous_observation=None, terminal=None):
    """Full-horizon reward with a bounded ordinary cost and absorbing failure.

    Setting terminal potential to zero is essential: otherwise a final large
    error could change the finite-horizon objective through its potential.
    Physical failure flags are cumulative, not just the failing transition.
    """
    w = REWARDS[reward_name]
    obs = np.asarray(observation, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 2 or not np.all(np.isfinite(obs)):
        raise ValueError('reward observation must be finite (n,2)')
    n = len(obs)
    elapsed = np.broadcast_to(np.asarray(elapsed, float), (n,))
    actual_cost = np.broadcast_to(np.asarray(actual_cost, float), (n,))
    failed = np.broadcast_to(np.asarray(failed, bool), (n,))
    if np.any(~np.isfinite(elapsed)) or np.any(~np.isfinite(actual_cost)):
        raise ValueError('elapsed and physical torque costs must be finite')
    if np.any(actual_cost < -1e-12) or np.any(actual_cost > p.DT*p.LIMIT**2+1e-10):
        raise ValueError('physical interval cost exceeds the common actuator bound')
    e, rate = np.abs(obs[:, 0]), np.abs(obs[:, 1])
    eratio, wratio = e/np.deg2rad(p.ANGLE_TOL), rate/np.deg2rad(p.RATE_TOL)
    outside = (eratio > 1.) | (wratio > 1.)
    violation = (outside.astype(float)+np.minimum(np.maximum(eratio-1., 0.)**2, 25.)
                 +np.minimum(np.maximum(wratio-1., 0.)**2, 25.))
    ordinary = (w['torque_weight']*np.clip(actual_cost/(p.DT*p.LIMIT**2), 0., 1.)
                +w['error_weight']*np.minimum((e/ERROR_SCALE)**2, 100.)
                +w['rate_weight']*np.minimum((rate/RATE_SCALE)**2, 100.)
                +(elapsed >= p.DEADLINE-1e-10)*w['violation_weight']*violation)
    cost = np.where(failed, maximum_rate_cost(reward_name)+1., ordinary)
    reward = -p.DT*cost
    if previous_observation is not None:
        prev = np.asarray(previous_observation, dtype=float)
        if prev.shape != obs.shape:
            raise ValueError('previous observation differs from current shape')
        done = np.broadcast_to(np.asarray(terminal, bool), (n,)) if terminal is not None else elapsed >= p.DURATION-1e-10
        next_phi = np.where(done, 0., potential(obs))
        reward += p.GAMMA*next_phi-potential(prev)
    return reward.astype(np.float32)


class FeatureHistory:
    def __init__(self, mode):
        if mode not in p.METHODS:
            raise ValueError(mode)
        self.mode = mode
        self.n = 0

    @property
    def dimension(self):
        return 3*p.HISTORY+4+(2 if self.mode == 'residual' else 0)

    def reset(self, n, initial_observation):
        obs = np.asarray(initial_observation, float)
        if obs.shape != (n, 2) or not np.all(np.isfinite(obs)):
            raise ValueError('initial observation must be finite (n,2)')
        self.n = n
        self.measurements = np.repeat(obs[:, None, :], p.HISTORY+1, axis=1)
        self.commands = np.zeros((n, p.HISTORY))
        self.valid = np.zeros(n, dtype=np.int64)
        self.pending = np.zeros(n, dtype=bool)

    def reset_rows(self, mask, observation):
        mask = np.asarray(mask, bool)
        obs = np.asarray(observation, float)
        if mask.shape != (self.n,) or obs.shape != (self.n, 2):
            raise ValueError('invalid partial reset shape')
        self.measurements[mask] = obs[mask, None, :]
        self.commands[mask] = 0.
        self.valid[mask] = 0
        self.pending[mask] = False

    def observe(self, observation):
        obs = np.asarray(observation, float)
        if obs.shape != (self.n, 2) or not np.all(np.isfinite(obs)):
            raise ValueError('observation must be finite (n,2)')
        rows = np.flatnonzero(self.pending)
        if len(rows):
            self.measurements[rows, :-1] = self.measurements[rows, 1:]
            self.measurements[rows, -1] = obs[rows]
            self.pending[rows] = False
        if not np.array_equal(self.measurements[:, -1], obs):
            raise ValueError('observation changed without an intervening command')

    def record_command(self, command):
        command = np.asarray(command, float).reshape(-1)
        if command.shape != (self.n,) or not np.all(np.isfinite(command)):
            raise ValueError('command must be finite length n')
        if np.any(np.abs(command) > p.LIMIT+1e-12) or np.any(self.pending):
            raise ValueError('command must be clipped and followed by an observation')
        self.commands[:, :-1] = self.commands[:, 1:]
        self.commands[:, -1] = command
        self.valid = np.minimum(self.valid+1, p.HISTORY)
        self.pending[:] = True

    def features(self, elapsed, raw_pid=None, integral=None):
        elapsed = np.broadcast_to(np.asarray(elapsed, float), (self.n,))
        if np.any(~np.isfinite(elapsed)) or np.any(elapsed < 0.):
            raise ValueError('elapsed must be finite and nonnegative')
        result = np.empty((self.n, self.dimension), dtype=np.float32)
        nstate, ncommand = 2*(p.HISTORY+1), p.HISTORY
        result[:, :nstate] = (self.measurements/np.array([ERROR_SCALE, RATE_SCALE])).reshape(self.n, nstate)
        result[:, nstate:nstate+ncommand] = self.commands/p.LIMIT
        result[:, nstate+ncommand] = elapsed/p.DURATION
        result[:, nstate+ncommand+1] = self.valid/p.HISTORY
        if self.mode == 'residual':
            if raw_pid is None or integral is None:
                raise ValueError('residual features require own PID state')
            result[:, -2] = np.asarray(raw_pid)/p.LIMIT
            result[:, -1] = np.asarray(integral)/INTEGRAL_SCALE
        return result


class PolicyInterface:
    """Exactly shared training/deployment feature, clipping and PID state path."""
    def __init__(self, mode, base_parameters=None):
        self.mode = mode
        self.base_parameters = base_parameters
        self.history = FeatureHistory(mode)
        self.pid = None
        if mode == 'residual':
            if base_parameters is None:
                raise ValueError('residual control requires fixed PID parameters')
            from .baselines import PIDController
            self.pid = PIDController(base_parameters)

    def reset(self, n, observation):
        self.history.reset(n, observation)
        if self.pid is not None:
            self.pid.reset(n, observation)

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
        actions = np.asarray(actions, float).reshape(-1)
        if actions.shape != (self.history.n,) or not np.all(np.isfinite(actions)):
            raise ValueError('finite length-n actions required')
        actions = np.clip(actions, -1., 1.)
        self.history.observe(observation)
        if self.pid is None:
            command = p.LIMIT*actions
        else:
            raw = self.pid.propose(observation)+p.ALPHA*p.LIMIT*actions
            command = np.clip(raw, -p.LIMIT, p.LIMIT)
            self.pid.commit(observation, command, raw=raw)
        self.history.record_command(command)
        return command


class Controller:
    def __init__(self, model, mode, base_parameters=None):
        self.model = model
        self.mode = mode
        self.interface = PolicyInterface(mode, base_parameters)

    def reset(self, n, observation):
        self.interface.reset(n, observation)

    def features(self, observation, elapsed):
        return self.interface.features(observation, elapsed)

    def act(self, observation, elapsed):
        features = self.features(observation, elapsed)
        actions, _ = self.model.predict(features, deterministic=True)
        return self.interface.command(observation, actions)


def load_controller(model_path, algorithm, mode, base_parameters=None):
    import stable_baselines3 as sb3
    if algorithm not in p.ALGORITHMS:
        raise ValueError(algorithm)
    model = getattr(sb3, algorithm).load(model_path, device='cpu')
    return Controller(model, mode, base_parameters)


def _make_vec_class():
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import VecEnv
    from .physics import BatchPlant

    class StudyVecEnv(VecEnv):
        render_mode = None

        def __init__(self, kind, mode, reward_name, seed, base_parameters=None,
                     n_envs=p.N_ENVS, log_directory=None):
            if kind not in (*p.KINDS, 'ideal') or mode not in p.METHODS or reward_name not in REWARDS:
                raise ValueError('unknown environment configuration')
            self.kind, self.mode, self.reward_name = kind, mode, reward_name
            self.run_seed = int(seed)
            self.interface = PolicyInterface(mode, base_parameters)
            self.plant = BatchPlant('lag' if kind == 'ideal' else kind, n_envs)
            kind_id = (*p.KINDS, 'ideal').index(kind)
            self.rngs = [np.random.default_rng(np.random.SeedSequence([self.run_seed, kind_id, p.METHODS.index(mode), rank])) for rank in range(n_envs)]
            self.episode_ids = np.zeros(n_envs, dtype=np.int64)
            self.episode_returns = np.zeros(n_envs)
            self.episode_discounted_returns = np.zeros(n_envs)
            self.episode_actual_cost = np.zeros(n_envs)
            self.episode_command_cost = np.zeros(n_envs)
            self.episode_success = np.ones(n_envs, bool)
            self.total_steps = 0
            self.actions = None
            self.closed = False
            self._initial_log = self._episode_log = None
            if log_directory is not None:
                directory = Path(log_directory)
                directory.mkdir(parents=True, exist_ok=True)
                self._initial_log = (directory/'training_episode_initials.jsonl').open('x', encoding='utf-8')
                self._episode_log = (directory/'training_episodes.jsonl').open('x', encoding='utf-8')
            super().__init__(n_envs, Box(-np.inf, np.inf, (self.interface.history.dimension,), dtype=np.float32), Box(-1., 1., (1,), dtype=np.float32))

        def seed(self, seed=None):
            if seed is not None and int(seed) != self.run_seed:
                raise ValueError('training random seed is fixed')
            return [int(np.random.SeedSequence([self.run_seed, (*p.KINDS, 'ideal').index(self.kind), p.METHODS.index(self.mode), rank]).generate_state(1)[0]) for rank in range(self.num_envs)]

        def _reset_rows(self, mask, first=False):
            theta, omega, parameters = [], [], []
            for rank in np.flatnonzero(mask):
                angle, rate = p.training_initial(self.rngs[rank], 1, sanity=self.kind == 'ideal')
                parameter = float(p.training_parameters(self.kind, self.rngs[rank], 1)[0])
                theta.append(np.deg2rad(float(angle[0])))
                omega.append(np.deg2rad(float(rate[0])))
                parameters.append(parameter)
                self.reset_infos[rank] = dict(rank=int(rank), episode=int(self.episode_ids[rank]), seed=self.run_seed, kind=self.kind, mode=self.mode, initial_angle_deg=float(angle[0]), initial_omega_deg_s=float(rate[0]), parameter_s=parameter, total_training_steps=int(self.total_steps))
                self.episode_ids[rank] += 1
                if self._initial_log is not None:
                    self._initial_log.write(json.dumps(self.reset_infos[rank], allow_nan=False)+'\n')
            self.plant.reset(np.array(theta), np.array(omega), np.array(parameters), mask=mask)
            obs = self.plant.observation()
            if first:
                self.interface.reset(self.num_envs, obs)
            else:
                self.interface.reset_rows(mask, obs)
            self.episode_returns[mask] = 0.
            self.episode_discounted_returns[mask] = 0.
            self.episode_actual_cost[mask] = 0.
            self.episode_command_cost[mask] = 0.
            self.episode_success[mask] = True
            if self._initial_log is not None:
                self._initial_log.flush()

        def reset(self):
            self._reset_rows(np.ones(self.num_envs, bool), first=True)
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
            previous = self.plant.observation()
            command = self.interface.command(previous, self.actions)
            self.actions = None
            step = self.plant.step(command)
            self.total_steps += self.num_envs
            measured = self.plant.observation()
            dones = step['done']
            reward = reward_values(measured, self.plant.elapsed, step['actual_cost'], self.plant.failed, self.reward_name, previous, dones)
            self.episode_returns += reward
            self.episode_discounted_returns += p.GAMMA**(self.plant.step_count-1)*reward
            self.episode_actual_cost += step['actual_cost']
            self.episode_command_cost += step['command_cost']
            in_tolerance = (np.abs(measured[:, 0]) <= np.deg2rad(p.ANGLE_TOL)) & (np.abs(measured[:, 1]) <= np.deg2rad(p.RATE_TOL)) & ~self.plant.failed
            self.episode_success &= (self.plant.elapsed < p.DEADLINE-1e-10) | in_tolerance
            features = self.interface.features(measured, self.plant.elapsed)
            infos = []
            for rank in range(self.num_envs):
                info = {'TimeLimit.truncated': False}
                if dones[rank]:
                    episode = dict(r=float(self.episode_returns[rank]), l=int(self.plant.step_count[rank]), discounted_return=float(self.episode_discounted_returns[rank]), actual_cost=float(self.episode_actual_cost[rank]), command_cost=float(self.episode_command_cost[rank]), failed=bool(self.plant.failed[rank]), success=bool(self.episode_success[rank]))
                    info.update(terminal_observation=features[rank].copy(), episode=episode, termination_reason='finite_horizon_22s_after_failure' if self.plant.failed[rank] else 'finite_horizon_22s')
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
            return [getattr(self, attr_name) for _ in self._get_indices(indices)]

        def set_attr(self, attr_name, value, indices=None):
            raise AttributeError('environment settings are fixed')

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            raise AttributeError('no public mutation of individual environments')

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return StudyVecEnv


_VEC_CLASS = None


def StudyVecEnv(*args, **kwargs):
    global _VEC_CLASS
    if _VEC_CLASS is None:
        _VEC_CLASS = _make_vec_class()
    return _VEC_CLASS(*args, **kwargs)


def build_model(algorithm, env, seed, steps, optimizer_log=None):
    """Construct a real SB3 learner and explicitly zero its action mean."""
    import stable_baselines3 as sb3
    from stable_baselines3.common.noise import NormalActionNoise
    import torch
    if algorithm not in p.ALGORITHMS:
        raise ValueError(algorithm)
    torch.set_num_threads(COMMON_SETTINGS['torch_threads'])
    torch.use_deterministic_algorithms(True)
    settings = ALGORITHM_SETTINGS[algorithm]
    kwargs = dict(seed=int(seed), device='cpu', verbose=0, learning_rate=COMMON_SETTINGS['learning_rate'], gamma=p.GAMMA,
                  policy_kwargs=dict(net_arch=COMMON_SETTINGS['net_arch']))
    if algorithm in ('SAC', 'TD3'):
        kwargs.update(batch_size=settings['batch_size'], tau=settings['tau'], buffer_size=min(settings['buffer_size'], max(steps, settings['learning_starts']+8)),
                      learning_starts=settings['learning_starts'], train_freq=(settings['train_freq_vector_steps'], 'step'), gradient_steps=settings['gradient_steps'])
        if algorithm == 'SAC':
            kwargs.update(ent_coef=settings['ent_coef'], target_entropy=settings['target_entropy'])
        else:
            kwargs.update(action_noise=NormalActionNoise(np.zeros(1), settings['exploration_std']*np.ones(1)), policy_delay=settings['policy_delay'],
                          target_policy_noise=settings['target_policy_noise'], target_noise_clip=settings['target_noise_clip'])
    else:
        kwargs.update(n_steps=settings['n_steps'], batch_size=settings['batch_size'], n_epochs=settings['n_epochs'],
                      gae_lambda=settings['gae_lambda'], clip_range=settings['clip_range'], ent_coef=settings['ent_coef'],
                      vf_coef=settings['vf_coef'], max_grad_norm=settings['max_grad_norm'], normalize_advantage=settings['normalize_advantage'], target_kl=settings['target_kl'])
        kwargs['policy_kwargs']['log_std_init'] = settings['initial_log_std']
    model = getattr(sb3, algorithm)('MlpPolicy', env, **kwargs)
    with torch.no_grad():
        if algorithm == 'SAC':
            model.actor.mu.weight.zero_()
            model.actor.mu.bias.zero_()
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(settings['initial_log_std'])
        elif algorithm == 'TD3':
            final = [layer for layer in model.actor.mu.modules() if isinstance(layer, torch.nn.Linear)][-1]
            final.weight.zero_()
            final.bias.zero_()
            model.actor_target.load_state_dict(model.actor.state_dict())
        else:
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    counters = {}
    optimizers = []
    if algorithm == 'SAC':
        optimizers = [('actor', model.actor.optimizer), ('critic', model.critic.optimizer), ('entropy', model.ent_coef_optimizer)]
    elif algorithm == 'TD3':
        optimizers = [('actor', model.actor.optimizer), ('critic', model.critic.optimizer)]
    else:
        optimizers = [('policy_value', model.policy.optimizer)]
    for name, optimizer in optimizers:
        if optimizer is None:
            continue
        counters[name] = 0
        original = optimizer.step
        def counted_step(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            counters[_name] += 1
            return result
        optimizer.step = counted_step
    original_train = model.train
    def logged_train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        if optimizer_log is not None:
            metrics = {key: float(value) for key, value in model.logger.name_to_value.items()
                       if key.startswith('train/') and isinstance(value, (float, int, np.number)) and np.isfinite(value)}
            metrics.update(timesteps=int(model.num_timesteps), sb3_n_updates=int(model._n_updates), optimizer_step_counts=dict(counters))
            optimizer_log.write(json.dumps(metrics, allow_nan=False)+'\n')
        return result
    model.train = logged_train
    # These runtime closures are deliberately excluded from SB3's data archive.
    # Optimizer state_dicts are still saved normally in every model.zip.
    model._continuous_original_train = original_train
    model._continuous_optimizer_counts = counters
    original_excluded = model._excluded_save_params
    model._excluded_save_params = lambda: original_excluded()+['train', '_continuous_original_train', '_continuous_optimizer_counts', '_excluded_save_params']
    return model


def _actor_statistics(model, algorithm, features):
    import torch
    actions, _ = model.predict(features, deterministic=True)
    result = dict(deterministic_action_mean=float(np.mean(actions)), deterministic_action_std=float(np.std(actions)),
                  deterministic_action_abs_max=float(np.max(np.abs(actions))),
                  deterministic_action_abs_mean=float(np.mean(np.abs(actions))))
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=model.device)
        if algorithm == 'SAC':
            _, log_std, _ = model.actor.get_action_dist_params(x)
            std = log_std.exp()
            result.update(exploration_std_mean=float(std.mean()), exploration_std_min=float(std.min()), exploration_std_max=float(std.max()))
        elif algorithm == 'PPO':
            std = model.policy.log_std.exp()
            result['exploration_std_mean'] = float(std.mean())
        else:
            result['external_exploration_std'] = ALGORITHM_SETTINGS['TD3']['exploration_std']
    return result


def _rank(checkpoint):
    from .evaluation import ranking
    return (*ranking(checkpoint['validation']), -checkpoint['steps'])


def predeclare():
    data = json.loads(json.dumps(configuration()))
    path = p.STUDY/'03_训练记录'/'predeclared_learning.json'
    if path.exists() and p.read_json(path) != data:
        raise RuntimeError('learning declaration changed')
    if not path.exists():
        p.write_json(path, data)
    return data


def _relative(path):
    return Path(path).relative_to(p.STUDY).as_posix()


def _run_directory(kind, algorithm, seed, stage, mode, reward_name):
    return p.STUDY/'03_训练记录'/stage/kind/mode/algorithm/f'reward_{reward_name}'/f'seed_{seed}'


def _load_base(kind):
    return p.read_json(p.STUDY/'02_传统控制'/f'selected_{kind}.json')['pid']['parameters']


def train_run(kind, algorithm, seed, steps, stage, base_parameters=None,
              validation_scenes=None, mode='residual', reward_name='A'):
    """Run an exact predeclared SB3 budget and retain all checkpoint evidence."""
    if (p.STUDY/'04_独立测试'/'freeze.json').exists():
        raise RuntimeError('held-out freeze prohibits any further fitting')
    if algorithm not in p.ALGORITHMS or reward_name not in REWARDS or mode not in p.METHODS:
        raise ValueError('unknown training configuration')
    budgets = dict(sanity=p.SANITY_STEPS, development=p.DEVELOPMENT_STEPS, formal=p.FORMAL_STEPS)
    if stage not in budgets or steps != budgets[stage]:
        raise ValueError('step count differs from predeclared stage budget')
    if steps % p.VALIDATE_EVERY or p.VALIDATE_EVERY % (p.N_ENVS*ALGORITHM_SETTINGS['PPO']['n_steps']):
        raise ValueError('all checkpoint budgets must align to the complete PPO rollout')
    if stage == 'sanity':
        if kind != 'ideal' or mode != 'direct' or reward_name != 'A' or seed != p.SANITY_SEEDS[algorithm]:
            raise ValueError('unregistered sanity run')
    else:
        if kind not in p.KINDS or mode != 'residual':
            raise ValueError('research uses residual control of lag or delay')
        if stage == 'development' and seed != p.DEVELOPMENT_SEEDS[reward_name]:
            raise ValueError('unregistered development seed')
        if stage == 'formal' and seed not in p.FORMAL_SEEDS:
            raise ValueError('unregistered formal seed')
        selected_base = _load_base(kind)
        if base_parameters is None:
            base_parameters = selected_base
        if base_parameters != selected_base:
            raise ValueError('residual base differs from independently selected PID')
    declaration = p.STUDY/'03_训练记录'/'predeclared_learning.json'
    if not declaration.exists() or p.read_json(declaration) != json.loads(json.dumps(configuration())):
        raise RuntimeError('freeze learning configuration before training')
    if stage == 'formal':
        decision_path = p.STUDY/'03_训练记录'/'decisions'/f'{kind}.json'
        decision = p.read_json(decision_path)
        selected = decision.get('selected') or {}
        if not decision['formal_eligible'] or selected.get('algorithm') != algorithm or selected.get('reward_name') != reward_name:
            raise RuntimeError('formal run differs from the selected eligible development configuration')
    from .evaluation import evaluate, summary, write_csv
    from stable_baselines3.common.logger import configure
    import stable_baselines3
    import torch
    import gymnasium
    directory = _run_directory(kind, algorithm, seed, stage, mode, reward_name)
    directory.mkdir(parents=True, exist_ok=False)
    before = p.source_hashes()
    for key in before:
        source = p.STUDY/key
        destination = directory/'source_snapshot'/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    metadata = dict(status='running', kind=kind, algorithm=algorithm, mode=mode, stage=stage,
                    seed=int(seed), reward_name=reward_name, requested_steps=int(steps), actual_steps=0,
                    optimizer_updates=0, optimizer_step_counts={}, best=None, checkpoints=[],
                    base_parameters=base_parameters, source_sha256=before, source_sha256_after=None,
                    configuration=configuration(), declaration_sha256=p.sha256(declaration),
                    run_path=_relative(directory), training_elapsed_seconds=0., validation_elapsed_seconds=0.,
                    replay_buffer_saved=False, test_responses_used=False, trained_checkpoint_improved_initial=False,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    versions=dict(numpy=np.__version__, torch=torch.__version__, stable_baselines3=stable_baselines3.__version__, gymnasium=gymnasium.__version__))
    metadata_path = directory/'training_metadata.json'
    p.write_json(metadata_path, metadata)
    env = model = optimizer_log = None
    start = time.monotonic()
    try:
        env = StudyVecEnv(kind, mode, reward_name, seed, base_parameters, log_directory=directory)
        optimizer_log = (directory/'optimizer_steps.jsonl').open('x', encoding='utf-8')
        model = build_model(algorithm, env, seed, steps, optimizer_log)
        model.set_logger(configure(str(directory/'sb3_logs'), ['csv']))
        scenes = validation_scenes if validation_scenes is not None else p.scenes(kind, 'validation')
        physical_kind = 'lag' if kind == 'ideal' else kind
        validation_obs = np.column_stack((np.deg2rad([-s['angle_deg'] for s in scenes]), np.deg2rad([s['omega_deg_s'] for s in scenes])))
        inspector = PolicyInterface(mode, base_parameters)
        inspector.reset(len(scenes), validation_obs)
        initial_features = inspector.features(validation_obs, np.zeros(len(scenes)))
        for checkpoint_steps in range(0, steps+1, p.VALIDATE_EVERY):
            if checkpoint_steps:
                started = time.monotonic()
                model.learn(p.VALIDATE_EVERY, reset_num_timesteps=(checkpoint_steps == p.VALIDATE_EVERY), log_interval=1)
                metadata['training_elapsed_seconds'] += time.monotonic()-started
                if model.num_timesteps != checkpoint_steps or env.total_steps != checkpoint_steps:
                    raise RuntimeError('actual aggregate steps differ from the fixed budget')
            checkpoint_dir = directory/'checkpoints'/f'steps_{checkpoint_steps:06d}'
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            model_path = checkpoint_dir/'model.zip'
            model.save(model_path)
            started = time.monotonic()
            result = evaluate(physical_kind, scenes, Controller(model, mode, base_parameters))
            metadata['validation_elapsed_seconds'] += time.monotonic()-started
            stats = summary(result['rows'])
            stats['mean_deadline_violation'] = float(np.mean([row['deadline_violation'] for row in result['rows']]))
            write_csv(checkpoint_dir/'validation.csv', result['rows'])
            actor_stats = _actor_statistics(model, algorithm, initial_features)
            if checkpoint_steps == 0 and actor_stats['deterministic_action_abs_max'] != 0.:
                raise RuntimeError('actor mean was not initialized to exact zero')
            checkpoint = dict(model_path=_relative(model_path), model_sha256=p.sha256(model_path), steps=int(model.num_timesteps),
                              validation=stats, optimizer_updates=int(model._n_updates),
                              optimizer_step_counts=dict(model._continuous_optimizer_counts), actor_statistics=actor_stats)
            p.write_json(checkpoint_dir/'checkpoint_metadata.json', dict(checkpoint, source_sha256=before, base_parameters=base_parameters,
                                                                        reward_name=reward_name, algorithm=algorithm, seed=int(seed)))
            metadata['checkpoints'].append(checkpoint)
            if checkpoint_steps == 0:
                metadata['initial'] = checkpoint
                metadata['initial_zero_residual_validation'] = stats if mode == 'residual' else None
            if metadata['best'] is None or _rank(checkpoint) > _rank(metadata['best']):
                shutil.copy2(model_path, directory/'best.zip')
                metadata['best'] = dict(checkpoint, checkpoint_model_path=checkpoint['model_path'], model_path=_relative(directory/'best.zip'))
            metadata['trained_checkpoint_improved_initial'] = bool(metadata['best']['steps'] > 0 and _rank(metadata['best']) > _rank(metadata['initial']))
            metadata['actual_steps'] = int(model.num_timesteps)
            metadata['optimizer_updates'] = int(model._n_updates)
            metadata['optimizer_step_counts'] = dict(model._continuous_optimizer_counts)
            metadata['source_sha256_after'] = p.source_hashes()
            if metadata['source_sha256_after'] != before:
                raise RuntimeError('production source changed during training')
            metadata['elapsed_seconds'] = time.monotonic()-start
            if checkpoint_steps:
                metadata['training_steps_per_second'] = model.num_timesteps/max(metadata['training_elapsed_seconds'], 1e-12)
            optimizer_log.flush()
            p.write_json(metadata_path, metadata)
            print(json.dumps(dict(event='continuous_control_checkpoint', kind=kind, algorithm=algorithm, mode=mode, stage=stage,
                                  seed=seed, reward=reward_name, steps=checkpoint_steps, validation=stats,
                                  actor_statistics=actor_stats, elapsed_seconds=metadata['elapsed_seconds']), ensure_ascii=False), flush=True)
        model.save(directory/'final.zip')
        metadata['final'] = dict(model_path=_relative(directory/'final.zip'), model_sha256=p.sha256(directory/'final.zip'), steps=int(model.num_timesteps))
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
