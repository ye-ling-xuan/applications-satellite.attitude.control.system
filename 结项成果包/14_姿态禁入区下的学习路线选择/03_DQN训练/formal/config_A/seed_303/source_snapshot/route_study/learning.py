"""True epsilon-greedy single-decision DQN using cached physical rollouts.

One agent step is one complete route/feedback trajectory, not one 0.05 s plant
step. The cache contains rewards for both actions but the agent observes only
the selected action's outcome. No direction oracle, expert label, correction or
supervised imitation loss is used by the learning policy.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
from . import protocol as p
from .baselines import selected, write_csv
from .physics import features, simulate, summary, rewards

CONFIGS = dict(A=dict(learning_rate=3e-4, exploration_fraction=.4, exploration_final_eps=.05),
               B=dict(learning_rate=1e-3, exploration_fraction=.6, exploration_final_eps=.02))
SETTINGS = dict(net_arch=[64, 64], gamma=0., batch_size=128, buffer_size=50000,
                learning_starts=512, train_freq_vector_steps=1, gradient_steps=2,
                n_envs=p.N_ENVS, target_update_interval=2000, device='cpu', torch_threads=1)


def declaration():
    return dict(configs=CONFIGS, settings=SETTINGS,
                feature_order=['sin(initial angle)', 'cos(initial angle)', 'sin(target angle)', 'cos(target angle)'],
                omega0=0., reward='(-10 if collision else +10 if full task success else -2) -0.1*torque_cost/(limit^2*duration) -0.05*settling_or_duration/duration',
                gamma_explanation='every contextual decision is terminal, no next-decision value is bootstrapped',
                optimizer_ratio='two gradient updates per 16 aggregate route decisions after warmup512',
                validation_selection='maximize task successes, then minimize collisions, then torque cost, then earlier checkpoint',
                no_oracle_actions=True, no_rule_correction=True, no_test_calibration=True)


def predeclare():
    path = p.STUDY/'01_方案与场景'/'DQN预声明.json'
    value = declaration()
    if path.exists() and p.read_json(path) != value:
        raise RuntimeError('DQN declaration changed')
    p.write_json(path, value)
    return value


def prepare_database():
    """No optimizer work: compute exact physical environment outcomes once."""
    directory = p.STUDY/'03_DQN训练'/'physical_environment_database'
    if (directory/'database.npz').exists():
        raise FileExistsError('physical environment database is immutable')
    baseline = selected()
    if not baseline['analytic_feasibility_passed']:
        raise RuntimeError('analytic route + PID must establish task feasibility first')
    before = p.source_hashes()
    started = time.monotonic()
    population = p.sample(np.random.default_rng(p.SPLIT_SEEDS['training']), p.COUNTS['training'])
    count = len(population)
    arrays = dict(population=population, features=features(population), rewards=np.empty((count, 2), np.float32),
                  collision=np.empty((count, 2), bool), success=np.empty((count, 2), bool),
                  torque_cost=np.empty((count, 2)), physical_intervals=np.empty((count, 2), np.int32),
                  final_theta=np.empty((count, 2)))
    for action in (0, 1):
        for first in range(0, count, 2000):
            last = min(count, first+2000)
            records = simulate(population[first:last], np.full(last-first, action), baseline['parameters'])['rows']
            arrays['rewards'][first:last, action] = rewards(records)
            for key in ('collision', 'success', 'torque_cost', 'physical_intervals'):
                arrays[key][first:last, action] = [row[key] for row in records]
            arrays['final_theta'][first:last, action] = [row['final_theta_rad'] for row in records]
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'database.npz'
    np.savez_compressed(path, **arrays)
    after = p.source_hashes()
    if before != after:
        raise RuntimeError('source changed while computing physical database')
    metadata = dict(unique_independent_training_states=count, simulated_trajectories=2*count,
                    complete_horizon_record_slots=int(2*count*round(p.DURATION/p.DT)),
                    physically_active_hold_intervals=int(arrays['physical_intervals'].sum()),
                    failed_postcontact_padding='not physical propagation; no cost after failing interval',
                    database_sha256=p.sha256(path), parameters=baseline['parameters'],
                    source_sha256=before, source_sha256_after=after,
                    elapsed_seconds=time.monotonic()-started, test_responses_used=False)
    p.write_json(directory/'database_metadata.json', metadata)
    return metadata


def _vec_class():
    from gymnasium.spaces import Box, Discrete
    from stable_baselines3.common.vec_env import VecEnv

    class RouteVecEnv(VecEnv):
        render_mode = None

        def __init__(self, data, seed, log_path=None):
            self.data, self.seed_value = data, int(seed)
            self.rngs = [np.random.default_rng(np.random.SeedSequence([int(seed), 14, rank])) for rank in range(p.N_ENVS)]
            self.row_ids = np.zeros(p.N_ENVS, np.int64)
            self.actions = None
            self.decisions = 0
            self.seen_pairs = set()
            self.seen_states = set()
            self.log = None
            if log_path is not None:
                self.log = Path(log_path).open('x', encoding='utf-8', newline='')
                self.writer = csv.writer(self.log)
                self.writer.writerow(['decision', 'rank', 'training_row', 'action', 'reward', 'success', 'collision', 'equivalent_physical_intervals'])
            super().__init__(p.N_ENVS, Box(-1., 1., (4,), dtype=np.float32), Discrete(2))

        def _sample(self):
            self.row_ids[:] = [rng.integers(len(self.data['features'])) for rng in self.rngs]
            return self.data['features'][self.row_ids].copy()

        def reset(self):
            return self._sample()

        def seed(self, seed=None):
            if seed is not None and int(seed) != self.seed_value:
                raise ValueError('route stream seed fixed at construction')
            return [self.seed_value]*p.N_ENVS

        def step_async(self, actions):
            self.actions = np.asarray(actions, dtype=int).reshape(-1)

        def step_wait(self):
            if self.actions is None or self.actions.shape != (p.N_ENVS,) or np.any((self.actions < 0) | (self.actions > 1)):
                raise ValueError('invalid route choices')
            actions = self.actions
            outcome = self.data['rewards'][self.row_ids, actions].copy()
            infos = []
            for rank in range(p.N_ENVS):
                row, action = int(self.row_ids[rank]), int(actions[rank])
                self.seen_states.add(row)
                self.seen_pairs.add((row, action))
                end = self.data['final_theta'][row, action]
                goal = self.data['population'][row, 1]
                terminal = np.array([np.sin(end), np.cos(end), np.sin(goal), np.cos(goal)], np.float32)
                infos.append({'TimeLimit.truncated': False, 'terminal_observation': terminal,
                              'episode': dict(r=float(outcome[rank]), l=1),
                              'training_row_id': row, 'chosen_action': action})
                if self.log is not None:
                    self.writer.writerow([self.decisions+rank+1, rank, row, action, float(outcome[rank]),
                                          bool(self.data['success'][row, action]), bool(self.data['collision'][row, action]),
                                          int(self.data['physical_intervals'][row, action])])
            self.decisions += p.N_ENVS
            self.actions = None
            return self._sample(), outcome, np.ones(p.N_ENVS, bool), infos

        def close(self):
            if self.log is not None:
                self.log.close()
                self.log = None

        def get_attr(self, name, indices=None):
            return [getattr(self, name) for _ in self._get_indices(indices)]

        def set_attr(self, name, value, indices=None):
            raise AttributeError('immutable contextual environment')

        def env_method(self, name, *args, indices=None, **kwargs):
            raise AttributeError('no per-rank mutation')

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return RouteVecEnv


def _rank(item):
    stats = item['validation']
    return (stats['successes'], -stats['collisions'], -stats['mean_torque_cost'], -item['decisions'])


def run_path(stage, config_name, seed):
    return p.STUDY/'03_DQN训练'/stage/f'config_{config_name}'/f'seed_{seed}'


def train_run(stage, config_name, seed):
    if (p.STUDY/'04_独立测试'/'freeze.json').exists():
        raise RuntimeError('held-out response freeze prohibits subsequent training')
    if config_name not in CONFIGS or stage not in ('development', 'formal'):
        raise ValueError('unknown DQN study run')
    if stage == 'development' and seed != p.DEVELOPMENT_SEEDS[config_name]:
        raise ValueError('undeclared development seed')
    if stage == 'formal':
        decision = p.read_json(p.STUDY/'03_DQN训练'/'development_decision.json')
        if not decision['passed_gate'] or decision['config_name'] != config_name or seed not in p.FORMAL_SEEDS:
            raise RuntimeError('formal route training has not passed its declared gate')
    if p.read_json(p.STUDY/'01_方案与场景'/'DQN预声明.json') != declaration():
        raise RuntimeError('DQN declaration differs')
    baseline = selected()
    database_directory = p.STUDY/'03_DQN训练'/'physical_environment_database'
    database_metadata = p.read_json(database_directory/'database_metadata.json')
    if p.sha256(database_directory/'database.npz') != database_metadata['database_sha256'] or database_metadata['parameters'] != baseline['parameters']:
        raise RuntimeError('physical environment database integrity/PID mismatch')
    from stable_baselines3 import DQN
    from stable_baselines3.common.logger import configure
    import torch
    torch.set_num_threads(SETTINGS['torch_threads'])
    torch.use_deterministic_algorithms(True)
    directory = run_path(stage, config_name, seed)
    directory.mkdir(parents=True, exist_ok=False)
    source = p.source_hashes()
    for file in sorted((p.STUDY/'代码').rglob('*.py')):
        if '__pycache__' not in file.parts:
            destination = directory/'source_snapshot'/file.relative_to(p.STUDY/'代码')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    maximum = p.DEVELOPMENT_DECISIONS if stage == 'development' else p.FORMAL_DECISIONS
    metadata = dict(status='running', stage=stage, config_name=config_name, seed=int(seed),
                    requested_decisions=maximum, actual_decisions=0, best=None, checkpoints=[],
                    source_sha256=source, source_sha256_after=None, parameters=baseline['parameters'],
                    declaration=declaration(), database_sha256=database_metadata['database_sha256'],
                    cached_environment=True, new_physical_rollouts_during_optimizer_training=0,
                    test_responses_used=False, training_seconds=0., validation_seconds=0.)
    path = directory/'training_metadata.json'
    p.write_json(path, metadata)
    log = (directory/'optimizer_steps.jsonl').open('x', encoding='utf-8')
    env = model = None
    try:
        with np.load(database_directory/'database.npz') as archive:
            data = {name: archive[name] for name in archive.files}
        env = _vec_class()(data, seed, directory/'selected_action_experience.csv')

        class LoggedDQN(DQN):
            def train(self, gradient_steps, batch_size=100):
                super().train(gradient_steps, batch_size)
                stats = {key: float(value) for key, value in self.logger.name_to_value.items()
                         if key.startswith('train/') and isinstance(value, (float, int, np.number))}
                log.write(json.dumps(dict(decisions=int(self.num_timesteps), updates=int(self._n_updates), **stats), allow_nan=False)+'\n')

        model = LoggedDQN('MlpPolicy', env, policy_kwargs=dict(net_arch=SETTINGS['net_arch']),
                          learning_rate=CONFIGS[config_name]['learning_rate'], gamma=0.,
                          buffer_size=SETTINGS['buffer_size'], learning_starts=SETTINGS['learning_starts'],
                          batch_size=SETTINGS['batch_size'], train_freq=(1, 'step'), gradient_steps=2,
                          target_update_interval=SETTINGS['target_update_interval'],
                          exploration_fraction=CONFIGS[config_name]['exploration_fraction'],
                          exploration_final_eps=CONFIGS[config_name]['exploration_final_eps'],
                          seed=int(seed), device='cpu', verbose=0)
        model.set_logger(configure(str(directory/'sb3_logs'), ['csv']))
        # One continuous learn call keeps epsilon decay tied to the complete
        # declared budget. Checkpoints run after completed optimizer updates.
        from stable_baselines3.common.callbacks import BaseCallback

        class EvaluateCallback(BaseCallback):
            def _on_step(self):
                return True

        callback = EvaluateCallback()
        callback.init_callback(model)
        # DQN's learn chunks reset exploration schedules unless the total horizon
        # remains fixed. Collect/train explicitly with SB3's own collect_rollouts
        # and train methods after one _setup_learn(maximum).
        _, callback = model._setup_learn(maximum, callback=callback, reset_num_timesteps=True)
        callback.on_training_start(locals(), globals())
        scene_list = p.scenes('validation')
        population = np.array([[row['theta_rad'], row['target_rad']] for row in scene_list])
        while model.num_timesteps < maximum:
            before = time.monotonic()
            rollout = model.collect_rollouts(model.env, callback=callback, train_freq=model.train_freq,
                                              replay_buffer=model.replay_buffer, action_noise=None,
                                              learning_starts=model.learning_starts, log_interval=128)
            if not rollout.continue_training:
                raise RuntimeError('unexpected early stop')
            if model.num_timesteps > model.learning_starts:
                model.train(gradient_steps=2, batch_size=model.batch_size)
            metadata['training_seconds'] += time.monotonic()-before
            if model.num_timesteps % p.CHECK_EVERY:
                continue
            elapsed = int(model.num_timesteps)
            expected_updates = 2*max(0, (elapsed-model.learning_starts)//p.N_ENVS)
            if elapsed > maximum or env.decisions != elapsed or model._n_updates != expected_updates:
                raise RuntimeError('DQN aggregate decision/update budget differs')
            checkpoint_dir = directory/'checkpoints'/f'decisions_{elapsed:06d}'
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            model.save(checkpoint_dir/'model.zip')
            before = time.monotonic()
            actions = model.predict(features(population), deterministic=True)[0]
            validation = simulate(scene_list, actions, baseline['parameters'])['rows']
            metadata['validation_seconds'] += time.monotonic()-before
            write_csv(checkpoint_dir/'validation.csv', validation)
            item = dict(decisions=elapsed, validation=summary(validation),
                        model_path=(checkpoint_dir/'model.zip').relative_to(p.STUDY).as_posix(),
                        model_sha256=p.sha256(checkpoint_dir/'model.zip'), updates=int(model._n_updates))
            metadata['checkpoints'].append(item)
            if metadata['best'] is None or _rank(item) > _rank(metadata['best']):
                shutil.copy2(checkpoint_dir/'model.zip', directory/'best.zip')
                metadata['best'] = dict(item, model_path=(directory/'best.zip').relative_to(p.STUDY).as_posix())
            metadata.update(actual_decisions=elapsed, optimizer_updates=int(model._n_updates),
                            unique_training_states_seen=len(env.seen_states), unique_state_action_pairs_seen=len(env.seen_pairs),
                            repeat_state_action_replays=elapsed-len(env.seen_pairs),
                            source_sha256_after=p.source_hashes())
            if source != metadata['source_sha256_after']:
                raise RuntimeError('source changed during route learning')
            p.write_json(path, metadata)
            log.flush()
            if env.log is not None:
                env.log.flush()
            print(json.dumps(dict(event='DQN route checkpoint', stage=stage, config=config_name, seed=seed,
                                  decisions=elapsed, validation=item['validation']), ensure_ascii=False), flush=True)
        callback.on_training_end()
        model.save(directory/'final.zip')
        metadata.update(status='complete', final=dict(model_path=(directory/'final.zip').relative_to(p.STUDY).as_posix(),
                                                      model_sha256=p.sha256(directory/'final.zip')))
    except BaseException as error:
        metadata.update(status='incomplete', error=repr(error), actual_decisions=int(model.num_timesteps) if model else 0)
        if model is not None:
            model.save(directory/'incomplete.zip')
        raise
    finally:
        metadata['source_sha256_after'] = p.source_hashes()
        p.write_json(path, metadata)
        log.close()
        if model is not None:
            model.logger.close()
        if env is not None:
            env.close()
    return metadata


def decide():
    attempts = []
    complete = []
    for config_name, seed in p.DEVELOPMENT_SEEDS.items():
        path = run_path('development', config_name, seed)/'training_metadata.json'
        if path.exists():
            metadata = p.read_json(path)
            attempts.append(dict(metadata_path=path.relative_to(p.STUDY).as_posix(), **metadata))
            if metadata['status'] == 'complete' and metadata['actual_decisions'] == p.DEVELOPMENT_DECISIONS:
                complete.append(metadata)
    chosen = max(complete, key=lambda item: _rank(item['best'])) if complete else None
    stats = chosen['best']['validation'] if chosen else None
    decision = dict(passed_gate=bool(stats and stats['collisions'] == 0 and stats['success_rate'] >= .95),
                    config_name=chosen['config_name'] if chosen else None,
                    best=chosen['best'] if chosen else None, attempts=attempts,
                    reason='only a development resource gate; no assumed superiority over analytic planning',
                    source_sha256=p.source_hashes(), test_responses_used=False)
    path = p.STUDY/'03_DQN训练'/'development_decision.json'
    if path.exists() and p.read_json(path) != decision:
        raise RuntimeError('development decision already immutable')
    p.write_json(path, decision)
    return decision
