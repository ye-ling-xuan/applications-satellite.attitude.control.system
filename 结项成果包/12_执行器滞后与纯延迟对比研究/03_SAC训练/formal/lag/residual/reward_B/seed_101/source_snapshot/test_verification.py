"""Adversarial checks of the independent verifier, without research training."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from actuator_study import protocol as p
from actuator_study.baselines import PIDController, ModelCompensatedController
from actuator_study.evaluation import evaluate
from actuator_study.learning import SACController
from actuator_study import verification as v


def _scenes(kind, parameter, initials=None, region='in_domain'):
    if initials is None:
        initials = [dict(sample_id=i, angle_deg=angle, omega_deg_s=rate)
                    for i, (angle, rate) in enumerate([(35., 1.), (-45., -2.), (0., 0.)])]
    return [dict(row, kind=kind, split='test', region=region, parameter_s=parameter,
                 scenario_id=f'{kind}_test_{parameter:.2f}_{row["sample_id"]:03d}')
            for row in initials]


def _descriptor(kind, family, parameters=None, **extra):
    return dict(id='test_controller', kind=kind, family=family, stage='baseline',
                parameters=parameters or {}, **extra)


class _FeaturePolicy:
    """A deterministic float32 policy sensitive to every input coordinate."""
    def __init__(self):
        self.seen = []

    def predict(self, features, deterministic=True):
        assert deterministic
        self.seen.append(features.copy())
        weights = np.linspace(-.03, .04, features.shape[1], dtype=np.float32)
        return np.tanh(np.sum(features*weights[None, :], axis=1))[:, None], None


class IndependentShardTests(unittest.TestCase):
    def _run(self, kind, family, parameter, parameters, initials=None):
        scenes = _scenes(kind, parameter, initials)
        controller = (PIDController(parameters) if family == 'pid'
                      else ModelCompensatedController(kind, parameters))
        result = evaluate(kind, scenes, controller, keep_traces=True)
        return scenes, result, _descriptor(kind, family, parameters)

    def test_lag_pid_and_ideal_are_verified_by_independent_matrix_and_quadrature(self):
        for parameter in [0., .05, .17, .5]:
            scenes, result, descriptor = self._run('lag', 'pid', parameter,
                                                   dict(kp=1.5, kd=2., ki=.005))
            audit = v.verify_shard('lag', result['traces'], scenes, result['rows'], descriptor)
            self.assertTrue(audit['passed'], audit['errors'])
            self.assertLess(audit['max_absolute_errors']['state'], 1e-11)
            self.assertLess(audit['max_absolute_errors']['interval_cost'], 1e-13)

    def test_compensated_lag_observer_and_delay_predictor_actions(self):
        for kind in ['lag', 'delay']:
            parameters = dict(kp=1.6, kd=2.1, ki=.003, nominal_parameter_s=.2,
                              actuator_feedback=.6, observer_gain=.55)
            scenes, result, descriptor = self._run(kind, 'model_compensated', .1, parameters)
            audit = v.verify_shard(kind, result['traces'], scenes, result['rows'], descriptor)
            self.assertTrue(audit['passed'], audit['errors'])
            self.assertLess(audit['max_absolute_errors']['action'], 1e-11)

    def test_direct_and_residual_features_rebuilt_without_production_history(self):
        for family in ['direct', 'residual']:
            scenes = _scenes('lag', .2)
            base = dict(kp=.7, kd=.9, ki=.2, integral_limit=.6)
            policy = _FeaturePolicy()
            result = evaluate('lag', scenes, SACController(policy, family, base), keep_traces=True)
            independent = _FeaturePolicy()
            descriptor = _descriptor('lag', family, base_parameters=base)
            audit = v.verify_shard('lag', result['traces'], scenes, result['rows'], descriptor, independent)
            self.assertTrue(audit['passed'], audit['errors'])
            for actual, expected in zip(independent.seen, policy.seen):
                # Healthy trajectories have exact shared information; padding
                # after physical failure is excluded from controller decisions.
                if not np.any(result['traces']['failed']):
                    np.testing.assert_array_equal(actual, expected)

    def test_physical_failure_padding_and_first_interval_cost_are_verified(self):
        initial = [dict(sample_id=0, angle_deg=169.999, omega_deg_s=np.rad2deg(.1))]
        scenes, result, descriptor = self._run('lag', 'pid', .1, dict(kp=2., kd=2.), initial)
        self.assertTrue(result['rows'][0]['failed'])
        audit = v.verify_shard('lag', result['traces'], scenes, result['rows'], descriptor)
        self.assertTrue(audit['passed'], audit['errors'])
        changed = deepcopy(result['traces'])
        changed['command'][1, 0] = .1
        audit = v.verify_shard('lag', changed, scenes, result['rows'], descriptor)
        self.assertFalse(audit['passed'])
        self.assertTrue(any('padded with zero' in error for error in audit['errors']))

    def test_state_torque_cost_and_reported_metrics_corruption_are_detected(self):
        scenes, result, descriptor = self._run('delay', 'pid', .1, dict(kp=1.2, kd=1.8))
        for key, index, increment in [('theta', (100, 0), .001),
                                      ('tau_start', (0, 0), .01),
                                      ('tau_end', (2, 1), .01),
                                      ('actual_cost_interval', (10, 0), .001)]:
            changed = deepcopy(result['traces'])
            changed[key][index] += increment
            audit = v.verify_shard('delay', changed, scenes, result['rows'], descriptor)
            self.assertFalse(audit['passed'], key)
        changed_rows = deepcopy(result['rows'])
        changed_rows[0]['success'] = not changed_rows[0]['success']
        changed_rows[1]['actual_cost'] += .1
        changed_rows[2]['rms_error_deg'] += .1
        audit = v.verify_shard('delay', result['traces'], scenes, changed_rows, descriptor)
        self.assertFalse(audit['passed'])
        self.assertTrue(any('success' in error for error in audit['errors']))

    def test_physically_consistent_but_wrong_controller_is_detected(self):
        scenes = _scenes('lag', .1)
        result = evaluate('lag', scenes, PIDController({}), keep_traces=True)
        claimed = _descriptor('lag', 'pid', dict(kp=2., kd=2.))
        audit = v.verify_shard('lag', result['traces'], scenes, result['rows'], claimed)
        self.assertFalse(audit['passed'])
        self.assertTrue(any('command at step' in error for error in audit['errors']))
        self.assertLess(audit['max_absolute_errors']['state'], 1e-11)

    def test_incomplete_grid_nonfinite_and_hidden_actual_state_are_rejected(self):
        scenes, result, descriptor = self._run('lag', 'pid', .1, dict(kp=1., kd=2.))
        changed = deepcopy(result['traces'])
        changed['time'][-1] = 21.99
        self.assertFalse(v.verify_shard('lag', changed, scenes, result['rows'], descriptor)['passed'])
        changed = deepcopy(result['traces'])
        changed['theta'][5, 0] = np.nan
        self.assertFalse(v.verify_shard('lag', changed, scenes, result['rows'], descriptor)['passed'])
        tree = ast.parse(Path(v.__file__).read_text(encoding='utf-8'))
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        for forbidden in ('physics', 'evaluation', 'learning', 'baselines'):
            self.assertNotIn(forbidden, imported)


class FrozenManifestTests(unittest.TestCase):
    def test_full_manifest_binding_coverage_hashes_and_trajectory_audit(self):
        with tempfile.TemporaryDirectory(prefix='actuator-independent-audit-') as temporary:
            study = Path(temporary)
            test = study/'04_独立测试'
            test.mkdir()
            rng = np.random.default_rng(p.SPLIT_SEEDS['test'])
            angles = rng.uniform(-60., 60., p.COUNTS['test'])
            rates = rng.uniform(-5., 5., p.COUNTS['test'])
            initials = [dict(sample_id=i, angle_deg=float(angle), omega_deg_s=float(rate))
                        for i, (angle, rate) in enumerate(zip(angles, rates))]
            initial_relative = '01_实验方案/test_initial_states.json'
            v._write(study/initial_relative, initials)
            parameters = dict(kp=1.3, kd=2.)
            descriptor = _descriptor('lag', 'pid', parameters)
            freeze = dict(controllers=[descriptor], artifacts={initial_relative: v._hash(study/initial_relative)})
            freeze_path = test/'freeze.json'
            v._write(freeze_path, freeze)
            manifest = dict(entries=[], artifacts={}, freeze_sha256=v._hash(freeze_path), complete=True)
            for region, values in [('ideal', (0.,)), ('in_domain', p.IN_DOMAIN),
                                   ('out_of_domain', p.OUT_DOMAIN['lag'])]:
                for parameter in values:
                    scenes = _scenes('lag', parameter, initials, region)
                    result = evaluate('lag', scenes, PIDController(parameters), keep_traces=True)
                    stem = f'04_独立测试/{region}_{parameter:.2f}'
                    trace_relative, data_relative = stem+'.npz', stem+'.json'
                    np.savez_compressed(study/trace_relative, **result['traces'])
                    v._write(study/data_relative, dict(scenes=scenes, rows=result['rows']))
                    manifest['entries'].append(dict(controller_id=descriptor['id'], kind='lag', region=region,
                                                    parameter_s=parameter, trace_path=trace_relative, data_path=data_relative))
                    for relative in [trace_relative, data_relative]:
                        manifest['artifacts'][relative] = v._hash(study/relative)
            manifest_path = test/'test_manifest.json'
            v._write(manifest_path, manifest)
            report = v.verify_all(freeze_path)
            self.assertTrue(report['passed'], report['errors'])
            self.assertEqual(report['shards_verified'], 9)
            self.assertEqual(report['episodes_verified'], 9*256)
            self.assertEqual(report['intervals_verified'], 9*256*440)
            self.assertEqual(report['freeze_sha256'], hashlib.sha256(freeze_path.read_bytes()).hexdigest())
            self.assertTrue((study/'05_独立核验'/'核验结果.json').exists())

            # Re-sealing changed metrics cannot defeat the independent audit.
            relative = manifest['entries'][0]['data_path']
            changed = v._read(study/relative)
            changed['rows'][0]['actual_cost'] += .1
            v._write(study/relative, changed)
            manifest['artifacts'][relative] = v._hash(study/relative)
            v._write(manifest_path, manifest)
            report = v.verify(freeze_path)
            self.assertFalse(report['passed'])
            self.assertEqual(report['shards_verified'], 9)
            self.assertTrue(any('failed shard' in error for error in report['errors']))

            # An unsealed byte change fails before trusting any trajectory.
            (study/relative).write_text('{}', encoding='utf-8')
            report = v.verify(freeze_path)
            self.assertFalse(report['passed'])
            self.assertEqual(report['shards_verified'], 0)
            self.assertTrue(any('hash mismatch' in error for error in report['errors']))

            manifest['freeze_sha256'] = '0'*64
            manifest['entries'] = manifest['entries'][:-1]
            v._write(manifest_path, manifest)
            report = v.verify(freeze_path)
            self.assertFalse(report['passed'])
            self.assertTrue(any('not bound' in error for error in report['errors']))
            self.assertTrue(any('coverage differs' in error for error in report['errors']))

    def test_artifact_paths_cannot_escape_study(self):
        with tempfile.TemporaryDirectory(prefix='actuator-path-check-') as temporary:
            with self.assertRaises(ValueError):
                v._path(Path(temporary), '../unrelated.txt')


if __name__ == '__main__':
    unittest.main()
