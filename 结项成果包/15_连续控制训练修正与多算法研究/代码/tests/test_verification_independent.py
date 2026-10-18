import numpy as np
import pytest
from continuous_study.baselines import PIDController, ModelCompensatedController
from continuous_study.evaluation import evaluate
from continuous_study.verification import verify_shard, _policy_features


def scenarios(kind):
    return [dict(kind=kind, parameter_s=t, split='validation', region='in_domain',
                 sample_id=i, scenario_id=f'check_{i}', angle_deg=a, omega_deg_s=w)
            for i, (t, a, w) in enumerate([(.05, 35., 3.), (.30, -50., -4.), (.15, .2, 0.)])]


@pytest.mark.parametrize('kind', ['lag', 'delay'])
def test_independent_full_response_and_mutated_cost(kind):
    parameters = dict(kp=.5, kd=1.1, ki=.001, integral_limit=2.)
    cases = scenarios(kind)
    result = evaluate(kind, cases, PIDController(parameters), keep_traces=True)
    entry = dict(family='pid', kind=kind, parameters=parameters)
    audit = verify_shard(kind, result['traces'], cases, result['rows'], entry)
    assert audit['passed'], audit['errors']
    result['rows'][0]['actual_cost'] += .001
    assert not verify_shard(kind, result['traces'], cases, result['rows'], entry)['passed']


@pytest.mark.parametrize('kind', ['lag', 'delay'])
def test_model_compensation_commands_are_independently_rebuilt(kind):
    parameters = dict(kp=.5, kd=1.1, ki=.001, integral_limit=2.,
                      nominal_parameter_s=.15, actuator_feedback=.5, observer_gain=.8)
    cases = scenarios(kind)
    result = evaluate(kind, cases, ModelCompensatedController(kind, parameters), keep_traces=True)
    entry = dict(family='model_compensated', kind=kind, parameters=parameters)
    audit = verify_shard(kind, result['traces'], cases, result['rows'], entry)
    assert audit['passed'], audit['errors']
    result['traces']['command'][5, 0] += .01
    assert not verify_shard(kind, result['traces'], cases, result['rows'], entry)['passed']


def test_new_policy_window_dimensions_and_absolute_time():
    traces = dict(theta=np.arange(51., dtype=float)[:, None]/100.,
                  omega=np.zeros((51, 1)), command=np.ones((50, 1))*.1)
    early = _policy_features(traces, 0, 'direct')
    late = _policy_features(traces, 30, 'residual', np.array([.2]), np.array([.3]))
    assert early.shape == (1, 40)
    assert late.shape == (1, 42)
    assert late[0, 38] == pytest.approx(30*.05/22.)
    assert late[0, 39] == 1.
    assert late[0, 40] == pytest.approx(.4)
    assert late[0, 41] == pytest.approx(.15)


@pytest.mark.parametrize('kind', ['lag', 'delay'])
def test_policy_history_after_physical_failure_keeps_issued_commands(kind):
    from continuous_study.learning import Controller

    class BadHistoryPolicy:
        def predict(self, features, deterministic=True):
            # Deliberately wrong continuous policy with a command-history
            # dependency: the audit must agree even after safety shutdown.
            return (.85+.10*features[:, 37]).reshape(-1, 1), None

    model = BadHistoryPolicy()
    cases = scenarios(kind)
    result = evaluate(kind, cases, Controller(model, 'direct'), keep_traces=True)
    assert all(row['failed'] for row in result['rows'])
    assert np.any(result['traces']['sent_command'] != result['traces']['command'])
    entry = dict(family='direct', kind=kind, algorithm='SAC')
    audited = verify_shard(kind, result['traces'], cases, result['rows'], entry, model)
    assert audited['passed'], audited['errors']
    # Shutdown zeros may never be substituted for issued command history.
    result['traces']['sent_command'] = result['traces']['command'].copy()
    assert not verify_shard(kind, result['traces'], cases, result['rows'], entry, model)['passed']
