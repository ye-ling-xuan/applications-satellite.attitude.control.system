"""Check a physical equilibrium consequence of policy offset compensation."""
import numpy as np

from multi_algorithm_benchmark.controllers import ModelController
from multi_algorithm_benchmark.improvements import CalibratedModelController
from multi_algorithm_benchmark.metrics import evaluate_batch


class BiasedStablePolicy:
    def predict(self, observations, deterministic=True):
        obs = np.asarray(observations)
        action = -4. * obs[:, :1] - 3. * obs[:, 1:2] + .1
        return np.clip(action, -1., 1.), None


def test_offset_compensation_restores_nominal_closed_loop_equilibrium():
    scene = [{"theta_deg": 35., "omega_deg_s": 2., "noise_seed": 17}]
    model = BiasedStablePolicy()
    raw = evaluate_batch("single", scene, ModelController(model, "single"))[0]
    calibrated = evaluate_batch("single", scene, CalibratedModelController(model, "single"))[0]
    # Raw equilibrium: normalized action .1/(4 rad^-1) = .025 rad (1.43 deg).
    assert abs(raw["metrics"]["final_error_deg"] - np.degrees(.025)) < .005
    assert not raw["metrics"]["success"]
    assert calibrated["metrics"]["success"]
    assert calibrated["metrics"]["tail_mean_error_deg"] < .01
    assert np.max(np.abs(calibrated["trace"]["torque"])) <= .5
