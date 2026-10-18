"""Independent ODE, conservation and actuator-integral checks before fitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import physics as p


def scenario(**changes):
    value = dict(sample_id=0, euler_deg=[10., -8., 5.], omega_deg_s=[.5, -.2, .3],
                 phase_rad=[.1, 2.1, 4.2], frequency_rad_s=[.6, .8, 1.1],
                 amplitude_1_Nm=[.08, .06, .04], amplitude_2_Nm=[.02, .01, .015],
                 attitude_coefficient_Nm=[.006, .01, .005], rate_coefficient_Nm=[.004, .006, .005])
    value.update(changes)
    return value


def independent_derivative(time, state, data, command, start, torque0):
    # SciPy's rotation-vector implementation and explicit scalar quaternion
    # equations are independent of the production quaternion algebra.
    q, w = state[:4], state[4:7]
    theta = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
    phase = np.asarray(data['phase_rad'])+time*np.asarray(data['frequency_rad_s'])
    torque = command+(torque0-command)*np.exp(-(time-start)/.15)
    disturbance = (np.asarray(data['amplitude_1_Nm'])*np.sin(phase)
                   +np.asarray(data['amplitude_2_Nm'])*np.sin(2.*phase+.4)
                   +np.asarray(data['attitude_coefficient_Nm'])*np.sin(2.*theta)
                   +np.asarray(data['rate_coefficient_Nm'])*np.tanh(w/.05))
    s, x, y, z = q
    a, b, c = w
    qdot = .5*np.array([-x*a-y*b-z*c, s*a+y*c-z*b, s*b+z*a-x*c, s*c+x*b-y*a])
    ix, iy, iz = [1.2, 1., .8]
    total = torque+disturbance
    wdot = np.array([(total[0]+(iy-iz)*b*c)/ix, (total[1]+(iz-ix)*c*a)/iy,
                     (total[2]+(ix-iy)*a*b)/iz])
    return np.r_[qdot, wdot, np.sum(torque**2)]


def test_quaternion_conventions_and_sign_equivalence():
    angle = np.deg2rad(np.array([[20., -15., 32.], [-170., 2., -40.]]))
    q = p.from_euler(angle)
    expected = Rotation.from_euler('xyz', angle).as_quat()[:, [3, 0, 1, 2]]
    np.testing.assert_allclose(q, expected, atol=3e-16)
    np.testing.assert_allclose(p.rotation_vector(q), p.rotation_vector(-q), atol=1e-15)
    np.testing.assert_allclose(p.quaternion_angle(q), np.linalg.norm(p.rotation_vector(q), axis=1), atol=1e-15)
    vector = np.array([[1., 2., 3.], [2., 1., -1.]])
    np.testing.assert_allclose(p.rotate(q, vector), Rotation.from_quat(q[:, [1, 2, 3, 0]]).apply(vector), atol=2e-15)


def test_entire_32_second_open_loop_replay_against_independent_dop853(monkeypatch):
    data = scenario()
    plant = p.Plant([data])
    independent = np.r_[plant.quaternion[0], plant.omega[0], 0.]
    independent_torque = np.zeros(3)
    largest = 0.
    cost = 0.
    commands = []
    for k in range(640):
        command = np.clip(-np.array([30., 25., 20.])*plant.theta[0]
                          -np.array([14., 12., 10.])*plant.omega[0], -.5, .5)
        commands.append(command.copy())
        start = k*.05
        result = solve_ivp(independent_derivative, (start, start+.05), independent,
                           args=(data, command, start, independent_torque), method='DOP853',
                           rtol=2e-12, atol=2e-14)
        assert result.success
        independent = result.y[:, -1]
        independent_torque = command+(independent_torque-command)*np.exp(-.05/.15)
        interval = plant.step(command[None, :])
        cost += interval['actual_cost'][0]
        state = np.r_[plant.quaternion[0], plant.omega[0]]
        largest = max(largest, float(np.max(np.abs(state-independent[:7]))))
        assert not plant.failed[0]
    # The agreed trajectory-audit threshold is 2e-7 rad, about 0.0000115
    # degrees; the actual open-loop error is reported by this assertion.
    assert largest < 2e-7, largest
    np.testing.assert_allclose(cost, independent[7], rtol=1e-10, atol=1e-12)
    monkeypatch.setattr(p, 'SUBSTEPS', 16)
    refined = p.Plant([data])
    for command in commands:
        refined.step(command[None, :])
    finer_error = np.max(np.abs(np.r_[refined.quaternion[0], refined.omega[0]]-independent[:7]))
    final_error = np.max(np.abs(state-independent[:7]))
    assert finer_error < final_error/5., (final_error, finer_error)


def test_torque_free_energy_and_inertial_angular_momentum():
    data = scenario(omega_deg_s=[8., -6., 7.], amplitude_1_Nm=[0.]*3,
                    amplitude_2_Nm=[0.]*3, attitude_coefficient_Nm=[0.]*3,
                    rate_coefficient_Nm=[0.]*3, euler_deg=[0., 0., 0.])
    plant = p.Plant([data])
    h0 = p.rotate(plant.quaternion, p.INERTIA*plant.omega)
    energy = .5*np.sum(p.INERTIA*plant.omega**2)
    # Four seconds stays within the declared 90-degree linear pointing domain.
    for k in range(80):
        plant.step(np.zeros((1, 3)))
    np.testing.assert_allclose(p.rotate(plant.quaternion, p.INERTIA*plant.omega), h0, atol=3e-13)
    np.testing.assert_allclose(.5*np.sum(p.INERTIA*plant.omega**2), energy, atol=1e-14)


def test_absorbing_failure_clock_and_mask_reset():
    data = [scenario(), scenario(sample_id=1)]
    plant = p.Plant(data)
    plant.failed[0] = True
    before = np.r_[plant.quaternion[0], plant.omega[0]]
    for k in range(3):
        result = plant.step(np.full((2, 3), .1))
        assert result['actual_cost'][0] == 0.
        assert np.all(result['command'][0] == 0.)
    np.testing.assert_array_equal(np.r_[plant.quaternion[0], plant.omega[0]], before)
    assert plant.count[0] == 3
    preserved = plant.quaternion[1].copy()
    plant.reset([scenario(euler_deg=[1., 2., 3.])], np.array([True, False]))
    assert not plant.failed[0] and plant.count[0] == 0 and plant.count[1] == 3
    np.testing.assert_array_equal(plant.quaternion[1], preserved)


def test_measured_variables_do_not_reveal_disturbance_amplitudes():
    first = p.Plant([scenario()])
    second = p.Plant([scenario(amplitude_1_Nm=[.2, .1, .3])])
    np.testing.assert_array_equal(first.measured(), second.measured())
    assert first.measured().shape == (1, 15)
