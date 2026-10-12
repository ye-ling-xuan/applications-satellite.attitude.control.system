"""Quaternion three-axis PID demonstration based on the preserved week5 model.

Rigid-body dynamics include gyroscopic coupling. Internal units are rad/s and
N m; figures and CSV use degrees. This is a PID extension, not three-axis RL.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def finite_vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain {size} finite values')
    return result.copy()


def normalize(q):
    q = finite_vector(q, 4, 'quaternion')
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('zero quaternion is invalid')
    return q / norm


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def conjugate(q):
    return q * np.array([1., -1., -1., -1.])


def rotation_quaternion(rotation):
    rotation = finite_vector(rotation, 3, 'rotation')
    angle = np.linalg.norm(rotation)
    if angle < 1e-12:
        return normalize(np.concatenate(([1.], rotation / 2)))
    return np.concatenate(([np.cos(angle / 2)], rotation * np.sin(angle / 2) / angle))


def euler_quaternion(degrees):
    roll, pitch, yaw = np.deg2rad(finite_vector(degrees, 3, 'euler_degrees')) / 2
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return normalize(np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                               cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]))


def euler_degrees(q):
    w, x, y, z = normalize(q)
    return np.rad2deg([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                      np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                      np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def attitude_error(target, current):
    """Shortest rotation in the body frame; q and -q give equal commands.

    q maps body to inertial coordinates and qdot = 0.5*q*(0, omega).
    Thus conjugate(current)*target is the error needed by body-frame torque.
    A deterministic axis sign also resolves the exact 180-degree tie.
    """
    error = normalize(multiply(conjugate(normalize(current)), normalize(target)))
    if error[0] < -1e-12:
        error = -error
    elif abs(error[0]) <= 1e-12:
        first = next((v for v in error[1:] if abs(v) > 1e-12), 1.)
        if first < 0:
            error = -error
        error[0] = 0.
    sine = np.linalg.norm(error[1:])
    if sine < 1e-12:
        return np.zeros(3)
    angle = 2 * np.arctan2(sine, max(error[0], 0.))
    return error[1:] * angle / sine


class Satellite3D:
    def __init__(self, inertia=(1.2, 1.0, 0.8)):
        self.inertia = finite_vector(inertia, 3, 'inertia')
        if np.any(self.inertia <= 0):
            raise ValueError('inertia must be positive')
        self.reset()

    def reset(self, q=(1., 0., 0., 0.), omega=(0., 0., 0.)):
        self.q = normalize(q)
        self.omega = finite_vector(omega, 3, 'omega')

    def derivative(self, state, torque):
        q, omega = state[:4], state[4:]
        qdot = 0.5 * multiply(q, np.concatenate(([0.], omega)))
        wdot = (torque - np.cross(omega, self.inertia * omega)) / self.inertia
        return np.concatenate((qdot, wdot))

    def step(self, torque, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        torque = finite_vector(torque, 3, 'torque')
        state = np.concatenate((self.q, self.omega))
        k1 = self.derivative(state, torque)
        k2 = self.derivative(state + dt*k1/2, torque)
        k3 = self.derivative(state + dt*k2/2, torque)
        k4 = self.derivative(state + dt*k3, torque)
        result = state + dt*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('non-finite rigid-body state')
        self.q, self.omega = normalize(result[:4]), result[4:].copy()


class PID3D:
    """Geometric attitude PID with measured-rate damping and anti-windup."""
    def __init__(self, dt=.02, kp=(1.8, 1.5, 1.2), ki=(.025, .020, .015),
                 kd=(3.1, 2.6, 2.1), output_limit=.2, integral_limit=.25):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.dt = dt
        self.kp = finite_vector(kp, 3, 'kp')
        self.ki = finite_vector(ki, 3, 'ki')
        self.kd = finite_vector(kd, 3, 'kd')
        self.limit = np.full(3, output_limit, dtype=float)
        self.integral_limit = np.full(3, integral_limit, dtype=float)
        if np.any(self.limit <= 0) or np.any(self.integral_limit < 0):
            raise ValueError('invalid controller limits')
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)

    def compute(self, error, measured_omega):
        error = finite_vector(error, 3, 'error')
        measured_omega = finite_vector(measured_omega, 3, 'measured_omega')
        candidate = np.clip(self.integral + error*self.dt, -self.integral_limit, self.integral_limit)
        unsaturated = self.kp*error + self.ki*candidate - self.kd*measured_omega
        saturated = np.abs(unsaturated) > self.limit
        driving_saturation = saturated & (np.sign(unsaturated) == np.sign(error))
        self.integral = np.where(driving_saturation, self.integral, candidate)
        command = self.kp*error + self.ki*self.integral - self.kd*measured_omega
        return command, np.clip(command, -self.limit, self.limit)


@dataclass(frozen=True)
class Config:
    dt: float = .02
    duration: float = 20.
    seed: int = 20261009
    inertia_kg_m2: tuple = (1.2, 1., .8)
    initial_euler_deg: tuple = (30., -20., 45.)
    initial_omega_deg_s: tuple = (2., -1., 1.5)
    kp: tuple = (1.8, 1.5, 1.2)
    ki: tuple = (.025, .020, .015)
    kd: tuple = (3.1, 2.6, 2.1)
    torque_limit_Nm: float = .2
    integral_limit_rad_s: float = .25
    deadzone_Nm: float = .0005
    attitude_noise_std_deg: float = .10
    rate_noise_std_deg_s: float = .02
    attitude_threshold_deg: float = 1.
    rate_threshold_deg_s: float = .5
    hold_window_s: float = 5.


def disturbance(t):
    bias = np.array([.002, -.003, .002])
    periodic = .006 * np.sin(2*np.pi*np.array([.13, .17, .11])*t + [0., .7, 1.1])
    pulse = np.array([.060, -.040, .050]) if 8. <= t < 10. else np.zeros(3)
    return bias + periodic + pulse


def simulate(config=Config(), q_initial=None, disturbances=True, noise=True):
    if config.duration < config.hold_window_s or config.hold_window_s <= 0:
        raise ValueError('duration must cover the positive hold window')
    ratio = config.duration / config.dt
    if not np.isfinite(ratio) or abs(ratio-round(ratio)) > 1e-9:
        raise ValueError('duration must be an integer multiple of dt')
    count = int(round(ratio))
    sat = Satellite3D(config.inertia_kg_m2)
    initial = euler_quaternion(config.initial_euler_deg) if q_initial is None else q_initial
    sat.reset(initial, np.deg2rad(config.initial_omega_deg_s))
    pid = PID3D(config.dt, config.kp, config.ki, config.kd,
                config.torque_limit_Nm, config.integral_limit_rad_s)
    pid.reset()
    rng = np.random.default_rng(config.seed)
    target = np.array([1., 0., 0., 0.])
    time = np.arange(count+1)*config.dt
    q = np.zeros((count+1, 4))
    omega, command, output, dist = [np.zeros((count+1, 3)) for _ in range(4)]
    error, integral = np.zeros(count+1), np.zeros((count+1, 3))
    for index, t in enumerate(time):
        q[index], omega[index] = sat.q, sat.omega
        error[index] = np.rad2deg(np.linalg.norm(attitude_error(target, sat.q)))
        integral[index] = pid.integral
        if index == count:
            break
        measured_q, measured_omega = sat.q.copy(), sat.omega.copy()
        if noise:
            attitude_noise = np.deg2rad(rng.normal(0., config.attitude_noise_std_deg, 3))
            measured_q = normalize(multiply(measured_q, rotation_quaternion(attitude_noise)))
            measured_omega += np.deg2rad(rng.normal(0., config.rate_noise_std_deg_s, 3))
        command[index], output[index] = pid.compute(attitude_error(target, measured_q), measured_omega)
        output[index, np.abs(output[index]) < config.deadzone_Nm] = 0.
        dist[index] = disturbance(t) if disturbances else np.zeros(3)
        sat.step(output[index] + dist[index], config.dt)
    all_finite = bool(all(np.all(np.isfinite(v)) for v in [q, omega, command, output, dist, error, integral]))
    omega_norm = np.rad2deg(np.linalg.norm(omega, axis=1))
    within = (error <= config.attitude_threshold_deg) & (omega_norm <= config.rate_threshold_deg_s)
    last_window = time >= config.duration-config.hold_window_s-1e-9
    success = bool(all_finite and np.all(within[last_window]))
    stable_suffix = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(stable_suffix & (time <= config.duration-config.hold_window_s+1e-9))
    metrics = {
        'config': asdict(config), 'controller': 'three_axis_geometric_PID',
        'scope': 'three-axis PID extension demonstration; not three-axis reinforcement learning',
        'success_definition': f'Every true state sample in the final {config.hold_window_s:g} s has geodesic attitude error <= {config.attitude_threshold_deg:g} degree and angular-rate norm <= {config.rate_threshold_deg_s:g} degree/s.',
        'integration': 'RK4 with per-step quaternion normalization; torque held during each interval',
        'disturbance': 'bias [0.002,-0.003,0.002] Nm + 0.006 Nm sinusoidal components; [0.060,-0.040,0.050] Nm pulse during 8-10 s' if disturbances else 'none',
        'noise_enabled': noise, 'all_states_and_controls_finite': all_finite,
        'success_last_5s': success,
        'settling_time_s': float(time[candidates[0]]) if candidates.size else None,
        'final_attitude_error_deg': float(error[-1]),
        'max_attitude_error_last_5s_deg': float(np.max(error[last_window])),
        'final_rate_norm_deg_s': float(omega_norm[-1]),
        'max_rate_norm_last_5s_deg_s': float(np.max(omega_norm[last_window])),
        'max_quaternion_norm_error': float(np.max(np.abs(np.linalg.norm(q, axis=1)-1))),
        'max_output_torque_Nm': float(np.max(np.abs(output))),
        'max_integral_abs_rad_s': float(np.max(np.abs(integral))),
        'control_cost_N2m2_s': float(np.sum(output[:-1]**2)*config.dt),
        'saturation_interval_fraction': float(np.mean(np.any(np.abs(command[:-1]) > config.torque_limit_Nm, axis=1))),
        'note': 'One deterministic demonstration validates implementation only; it is not a statistical robustness claim.',
    }
    return {'time': time, 'quaternion': q, 'euler_deg': np.array([euler_degrees(v) for v in q]),
            'omega_deg_s': np.rad2deg(omega), 'command_Nm': command, 'output_Nm': output,
            'disturbance_Nm': dist, 'error_deg': error, 'integral_rad_s': integral}, metrics


def regression_checks():
    config = Config()
    initial = euler_quaternion(config.initial_euler_deg)
    plus, metrics = simulate(config, initial)
    minus, _ = simulate(config, -initial)
    state_error = float(np.max(np.abs(plus['quaternion'] + minus['quaternion'])))
    control_error = float(np.max(np.abs(plus['output_Nm'] - minus['output_Nm'])))
    pi_plus = attitude_error([1., 0., 0., 0.], [0., 1., 0., 0.])
    pi_minus = attitude_error([1., 0., 0., 0.], [0., -1., 0., 0.])
    zero = Config(initial_euler_deg=(0., 0., 0.), initial_omega_deg_s=(0., 0., 0.))
    trace, zero_metrics = simulate(zero, disturbances=False, noise=False)
    pid = PID3D()
    pid.compute([.1, -.2, .3], [.01, .02, .03])
    pid.reset()
    command, output = pid.compute([0., 0., 0.], [0., 0., 0.])
    invalid_rejected = False
    try:
        normalize([0., 0., 0., 0.])
    except ValueError:
        invalid_rejected = True
    results = {
        'q_sign_equivalent': state_error < 1e-11 and control_error < 1e-11,
        'q_sign_max_opposite_state_difference': state_error,
        'q_sign_max_control_difference': control_error,
        'exact_180_degree_sign_equivalent': bool(np.allclose(pi_plus, pi_minus, atol=1e-12)),
        'zero_error_stays_at_rest': bool(np.max(np.abs(trace['output_Nm'])) == 0 and np.max(trace['error_deg']) == 0 and zero_metrics['success_last_5s']),
        'controller_reset_clears_integral': bool(np.all(pid.integral == 0) and np.all(command == 0) and np.all(output == 0)),
        'zero_quaternion_rejected': invalid_rejected,
        'quaternion_norm_preserved': metrics['max_quaternion_norm_error'] < 1e-12,
        'bounded_finite_output': bool(metrics['all_states_and_controls_finite'] and metrics['max_output_torque_Nm'] <= config.torque_limit_Nm+1e-12 and metrics['max_integral_abs_rad_s'] <= config.integral_limit_rad_s+1e-12),
    }
    passed = all(results[key] for key in ['q_sign_equivalent', 'exact_180_degree_sign_equivalent', 'zero_error_stays_at_rest', 'controller_reset_clears_integral', 'zero_quaternion_rejected', 'quaternion_norm_preserved', 'bounded_finite_output'])
    results['all_passed'] = bool(passed)
    if not passed:
        raise AssertionError(json.dumps(results, indent=2))
    return results


def save_outputs(trace, metrics, directory):
    directory.mkdir(parents=True, exist_ok=True)
    header = ['time_s', 'interval_starts_here', 'qw', 'qx', 'qy', 'qz', 'roll_deg', 'pitch_deg', 'yaw_deg',
              'omega_x_deg_s', 'omega_y_deg_s', 'omega_z_deg_s', 'attitude_error_deg',
              'command_x_Nm', 'command_y_Nm', 'command_z_Nm', 'output_x_Nm', 'output_y_Nm', 'output_z_Nm',
              'disturbance_x_Nm', 'disturbance_y_Nm', 'disturbance_z_Nm', 'integral_x_rad_s', 'integral_y_rad_s', 'integral_z_rad_s']
    with (directory / 'trace.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, t in enumerate(trace['time']):
            row = [t, int(index < len(trace['time'])-1), *trace['quaternion'][index], *trace['euler_deg'][index],
                   *trace['omega_deg_s'][index], trace['error_deg'][index], *trace['command_Nm'][index],
                   *trace['output_Nm'][index], *trace['disturbance_Nm'][index], *trace['integral_rad_s'][index]]
            writer.writerow(row)
    (directory / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, label in enumerate(['x / roll', 'y / pitch', 'z / yaw']):
        axes[0, 0].plot(trace['time'], trace['euler_deg'][:, axis], label=label)
        axes[0, 1].plot(trace['time'], trace['omega_deg_s'][:, axis], label=label)
        axes[1, 0].plot(trace['time'][:-1], trace['output_Nm'][:-1, axis], label=label)
    axes[0, 0].set_ylabel('Euler angle [deg]')
    axes[0, 1].set_ylabel('Body angular rate [deg/s]')
    axes[1, 0].set_ylabel('Applied actuator torque [N m]')
    axes[1, 1].plot(trace['time'], trace['error_deg'], label='Geodesic attitude error')
    axes[1, 1].axhline(metrics['config']['attitude_threshold_deg'], color='black', linestyle='--', label='1 deg threshold')
    axes[1, 1].set_ylabel('Attitude error [deg]')
    for axis in axes.flat:
        axis.set_xlabel('Time [s]')
        axis.axvspan(8, 10, alpha=.13, color='orange', label='Disturbance pulse' if axis is axes[1, 1] else None)
        axis.axvspan(15, 20, alpha=.08, color='green', label='Final 5 s' if axis is axes[1, 1] else None)
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    fig.suptitle('Three-axis quaternion PID: coupled dynamics, noise and torque limits')
    fig.savefig(directory / '3axis_response.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--check', action='store_true', help='Run q/-q, zero-error and reset regression checks')
    arguments = parser.parse_args()
    trace, metrics = simulate(Config(seed=arguments.seed))
    save_outputs(trace, metrics, arguments.output)
    if arguments.check:
        checks = regression_checks()
        (arguments.output / 'regression_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
