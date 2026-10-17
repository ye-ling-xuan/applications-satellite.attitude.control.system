"""Exact sampled single-axis dynamics for lag and pure input delay.

Angles and rates are radians, commands and torques are N m, and time is seconds.
Commands are constant over each sample interval. Failure is checked at sampled
endpoints. A failed trajectory is physically frozen after its failing interval;
later calls only advance its recorded clock to the declared horizon.
"""
from __future__ import annotations

import math
import numpy as np

from .protocol import DT, DURATION, FAIL_ANGLE, FAIL_RATE, INERTIA, LIMIT


def _lag_coefficients(r):
    """Stable dimensionless integrals for r = interval / time constant."""
    r = np.asarray(r, dtype=float)
    small = r < .05
    a = np.empty_like(r)       # mean(exp(-r*x)), x in [0, 1]
    b = np.empty_like(r)       # integral((1-x)*exp(-r*x), x in [0, 1])
    c1 = np.empty_like(r)      # mean(1-exp(-r*x))
    c2 = np.empty_like(r)      # integral((1-x)*(1-exp(-r*x)))
    variance = np.empty_like(r)

    z = r[small]
    # Expansions avoid subtracting almost equal numbers for very slow actuators.
    p1 = np.array([1/2, -1/6, 1/24, -1/120, 1/720, -1/5040,
                   1/40320, -1/362880, 1/3628800])
    p2 = np.array([1/6, -1/24, 1/120, -1/720, 1/5040, -1/40320,
                   1/362880, -1/3628800, 1/39916800])
    pq = np.array([1/3, -1/4, 7/60, -1/24, 31/2520, -1/320,
                   127/181440, -17/120960, 511/19958400])
    c1[small] = z * np.polynomial.polynomial.polyval(z, p1)
    c2[small] = z * np.polynomial.polynomial.polyval(z, p2)
    q = z*z * np.polynomial.polynomial.polyval(z, pq)
    variance[small] = np.maximum(q-c1[small]**2, 0.)
    a[small] = 1.-c1[small]
    b[small] = .5-c2[small]

    z = r[~small]
    a[~small] = -np.expm1(-z)/z
    b[~small] = (1.-a[~small])/z
    c1[~small] = 1.-a[~small]
    c2[~small] = .5-b[~small]
    c = -np.expm1(-2.*z)/(2.*z)
    variance[~small] = np.maximum(c-a[~small]**2, 0.)
    return a, b, c1, c2, variance


def ideal_interval(theta, omega, command, interval, inertia=INERTIA):
    """Exact double-integrator state and torque-square integral for one hold."""
    theta, omega, command, interval = np.broadcast_arrays(
        np.asarray(theta, dtype=float), np.asarray(omega, dtype=float),
        np.asarray(command, dtype=float), np.asarray(interval, dtype=float))
    return (np.asarray(theta+omega*interval+.5*command*interval**2/inertia),
            np.asarray(omega+command*interval/inertia),
            np.asarray(command**2*interval))


def lag_interval(theta, omega, tau_start, command, parameter_s, interval,
                 inertia=INERTIA):
    """Exact joint state/actuator propagation, including the T=0 ideal branch.

    Returns (theta_end, omega_end, tau_end, integral(tau(t)**2 dt)). The energy
    integral uses mean-square plus variance, which remains nonnegative even
    when a command reversal causes the torque to cross zero within an interval.
    """
    theta, omega, tau_start, command, parameter_s, interval = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in
          (theta, omega, tau_start, command, parameter_s, interval)])
    if (not all(np.all(np.isfinite(value)) for value in
                (theta, omega, tau_start, command, parameter_s, interval))
            or np.any(parameter_s < 0.) or np.any(interval < 0.)
            or not np.isfinite(inertia) or inertia <= 0.):
        raise ValueError('lag integration requires finite states and nonnegative times')
    theta_end, omega_end, cost = ideal_interval(theta, omega, command, interval, inertia)
    tau_end = command.copy()
    dynamic = (parameter_s > 0.) & (interval > 0.)
    if np.any(dynamic):
        t = interval[dynamic]
        old = tau_start[dynamic]
        u = command[dynamic]
        with np.errstate(over='ignore', divide='ignore'):
            r = t/parameter_s[dynamic]
            a, b, c1, c2, variance = _lag_coefficients(r)
        delta = u-old
        small = r < .05
        mean = np.where(small, old+delta*c1, u-delta*a)
        angle_integral = np.where(small, .5*old+delta*c2, .5*u-delta*b)
        theta_end[dynamic] = theta[dynamic]+omega[dynamic]*t+t*t*angle_integral/inertia
        omega_end[dynamic] = omega[dynamic]+t*mean/inertia
        tau_end[dynamic] = np.where(small, old+delta*(-np.expm1(-r)),
                                   u-delta*np.exp(-r))
        cost[dynamic] = t*(mean**2+delta**2*variance)
    zero_interval = interval == 0.
    tau_end[zero_interval] = tau_start[zero_interval]
    return theta_end, omega_end, tau_end, cost


class BatchPlant:
    """Vectorized independent plants with fixed parameters within each episode.

    ``reset`` accepts scalars, full-size arrays, or selected-row arrays when a
    boolean mask is supplied. Actual actuator state, delay history and clocks
    are reset only in selected rows. Observations expose only [-theta, omega].
    """

    def __init__(self, kind, n, dt=DT, inertia=INERTIA, limit=LIMIT,
                 duration=DURATION):
        if kind not in ('lag', 'delay'):
            raise ValueError("kind must be 'lag' or 'delay'")
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError('n must be a positive integer')
        for name, value in (('dt', dt), ('inertia', inertia), ('limit', limit),
                            ('duration', duration)):
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0.:
                raise ValueError(name+' must be finite and positive')
        self.kind, self.n = kind, int(n)
        self.dt, self.inertia = float(dt), float(inertia)
        self.limit, self.duration = float(limit), float(duration)
        ratio = self.duration/self.dt
        if not np.isfinite(ratio) or ratio > np.iinfo(np.int64).max:
            raise ValueError('duration/dt exceeds the supported clock range')
        self._max_steps = max(1, int(math.ceil(ratio-1e-12)))
        self.theta = np.zeros(self.n)
        self.omega = np.zeros(self.n)
        self.actual_torque = np.zeros(self.n)
        self.failed = np.zeros(self.n, dtype=bool)
        self.step_count = np.zeros(self.n, dtype=np.int64)
        self.elapsed = np.zeros(self.n)
        self._parameter_s = np.zeros(self.n)
        self._delay_steps = np.zeros(self.n, dtype=np.int64)
        # Column 0 is the preceding command, column m-1 the command m holds ago.
        self._history = np.zeros((self.n, 0))

    @property
    def parameter_s(self):
        value = self._parameter_s.copy()
        value.setflags(write=False)
        return value

    def _reset_values(self, value, mask, name):
        value = np.asarray(value, dtype=float)
        count = int(np.count_nonzero(mask))
        if value.ndim == 0:
            selected = np.full(count, float(value))
        elif value.shape == (self.n,):
            selected = value[mask].copy()
        elif value.shape == (count,):
            selected = value.copy()
        else:
            raise ValueError(name+' must be scalar, length n, or length selected rows')
        if not np.all(np.isfinite(selected)):
            raise ValueError(name+' must be finite')
        return selected

    def reset(self, theta_rad, omega_rad_s, parameter_s, mask=None):
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        else:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != (self.n,):
                raise ValueError('mask must be a boolean array of length n')
        theta = self._reset_values(theta_rad, mask, 'theta_rad')
        omega = self._reset_values(omega_rad_s, mask, 'omega_rad_s')
        parameter = self._reset_values(parameter_s, mask, 'parameter_s')
        if np.any(parameter < 0.):
            raise ValueError('parameter_s must be nonnegative')
        delays = None
        if self.kind == 'delay':
            ratios = parameter/self.dt
            rounded = np.rint(ratios)
            if (not np.all(np.isfinite(ratios))
                    or np.any(rounded >= np.iinfo(np.int64).max)
                    or not np.all(np.isclose(ratios, rounded, rtol=0., atol=1e-9))):
                raise ValueError('pure delay must be a nonnegative integer multiple of dt')
            delays = rounded.astype(np.int64)
            required = int(np.max(delays, initial=0))
            if required > self._history.shape[1]:
                extra = np.zeros((self.n, required-self._history.shape[1]))
                self._history = np.concatenate((self._history, extra), axis=1)
        self.theta[mask], self.omega[mask] = theta, omega
        self._parameter_s[mask] = parameter
        self.actual_torque[mask] = 0.
        self.failed[mask] = False
        self.step_count[mask] = 0
        self.elapsed[mask] = 0.
        self._history[mask] = 0.
        if delays is not None:
            self._delay_steps[mask] = delays
        return self.observation()

    def observation(self):
        return np.column_stack((-self.theta, self.omega))

    def step(self, commands_Nm):
        commands = np.asarray(commands_Nm, dtype=float)
        if commands.shape != (self.n,) or not np.all(np.isfinite(commands)):
            raise ValueError('commands_Nm must be a finite array of length n')
        commands = np.clip(commands, -self.limit, self.limit)
        ticking = self.step_count < self._max_steps
        active = ticking & ~self.failed
        commands = np.where(active, commands, 0.)
        rows = np.flatnonzero(active)
        tau_start = np.zeros(self.n)
        tau_end = np.zeros(self.n)
        actual_cost = np.zeros(self.n)
        command_cost = np.zeros(self.n)
        new_failure = np.zeros(self.n, dtype=bool)
        interval = np.minimum(self.dt, self.duration-self.elapsed[active])
        if rows.size:
            u = commands[active]
            command_cost[active] = u*u*interval
            if self.kind == 'lag':
                tau_start[active] = np.where(self._parameter_s[active] == 0.,
                                              u, self.actual_torque[active])
                theta, omega, torque, cost = lag_interval(
                    self.theta[active], self.omega[active], self.actual_torque[active],
                    u, self._parameter_s[active], interval, self.inertia)
            else:
                delays = self._delay_steps[active]
                torque = u.copy()
                delayed = delays > 0
                torque[delayed] = self._history[rows[delayed], delays[delayed]-1]
                tau_start[active] = torque
                theta, omega, cost = ideal_interval(
                    self.theta[active], self.omega[active], torque, interval, self.inertia)
                if self._history.shape[1]:
                    self._history[rows, 1:] = self._history[rows, :-1]
                    self._history[rows, 0] = u
            self.theta[active], self.omega[active] = theta, omega
            tau_end[active], actual_cost[active] = torque, cost
            new_failure[active] = ((np.abs(theta) > np.deg2rad(FAIL_ANGLE))
                                    | (np.abs(omega) > np.deg2rad(FAIL_RATE)))
        self.actual_torque[:] = tau_end
        self.failed |= new_failure
        self.step_count[ticking] += 1
        # Multiplying the integer counter prevents cumulative floating point drift.
        self.elapsed[ticking] = np.minimum(
            self.step_count[ticking]*self.dt, self.duration)
        self.elapsed[self.step_count >= self._max_steps] = self.duration
        done = self.failed | (self.step_count >= self._max_steps)
        return dict(command=commands, tau_start=tau_start, tau_end=tau_end,
                    actual_cost=actual_cost, command_cost=command_cost,
                    new_failure=new_failure, done=done.copy())
