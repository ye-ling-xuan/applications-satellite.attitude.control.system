"""Coupled Euler rigid-body dynamics with quaternion attitude and actuator lag.

Quaternions are scalar first and rotate body coordinates into world coordinates.
All torques and angular velocities are expressed in the body frame. The actuator
is propagated analytically; RK4 only integrates quaternion and angular velocity.
This removes the stiff actuator integration error from numerical comparisons.
"""
from __future__ import annotations
import numpy as np

DT = .05
DURATION = 32.
DEADLINE = 8.
LAG = .15
INERTIA = np.array([1.2, 1., .8])
LIMIT = .5
STEPS = 640
SUBSTEPS = 8


def normalize(quaternion):
    q = np.asarray(quaternion, float)
    return q/np.linalg.norm(q, axis=-1, keepdims=True)


def multiply(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    scalar = a[..., :1]*b[..., :1]-np.sum(a[..., 1:]*b[..., 1:], axis=-1, keepdims=True)
    vector = a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((scalar, vector), axis=-1)


def from_euler(euler):
    """Body-to-world Rz(yaw) Ry(pitch) Rx(roll), radians."""
    r, p, y = (np.asarray(euler, float)/2.).T
    cr, cp, cy, sr, sp, sy = np.cos(r), np.cos(p), np.cos(y), np.sin(r), np.sin(p), np.sin(y)
    return np.column_stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def rotation_vector(quaternion):
    q = normalize(quaternion)
    q = np.where((q[..., :1] < 0.), -q, q)
    vector = q[..., 1:]
    length = np.linalg.norm(vector, axis=-1)
    angle = 2.*np.arctan2(length, np.maximum(q[..., 0], 0.))
    factor = np.divide(angle, length, out=np.full_like(length, 2.), where=length > 1e-12)
    return vector*factor[..., None]


def quaternion_angle(quaternion):
    q = normalize(quaternion)
    return 2.*np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0]))


def rotate(quaternion, vectors):
    q = normalize(quaternion)
    vector = np.asarray(vectors, float)
    cross = 2.*np.cross(q[..., 1:], vector)
    return vector+q[..., :1]*cross+np.cross(q[..., 1:], cross)


class Plant:
    def __init__(self, data):
        self.n = len(data)
        if not self.n:
            raise ValueError('nonempty scenarios required')
        for name, width in [('quaternion', 4), ('omega', 3), ('tau', 3), ('phase', 3),
                            ('frequency', 3), ('a1', 3), ('a2', 3), ('ctheta', 3), ('comega', 3)]:
            setattr(self, name, np.zeros((self.n, width)))
        self.count = np.zeros(self.n, dtype=np.int64)
        self.failed = np.zeros(self.n, dtype=bool)
        self.reset(data)

    def reset(self, data, mask=None):
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        else:
            mask = np.asarray(mask, bool)
        if mask.shape != (self.n,):
            raise ValueError('reset mask shape mismatch')
        count = int(mask.sum())
        selected = data if len(data) == count else [s for s, chosen in zip(data, mask) if chosen]
        if len(selected) != count:
            raise ValueError('reset data coverage mismatch')
        if not count:
            return
        def field(key):
            value = np.asarray([s[key] for s in selected], float)
            if value.shape != (count, 3) or not np.all(np.isfinite(value)):
                raise ValueError('scenario must have finite 3-vector '+key)
            return value
        self.quaternion[mask] = from_euler(np.deg2rad(field('euler_deg')))
        self.omega[mask] = np.deg2rad(field('omega_deg_s'))
        self.phase[mask] = field('phase_rad')
        for attribute, key in [('frequency', 'frequency_rad_s'), ('a1', 'amplitude_1_Nm'),
                               ('a2', 'amplitude_2_Nm'), ('ctheta', 'attitude_coefficient_Nm'),
                               ('comega', 'rate_coefficient_Nm')]:
            getattr(self, attribute)[mask] = field(key)
        self.tau[mask] = 0.
        self.count[mask] = 0
        self.failed[mask] = False

    @property
    def theta(self):
        return rotation_vector(self.quaternion)

    @property
    def elapsed(self):
        return self.count*DT

    def measured(self):
        return np.column_stack((-self.theta, self.omega, np.sin(self.phase),
                                np.cos(self.phase), self.frequency))

    def disturbance(self, quaternion, omega, phase):
        return (self.a1*np.sin(phase)+self.a2*np.sin(2.*phase+.4)
                +self.ctheta*np.sin(2.*rotation_vector(quaternion))
                +self.comega*np.tanh(omega/.05))

    def derivative(self, quaternion, omega, tau, phase):
        qdot = .5*multiply(quaternion, np.column_stack((np.zeros(self.n), omega)))
        applied = tau+self.disturbance(quaternion, omega, phase)
        wdot = (applied-np.cross(omega, INERTIA*omega))/INERTIA
        return qdot, wdot

    def step(self, commands):
        requested = np.asarray(commands, float)
        if requested.shape != (self.n, 3) or not np.all(np.isfinite(requested)):
            raise ValueError('commands must be finite (n,3)')
        active = ~self.failed & (self.count < STEPS)
        command = np.where(active[:, None], np.clip(requested, -LIMIT, LIMIT), 0.)
        q0, w0, tau0 = self.quaternion.copy(), self.omega.copy(), self.tau.copy()
        phase0 = self.phase.copy()
        delta = tau0-command
        h = DT/SUBSTEPS
        q, w = q0.copy(), w0.copy()
        def slopes(at, xq, xw):
            tau = command+delta*np.exp(-at/LAG)
            return self.derivative(xq, xw, tau, phase0+at*self.frequency)
        for index in range(SUBSTEPS):
            t = index*h
            a, b = slopes(t, q, w)
            c, d = slopes(t+h/2., q+h/2.*a, w+h/2.*b)
            e, f = slopes(t+h/2., q+h/2.*c, w+h/2.*d)
            g, j = slopes(t+h, q+h*e, w+h*f)
            q = normalize(q+h/6.*(a+2.*c+2.*e+g))
            w = w+h/6.*(b+2.*d+2.*f+j)
        self.quaternion[active] = q[active]
        self.omega[active] = w[active]
        tau_end = command+delta*np.exp(-DT/LAG)
        self.tau[:] = np.where(active[:, None], tau_end, 0.)
        mean_exp = -np.expm1(-DT/LAG)/(DT/LAG)
        variance_exp = -np.expm1(-2.*DT/LAG)/(2.*DT/LAG)-mean_exp**2
        mean_torque = command+delta*mean_exp
        interval_cost = DT*np.sum(mean_torque**2+delta**2*variance_exp, axis=1)
        interval_cost[~active] = 0.
        self.phase[active] = (phase0[active]+DT*self.frequency[active])%(2.*np.pi)
        new_failure = active & ((np.linalg.norm(self.theta, axis=1) > np.deg2rad(90.))
                                | (np.linalg.norm(self.omega, axis=1) > np.deg2rad(60.)))
        self.failed |= new_failure
        self.count = np.minimum(self.count+1, STEPS)
        return dict(command=command, tau_start=np.where(active[:, None], tau0, 0.),
                    tau_end=np.where(active[:, None], tau_end, 0.),
                    actual_cost=interval_cost, command_cost=DT*np.sum(command**2, axis=1),
                    new_failure=new_failure, done=self.count >= STEPS)
