"""Exact held-torque dynamics and continuous forbidden-arc intersection."""
from __future__ import annotations
import numpy as np
from . import protocol as p


def wrap(angle):
    return (np.asarray(angle)+np.pi)%(2*np.pi)-np.pi


def features(population):
    values = np.asarray(population, dtype=float)
    return np.column_stack((np.sin(values[:, 0]), np.cos(values[:, 0]),
                            np.sin(values[:, 1]), np.cos(values[:, 1]))).astype(np.float32)


def analytic_actions(population):
    values = np.asarray(population, dtype=float)
    # This oracle belongs exclusively to the disclosed classical baseline.
    return (wrap(values[:, 1]) > wrap(values[:, 0])).astype(np.int64)


def route_target(theta, goal, actions):
    positive = np.mod(goal-theta, 2*np.pi)
    negative = -np.mod(theta-goal, 2*np.pi)
    return theta+np.where(actions == 1, positive, negative)


def interval_intersects(theta, omega, torque, interval=p.DT):
    """Intersect a continuous quadratic's exact range with periodic closed arcs."""
    theta, omega, torque = np.broadcast_arrays(np.asarray(theta, float), np.asarray(omega, float), np.asarray(torque, float))
    acceleration = torque/p.INERTIA
    end = theta+omega*interval+.5*acceleration*interval**2
    minimum, maximum = np.minimum(theta, end), np.maximum(theta, end)
    turning = np.divide(-omega, acceleration, out=np.full(theta.shape, np.inf), where=acceleration != 0.)
    inside = (turning > 0.) & (turning < interval)
    vertex = theta+omega*np.where(inside, turning, 0.)+.5*acceleration*np.where(inside, turning, 0.)**2
    minimum = np.minimum(minimum, vertex)
    maximum = np.maximum(maximum, vertex)
    lower, upper = p.CENTER-p.HALF_WIDTH, p.CENTER+p.HALF_WIDTH
    first = np.ceil((minimum-upper-1e-12)/(2*np.pi))
    last = np.floor((maximum-lower+1e-12)/(2*np.pi))
    return first <= last


def summary(rows):
    n = len(rows)
    return dict(count=n, successes=sum(row['success'] for row in rows),
                collisions=sum(row['collision'] for row in rows),
                success_rate=sum(row['success'] for row in rows)/n,
                safety_rate=1-sum(row['collision'] for row in rows)/n,
                mean_torque_cost=float(np.mean([row['torque_cost'] for row in rows])),
                mean_success_cost=float(np.mean([row['torque_cost'] for row in rows if row['success']])) if any(row['success'] for row in rows) else None)


def simulate(scenarios, actions, parameters, keep_traces=False):
    """Actions -1=direct wrapped PID; 0/1=one committed learned/planned route.

    PID proposes a candidate integral, clips the total physical command, and
    rejects only an integral increment which deepens command saturation.
    A collision at any time inside an interval is a task failure, even if both
    sampled endpoints are outside the arc. The full failing interval is retained.
    """
    if isinstance(scenarios, np.ndarray):
        population = np.asarray(scenarios, dtype=float)
        identities = [str(i) for i in range(len(population))]
    else:
        population = np.array([[row['theta_rad'], row['target_rad']] for row in scenarios])
        identities = [row['scenario_id'] for row in scenarios]
    n = len(population)
    if not n:
        raise ValueError('no scenarios')
    action = np.broadcast_to(np.asarray(actions, dtype=int), (n,)).copy()
    if not np.all(np.isin(action, [-1, 0, 1])):
        raise ValueError('unknown route action')
    theta, goal = population[:, 0].copy(), population[:, 1].copy()
    omega, integral, costs = np.zeros(n), np.zeros(n), np.zeros(n)
    kp, kd, ki = [np.broadcast_to(np.asarray(parameters.get(key, 0.), float), (n,)) for key in ('kp', 'kd', 'ki')]
    lifted = route_target(theta, goal, action)
    failed = np.zeros(n, bool)
    contact_interval = np.full(n, -1, int)
    success = np.ones(n, bool)
    last_bad = np.zeros(n, int)
    initial_outside = np.abs(wrap(goal-theta)) > p.ANGLE_TOL
    last_bad[~initial_outside] = -1
    steps = int(round(p.DURATION/p.DT))
    active_intervals = np.zeros(n, int)
    traces = None
    if keep_traces:
        traces = dict(time=np.arange(steps+1)*p.DT, theta=np.empty((steps+1, n)), omega=np.empty((steps+1, n)),
                      collision=np.empty((steps+1, n), bool), command=np.empty((steps, n)),
                      interval_cost=np.empty((steps, n)), action=action.copy(), population=population.copy())
        traces['theta'][0], traces['omega'][0], traces['collision'][0] = theta, omega, failed
    for k in range(steps):
        active = ~failed
        error = np.where(action == -1, wrap(goal-theta), lifted-theta)
        proposed_integral = np.clip(integral+p.DT*error, -2., 2.)
        raw = kp*error-kd*omega+ki*proposed_integral
        command = np.where(active, np.clip(raw, -p.LIMIT, p.LIMIT), 0.)
        accept = (raw-command)*ki*(proposed_integral-integral) <= 1e-15
        integral = np.where(active & accept, proposed_integral, integral)
        crossing = active & interval_intersects(theta, omega, command)
        theta = np.where(active, theta+omega*p.DT+.5*command*p.DT**2/p.INERTIA, theta)
        omega = np.where(active, omega+command*p.DT/p.INERTIA, omega)
        interval_cost = command**2*p.DT
        costs += interval_cost
        active_intervals += active
        contact_interval[crossing] = k
        failed |= crossing
        outside = (np.abs(wrap(goal-theta)) > p.ANGLE_TOL) | (np.abs(omega) > p.RATE_TOL) | failed
        last_bad = np.where(outside, k+1, last_bad)
        if (k+1)*p.DT >= p.DEADLINE-1e-10:
            success &= ~outside
        if traces is not None:
            traces['theta'][k+1], traces['omega'][k+1], traces['collision'][k+1] = theta, omega, failed
            traces['command'][k], traces['interval_cost'][k] = command, interval_cost
    rows = [dict(scenario_id=identities[i], action=int(action[i]), success=bool(success[i] and not failed[i]),
                 collision=bool(failed[i]), collision_interval=int(contact_interval[i]),
                 torque_cost=float(costs[i]), physical_intervals=int(active_intervals[i]),
                 settling_time_s=float(max(0, last_bad[i]+1)*p.DT) if success[i] and not failed[i] else None,
                 final_error_deg=float(np.rad2deg(abs(wrap(goal[i]-theta[i])))),
                 final_rate_deg_s=float(np.rad2deg(abs(omega[i]))),
                 final_theta_rad=float(theta[i])) for i in range(n)]
    return dict(rows=rows, traces=traces)


def rewards(rows):
    # The simulator's physical outcomes are the only reward source. No correct
    # direction label is ever supplied to the DQN/replay buffer.
    return np.array([(-10. if row['collision'] else 10. if row['success'] else -2.)
                     -.1*row['torque_cost']/(p.LIMIT**2*p.DURATION)
                     -.05*((row['settling_time_s'] or p.DURATION)/p.DURATION)
                     for row in rows], dtype=np.float32)
