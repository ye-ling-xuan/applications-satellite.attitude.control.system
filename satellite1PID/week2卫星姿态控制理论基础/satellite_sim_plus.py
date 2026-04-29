import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ==================== Quaternion Utilities ====================
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

# ==================== Satellite Dynamics ====================
class Satellite:
    def __init__(self):
        self.I = np.diag([1.2, 1.0, 0.8])
        self.I_inv = np.linalg.inv(self.I)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.array([0.1, -0.05, 0.08])

    def dynamics(self, torque):
        Iw = self.I @ self.omega
        return self.I_inv @ (torque - np.cross(self.omega, Iw))

    def kinematics(self):
        omega_quat = np.array([0, *self.omega])
        return 0.5 * quat_multiply(self.q, omega_quat)

    def step(self, torque, dt):
        # RK4 integration
        def f(state):
            q, w = state[:4], state[4:]
            self.q, self.omega = q, w
            dq = self.kinematics()
            dw = self.dynamics(torque)
            return np.concatenate([dq, dw])

        state = np.concatenate([self.q, self.omega])
        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)

        state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.q = quat_normalize(state[:4])
        self.omega = state[4:]

# ==================== Quaternion Controller ====================
class AttitudeController:
    def __init__(self):
        self.Kp = 8.0
        self.Kd = 2.5

    def compute(self, q, omega, q_target):
        q_inv = quat_conjugate(q)
        q_err = quat_multiply(q_target, q_inv)
        q_err_vec = q_err[1:]
        return -self.Kp * q_err_vec - self.Kd * omega

# ==================== Simulation ====================
def simulate():
    sat = Satellite()
    ctrl = AttitudeController()

    q_target = np.array([1, 0, 0, 0])

    dt = 0.02
    T = 10
    steps = int(T/dt)

    history_q = []

    for _ in range(steps):
        torque = ctrl.compute(sat.q, sat.omega, q_target)
        sat.step(torque, dt)
        history_q.append(sat.q.copy())

    return np.array(history_q)

# ==================== 3D Animation ====================
def animate(history_q):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.cla()
        R = quat_to_rotation_matrix(history_q[i])

        origin = np.array([0,0,0])
        axes = np.eye(3)
        colors = ['r','g','b']

        for j in range(3):
            vec = R @ axes[j]
            ax.quiver(*origin, *vec)

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_title(f"Step {i}")

    ani = animation.FuncAnimation(fig, update, frames=len(history_q), interval=50)
    plt.show()

# ==================== Main ====================
if __name__ == '__main__':
    history_q = simulate()
    animate(history_q)
