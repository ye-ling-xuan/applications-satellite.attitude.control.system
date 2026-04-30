import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ---------- 四元数工具函数 ----------
def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
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

def quat_to_euler(q):
    w,x,y,z = q
    roll = np.arctan2(2*(w*x + y*z), 1-2*(x*x + y*y))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1-2*(y*y + z*z))
    return np.array([roll, pitch, yaw])

def axis_angle_to_quaternion(axis, angle):
    half = angle/2.0; s = np.sin(half)
    return np.array([np.cos(half), s*axis[0], s*axis[1], s*axis[2]])

def quaternion_to_rotation_matrix(q):
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

# ---------- 卫星动力学类 ----------
class Satellite:
    def __init__(self, I=np.diag([1.2, 1.0, 0.8])):
        self.I = I
        self.I_inv = np.linalg.inv(I)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)

    def set_initial_attitude(self, roll_deg, pitch_deg, yaw_deg):
        q_yaw = axis_angle_to_quaternion([0,0,1], np.radians(yaw_deg))
        q_pitch = axis_angle_to_quaternion([0,1,0], np.radians(pitch_deg))
        q_roll = axis_angle_to_quaternion([1,0,0], np.radians(roll_deg))
        self.q = quat_multiply(q_roll, quat_multiply(q_pitch, q_yaw))

    def set_initial_omega(self, omega_deg_s):
        self.omega = np.radians(omega_deg_s)

    def dynamics(self, torque):
        I_omega = self.I @ self.omega
        omega_dot = self.I_inv @ (torque - np.cross(self.omega, I_omega))
        return omega_dot

    def kinematics(self):
        w,x,y,z = self.q
        wx,wy,wz = self.omega
        return 0.5 * np.array([
            -x*wx - y*wy - z*wz,
             w*wx + y*wz - z*wy,
             w*wy - x*wz + z*wx,
             w*wz + x*wy - y*wx
        ])

    def update(self, torque, dt):
        self.omega += self.dynamics(torque) * dt
        self.q += self.kinematics() * dt
        self.q = quat_normalize(self.q)
        return self.q, self.omega

# ---------- 四元数误差 PID ----------
def quaternion_error(q_target, q_current):
    qc_conj = quat_conjugate(q_current)
    return quat_multiply(q_target, qc_conj)

class QuaternionPID:
    def __init__(self, Kp=8.0, Ki=0.5, Kd=2.0, max_torque=2.0, dt=0.01):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.max_torque = max_torque
        self.dt = dt
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute(self, q_target, q_current, omega_current):
        q_err = quaternion_error(q_target, q_current)
        error_vec = 2 * q_err[1:4]
        self.integral += error_vec * self.dt
        derivative = -omega_current
        torque = (self.Kp * error_vec +
                  self.Ki * self.integral +
                  self.Kd * derivative)
        return np.clip(torque, -self.max_torque, self.max_torque)

# ---------- 3D 动画函数 ----------
def animate_3d(q_log, dt):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Satellite Body Axes (Quaternion Control)')

    origin = np.array([0, 0, 0])
    colors = ['r', 'g', 'b']
    lines = [ax.plot([], [], [], color=c, linewidth=3)[0] for c in colors]

    def update(frame):
        q = q_log[frame]
        R = quaternion_to_rotation_matrix(q)
        axes_end = [R @ np.array([1,0,0]), R @ np.array([0,1,0]), R @ np.array([0,0,1])]
        for i, line in enumerate(lines):
            end = axes_end[i]
            line.set_data([origin[0], end[0]], [origin[1], end[1]])
            line.set_3d_properties([origin[2], end[2]])
        return lines

    anim = animation.FuncAnimation(fig, update, frames=len(q_log), interval=dt*1000, blit=False)
    plt.show()
    return anim

# ---------- 仿真主程序 ----------
def run_simulation():
    dt = 0.01
    duration = 10.0
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)

    sat = Satellite()
    sat.set_initial_attitude(30, 20, 10)   # 初始滚转30°，俯仰20°，偏航10°
    sat.set_initial_omega([0, 0, 0])
    q_target = np.array([1.0, 0.0, 0.0, 0.0])

    pid = QuaternionPID(Kp=8.0, Ki=0.5, Kd=2.0, max_torque=2.0, dt=dt)

    q_log = np.zeros((steps, 4))

    for i in range(steps):
        q_log[i] = sat.q
        torque = pid.compute(q_target, sat.q, sat.omega)
        sat.update(torque, dt)

    # 只显示 3D 动画，不显示静态曲线
    animate_3d(q_log, dt)

if __name__ == "__main__":
    run_simulation()