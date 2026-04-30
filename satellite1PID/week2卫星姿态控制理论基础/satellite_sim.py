import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ==================== 四元数工具函数 ====================
def quat_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    """四元数共轭"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    """四元数归一化"""
    return q / np.linalg.norm(q)

def quat_to_euler(q):
    """四元数转欧拉角 (ZYX 顺序)"""
    w, x, y, z = q
    # 滚转 (x轴)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # 俯仰 (y轴)
    sinp = 2 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    # 偏航 (z轴)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def quat_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

# ==================== 卫星动力学类 ====================
class Satellite:
    """刚体卫星姿态动力学仿真器（四元数版本）"""
    
    def __init__(self, I=np.diag([1.2, 1.0, 0.8])):
        self.I = I
        self.I_inv = np.linalg.inv(I)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.array([0.0, 0.0, 0.0])
        self.time = 0.0
    
    def set_initial_attitude(self, roll_deg, pitch_deg, yaw_deg):
        """用欧拉角设置初始姿态"""
        roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y_val = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        self.q = quat_normalize(np.array([w, x, y_val, z]))
    
    def dynamics(self, torque):
        """欧拉动力学方程"""
        I_omega = self.I @ self.omega
        gyro = np.cross(self.omega, I_omega)
        return self.I_inv @ (torque - gyro)
    
    def kinematics(self):
        """四元数运动学方程"""
        w, x, y, z = self.q
        wx, wy, wz = self.omega
        return 0.5 * np.array([
            -x*wx - y*wy - z*wz,
             w*wx + y*wz - z*wy,
             w*wy - x*wz + z*wx,
             w*wz + x*wy - y*wx
        ])
    
    def update(self, torque, dt):
        """欧拉法积分一步"""
        self.omega += self.dynamics(torque) * dt
        self.q += self.kinematics() * dt
        self.q = quat_normalize(self.q)
        self.time += dt
        return self.q, self.omega

# ==================== PID 控制器 ====================
class PIDController:
    def __init__(self, Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01, max_torque=2.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt, self.max_torque = dt, max_torque
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        output = np.clip(output, -self.max_torque, self.max_torque)
        self.prev_error = error
        return output

# ==================== 仿真运行 ====================
def run_simulation(duration=8.0, dt=0.01, init_angle=30.0):
    sat = Satellite()
    sat.set_initial_attitude(init_angle, 0, 0)
    pid_x = PIDController(dt=dt); pid_y = PIDController(dt=dt); pid_z = PIDController(dt=dt)
    
    steps = int(duration / dt)
    time, euler, omega, torque = np.zeros(steps), np.zeros((steps,3)), np.zeros((steps,3)), np.zeros((steps,3))
    
    for i in range(steps):
        time[i] = i * dt
        euler_rad = quat_to_euler(sat.q)
        euler[i] = np.degrees(euler_rad)
        omega[i] = np.degrees(sat.omega)
        tau = np.array([pid_x.compute(0, euler_rad[0]), pid_y.compute(0, euler_rad[1]), pid_z.compute(0, euler_rad[2])])
        torque[i] = tau
        sat.update(tau, dt)
    
    return {'time': time, 'euler': euler, 'omega': omega, 'torque': torque, 'satellite': sat}

# ==================== 可视化 ====================
def plot_results(data):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    # 姿态角
    axes[0,0].plot(data['time'], data['euler'][:,0], 'r-'); axes[0,0].set_ylabel('Roll (deg)'); axes[0,0].grid(True)
    axes[1,0].plot(data['time'], data['euler'][:,1], 'g-'); axes[1,0].set_ylabel('Pitch (deg)'); axes[1,0].grid(True)
    axes[2,0].plot(data['time'], data['euler'][:,2], 'b-'); axes[2,0].set_ylabel('Yaw (deg)'); axes[2,0].set_xlabel('Time (s)'); axes[2,0].grid(True)
    # 角速度
    axes[0,1].plot(data['time'], data['omega'][:,0], 'r-'); axes[0,1].set_ylabel('ωx (deg/s)'); axes[0,1].grid(True)
    axes[1,1].plot(data['time'], data['omega'][:,1], 'g-'); axes[1,1].set_ylabel('ωy (deg/s)'); axes[1,1].grid(True)
    axes[2,1].plot(data['time'], data['omega'][:,2], 'b-'); axes[2,1].set_ylabel('ωz (deg/s)'); axes[2,1].set_xlabel('Time (s)'); axes[2,1].grid(True)
    plt.suptitle('Satellite Attitude Response (PID Control)')
    plt.tight_layout()
    plt.savefig('satellite_response.png', dpi=150)
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("卫星姿态动力学仿真运行中...")
    data = run_simulation(duration=8.0, dt=0.01, init_angle=30.0)
    plot_results(data)
    euler_final = data['euler'][-1]
    print(f"最终姿态角: Roll={euler_final[0]:.2f}°, Pitch={euler_final[1]:.2f}°, Yaw={euler_final[2]:.2f}°")
    print("仿真完成！结果已保存为 satellite_response.png")