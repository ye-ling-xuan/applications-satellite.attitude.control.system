import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ========================= 四元数工具函数 =========================
def quat_multiply(q1, q2):
    """四元数乘法：q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    """四元数共轭（逆）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    """归一化四元数"""
    return q / np.linalg.norm(q)

def quat_to_euler(q):
    """四元数转欧拉角（ZYX 顺序，即偏航-俯仰-滚转）"""
    w, x, y, z = q
    # 滚转 (绕X)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # 俯仰 (绕Y)
    sinp = 2 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    # 偏航 (绕Z)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def axis_angle_to_quaternion(axis, angle):
    """轴-角表示转四元数 (axis: 单位向量, angle: 弧度)"""
    half = angle / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), s*axis[0], s*axis[1], s*axis[2]])

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵 (用于3D可视化)"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

# ========================= 卫星动力学类 =========================
class Satellite:
    """
    刚体卫星，使用四元数姿态，考虑完整动力学和运动学
    """
    def __init__(self, I=np.diag([1.2, 1.0, 0.8])):
        """
        I: 转动惯量矩阵 (假设与主轴对齐，所以是对角矩阵)
        """
        self.I = I                      # 惯量张量
        self.I_inv = np.linalg.inv(I)   # 逆惯量
        self.q = np.array([1.0, 0.0, 0.0, 0.0])   # 初始姿态（无偏转）
        self.omega = np.zeros(3)                   # 初始角速度 (rad/s)

    def set_initial_attitude(self, roll_deg, pitch_deg, yaw_deg):
        """使用欧拉角（度）设置初始姿态（顺序：ZYX）"""
        # 分别构造三个基本旋转的四元数，并按 Z→Y→X 顺序复合
        q_yaw   = axis_angle_to_quaternion([0,0,1], np.radians(yaw_deg))
        q_pitch = axis_angle_to_quaternion([0,1,0], np.radians(pitch_deg))
        q_roll  = axis_angle_to_quaternion([1,0,0], np.radians(roll_deg))
        # 注意顺序：先绕Z，再绕Y，最后绕X -> q = q_roll ⊗ q_pitch ⊗ q_yaw
        self.q = quat_multiply(q_roll, quat_multiply(q_pitch, q_yaw))

    def set_initial_omega(self, omega_deg_s):
        """设置初始角速度，输入为 [ωx, ωy, ωz] 度/秒"""
        self.omega = np.radians(omega_deg_s)

    def dynamics(self, torque):
        """
        欧拉动力学方程：计算角加速度
        τ = I·ω̇ + ω × (I·ω)  =>  ω̇ = I⁻¹ (τ - ω × Iω)
        """
        I_omega = self.I @ self.omega               # I · ω
        gyro = np.cross(self.omega, I_omega)        # ω × (I·ω)
        omega_dot = self.I_inv @ (torque - gyro)    # 角加速度
        return omega_dot

    def kinematics(self):
        """
        四元数运动学方程：q̇ = 1/2 q ⊗ (0, ω)
        返回四元数导数
        """
        w, x, y, z = self.q
        wx, wy, wz = self.omega
        # 直接使用导出公式，避免重复计算四元数乘法
        q_dot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
             w*wx + y*wz - z*wy,
             w*wy - x*wz + z*wx,
             w*wz + x*wy - y*wx
        ])
        return q_dot

    def update(self, torque, dt):
        """
        欧拉法积分一步
        torque: 当前施加的控制力矩 (3,)
        dt: 时间步长 (s)
        """
        # 1. 更新角速度
        omega_dot = self.dynamics(torque)
        self.omega += omega_dot * dt

        # 2. 更新四元数
        q_dot = self.kinematics()
        self.q += q_dot * dt

        # 3. 归一化，防止数值误差导致模长偏离1
        self.q = quat_normalize(self.q)

        return self.q, self.omega

# ========================= PID 控制器 =========================
def quaternion_error(q_target, q_current):
    """计算误差四元数：q_error = q_target ⊗ q_current⁻¹"""
    q_current_conj = quat_conjugate(q_current)
    return quat_multiply(q_target, q_current_conj)

class QuaternionPID:
    """
    基于四元数误差的PID控制器（连续型，输出力矩）
    只考虑姿态误差，忽略角速度误差（即简单P控制，也可扩展）
    """
    def __init__(self, Kp=10.0, Ki=0.0, Kd=2.0, max_torque=2.0, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_torque = max_torque
        self.dt = dt
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute(self, q_target, q_current, omega_current):
        """
        输入目标四元数和当前四元数，输出控制力矩（在本体系下）
        使用四元数误差的向量部分作为比例误差
        """
        # 误差四元数
        q_err = quaternion_error(q_target, q_current)
        # 提取向量部分（x,y,z）作为姿态误差（对于小角度近似）
        error_xyz = q_err[1:4]      # 2*sin(θ/2)*u，约为θ*u
        # 乘以2使其近似等于角度误差（对于小角度）
        error_vec = 2 * error_xyz

        # 积分项
        self.integral += error_vec * self.dt
        # 微分项：这里使用角速度的负值作为微分项（更常见的做法）
        derivative = -omega_current   # 因为角速度就是误差的变化率
        # 也可以直接用角速度，但符号视控制律而定

        # PID输出力矩
        torque = (self.Kp * error_vec +
                  self.Ki * self.integral +
                  self.Kd * derivative)

        # 力矩限幅
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        return torque

# ========================= 仿真主程序 =========================
def run_simulation():
    # 仿真参数
    dt = 0.01
    duration = 10.0
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)

    # 创建卫星
    sat = Satellite()
    # 初始姿态：滚转30度，俯仰20度，偏航10度
    sat.set_initial_attitude(30, 20, 10)
    # 初始角速度：零
    sat.set_initial_omega([0, 0, 0])

    # 目标姿态：零
    q_target = np.array([1.0, 0.0, 0.0, 0.0])

    # PID控制器
    pid = QuaternionPID(Kp=8.0, Ki=0.5, Kd=2.0, max_torque=2.0, dt=dt)

    # 数据记录
    euler_deg = np.zeros((steps, 3))
    torque_log = np.zeros((steps, 3))
    omega_deg = np.zeros((steps, 3))
    q_log = np.zeros((steps, 4))

    # 主循环
    for i in range(steps):
        # 记录数据
        euler_rad = quat_to_euler(sat.q)
        euler_deg[i] = np.degrees(euler_rad)
        omega_deg[i] = np.degrees(sat.omega)
        q_log[i] = sat.q

        # 计算控制力矩
        torque = pid.compute(q_target, sat.q, sat.omega)
        torque_log[i] = torque

        # 更新卫星状态
        sat.update(torque, dt)

    # 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    # 左列：欧拉角响应
    axes[0,0].plot(time, euler_deg[:,0], 'r', label='Roll')
    axes[0,0].set_ylabel('Roll (deg)')
    axes[0,0].grid(True); axes[0,0].legend()
    axes[1,0].plot(time, euler_deg[:,1], 'g', label='Pitch')
    axes[1,0].set_ylabel('Pitch (deg)')
    axes[1,0].grid(True); axes[1,0].legend()
    axes[2,0].plot(time, euler_deg[:,2], 'b', label='Yaw')
    axes[2,0].set_ylabel('Yaw (deg)')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].grid(True); axes[2,0].legend()
    axes[0,0].set_title('Euler angles')

    # 右列：角速度
    axes[0,1].plot(time, omega_deg[:,0], 'r', label='ωx')
    axes[0,1].set_ylabel('ωx (deg/s)')
    axes[0,1].grid(True); axes[0,1].legend()
    axes[1,1].plot(time, omega_deg[:,1], 'g', label='ωy')
    axes[1,1].set_ylabel('ωy (deg/s)')
    axes[1,1].grid(True); axes[1,1].legend()
    axes[2,1].plot(time, omega_deg[:,2], 'b', label='ωz')
    axes[2,1].set_ylabel('ωz (deg/s)')
    axes[2,1].set_xlabel('Time (s)')
    axes[2,1].grid(True); axes[2,1].legend()
    axes[0,1].set_title('Angular velocity')

    plt.tight_layout()
    plt.savefig('attitude_response.png', dpi=150)
    plt.show()

    # 可选：3D动画显示卫星本体轴指向
    animate_3d(q_log, dt)

def animate_3d(q_log, dt):
    """创建3D动画，显示卫星本体坐标系（三轴）随时间旋转"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Satellite Body Axes (Quaternion Control)')

    # 初始化三个轴线
    origin = np.array([0, 0, 0])
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    lines = []
    for col in colors:
        line, = ax.plot([], [], [], color=col, linewidth=3)
        lines.append(line)
    ax.legend(lines, labels)

    def update(frame):
        q = q_log[frame]
        R = quaternion_to_rotation_matrix(q)
        # 三个轴在本体系中的方向 (单位向量在惯性系中的表示)
        x_axis = R @ np.array([1, 0, 0])
        y_axis = R @ np.array([0, 1, 0])
        z_axis = R @ np.array([0, 0, 1])
        axes_ends = [x_axis, y_axis, z_axis]
        for i, line in enumerate(lines):
            end = axes_ends[i]
            line.set_data([origin[0], end[0]], [origin[1], end[1]])
            line.set_3d_properties([origin[2], end[2]])
        return lines

    anim = animation.FuncAnimation(fig, update, frames=len(q_log), interval=dt*1000, blit=False)
    # 保存为gif（可选，需安装pillow）
    anim.save('satellite_rotation.gif', writer='pillow', fps=20)
    #plt.show()

if __name__ == "__main__":
    run_simulation()