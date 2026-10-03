"""
三轴刚体卫星动力学模型（satellite_ai 复制版）

ds13: 从 satellite1PID/week5：三轴卫星姿态控制/satellite3d.py 复制而来，动力学未改动：
      动力学  I·ω̇ + ω×(Iω) = τ
      运动学  q̇ = 0.5·q⊗(0,ω)，欧拉积分 + 四元数归一化
      默认转动惯量 diag([1.2, 1.0, 0.8]) 与 PID 三轴侧一致，保证对比公平。
"""

import numpy as np
from three_axis.quaternion_utils import quaternion_to_euler


class Satellite3D:
    """三轴刚体卫星，四元数描述姿态。"""

    def __init__(self, I=None):
        if I is None:
            I = np.diag([1.2, 1.0, 0.8])
        self.I = I
        self.I_inv = np.linalg.inv(I)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数（零姿态）
        self.omega = np.zeros(3)                  # 角速度 (rad/s)

    def set_state(self, q=None, omega=None):
        """设置初始状态（四元数会被归一化，角速度单位 rad/s）。"""
        if q is not None:
            self.q = q / np.linalg.norm(q)
        if omega is not None:
            self.omega = omega.copy()

    def dynamics(self, torque):
        """欧拉动力学：ω̇ = I⁻¹(τ - ω×(Iω))，返回角加速度 (3,)。"""
        I_omega = self.I @ self.omega
        return self.I_inv @ (torque - np.cross(self.omega, I_omega))

    def kinematics(self):
        """四元数运动学：dq/dt = 0.5·q⊗(0,ω)，返回 q_dot (4,)。"""
        w, x, y, z = self.q
        wx, wy, wz = self.omega
        q_dot = 0.5 * np.array([
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy - x * wz + z * wx,
            w * wz + x * wy - y * wx,
        ])
        return q_dot

    def update(self, torque, dt):
        """欧拉积分更新状态（角速度 + 四元数，含四元数归一化）。"""
        self.omega += self.dynamics(torque) * dt
        self.q += self.kinematics() * dt
        self.q = self.q / np.linalg.norm(self.q)

    def get_euler_deg(self):
        """返回欧拉角 (roll, pitch, yaw)，单位度。"""
        return np.degrees(quaternion_to_euler(self.q))

    def get_omega_deg(self):
        """返回角速度 (deg/s)。"""
        return np.degrees(self.omega)
