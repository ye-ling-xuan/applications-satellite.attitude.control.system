"""
三轴卫星姿态动力学模型，使用四元数表示姿态。
支持欧拉积分、干扰力矩、状态获取。
"""
import numpy as np
from quaternion_utils import quaternion_to_euler

class Satellite3D:
    """
    三轴刚体卫星，动力学方程为：
    I * dω/dt + ω × (I ω) = τ
    运动学方程：dq/dt = 0.5 * q ⊗ (0, ω)
    四元数形式的姿态运动学方程，描述了「四元数随时间的变化率（姿态变化快慢）」和「刚体的瞬时角速度」之间的关系
    (0, ω)是「角速度四元数」：实部为 0，虚部是刚体的瞬时角速度向量 ω=(ω_x, ω_y, ω_z)\)（单位：rad/s），完整形式是 (0, ω_x, ω_y, ω_z)
    """

    def __init__(self, I=None):
        """
        参数:
            I: 3x3 转动惯量矩阵（对角阵或全矩阵）。
            默认 diag([1.2, 1.0, 0.8])
        """
        if I is None:
            I = np.diag([1.2, 1.0, 0.8])
        self.I = I                                #把转动惯量矩阵保存为实例属性
        self.I_inv = np.linalg.inv(I)             #计算并保存转动惯量矩阵的逆矩阵
        self.q = np.array([1.0, 0.0, 0.0, 0.0])   #单位四元数，对应刚体的初始姿态：没有任何旋转（与世界坐标系对齐）
        self.omega = np.zeros(3)                  #角速度 (rad/s)，表示初始角速度为 0，刚体处于静止状态


    def set_state(self, q=None, omega=None):
        """设置初始状态（四元数需归一化，角速度弧度/秒）"""
        if q is not None:
            self.q = q / np.linalg.norm(q)
        if omega is not None:
            '''
            omega.copy() 会创建一个全新的副本
            把传入的角速度值复制一份保存到 self.omega 中
            后续外部修改 omega 不会影响刚体的内部状态
            保证了状态的独立性和安全性
            '''
            self.omega = omega.copy()

    def dynamics(self, torque):
        """
        欧拉动力学方程，计算角加速度。
        torque: 三维总力矩 (N·m)
        ω˙=I−1 (M−ω×(Iω))
        返回: 角加速度 (rad/s²)
        """
        #@ 是 NumPy 中的矩阵乘法运算符
        #这里计算的是刚体的角动量 L = Iω。
        I_omega = self.I @ self.omega

        #实现了变形后的欧拉方程
        #ω˙=I−1 (M−ω×(Iω))
        #np.cross(...) 是计算向量叉乘（Cross Product） 的函数
        omega_dot = self.I_inv @ (torque - np.cross(self.omega, I_omega))
        return omega_dot

    def kinematics(self):
        """
        四元数运动学方程，计算四元数导数。
        返回: dq/dt (4,)
        """
        w, x, y, z = self.q
        wx, wy, wz = self.omega
        q_dot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
             w*wx + y*wz - z*wy,
             w*wy - x*wz + z*wx,
             w*wz + x*wy - y*wx
        ])
        return q_dot

    def update(self, torque, dt):
        """
        欧拉积分更新卫星状态。
        torque: 三维控制力矩 (N·m)
        dt: 时间步长 (s)
        """
        # 动力学
        omega_dot = self.dynamics(torque)
        self.omega += omega_dot * dt
        # 运动学
        q_dot = self.kinematics()
        self.q += q_dot * dt
        # 归一化四元数
        self.q = self.q / np.linalg.norm(self.q)

    def get_euler_deg(self):
        """返回当前欧拉角 (度) : (roll, pitch, yaw)"""
        roll, pitch, yaw = quaternion_to_euler(self.q)
        return np.degrees([roll, pitch, yaw])

    def get_omega_deg(self):
        """返回角速度 (度/秒)"""
        return np.degrees(self.omega)