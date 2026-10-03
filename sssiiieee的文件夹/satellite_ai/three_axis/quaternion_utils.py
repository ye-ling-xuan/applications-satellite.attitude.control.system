"""
四元数基本运算与坐标变换工具（satellite_ai 复制版）

ds13: 从 satellite1PID/week5：三轴卫星姿态控制/quaternion_utils.py 复制而来，
      数学逻辑未做任何改动，仅精简注释，用于在 satellite_ai 内保持三轴模块自包含。
"""

import numpy as np


def quat_multiply(q1, q2):
    """四元数乘法 q1 ⊗ q2，[w, x, y, z] 顺序。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    """四元数共轭（单位四元数下即逆）。"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    """归一化四元数。"""
    return q / np.linalg.norm(q)


def euler_to_quaternion(roll, pitch, yaw):
    """欧拉角（弧度，ZYX 顺序）转四元数 [w, x, y, z]。"""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quaternion_to_euler(q):
    """四元数转欧拉角（弧度，ZYX 顺序），返回 (roll, pitch, yaw)。"""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quaternion_error(q_desired, q_current):
    """误差四元数 q_err = q_desired ⊗ q_current⁻¹ 的等效轴角向量（弧度，3 维）。"""
    q_err = quat_multiply(q_desired, quat_conjugate(q_current))
    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis
