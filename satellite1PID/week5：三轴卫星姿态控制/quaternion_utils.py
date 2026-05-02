"""
四元数基本运算与坐标变换工具。
"""
import numpy as np

def quat_multiply(q1, q2):
    """
    四元数乘法。
    参数:
        q1, q2: 形状 (4,) 的四元数 [w, x, y, z]
    返回:
        q1 ⊗ q2
    """
    #把传入的两个四元数的八个值提取出来
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    #返回四元数乘法结果
    '''
    核心作用是将四元数乘法计算得到的 4 个标量结果
    从 Python 原生列表转换为 NumPy 的核心数据结构 
    ——ndarray（多维数组）对象
    '''
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    """返回四元数的共轭（逆）。"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    """归一化四元数。"""
    return q / np.linalg.norm(q)  #np.linalg.norm(q)代表计算模长

def euler_to_quaternion(roll, pitch, yaw):
    """
    欧拉角（弧度，ZYX 顺序）转四元数。
    参数:
        roll, pitch, yaw: 弧度
    返回:
        四元数 [w, x, y, z]
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])

def quaternion_to_euler(q):
    """
    四元数转欧拉角（弧度，ZYX 顺序）。
    返回: (roll, pitch, yaw)
    """
    w, x, y, z = q
    # roll (x轴)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y轴)
    sinp = 2 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    # yaw (z轴)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def quaternion_error(q_desired, q_current):
    """
    计算误差四元数 q_err = q_desired ⊗ q_current⁻¹，
    误差姿态四元数 = 目标姿态四元数⊗当前姿态四元数的共轭（逆）	
    并转换为等效轴角表示的误差向量（用于 PID 控制）。
    返回: 三维误差向量 (弧度)，每个分量为该轴上的误差角。
    """
    q_conj = quat_conjugate(q_current)
    q_err = quat_multiply(q_desired, q_conj)
    # 误差角
    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis