"""
三轴卫星闭环仿真引擎，支持干扰力矩、传感器噪声、执行器饱和/死区。
返回完整的历史数据。
"""
import numpy as np
from quaternion_utils import quaternion_error

def run_simulation_3d(sat, pid, target_q, duration, dt=0.01,
                      disturbance_func=None,
                      noise_std_deg=0.0,
                      max_torque=None,
                      deadzone=0.0):
    """
    参数:
        sat: Satellite3D 实例
        pid: PID3D 实例
        target_q: 目标四元数 (4,)
        duration: 仿真时长 (s)
        dt: 步长 (s)
        disturbance_func: 函数 disturbance(t) 返回三维干扰力矩 (N·m)
        noise_std_deg: 角度测量噪声标准差 (度)，对误差向量添加噪声
        max_torque: 执行器饱和上限 (N·m)，标量或三维数组
        deadzone: 执行器死区 (N·m)，标量或三维数组
    返回:
        字典包含 time, euler_roll/pitch/yaw, omega_xyz,
        torque_cmd, torque_out, disturbance
    """
    steps = int(duration / dt)
    time = np.zeros(steps)

    # 欧拉角历史 (度)
    roll_hist = np.zeros(steps)
    pitch_hist = np.zeros(steps)
    yaw_hist = np.zeros(steps)

    # 角速度历史 (度/秒)
    omega_x = np.zeros(steps)
    omega_y = np.zeros(steps)
    omega_z = np.zeros(steps)

    # 力矩历史
    """
    torque_cmd_hist	记录 PID 控制器发出的控制指令力矩（每个时间步的三轴力矩指令）
    torque_out_hist	记录执行器实际输出的力矩（比如经过死区、限幅后的力矩，和指令可能不同）
    dist_hist	记录外部干扰力矩（比如风扰、模型误差带来的力矩，用于分析抗扰效果）
    """
    torque_cmd_hist = np.zeros((steps, 3))
    torque_out_hist = np.zeros((steps, 3))
    dist_hist = np.zeros((steps, 3))

    #重置 PID 控制器状态
    pid.reset()

    for i in range(steps):
        t = i * dt
        time[i] = t

        # 当前欧拉角和角速度（用于记录）
        roll, pitch, yaw = sat.get_euler_deg()
        roll_hist[i] = roll
        pitch_hist[i] = pitch
        yaw_hist[i] = yaw

        omega_deg = sat.get_omega_deg()
        omega_x[i], omega_y[i], omega_z[i] = omega_deg

        # 计算四元数误差向量（弧度）
        error_vec = quaternion_error(target_q, sat.q)

        # 模拟传感器噪声（对误差添加噪声）
        if noise_std_deg > 0:
            noise = np.radians(np.random.normal(0, noise_std_deg, size=3))
            error_vec += noise

        # PID 计算指令力矩
        u_cmd = pid.compute(error_vec)

        # 执行器模型：死区 + 饱和
        u_out = u_cmd.copy()
        # 死区
        if deadzone > 0:
            """
            这是一个参数兼容处理，让代码同时支持标量、列表、NumPy 数组三种输入方式：
            如果用户传入的是 list 或 np.ndarray（比如 deadzone=[0.1, 0.05, 0.1]），
            就用 np.asarray() 把它转换成 NumPy 数组；
            如果用户传入的是标量（比如 deadzone=0.1），就直接保留为标量；
            这样后续代码可以同时处理 “三轴共用死区” 和 “三轴独立死区” 两种场景。
            """
            dead = np.asarray(deadzone) if isinstance(deadzone, (list, np.ndarray)) else deadzone
            """
            标量死区模式（三轴共用同一个死区）
            np.isscalar(dead) 判断 dead 是否是标量，对应 “三轴共用同一个死区宽度” 的场景；
            np.abs(u_out) 计算控制指令每个分量的绝对值；
            u_out[np.abs(u_out) < dead] = 0.0 用布尔索引，一次性把所有绝对值小于死区宽度的分量设为 0。
            """
            if np.isscalar(dead):
                u_out[np.abs(u_out) < dead] = 0.0
            else:
                for ax in range(3):
                    if abs(u_out[ax]) < dead[ax]:
                        u_out[ax] = 0.0
            """
            数组死区模式（三轴独立死区）
            else 对应 dead 是 NumPy 数组的场景，也就是 “每个轴有独立的死区宽度”；
            for ax in range(3) 遍历三个轴（x/y/z）；
            abs(u_out[ax]) < dead[ax] 判断当前轴的指令是否小于该轴的死区宽度，如果是，就把指令置为 0。
            """
        # 饱和
        if max_torque is not None:
            limit = np.asarray(max_torque) if isinstance(max_torque, (list, np.ndarray)) else max_torque
            if np.isscalar(limit):
                u_out = np.clip(u_out, -limit, limit)
            else:
                u_out = np.clip(u_out, -limit, limit)

        torque_cmd_hist[i] = u_cmd
        torque_out_hist[i] = u_out

        # 干扰力矩
        if disturbance_func is not None:
            d = disturbance_func(t)
        else:
            d = np.zeros(3)
        dist_hist[i] = d

        # 总力矩 = 实际控制力矩 + 干扰
        total_torque = u_out + d
        sat.update(total_torque, dt)

    return {
        'time': time,
        'roll': roll_hist, 'pitch': pitch_hist, 'yaw': yaw_hist,
        'omega_x': omega_x, 'omega_y': omega_y, 'omega_z': omega_z,
        'torque_cmd': torque_cmd_hist,
        'torque_out': torque_out_hist,
        'disturbance': dist_hist
    }