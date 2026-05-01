'''
仿真引擎是整个卫星姿态控制系统的“总控中心”。
它负责按时间顺序调用控制器和卫星动力学模型，记录数据，最终输出系统的响应曲线。
没有仿真引擎，控制器和卫星模型只是孤立的部件，无法协同工作。

仿真引擎的核心任务
时间推进：从t=0 到 t=tend，以固定步长 Δt逐步前进。
在每个时间步：
1.获取卫星当前状态（角度、角速度）。
2.调用控制器计算当前时刻需要的控制力矩。
3.将该力矩传递给卫星动力学模型，更新卫星状态到下一时刻。
4.记录所有感兴趣的数据（时间、状态、力矩、误差等）。
5.输出：完整的时间序列数据，供可视化与性能分析。
'''
import numpy as np
from satellite import Satellite
from controller import PIDController

#def 函数名称（参数列表）：函数体
#return返回结果
'''
sat：（satellite类的一个形参）卫星实例，包含当前状态（角度、角速度）和 apply_torque 方法。
pid：（pidcontroller类的一个形参）PID控制器实例，包含 compute 方法和内部状态（积分、上一误差）。
target_deg：期望角度（度），作为PID的设定值。
duration：仿真总时长（秒）。
dt：时间步长（秒），默认0.01秒。
disturbance_func: 函数，输入时间 t（秒），输出干扰力矩（N·m），默认为 None
noise_std_deg: 角度测量噪声标准差（度），默认 0.0（无噪声）
max_torque: 执行器最大输出力矩（N·m），默认 None（不限幅）
deadzone: 执行器死区阈值（N·m），绝对值小于此值的力矩输出 0，默认 0.0
'''
def run_simulation(sat, pid, target_deg, duration, dt=0.01,
                   disturbance_func=None,
                   noise_std_deg=0.0,
                   max_torque=None,
                   deadzone=0.0):
    steps = int(duration / dt)   #总步数
    #这 7 行是 NumPy 里创建全零数组的核心语法
    #专门用来提前开辟内存、存储仿真过程中的数据（时间、角度、角速度、力矩）
    time = np.zeros(steps)
    angle_true = np.zeros(steps)
    angle_meas = np.zeros(steps)
    omega_true = np.zeros(steps)
    torque_cmd = np.zeros(steps)
    torque_out = np.zeros(steps)
    disturbance_hist = np.zeros(steps)
    
# 重置 PID 内部状态（避免多次仿真时积分残留）
    pid.integral = 0.0
    pid.prev_error = 0.0

    for i in range(steps):
        t = i * dt
        time[i] = t  #记录当前时间

        # 获取真实状态（无噪声）
        true_angle = sat.get_angle_deg()
        true_omega = sat.get_omega_deg()
        angle_true[i] = true_angle
        omega_true[i] = true_omega

        # 模拟传感器噪声：测量值 = 真实值 + 高斯噪声
        measured_angle = true_angle + np.random.normal(0, noise_std_deg)
        angle_meas[i] = measured_angle

        
        # 控制器计算力矩（使用带噪声的测量值）
        u_cmd = pid.compute(target_deg, measured_angle)
        torque_cmd[i] = u_cmd

        # 执行器模型：死区 + 饱和
        u_out = u_cmd
        if deadzone > 0 and abs(u_out) < deadzone:
            u_out = 0.0
        if max_torque is not None:
            u_out = np.clip(u_out, -max_torque, max_torque)
        torque_out[i] = u_out

        # 计算当前时刻的干扰力矩
        d = disturbance_func(t) if disturbance_func else 0.0
        disturbance_hist[i] = d

        # 更新卫星动力学（控制力矩 + 干扰力矩）
        sat.apply_torque(u_out, dt, disturbance=d)
    '''
    返回字典，包含：
    time: 时间数组
    angle_true: 真实角度（度）
    angle_meas: 测量角度（度，带噪声）
    omega_true: 真实角速度（度/秒）
    torque_cmd: PID 计算出的原始力矩（N·m）
    torque_out: 经过饱和/死区后的实际力矩（N·m）
    disturbance: 施加的干扰力矩（N·m）
    '''
    #可以用 results['time']、results['angle'] 等访问数据。
    return {
        'time': time,
        'angle_true': angle_true,
        'angle_meas': angle_meas,
        'omega_true': omega_true,
        'torque_cmd': torque_cmd,
        'torque_out': torque_out,
        'disturbance': disturbance_hist
    }