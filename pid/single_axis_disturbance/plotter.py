#这个函数用于将仿真引擎返回的数据（时间、角度、角速度、力矩）绘制成三个垂直排列的子图
#直观展示控制系统的动态响应。
import matplotlib.pyplot as plt

def plot_results(data):
    """
    绘制仿真结果，适配新版仿真引擎返回的数据。
    
    新版数据字典应包含以下键（至少包含前5个）：
        'time'           : 时间数组
        'angle_true'     : 真实角度 (deg)
        'angle_meas'     : 测量角度 (deg) —— 如果存在传感器噪声
        'omega_true'     : 真实角速度 (deg/s)
        'torque_out'     : 实际作用在卫星上的力矩 (Nm)（经过饱和/死区）
        'torque_cmd'     : PID 原始指令力矩 (Nm)（可选）
        'disturbance'    : 外部干扰力矩 (Nm)（可选）
    """
    time = data['time']
    
    # 判断是否存在测量角度和指令力矩，决定子图布局和显示内容
    has_meas = 'angle_meas' in data
    has_cmd = 'torque_cmd' in data
    has_dist = 'disturbance' in data
    
    # 确定子图数量：角度、角速度、力矩，以及可能的干扰图
    n_plots = 3 + (1 if has_dist else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots), sharex=True)
    # 如果只有一个子图，axes 不是列表，需要统一处理
    if n_plots == 1:
        axes = [axes]
    
    # ------ 子图1：角度 ------
    ax = axes[0]
    ax.plot(time, data['angle_true'], 'b-', linewidth=1.5, label='True angle')
    if has_meas:
        ax.plot(time, data['angle_meas'], 'r--', linewidth=1, alpha=0.7, label='Measured angle')
        ax.legend()
    ax.set_ylabel('Angle (deg)')
    ax.grid(True)
    
    # ------ 子图2：角速度 ------
    ax = axes[1]
    ax.plot(time, data['omega_true'], 'g-', linewidth=1.5)
    ax.set_ylabel('Angular velocity (deg/s)')
    ax.grid(True)
    
    # ------ 子图3：力矩 ------
    ax = axes[2]
    ax.plot(time, data['torque_out'], 'r-', linewidth=1.5, label='Actual torque')
    if has_cmd:
        ax.plot(time, data['torque_cmd'], 'b--', linewidth=1, alpha=0.6, label='Command torque')
        ax.legend()
    ax.set_ylabel('Torque (Nm)')
    ax.grid(True)
    
    # ------ 子图4：干扰（如果存在）------
    if has_dist:
        ax = axes[3]
        ax.plot(time, data['disturbance'], 'm-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Disturbance (Nm)')
        ax.grid(True)
    else:
        axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()