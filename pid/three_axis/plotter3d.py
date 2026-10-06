"""
绘制三轴仿真结果：欧拉角、角速度、力矩。
"""
import matplotlib.pyplot as plt

def plot_results_3d(data):
    """
    data: run_simulation_3d 返回的字典
    """
    time = data['time']

    # 创建子图布局：3行2列（欧拉角+角速度+力矩）
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # 第一列：欧拉角
    axes[0, 0].plot(time, data['roll'], 'b-', label='Roll')
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[1, 0].plot(time, data['pitch'], 'g-', label='Pitch')
    axes[1, 0].set_ylabel('Pitch (deg)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[2, 0].plot(time, data['yaw'], 'r-', label='Yaw')
    axes[2, 0].set_ylabel('Yaw (deg)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    # 第二列：角速度
    axes[0, 1].plot(time, data['omega_x'], 'b-', label='ωx')
    axes[0, 1].set_ylabel('ωx (deg/s)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 1].plot(time, data['omega_y'], 'g-', label='ωy')
    axes[1, 1].set_ylabel('ωy (deg/s)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    axes[2, 1].plot(time, data['omega_z'], 'r-', label='ωz')
    axes[2, 1].set_ylabel('ωz (deg/s)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_torque_3d(data):
    """
    单独绘制力矩曲线（指令力矩 vs 实际力矩）。
    """
    time = data['time']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time, data['torque_cmd'][:,0], 'b--', label='Cmd Tx')
    axes[0].plot(time, data['torque_out'][:,0], 'r-', label='Actual Tx')
    axes[0].set_ylabel('Tx (Nm)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, data['torque_cmd'][:,1], 'b--', label='Cmd Ty')
    axes[1].plot(time, data['torque_out'][:,1], 'r-', label='Actual Ty')
    axes[1].set_ylabel('Ty (Nm)')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time, data['torque_cmd'][:,2], 'b--', label='Cmd Tz')
    axes[2].plot(time, data['torque_out'][:,2], 'r-', label='Actual Tz')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Tz (Nm)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()